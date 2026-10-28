import inspect
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from networks import Actor, Critic


class DDPGAgent:
    """同一物种共享一套 DDPG 网络，支持批量推理。"""

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=128,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.98,
        tau=0.01,
        device="cpu",
        fused_adam=False,
        cuda_graph=False,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        # fused Adam 仅在 CUDA 上启用；CPU 保留默认后端以兼容无 GPU 的运行环境。
        self.fused_adam = bool(fused_adam and self.device.type == "cuda")
        # 图执行需要 CUDA 和 fused Adam；不满足条件时提前报错，默认仍走普通更新。
        if cuda_graph and (not self.fused_adam or not torch.cuda.is_available()):
            raise ValueError("cuda_graph requires CUDA and fused_adam")
        self.cuda_graph = bool(cuda_graph)
        self._graph_runner = None
        self._graph_failed = False
        self.graph_capture_seconds = 0.0

        self.actor = Actor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 默认路径不传 fused 参数，兼容尚未提供该选项的 PyTorch 版本。
        optimizer_kwargs = {}
        if self.fused_adam:
            if "fused" not in inspect.signature(torch.optim.Adam).parameters:
                raise ValueError("This PyTorch version does not support --fused_adam; use --no_fused_adam.")
            optimizer_kwargs["fused"] = True
        if self.cuda_graph:
            optimizer_kwargs["capturable"] = True
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, **optimizer_kwargs)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, **optimizer_kwargs)

    @torch.no_grad()
    def act(self, obs, noise_std=0.0):
        return self.act_batch(np.asarray(obs)[None, :], noise_std=noise_std)[0]

    @torch.no_grad()
    def act_batch(self, observations, noise_std=0.0):
        """将 (个体数, 观测维度) 批量映射为同顺序的 (个体数, 动作维度)。"""
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        # 每个物种只做一次前向计算和结果回传，避免逐个体触发 GPU/CPU 同步。
        action = self.actor(obs_t).cpu().numpy()
        if noise_std > 0:
            action += np.random.normal(0.0, noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def state_dict(self):
        """同时保存在线网络、目标网络和优化器，保留继续更新所需的训练状态。"""
        state = {
            name: getattr(self, name).state_dict()
            for name in ("actor", "actor_target", "critic", "critic_target", "actor_opt", "critic_opt")
        }
        state.update(gamma=self.gamma, tau=self.tau)
        return state

    def load_state_dict(self, state_dict):
        self._invalidate_graph()
        for name in ("actor", "actor_target", "critic", "critic_target"):
            getattr(self, name).load_state_dict(state_dict[name])
        for name in ("actor_opt", "critic_opt"):
            optimizer_state = dict(state_dict[name])
            # 先按当前配置选择后端，再加载优化器；这样切换后端时，Adam 步数张量
            # 也会放在正确的设备上。复制参数组，避免改写传入的检查点配置。
            optimizer_state["param_groups"] = [
                dict(group, fused=True if self.fused_adam else None, foreach=None,
                     capturable=self.cuda_graph)
                for group in optimizer_state["param_groups"]
            ]
            getattr(self, name).load_state_dict(optimizer_state)
        self.gamma = state_dict["gamma"]
        self.tau = state_dict["tau"]

    def update(self, replay_buffer, batch_size, return_losses=True):
        batch = replay_buffer.sample(batch_size)
        if self.cuda_graph and not self._graph_failed:
            from cuda_graph import CudaGraphPreparationError, CudaGraphUpdate
            # 捕获图依赖固定输入布局，批次形状或类型改变时必须重新捕获。
            signature = tuple((array.shape, array.dtype.str) for array in batch)
            if self._graph_runner is None or self._graph_runner.signature != signature:
                self._invalidate_graph()
                try:
                    runner = CudaGraphUpdate(self, batch, signature)
                except CudaGraphPreparationError as error:
                    # 只有成功回滚的准备失败才能回退到普通更新。
                    # 回滚失败或回放失败必须向外抛出，避免重复更新参数。
                    self._graph_failed = True
                    for optimizer in (self.actor_opt, self.critic_opt):
                        for group in optimizer.param_groups:
                            group['capturable'] = False
                    warnings.warn("CUDA Graph preparation failed; restored training state and "
                                  "using eager updates: {}".format(error), RuntimeWarning)
                else:
                    self._graph_runner = runner
                    self.graph_capture_seconds += runner.capture_seconds
            if self._graph_runner is not None:
                losses = self._graph_runner.replay(batch)
                if return_losses:
                    return tuple(loss.item() for loss in losses)
                return None
        tensors = tuple(torch.as_tensor(array, dtype=torch.float32, device=self.device) for array in batch)
        losses = self._update_tensors(tensors)
        # .item() 会等待 CUDA 结果回传；不记录损失时可跳过，梯度更新不受影响。
        if return_losses:
            return tuple(loss.item() for loss in losses)

    def _invalidate_graph(self):
        """等待已有回放完成后丢弃图；加载权重或训练状态后应重新捕获。"""
        if self._graph_runner is not None:
            torch.cuda.synchronize(self.device)
        self._graph_runner = None
        self._graph_failed = False

    @property
    def cuda_graph_status(self):
        """区分关闭、待捕获、运行和回退状态，供日志及检查点元数据使用。"""
        if not self.cuda_graph:
            return 'disabled'
        if self._graph_failed:
            return 'fallback'
        return 'active' if self._graph_runner is not None else 'pending'

    def _update_tensors(self, batch):
        """共用同一套更新算式，供普通执行和 CUDA Graph 捕获调用。"""
        obs, action, reward, next_obs, done = batch

        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_action)
            y = reward + self.gamma * (1.0 - done) * target_q

        critic_loss = F.mse_loss(self.critic(obs, action), y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor 更新仍需经过 Critic 对动作求导，只跳过不使用的 Critic 参数梯度。
        # 临时冻结的标志在异常路径上也必须恢复，保证后续 Critic 仍可训练。
        critic_parameters = tuple(self.critic.parameters())
        requires_grad = tuple(parameter.requires_grad for parameter in critic_parameters)
        try:
            for parameter in critic_parameters:
                parameter.requires_grad_(False)
            actor_loss = -self.critic(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
        finally:
            for parameter, required in zip(critic_parameters, requires_grad):
                parameter.requires_grad_(required)

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        return actor_loss, critic_loss

    def _soft_update(self, target, source):
        if self.device.type == 'cuda':
            # 批量执行原有的乘、乘、加三步，保留运算顺序和逐步舍入结果。
            targets = [parameter.data for parameter in target.parameters()]
            sources = [parameter.data for parameter in source.parameters()]
            torch._foreach_mul_(targets, 1.0 - self.tau)
            scaled_sources = torch._foreach_mul(sources, self.tau)
            torch._foreach_add_(targets, scaled_sources)
            return
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * source_param.data)
