"""捕获一次 DDPG 更新；准备过程可回滚，图缓冲不写入检查点。"""
from copy import deepcopy
import time

import numpy as np
import torch


# cuBLAS 按流缓存工作区。同一设备上的两个物种和重复捕获共用此流，
# 避免每次批次形状变化时再保留一份工作区。
_CAPTURE_STREAMS = {}


class CudaGraphPreparationError(RuntimeError):
    """捕获失败，但训练状态已成功恢复，可安全回退到普通更新。"""


def _capture_stream(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _CAPTURE_STREAMS:
        _CAPTURE_STREAMS[index] = torch.cuda.Stream(device=device)
    return _CAPTURE_STREAMS[index]


class CudaGraphUpdate:
    """使用固定输入地址捕获更新，再通过替换输入数据重复执行。"""

    def __init__(self, agent, batch, signature):
        started = time.perf_counter()
        self.signature = signature
        try:
            self.inputs = tuple(torch.as_tensor(value, dtype=torch.float32, device=agent.device).clone()
                                for value in batch)
            self.graph = torch.cuda.CUDAGraph()
            # 准备时会试跑更新，先独立备份训练状态和随机状态，避免额外训练。
            snapshot = deepcopy(agent.state_dict())
            numpy_rng = np.random.get_state()
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all()
            # 图内核记录的是张量地址，正常恢复时必须保留现有张量对象。
            parameter_values = [(parameter, parameter.detach().clone())
                                for network in (agent.actor, agent.actor_target, agent.critic, agent.critic_target)
                                for parameter in network.parameters()]
            optimizer_values = [(optimizer, {parameter: deepcopy(state)
                                            for parameter, state in optimizer.state.items()})
                                for optimizer in (agent.actor_opt, agent.critic_opt)]
        except Exception as error:
            # 此阶段还未修改训练值或随机状态；可恢复的分配失败允许回退。
            # 同步若暴露不可恢复的 CUDA 错误，则继续向外抛出。
            torch.cuda.synchronize(agent.device)
            raise CudaGraphPreparationError(str(error)) from error

        def restore_values():
            """原地恢复参数和 Adam 状态，保持捕获图记录的地址有效。"""
            with torch.no_grad():
                for parameter, saved in parameter_values:
                    parameter.copy_(saved)
                for optimizer, saved_states in optimizer_values:
                    for parameter, live in optimizer.state.items():
                        saved = saved_states.get(parameter)
                        for key, value in live.items():
                            if torch.is_tensor(value):
                                if saved is None:
                                    value.zero_()  # 新建的 Adam 动量和步数应恢复到尚未更新的零值。
                                else:
                                    value.copy_(saved[key])
                            elif saved is not None:
                                live[key] = saved[key]

        try:
            stream = _capture_stream(agent.device)
            stream.wait_stream(torch.cuda.current_stream(agent.device))
            # 在旁路流中预热，完成 Adam 延迟状态和底层计算库的初始化。
            with torch.cuda.stream(stream):
                for _ in range(3):
                    agent._update_tensors(self.inputs)
            stream.synchronize()
            restore_values()
            stream.wait_stream(torch.cuda.current_stream(agent.device))
            # 捕获本身也会执行一次更新，结束后再次恢复数值，首次回放才算训练。
            with torch.cuda.graph(self.graph, stream=stream):
                self.losses = agent._update_tensors(self.inputs)
            stream.synchronize()
            restore_values()
            torch.cuda.synchronize(agent.device)
        except Exception as error:
            # 失败后不再复用图地址，可以重新加载完整状态。恢复前先等待残留工作结束；
            # 同步或恢复失败必须向外抛出，不能在状态不确定时重试更新。
            torch.cuda.synchronize(agent.device)
            agent.load_state_dict(snapshot)
            for network in (agent.actor, agent.critic):
                for parameter in network.parameters():
                    parameter.grad = None
            raise CudaGraphPreparationError(str(error)) from error
        finally:
            # 无论捕获是否成功，准备阶段都不能改变后续采样的随机序列。
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
        self.capture_seconds = time.perf_counter() - started

    def replay(self, batch):
        """原地填充固定输入并回放一次；回放错误由调用方直接向外抛出。"""
        for destination, source in zip(self.inputs, batch):
            destination.copy_(torch.as_tensor(source, dtype=torch.float32))
        self.graph.replay()
        return self.losses
