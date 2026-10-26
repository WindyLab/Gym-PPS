import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def state_dict(self):
        """保存有效样本及环形写入位置，空余容量无需写入检查点。

        张量与原 NumPy 数组共享内存，调用方应在下一次写入前完成序列化。
        使用张量还可使检查点兼容 torch.load(weights_only=True)。
        """
        return {
            "capacity": self.capacity,
            "ptr": self.ptr,
            "size": self.size,
            "arrays": {
                name: torch.from_numpy(getattr(self, name)[:self.size])
                for name in ("obs", "actions", "rewards", "next_obs", "dones")
            },
        }

    def load_state_dict(self, state_dict):
        # 容量决定环形覆盖顺序；强制保持一致，才能还原后续写入和采样的行为。
        if self.capacity != state_dict["capacity"]:
            raise ValueError("buffer_size must match the saved replay buffer capacity")
        size = state_dict["size"]
        for name in ("obs", "actions", "rewards", "next_obs", "dones"):
            destination = getattr(self, name)
            source = state_dict["arrays"][name].cpu().numpy()
            if source.shape != destination[:size].shape:
                raise ValueError("Saved replay buffer dimensions do not match the environment")
            destination[:size] = source
        self.ptr = state_dict["ptr"]
        self.size = size
