"""经验回放缓冲区。

循环缓冲区存储 (state, action, reward, next_state, done) 转移，
支持均匀随机采样和优先级采样。
"""

import random
from collections import deque, namedtuple

import numpy as np

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    """固定大小的循环经验回放缓冲区。"""

    def __init__(self, capacity: int = 50000):
        self._buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self._buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._buffer, min(batch_size, len(self._buffer)))

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """缓冲区内样本是否足够开始训练（至少 1 个 batch）。"""
        return len(self._buffer) >= 64
