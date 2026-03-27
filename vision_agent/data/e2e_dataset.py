"""端到端数据集：图像嵌入 + 动作标签。

数据格式（.npz）：
  embeddings: (N, 576) float32  — MobileNetV3 视觉嵌入
  labels:     (N,) int64        — 动作索引
  action_map: JSON string       — {"attack": 0, "retreat": 1, ...}
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


_EMBED_DIM = 576
_INITIAL_CAPACITY = 1024


class E2EDataset:
    """端到端数据集：管理嵌入向量和动作标签。"""

    def __init__(self):
        # Pre-allocated buffer that doubles in capacity when full
        self._embeddings_buf: np.ndarray = np.empty((_INITIAL_CAPACITY, _EMBED_DIM), dtype=np.float32)
        self._count: int = 0
        self.labels: list[int] = []
        self.action_map: dict[str, int] = {}  # action_name → index
        self._action_list: list[str] = []      # index → action_name

    @property
    def embeddings(self) -> np.ndarray:
        """返回已填充部分的视图（无拷贝）。"""
        return self._embeddings_buf[:self._count]

    def set_actions(self, actions: list[str]):
        """设置动作空间。"""
        self._action_list = list(actions)
        self.action_map = {a: i for i, a in enumerate(actions)}

    def add_sample(self, embedding: np.ndarray, action: str):
        """添加一个样本。"""
        if action not in self.action_map:
            return
        # Grow buffer if full
        if self._count >= len(self._embeddings_buf):
            new_capacity = len(self._embeddings_buf) * 2
            new_buf = np.empty((new_capacity, self._embeddings_buf.shape[1]), dtype=np.float32)
            new_buf[:self._count] = self._embeddings_buf[:self._count]
            self._embeddings_buf = new_buf
        self._embeddings_buf[self._count] = embedding.astype(np.float32)
        self._count += 1
        self.labels.append(self.action_map[action])

    @property
    def num_actions(self) -> int:
        return len(self._action_list)

    @property
    def action_list(self) -> list[str]:
        return list(self._action_list)

    def __len__(self) -> int:
        return self._count

    def save(self, path: str):
        """保存数据集到 .npz 文件。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            embeddings=self._embeddings_buf[:self._count],  # slice view, no copy needed
            labels=np.array(self.labels, dtype=np.int64),
            action_map=json.dumps(self.action_map, ensure_ascii=False),
            action_list=json.dumps(self._action_list, ensure_ascii=False),
        )
        logger.info(f"E2E 数据集已保存: {path} ({len(self)} 样本, {self.num_actions} 动作)")

    @classmethod
    def load(cls, path: str) -> "E2EDataset":
        """从 .npz 文件加载数据集。"""
        data = np.load(path, allow_pickle=False)
        ds = cls()
        loaded_embeddings: np.ndarray = data["embeddings"].astype(np.float32)
        n = len(loaded_embeddings)
        embed_dim = loaded_embeddings.shape[1] if loaded_embeddings.ndim == 2 else _EMBED_DIM
        capacity = max(_INITIAL_CAPACITY, n)
        ds._embeddings_buf = np.empty((capacity, embed_dim), dtype=np.float32)
        ds._embeddings_buf[:n] = loaded_embeddings
        ds._count = n
        ds.labels = list(data["labels"])
        ds.action_map = json.loads(str(data["action_map"]))
        ds._action_list = json.loads(str(data["action_list"]))
        logger.info(f"E2E 数据集已加载: {path} ({len(ds)} 样本)")
        return ds

    def to_tensors(self):
        """转换为 PyTorch 张量，用于训练。"""
        import torch
        X = torch.tensor(self._embeddings_buf[:self._count])
        y = torch.tensor(np.array(self.labels, dtype=np.int64))
        return X, y

    def train_val_split(self, val_ratio: float = 0.2):
        """分割训练/验证集，返回 (X_train, y_train, X_val, y_val) 张量。"""
        import torch
        X, y = self.to_tensors()
        n = len(y)
        indices = torch.randperm(n)
        val_size = max(1, int(n * val_ratio))
        val_idx, train_idx = indices[:val_size], indices[val_size:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
