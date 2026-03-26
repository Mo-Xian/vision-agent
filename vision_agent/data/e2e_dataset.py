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


class E2EDataset:
    """端到端数据集：管理嵌入向量和动作标签。"""

    def __init__(self):
        self.embeddings: list[np.ndarray] = []  # 每个 (576,)
        self.labels: list[int] = []
        self.action_map: dict[str, int] = {}  # action_name → index
        self._action_list: list[str] = []      # index → action_name

    def set_actions(self, actions: list[str]):
        """设置动作空间。"""
        self._action_list = list(actions)
        self.action_map = {a: i for i, a in enumerate(actions)}

    def add_sample(self, embedding: np.ndarray, action: str):
        """添加一个样本。"""
        if action not in self.action_map:
            return
        self.embeddings.append(embedding.astype(np.float32))
        self.labels.append(self.action_map[action])

    @property
    def num_actions(self) -> int:
        return len(self._action_list)

    @property
    def action_list(self) -> list[str]:
        return list(self._action_list)

    def __len__(self) -> int:
        return len(self.labels)

    def save(self, path: str):
        """保存数据集到 .npz 文件。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            embeddings=np.array(self.embeddings, dtype=np.float32),
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
        ds.embeddings = list(data["embeddings"])
        ds.labels = list(data["labels"])
        ds.action_map = json.loads(str(data["action_map"]))
        ds._action_list = json.loads(str(data["action_list"]))
        logger.info(f"E2E 数据集已加载: {path} ({len(ds)} 样本)")
        return ds

    def to_tensors(self):
        """转换为 PyTorch 张量，用于训练。"""
        import torch
        X = torch.tensor(np.array(self.embeddings, dtype=np.float32))
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
