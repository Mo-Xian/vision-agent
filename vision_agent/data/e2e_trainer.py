"""端到端训练器：视觉嵌入 → MLP → 动作。

两个训练阶段：
  1. 行为克隆（Behavior Cloning）：用 LLM 标注数据监督训练
  2. 强化微调（RL Fine-tune）：策略梯度 + 简单奖励信号（可选）

产出：
  - model.pt       — MLP 权重
  - model.meta.json — 动作映射、训练指标、编码器信息
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class E2EMLP(nn.Module):
    """端到端决策 MLP：视觉嵌入 → 动作概率。"""

    def __init__(self, input_dim: int = 576, num_actions: int = 10,
                 hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_action(self, x: torch.Tensor) -> tuple[int, float]:
        """推理：返回 (action_idx, confidence)。"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            action = probs.argmax(dim=-1).item()
            confidence = probs[0, action].item() if probs.dim() == 2 else probs[action].item()
        return action, confidence


class E2ETrainer:
    """端到端训练器。

    用法:
        trainer = E2ETrainer(dataset, output_dir="runs/e2e/exp1")
        metrics = trainer.train()
    """

    def __init__(
        self,
        dataset,  # E2EDataset
        output_dir: str = "runs/e2e/exp1",
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
        val_ratio: float = 0.2,
        progress_callback=None,
        on_log=None,
    ):
        from .e2e_dataset import E2EDataset
        self._dataset: E2EDataset = dataset
        self._output_dir = Path(output_dir)
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._val_ratio = val_ratio
        self._progress_callback = progress_callback
        self._on_log = on_log

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass

    def train(self) -> dict:
        """执行行为克隆训练，返回指标。"""
        ds = self._dataset
        if len(ds) < 10:
            self._log(f"[E2E] 数据太少 ({len(ds)})，无法训练")
            return {"error": "insufficient_data", "count": len(ds)}

        self._log(f"[E2E] 开始训练: {len(ds)} 样本, {ds.num_actions} 动作, {self._epochs} 轮")

        # 分割数据
        X_train, y_train, X_val, y_val = ds.train_val_split(self._val_ratio)
        self._log(f"[E2E] 训练集: {len(y_train)}, 验证集: {len(y_val)}")

        # 构建模型
        model = E2EMLP(
            input_dim=X_train.shape[1],
            num_actions=ds.num_actions,
            hidden_dims=[256, 128],
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self._lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=10, factor=0.5,
        )

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self._batch_size, shuffle=True,
        )

        best_val_acc = 0.0
        best_epoch = 0
        history = {"train_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(1, self._epochs + 1):
            # 训练
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y_batch)
                correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total += len(y_batch)

            train_loss = total_loss / total
            train_acc = correct / total

            # 验证
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()

            scheduler.step(val_acc)

            history["train_loss"].append(round(train_loss, 4))
            history["train_acc"].append(round(train_acc, 4))
            history["val_acc"].append(round(val_acc, 4))

            # 保存最佳
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                self._save_model(model, ds, best_val_acc)

            if epoch % 10 == 0 or epoch == 1:
                self._log(
                    f"  Epoch {epoch}/{self._epochs}: "
                    f"loss={train_loss:.4f} train_acc={train_acc:.3f} "
                    f"val_acc={val_acc:.3f} (best={best_val_acc:.3f}@{best_epoch})"
                )

            if self._progress_callback:
                try:
                    self._progress_callback(epoch, self._epochs, train_loss, train_acc, val_acc)
                except Exception:
                    pass

        self._log(f"[E2E] 训练完成: best_val_acc={best_val_acc:.3f} @ epoch {best_epoch}")

        metrics = {
            "best_val_acc": round(best_val_acc, 4),
            "best_epoch": best_epoch,
            "total_epochs": self._epochs,
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "num_actions": ds.num_actions,
            "model_type": "e2e_mlp",
            "encoder": "mobilenet_v3_small",
            "embed_dim": X_train.shape[1],
        }

        # 保存历史
        history_path = self._output_dir / "train_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f)

        return metrics

    def _save_model(self, model: E2EMLP, dataset, val_acc: float):
        """保存模型和元数据。"""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # 保存权重
        model_path = self._output_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # 保存元数据
        meta = {
            "model_type": "e2e_mlp",
            "encoder": "mobilenet_v3_small",
            "embed_dim": model.net[0].in_features,
            "num_actions": dataset.num_actions,
            "action_map": dataset.action_map,
            "action_list": dataset.action_list,
            "hidden_dims": [256, 128],
            "best_val_acc": round(val_acc, 4),
            "train_samples": len(dataset),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path = self._output_dir / "model.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
