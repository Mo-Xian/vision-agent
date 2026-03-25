"""决策模型训练器。

支持两种模型：
  - MLP (PyTorch): 多层感知机，适合中大数据集，支持 GPU
  - RandomForest (sklearn): 随机森林，适合小数据集快速验证

训练产出:
  - 模型权重文件 (.pt 或 .joblib)
  - 元数据文件 (.meta.json): action_map、class_names、特征维度等
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── MLP 模型 (PyTorch) ──

class ActionMLP:
    """用于动作分类的多层感知机。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        import torch
        import torch.nn as nn

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims or [128, 64]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 构建网络
        layers = []
        prev_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers).to(self.device)

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_features: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 15,
        progress_callback=None,
    ) -> dict:
        """训练模型。

        Args:
            progress_callback: (epoch, total, train_loss, train_acc, val_acc) 回调

        Returns:
            训练指标 dict
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        X_train = torch.FloatTensor(features).to(self.device)
        y_train = torch.LongTensor(labels).to(self.device)
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        if val_features is not None:
            X_val = torch.FloatTensor(val_features).to(self.device)
            y_val = torch.LongTensor(val_labels).to(self.device)
        else:
            X_val = y_val = None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_state = None
        no_improve = 0
        history = {"train_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            # 训练
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y_batch)
                correct += (outputs.argmax(dim=1) == y_batch).sum().item()
                total += len(y_batch)

            train_loss = total_loss / total
            train_acc = correct / total
            scheduler.step(train_loss)

            # 验证
            val_acc = self._evaluate(X_val, y_val) if X_val is not None else train_acc

            history["train_loss"].append(round(train_loss, 4))
            history["train_acc"].append(round(train_acc, 4))
            history["val_acc"].append(round(val_acc, 4))

            if progress_callback:
                progress_callback(epoch, epochs, train_loss, train_acc, val_acc)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch}/{epochs} | loss={train_loss:.4f} "
                    f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
                )

            # 早停
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"早停 @ epoch {epoch}, best_val_acc={best_val_acc:.3f}")
                    break

        # 恢复最佳权重
        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return {
            "best_val_acc": round(best_val_acc, 4),
            "final_train_acc": history["train_acc"][-1],
            "epochs_trained": len(history["train_loss"]),
            "history": history,
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测动作类别。"""
        import torch
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features).to(self.device)
            outputs = self.model(X)
            return outputs.argmax(dim=1).cpu().numpy()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """预测各类别概率。"""
        import torch
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features).to(self.device)
            outputs = self.model(X)
            return torch.softmax(outputs, dim=1).cpu().numpy()

    def _evaluate(self, X, y) -> float:
        import torch
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            correct = (outputs.argmax(dim=1) == y).sum().item()
            return correct / len(y)

    def save(self, path: str):
        import torch
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        import torch
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)


# ── 训练流程 ──

class DecisionTrainer:
    """端到端训练流程：加载数据 → 训练模型 → 保存产出。

    用法:
        trainer = DecisionTrainer(
            data_dir="data/recordings",
            output_dir="runs/decision/exp1",
        )
        metrics = trainer.run()
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str = "runs/decision",
        action_map: dict[str, int] | None = None,
        min_detections: int = 0,
        model_type: str = "mlp",
        val_split: float = 0.2,
        # MLP 参数
        hidden_dims: list[int] | None = None,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 15,
        dropout: float = 0.3,
        # RandomForest 参数
        n_estimators: int = 100,
        # 回调
        progress_callback=None,
    ):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.action_map = action_map
        self.min_detections = min_detections
        self.model_type = model_type
        self.val_split = val_split
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.dropout = dropout
        self.n_estimators = n_estimators
        self.progress_callback = progress_callback

    def run(self) -> dict:
        """执行完整训练流程，返回指标。"""
        from .dataset import ActionDataset

        # 1. 加载数据
        logger.info(f"加载数据: {self.data_dir}")
        dataset = ActionDataset(self.data_dir)
        total = dataset.load()
        if total == 0:
            raise ValueError(f"未找到训练数据: {self.data_dir}")

        summary = dataset.summary()
        logger.info(f"数据概况: {json.dumps(summary, ensure_ascii=False, indent=2)}")

        # 使用指定或自动生成的 action_map
        action_map = self.action_map or dataset.action_map
        features, labels = dataset.to_feature_label(
            action_map=action_map,
            min_detections=self.min_detections,
        )

        if len(features) < 10:
            raise ValueError(f"有效样本太少 ({len(features)})，无法训练")

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)

        # 2. 划分训练/验证集
        X_train, X_val, y_train, y_val = self._split(X, y)
        logger.info(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")

        # 3. 特征标准化
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0  # 避免除零
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

        # 4. 训练
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.model_type == "mlp":
            metrics = self._train_mlp(X_train, y_train, X_val, y_val, len(action_map))
        elif self.model_type == "rf":
            metrics = self._train_random_forest(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")

        # 5. 保存元数据
        # action_map 反转为 label → action_key
        label_to_action = {v: k for k, v in action_map.items()}

        # 收集检测类别名（与特征顺序一致）
        all_classes = sorted({
            cls for s in dataset.samples
            for cls in s.get("object_counts", {}).keys()
        })

        meta = {
            "model_type": self.model_type,
            "input_dim": int(X_train.shape[1]),
            "num_classes": len(action_map),
            "action_map": action_map,
            "label_to_action": label_to_action,
            "detection_classes": all_classes,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "hidden_dims": self.hidden_dims or [128, 64],
            "dropout": self.dropout,
            "data_summary": summary,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "metrics": metrics,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        meta_path = self.output_dir / "model.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"元数据已保存: {meta_path}")

        return metrics

    def _split(self, X: np.ndarray, y: np.ndarray):
        """按比例划分训练/验证集（分层抽样）。"""
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        split_idx = int(len(X) * (1 - self.val_split))
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

    def _train_mlp(self, X_train, y_train, X_val, y_val, num_classes: int = 0) -> dict:
        """训练 PyTorch MLP。"""
        input_dim = X_train.shape[1]
        if num_classes <= 0:
            num_classes = int(max(y_train.max(), y_val.max())) + 1

        model = ActionMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )

        metrics = model.train(
            features=X_train,
            labels=y_train,
            val_features=X_val,
            val_labels=y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=self.patience,
            progress_callback=self.progress_callback,
        )

        model_path = self.output_dir / "model.pt"
        model.save(str(model_path))
        logger.info(f"MLP 模型已保存: {model_path}")

        return metrics

    def _train_random_forest(self, X_train, y_train, X_val, y_val) -> dict:
        """训练 sklearn 随机森林。"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            import joblib
        except ImportError:
            raise ImportError("需要安装 scikit-learn: pip install scikit-learn")

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        train_acc = rf.score(X_train, y_train)
        val_acc = rf.score(X_val, y_val)

        model_path = self.output_dir / "model.joblib"
        joblib.dump(rf, str(model_path))
        logger.info(f"RandomForest 已保存: {model_path} | train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        return {
            "best_val_acc": round(val_acc, 4),
            "final_train_acc": round(train_acc, 4),
        }
