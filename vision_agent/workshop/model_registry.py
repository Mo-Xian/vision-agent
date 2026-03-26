"""训练模型注册表：扫描、索引、管理训练产出的模型。"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """模型信息。"""
    name: str = ""
    model_dir: str = ""
    model_type: str = ""       # mlp / rf
    num_classes: int = 0
    actions: list[str] = field(default_factory=list)
    val_acc: float = 0.0
    train_acc: float = 0.0
    train_samples: int = 0
    trained_at: str = ""
    scene_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model_dir": self.model_dir,
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "actions": self.actions,
            "val_acc": self.val_acc,
            "train_acc": self.train_acc,
            "train_samples": self.train_samples,
            "trained_at": self.trained_at,
        }


class ModelRegistry:
    """训练模型注册表。

    扫描指定目录（默认 runs/）下的所有训练产出，
    索引模型元数据，提供查询和管理能力。

    用法:
        registry = ModelRegistry()
        registry.scan()
        models = registry.list_models()
        best = registry.get_best()
    """

    def __init__(self, scan_dirs: list[str] | None = None):
        self._scan_dirs = scan_dirs or ["runs/decision", "runs/workshop", "runs/auto_learn"]
        self._models: dict[str, ModelInfo] = {}

    def scan(self) -> int:
        """扫描目录，索引所有模型。返回找到的模型数。"""
        self._models.clear()
        for scan_dir in self._scan_dirs:
            root = Path(scan_dir)
            if not root.exists():
                continue
            # 递归查找 model.meta.json
            for meta_path in root.rglob("model.meta.json"):
                try:
                    info = self._load_meta(meta_path)
                    if info:
                        self._models[info.model_dir] = info
                except Exception as e:
                    logger.warning(f"加载模型元数据失败 {meta_path}: {e}")

        logger.info(f"ModelRegistry 扫描完成: {len(self._models)} 个模型")
        return len(self._models)

    def list_models(self, sort_by: str = "val_acc") -> list[ModelInfo]:
        """列出所有模型，按指定字段排序（默认按验证准确率降序）。"""
        models = list(self._models.values())
        if sort_by == "val_acc":
            models.sort(key=lambda m: m.val_acc, reverse=True)
        elif sort_by == "trained_at":
            models.sort(key=lambda m: m.trained_at, reverse=True)
        elif sort_by == "train_samples":
            models.sort(key=lambda m: m.train_samples, reverse=True)
        return models

    def get_best(self) -> ModelInfo | None:
        """获取验证准确率最高的模型。"""
        models = self.list_models(sort_by="val_acc")
        return models[0] if models else None

    def get_by_dir(self, model_dir: str) -> ModelInfo | None:
        """通过模型目录获取模型信息。"""
        return self._models.get(model_dir)

    @staticmethod
    def _load_meta(meta_path: Path) -> ModelInfo | None:
        """从 meta.json 加载模型信息。"""
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_dir = str(meta_path.parent)

        # 检查模型文件是否存在
        has_model = (
            (meta_path.parent / "model.pt").exists() or
            (meta_path.parent / "model.joblib").exists()
        )
        if not has_model:
            return None

        label_to_action = meta.get("label_to_action", {})
        actions = list(label_to_action.values())

        metrics = meta.get("metrics", {})

        return ModelInfo(
            name=meta_path.parent.name,
            model_dir=model_dir,
            model_type=meta.get("model_type", "unknown"),
            num_classes=meta.get("num_classes", 0),
            actions=actions,
            val_acc=metrics.get("best_val_acc", 0.0),
            train_acc=metrics.get("final_train_acc", 0.0),
            train_samples=meta.get("train_samples", 0),
            trained_at=meta.get("trained_at", ""),
        )
