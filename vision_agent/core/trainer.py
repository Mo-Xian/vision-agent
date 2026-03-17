"""YOLO 模型训练封装。"""

import logging
from pathlib import Path
from dataclasses import dataclass
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """训练配置。"""
    data_yaml: str          # 数据集 YAML 配置路径
    base_model: str = "yolov8n.pt"  # 基础预训练模型
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: str | None = None       # None=自动, "cpu", "0"
    project: str = "runs/train"     # 输出目录
    name: str = "custom"            # 实验名称
    patience: int = 10              # 早停耐心值
    lr0: float = 0.01               # 初始学习率


class Trainer:
    """YOLO 模型训练器。"""

    def __init__(self, config: TrainConfig, progress_callback=None):
        """
        Args:
            config: 训练配置
            progress_callback: 进度回调 (epoch, total_epochs, metrics_dict)
        """
        self.config = config
        self.progress_callback = progress_callback
        self._model: YOLO | None = None

    def train(self) -> str:
        """执行训练，返回最佳模型路径。"""
        cfg = self.config
        logger.info(f"开始训练: base={cfg.base_model}, data={cfg.data_yaml}, epochs={cfg.epochs}")

        self._model = YOLO(cfg.base_model)

        results = self._model.train(
            data=cfg.data_yaml,
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            device=cfg.device,
            project=cfg.project,
            name=cfg.name,
            patience=cfg.patience,
            lr0=cfg.lr0,
            verbose=True,
        )

        # 最佳模型路径
        best_path = Path(cfg.project) / cfg.name / "weights" / "best.pt"
        if best_path.exists():
            logger.info(f"训练完成, 最佳模型: {best_path}")
            return str(best_path)

        # fallback to last
        last_path = Path(cfg.project) / cfg.name / "weights" / "last.pt"
        logger.info(f"训练完成, 模型: {last_path}")
        return str(last_path)

    def validate(self, model_path: str = None, data_yaml: str = None) -> dict:
        """验证模型，返回指标。"""
        model = YOLO(model_path) if model_path else self._model
        if model is None:
            raise RuntimeError("没有可用的模型，请先训练或指定模型路径")

        results = model.val(data=data_yaml or self.config.data_yaml)
        return {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }

    @staticmethod
    def create_data_yaml(save_path: str, train_dir: str, val_dir: str,
                         class_names: list[str], test_dir: str = None) -> str:
        """生成 YOLO 数据集 YAML 配置文件。

        目录结构应为:
            train_dir/
                images/
                labels/
            val_dir/
                images/
                labels/
        """
        import yaml

        data = {
            "path": ".",
            "train": train_dir,
            "val": val_dir,
            "names": {i: name for i, name in enumerate(class_names)},
        }
        if test_dir:
            data["test"] = test_dir

        # 使用绝对路径
        data["train"] = str(Path(train_dir).resolve())
        data["val"] = str(Path(val_dir).resolve())
        if test_dir:
            data["test"] = str(Path(test_dir).resolve())

        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"数据集配置已保存: {save_path}")
        return save_path
