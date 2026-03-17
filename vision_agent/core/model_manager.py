"""模型管理器：注册、加载、运行时切换检测模型。"""

import logging
from pathlib import Path
from .detector import Detector

logger = logging.getLogger(__name__)


class ModelManager:
    """管理多个 YOLO 模型，支持运行时切换。"""

    def __init__(self, detector_kwargs: dict | None = None):
        """
        Args:
            detector_kwargs: 传给 Detector 的通用参数 (confidence, iou, device 等)
        """
        self._registry: dict[str, str] = {}          # name -> model_path
        self._detectors: dict[str, Detector] = {}     # name -> loaded Detector
        self._current_name: str | None = None
        self._detector_kwargs = detector_kwargs or {}

    def register(self, name: str, model_path: str) -> None:
        """注册一个模型（不立即加载）。"""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self._registry[name] = str(path)
        logger.info(f"注册模型: {name} -> {model_path}")

    def load(self, name: str) -> Detector:
        """加载模型到内存，返回 Detector 实例。"""
        if name not in self._registry:
            raise KeyError(f"模型 '{name}' 未注册。可用: {list(self._registry.keys())}")
        if name not in self._detectors:
            logger.info(f"加载模型: {name}")
            self._detectors[name] = Detector(
                model=self._registry[name],
                **self._detector_kwargs,
            )
        return self._detectors[name]

    def switch(self, name: str) -> Detector:
        """切换当前活跃模型，返回新 Detector。"""
        detector = self.load(name)
        old = self._current_name
        self._current_name = name
        logger.info(f"切换模型: {old} -> {name}")
        return detector

    @property
    def current(self) -> Detector | None:
        """获取当前活跃的 Detector。"""
        if self._current_name is None:
            return None
        return self._detectors.get(self._current_name)

    @property
    def current_name(self) -> str | None:
        return self._current_name

    def list_models(self) -> list[dict]:
        """列出所有已注册模型及其状态。"""
        return [
            {
                "name": name,
                "path": path,
                "loaded": name in self._detectors,
                "active": name == self._current_name,
            }
            for name, path in self._registry.items()
        ]

    def unload(self, name: str) -> None:
        """从内存释放指定模型。"""
        if name in self._detectors:
            del self._detectors[name]
            if self._current_name == name:
                self._current_name = None
            logger.info(f"卸载模型: {name}")
