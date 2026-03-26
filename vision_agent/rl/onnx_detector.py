"""ONNX 模型检测器：兼容 wzry_ai 项目的 start.onnx / death.onnx。

wzry_ai 的模型是 YOLO 格式的目标检测模型，输入 640x640，
输出检测框 + 类别 + 置信度。

用法：
    detector = OnnxDetector("models/start.onnx", classes=["started"])
    label = detector.detect(frame)  # "started" 或 None
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ort = None  # 延迟导入 onnxruntime


def _get_ort():
    global _ort
    if _ort is None:
        try:
            import onnxruntime as ort
            _ort = ort
        except ImportError:
            logger.warning("需要安装 onnxruntime: pip install onnxruntime")
            return None
    return _ort


class OnnxDetector:
    """轻量 ONNX 检测器，兼容 wzry_ai 的 YOLO 模型。"""

    def __init__(
        self,
        model_path: str,
        classes: list[str],
        input_size: int = 640,
        confidence_threshold: float = 0.5,
    ):
        self._model_path = model_path
        self._classes = classes
        self._input_size = input_size
        self._confidence_threshold = confidence_threshold
        self._session = None

    def _ensure_loaded(self) -> bool:
        """延迟加载模型。"""
        if self._session is not None:
            return True

        ort = _get_ort()
        if ort is None:
            return False

        if not Path(self._model_path).exists():
            logger.warning(f"ONNX 模型不存在: {self._model_path}")
            return False

        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(self._model_path, providers=providers)
            logger.info(f"ONNX 模型加载: {self._model_path} ({self._classes})")
            return True
        except Exception as e:
            logger.warning(f"ONNX 模型加载失败: {e}")
            return False

    def detect(self, frame: np.ndarray) -> str | None:
        """检测画面，返回最高置信度的类别名（或 None）。

        Args:
            frame: BGR 图像 (OpenCV 格式)

        Returns:
            类别名字符串，未检测到则返回 None
        """
        if not self._ensure_loaded():
            return None

        # 预处理：resize → RGB → normalize → NCHW
        resized = cv2.resize(frame, (self._input_size, self._input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC → CHW
        blob = np.expand_dims(blob, axis=0)     # → NCHW

        # 推理
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob})

        # 后处理（兼容 wzry_ai 的输出格式）
        result = np.squeeze(outputs[0])
        if result.ndim < 2:
            return None

        best_label = None
        best_score = 0.0

        for i in range(result.shape[0]):
            score = float(result[i][4]) if result.shape[1] > 4 else 0.0
            if score >= self._confidence_threshold and score > best_score:
                class_id = int(result[i][5]) if result.shape[1] > 5 else 0
                if class_id < len(self._classes):
                    best_label = self._classes[class_id]
                    best_score = score

        return best_label

    @property
    def is_available(self) -> bool:
        """模型文件是否存在。"""
        return Path(self._model_path).exists()
