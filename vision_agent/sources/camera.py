"""摄像头视频源。"""

import cv2
import numpy as np
from .base import BaseSource


class CameraSource(BaseSource):
    def __init__(self, device: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None

    def start(self):
        self._cap = cv2.VideoCapture(self.device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.device}")

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def stop(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
