"""视频文件/网络流源（支持本地文件、RTSP、HTTP 流）。"""

import cv2
import numpy as np
from .base import BaseSource


class VideoFileSource(BaseSource):
    def __init__(self, path: str, loop: bool = False):
        self.path = path
        self.loop = loop
        self._cap = None

    def start(self):
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.path}")

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret and self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        return frame if ret else None

    def stop(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
