"""屏幕捕获视频源。"""

import numpy as np
import cv2
from .base import BaseSource


class ScreenSource(BaseSource):
    def __init__(self, monitor: int = 1, region: list | None = None, fps: int = 30):
        self.monitor = monitor
        self.region = region
        self.fps = fps
        self._sct = None

    def start(self):
        import mss
        self._sct = mss.mss()

    def read(self) -> np.ndarray | None:
        if self._sct is None:
            return None
        if self.region:
            x, y, w, h = self.region
            mon = {"left": x, "top": y, "width": w, "height": h}
        else:
            mon = self._sct.monitors[self.monitor]

        img = self._sct.grab(mon)
        frame = np.array(img)
        # BGRA -> BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def stop(self):
        if self._sct is not None:
            self._sct.close()
            self._sct = None
