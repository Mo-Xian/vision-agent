"""检测结果可视化。"""

import time
import cv2
import numpy as np
from .detector import DetectionResult


class Visualizer:
    """在帧上绘制检测结果并显示。"""

    def __init__(self, window_name: str = "Vision Agent",
                 show_fps: bool = True, show_labels: bool = True,
                 show_confidence: bool = True):
        self.window_name = window_name
        self.show_fps = show_fps
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self._last_time = time.perf_counter()
        self._fps = 0.0
        self._fps_alpha = 0.9

    def draw(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """在帧上绘制检测框和标签，返回绘制后的帧。"""
        display = frame.copy()

        for det in result.detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = self._get_color(det.class_id)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            if self.show_labels:
                label = det.class_name
                if self.show_confidence:
                    label += f" {det.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(display, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if self.show_fps:
            now = time.perf_counter()
            dt = now - self._last_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self._fps = self._fps * self._fps_alpha + instant_fps * (1 - self._fps_alpha)
            self._last_time = now
            fps_text = f"FPS: {self._fps:.1f} | Inference: {result.inference_ms:.1f}ms"
            cv2.putText(display, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return display

    def show(self, frame: np.ndarray) -> bool:
        """显示帧，返回 False 表示用户按了 q 或关闭了窗口。"""
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def close(self):
        cv2.destroyAllWindows()

    @staticmethod
    def _get_color(class_id: int) -> tuple:
        """根据类别 ID 生成稳定的颜色。"""
        hue = (class_id * 47) % 180
        color_hsv = np.uint8([[[hue, 255, 200]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
        return tuple(int(c) for c in color_bgr[0][0])
