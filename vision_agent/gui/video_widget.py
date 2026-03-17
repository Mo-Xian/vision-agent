"""视频预览控件，将 OpenCV 帧渲染到 Qt 界面上。"""

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

from ..core.detector import DetectionResult


class VideoWidget(QLabel):
    """显示视频帧和检测结果的控件。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        self.setText("等待视频输入...")
        self.setStyleSheet(
            "background-color: #1a1a2e; border-radius: 8px; "
            "color: #888; font-size: 18px;"
        )
        self._colors: dict[int, tuple] = {}

    def update_frame(self, frame: np.ndarray, result: DetectionResult):
        """绘制检测结果并更新显示。"""
        display = frame.copy()
        self._draw_detections(display, result)
        self._show_image(display)

    def clear(self):
        self.setText("等待视频输入...")

    def _draw_detections(self, frame: np.ndarray, result: DetectionResult):
        for det in result.detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = self._get_color(det.class_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def _show_image(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def _get_color(self, class_id: int) -> tuple:
        if class_id not in self._colors:
            hue = (class_id * 47) % 180
            color_hsv = np.uint8([[[hue, 255, 200]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
            self._colors[class_id] = tuple(int(c) for c in color_bgr[0][0])
        return self._colors[class_id]
