"""视频预览控件，将 OpenCV 帧渲染到 Qt 界面上。"""

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QPainter, QFont, QColor, QPen
from PySide6.QtWidgets import QLabel

from ..core.detector import DetectionResult
from .styles import COLORS


class VideoWidget(QLabel):
    """显示视频帧和检测结果的控件。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet(
            f"background-color: {COLORS['bg_deep']}; "
            f"border: 1px solid {COLORS['border']}; "
            "border-radius: 10px;"
        )
        self._colors: dict[int, tuple] = {}
        self._action_text: str = ""
        self._action_expire: float = 0
        self._show_empty_state()

    def set_action(self, text: str, duration: float = 2.0):
        """设置要在画面上显示的决策动作文本。"""
        import time
        self._action_text = text
        self._action_expire = time.time() + duration

    def update_frame(self, frame: np.ndarray, result: DetectionResult):
        """绘制检测结果并更新显示。"""
        display = frame.copy()
        self._draw_detections(display, result)
        self._draw_overlay(display, result)
        self._show_image(display)

    def clear(self):
        self._show_empty_state()

    def _show_empty_state(self):
        """显示空状态占位。"""
        self.setText("")
        size = self.size()
        w, h = max(size.width(), 640), max(size.height(), 480)

        pixmap = QPixmap(w, h)
        pixmap.fill(QColor(COLORS['bg_deep']))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # 中央图标区域
        cx, cy = w // 2, h // 2

        # 播放按钮三角形
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(COLORS['border']))
        from PySide6.QtGui import QPolygon
        from PySide6.QtCore import QPoint
        tri_size = 30
        triangle = QPolygon([
            QPoint(cx - tri_size // 2, cy - tri_size),
            QPoint(cx - tri_size // 2, cy + tri_size),
            QPoint(cx + tri_size, cy),
        ])
        painter.drawPolygon(triangle)

        # 提示文字
        painter.setPen(QColor(COLORS['text_dim']))
        font = QFont("Segoe UI", 14)
        painter.setFont(font)
        painter.drawText(0, cy + tri_size + 20, w, 40, Qt.AlignCenter, "等待视频输入")

        font.setPointSize(11)
        painter.setFont(font)
        painter.setPen(QColor(COLORS['text_dim']))
        painter.drawText(0, cy + tri_size + 52, w, 30, Qt.AlignCenter,
                         "拖放视频/图片文件到窗口，或在左侧配置输入源")

        painter.end()
        self.setPixmap(pixmap)

    def _draw_detections(self, frame: np.ndarray, result: DetectionResult):
        for det in result.detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = self._get_color(det.class_id)

            # 绘制半透明填充
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

            # 绘制边框（圆角效果通过粗线模拟）
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # 角标装饰（四角小线段）
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            thick = 3
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thick, cv2.LINE_AA)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thick, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thick, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thick, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thick, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thick, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thick, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thick, cv2.LINE_AA)

            # 标签
            label = f"{det.class_name} {det.confidence:.0%}"
            font_scale = 0.55
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_h = th + baseline + 10
            label_y = max(y1 - label_h, 0)

            # 标签背景（带圆角）
            cv2.rectangle(frame, (x1, label_y), (x1 + tw + 12, label_y + label_h), color, -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1 + 6, label_y + th + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_overlay(self, frame: np.ndarray, result: DetectionResult):
        """在帧上绘制：右上角检测计数 + 左下角决策动作。"""
        h, w = frame.shape[:2]

        # 右上角：检测计数
        count = len(result.detections)
        if count > 0:
            text = f"{count} detected"
            font_scale = 0.6
            thickness = 1
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            pad = 8
            x = w - tw - pad * 2 - 10
            y = 10

            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + tw + pad * 2, y + th + pad * 2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, text, (x + pad, y + th + pad),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 220, 130), thickness, cv2.LINE_AA)

        # 左下角：决策动作
        import time
        if self._action_text and time.time() < self._action_expire:
            self._draw_action_badge(frame, self._action_text)

    def _draw_action_badge(self, frame: np.ndarray, text: str):
        """在画面左下角绘制决策动作标签。"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        pad = 10
        lines = text.split("\n")
        font_scale = 0.55
        thickness = 1
        line_height = 22

        # 计算背景尺寸
        max_tw = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_tw = max(max_tw, tw)

        box_w = max_tw + pad * 2
        box_h = line_height * len(lines) + pad * 2
        x1 = 10
        y1 = h - box_h - 10

        # 半透明深色背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 左侧彩色指示条
        cv2.rectangle(frame, (x1, y1), (x1 + 4, y1 + box_h), (80, 200, 255), -1)

        # 文本逐行绘制
        for i, line in enumerate(lines):
            ty = y1 + pad + line_height * (i + 1) - 4
            color = (80, 200, 255) if i == 0 else (200, 200, 200)
            cv2.putText(frame, line, (x1 + pad, ty), font, font_scale, color, thickness, cv2.LINE_AA)

    def _show_image(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def _get_color(self, class_id: int) -> tuple:
        if class_id not in self._colors:
            # 更鲜明的颜色生成
            hue = (class_id * 47 + 15) % 180
            color_hsv = np.uint8([[[hue, 220, 230]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
            self._colors[class_id] = tuple(int(c) for c in color_bgr[0][0])
        return self._colors[class_id]
