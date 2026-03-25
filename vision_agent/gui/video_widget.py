"""视频预览控件，将 OpenCV 帧渲染到 Qt 界面上。"""

import time
import cv2
import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QFont, QColor, QPen, QBrush
from PySide6.QtWidgets import QLabel

from ..core.detector import DetectionResult
from .styles import COLORS

# 动作图标映射（tool_name → 显示符号）
_ACTION_ICONS = {
    "keyboard": "⌨",
    "mouse": "🖱",
    "api_call": "🌐",
    "shell": "⚙",
}


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
        self._actions: list[dict] = []  # [{tool, params, reason, expire}, ...]
        self._show_empty_state()

    def set_action(self, tool_name: str, parameters: dict, reason: str = "", duration: float = 2.0):
        """设置要在画面上显示的决策动作。"""
        self._actions.append({
            "tool": tool_name,
            "params": parameters,
            "reason": reason,
            "expire": time.time() + duration,
        })
        # 最多保留 5 条
        if len(self._actions) > 5:
            self._actions = self._actions[-5:]

    def update_frame(self, frame: np.ndarray, result: DetectionResult):
        """绘制检测结果并更新显示。"""
        display = frame.copy()
        self._draw_detections(display, result)
        self._draw_count_badge(display, result)
        pixmap = self._to_pixmap(display)
        # 用 QPainter 在 pixmap 上绘制中文动作信息
        self._draw_actions_on_pixmap(pixmap)
        self.setPixmap(pixmap)

    def clear(self):
        self._actions.clear()
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

        cx, cy = w // 2, h // 2

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

        painter.setPen(QColor(COLORS['text_dim']))
        font = QFont("Microsoft YaHei", 14)
        painter.setFont(font)
        painter.drawText(0, cy + tri_size + 20, w, 40, Qt.AlignCenter, "等待视频输入")

        font.setPointSize(11)
        painter.setFont(font)
        painter.drawText(0, cy + tri_size + 52, w, 30, Qt.AlignCenter,
                         "拖放视频/图片文件到窗口，或在左侧配置输入源")

        painter.end()
        self.setPixmap(pixmap)

    def _draw_detections(self, frame: np.ndarray, result: DetectionResult):
        for det in result.detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = self._get_color(det.class_id)

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

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

            label = f"{det.class_name} {det.confidence:.0%}"
            font_scale = 0.55
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_h = th + baseline + 10
            label_y = max(y1 - label_h, 0)

            cv2.rectangle(frame, (x1, label_y), (x1 + tw + 12, label_y + label_h), color, -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1 + 6, label_y + th + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_count_badge(self, frame: np.ndarray, result: DetectionResult):
        """右上角检测计数（纯英文，用 OpenCV 即可）。"""
        count = len(result.detections)
        if count == 0:
            return
        h, w = frame.shape[:2]
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

    def _to_pixmap(self, frame: np.ndarray) -> QPixmap:
        """将 OpenCV 帧转为缩放后的 QPixmap。"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        return pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _draw_actions_on_pixmap(self, pixmap: QPixmap):
        """用 QPainter 在 pixmap 左下角绘制决策动作（支持中文）。"""
        now = time.time()
        # 清理过期动作
        self._actions = [a for a in self._actions if a["expire"] > now]
        if not self._actions:
            return

        pw, ph = pixmap.width(), pixmap.height()
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        font_main = QFont("Microsoft YaHei", 10)
        font_main.setBold(True)
        font_detail = QFont("Microsoft YaHei", 9)

        row_height = 28
        pad = 8
        margin = 10
        box_w = min(320, pw - 20)

        # 从底部往上逐条绘制
        y_bottom = ph - margin
        for action in reversed(self._actions):
            tool = action["tool"]
            params = action["params"]
            reason = action["reason"]

            # 格式化参数为可读文本
            param_text = self._format_params(tool, params)
            icon = _ACTION_ICONS.get(tool, "▶")

            # 计算透明度（快过期时淡出）
            remain = action["expire"] - now
            alpha = min(1.0, remain / 0.5) if remain < 0.5 else 1.0
            alpha_int = int(alpha * 220)

            # 总行数
            lines = [f"{icon}  {param_text}"]
            if reason:
                lines.append(f"    {reason}")
            total_h = row_height * len(lines) + pad * 2

            y_top = y_bottom - total_h

            # 半透明背景
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(15, 15, 15, alpha_int))
            painter.drawRoundedRect(QRectF(margin, y_top, box_w, total_h), 6, 6)

            # 左侧指示条
            bar_color = QColor(80, 200, 255, alpha_int)
            painter.setBrush(bar_color)
            painter.drawRoundedRect(QRectF(margin, y_top, 4, total_h), 2, 2)

            # 第一行：动作
            painter.setFont(font_main)
            painter.setPen(QColor(80, 200, 255, alpha_int))
            painter.drawText(
                int(margin + pad + 4), int(y_top + pad),
                int(box_w - pad * 2), row_height,
                Qt.AlignLeft | Qt.AlignVCenter,
                lines[0]
            )

            # 第二行：原因
            if len(lines) > 1:
                painter.setFont(font_detail)
                painter.setPen(QColor(180, 180, 180, alpha_int))
                painter.drawText(
                    int(margin + pad + 4), int(y_top + pad + row_height),
                    int(box_w - pad * 2), row_height,
                    Qt.AlignLeft | Qt.AlignVCenter,
                    lines[1]
                )

            y_bottom = y_top - 4  # 条目间距

        painter.end()

    @staticmethod
    def _format_params(tool: str, params: dict) -> str:
        """将工具参数格式化为直观的显示文本。"""
        if tool == "keyboard":
            action = params.get("action", "press")
            key = params.get("key", "?")
            action_labels = {"press": "按键", "hold": "长按", "release": "释放"}
            return f"{action_labels.get(action, action)}  [ {key.upper()} ]"
        elif tool == "mouse":
            action = params.get("action", "click")
            x, y = params.get("x", 0), params.get("y", 0)
            action_labels = {"click": "点击", "move": "移动", "right_click": "右键", "double_click": "双击"}
            label = action_labels.get(action, action)
            if x or y:
                return f"{label}  ({x}, {y})"
            return label
        elif tool == "api_call":
            url = params.get("url", "")
            return f"API → {url[:40]}"
        elif tool == "shell":
            cmd = params.get("command", "")
            return f"$ {cmd[:40]}"
        else:
            return f"{tool}: {params}"

    def _show_image(self, frame: np.ndarray):
        """旧接口，保留兼容。"""
        pixmap = self._to_pixmap(frame)
        self.setPixmap(pixmap)

    def _get_color(self, class_id: int) -> tuple:
        if class_id not in self._colors:
            hue = (class_id * 47 + 15) % 180
            color_hsv = np.uint8([[[hue, 220, 230]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
            self._colors[class_id] = tuple(int(c) for c in color_bgr[0][0])
        return self._colors[class_id]
