"""视频预览控件，将 OpenCV 帧渲染到 Qt 界面上。"""

import time
import cv2
import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QFont, QColor, QPen, QBrush
from PySide6.QtWidgets import QLabel

from ..core.detector import DetectionResult
from .styles import COLORS

# 动作图标映射
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
        self._actions: list[dict] = []
        self._last_frame_size: tuple = (640, 480)  # 原始帧尺寸，用于坐标映射
        self._show_empty_state()

    def set_action(self, tool_name: str, parameters: dict, reason: str = "",
                   duration: float = 2.0, target_bbox: tuple | None = None):
        """设置要在画面上显示的决策动作。"""
        self._actions.append({
            "tool": tool_name,
            "params": parameters,
            "reason": reason,
            "bbox": target_bbox,  # (x1,y1,x2,y2) 原始帧像素坐标
            "expire": time.time() + duration,
        })
        if len(self._actions) > 8:
            self._actions = self._actions[-8:]

    def update_frame(self, frame: np.ndarray, result: DetectionResult):
        """绘制检测结果并更新显示。"""
        self._last_frame_size = (frame.shape[1], frame.shape[0])
        display = frame.copy()
        self._draw_detections(display, result)
        self._draw_count_badge(display, result)
        pixmap = self._to_pixmap(display)
        self._draw_actions_on_pixmap(pixmap)
        self.setPixmap(pixmap)

    def clear(self):
        self._actions.clear()
        self._show_empty_state()

    def _show_empty_state(self):
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        return pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _frame_to_pixmap_coords(self, fx: float, fy: float, pixmap_w: int, pixmap_h: int) -> tuple:
        """将原始帧坐标转换为缩放后 pixmap 上的坐标。"""
        fw, fh = self._last_frame_size
        # 计算缩放比例（KeepAspectRatio）
        scale = min(pixmap_w / fw, pixmap_h / fh)
        # 缩放后的实际图像尺寸
        sw = fw * scale
        sh = fh * scale
        # 居中偏移
        ox = (pixmap_w - sw) / 2
        oy = (pixmap_h - sh) / 2
        return ox + fx * scale, oy + fy * scale

    def _draw_actions_on_pixmap(self, pixmap: QPixmap):
        """用 QPainter 在 pixmap 上绘制决策动作标签（定位到目标位置）。"""
        now = time.time()
        self._actions = [a for a in self._actions if a["expire"] > now]
        if not self._actions:
            return

        pw, ph = pixmap.width(), pixmap.height()
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        font_main = QFont("Microsoft YaHei", 10)
        font_main.setBold(True)
        font_detail = QFont("Microsoft YaHei", 9)

        for action in self._actions:
            tool = action["tool"]
            params = action["params"]
            reason = action["reason"]
            bbox = action["bbox"]

            # 计算透明度（淡出）
            remain = action["expire"] - now
            alpha = min(1.0, remain / 0.5) if remain < 0.5 else 1.0
            alpha_int = int(alpha * 230)

            # 格式化文本
            icon = _ACTION_ICONS.get(tool, "▶")
            param_text = self._format_params(tool, params)
            line1 = f"{icon} {param_text}"

            # 确定绘制位置
            if bbox:
                # 有目标 bbox → 画在目标右侧/下方
                bx1, by1, bx2, by2 = bbox
                cx = (bx1 + bx2) / 2
                cy = (by1 + by2) / 2
                px, py = self._frame_to_pixmap_coords(bx2 + 5, by1, pw, ph)

                # 画连接线：从目标中心到标签
                pcx, pcy = self._frame_to_pixmap_coords(cx, cy, pw, ph)
                pen = QPen(QColor(80, 200, 255, alpha_int), 1.5, Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(int(pcx), int(pcy), int(px + 4), int(py + 14))

                # 目标中心画十字准星
                cross_size = 8
                pen_cross = QPen(QColor(255, 80, 80, alpha_int), 2)
                painter.setPen(pen_cross)
                painter.drawLine(int(pcx - cross_size), int(pcy), int(pcx + cross_size), int(pcy))
                painter.drawLine(int(pcx), int(pcy - cross_size), int(pcx), int(pcy + cross_size))
            else:
                # 无位置 → 画在左下角
                px, py = 12, ph - 60

            # 计算标签尺寸
            fm_main = painter.fontMetrics()
            painter.setFont(font_main)
            fm_main = painter.fontMetrics()
            tw1 = fm_main.horizontalAdvance(line1)

            tw2 = 0
            if reason:
                painter.setFont(font_detail)
                fm_detail = painter.fontMetrics()
                tw2 = fm_detail.horizontalAdvance(reason)

            pad = 8
            box_w = max(tw1, tw2) + pad * 2
            row_h = 24
            box_h = row_h + pad * 2 + (row_h if reason else 0)

            # 防止超出画面边界
            px = min(px, pw - box_w - 5)
            py = max(py, 5)
            px = max(px, 5)
            if py + box_h > ph - 5:
                py = ph - box_h - 5

            # 半透明背景
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(15, 15, 15, alpha_int))
            painter.drawRoundedRect(QRectF(px, py, box_w, box_h), 6, 6)

            # 左侧指示条
            painter.setBrush(QColor(80, 200, 255, alpha_int))
            painter.drawRoundedRect(QRectF(px, py, 4, box_h), 2, 2)

            # 第一行：动作
            painter.setFont(font_main)
            painter.setPen(QColor(80, 200, 255, alpha_int))
            painter.drawText(
                int(px + pad + 4), int(py + pad),
                int(box_w - pad * 2), row_h,
                Qt.AlignLeft | Qt.AlignVCenter,
                line1,
            )

            # 第二行：原因
            if reason:
                painter.setFont(font_detail)
                painter.setPen(QColor(200, 200, 200, alpha_int))
                painter.drawText(
                    int(px + pad + 4), int(py + pad + row_h),
                    int(box_w - pad * 2), row_h,
                    Qt.AlignLeft | Qt.AlignVCenter,
                    reason,
                )

        painter.end()

    @staticmethod
    def _format_params(tool: str, params: dict) -> str:
        if tool == "keyboard":
            action = params.get("action", "press")
            key = params.get("key", "?")
            labels = {"press": "按键", "hold": "长按", "release": "释放"}
            return f"{labels.get(action, action)}  [ {key.upper()} ]"
        elif tool == "mouse":
            action = params.get("action", "click")
            x, y = params.get("x", 0), params.get("y", 0)
            labels = {"click": "点击", "move": "移动", "right_click": "右键", "double_click": "双击"}
            label = labels.get(action, action)
            if x or y:
                return f"{label}  ({x}, {y})"
            return label
        elif tool == "api_call":
            return f"API → {params.get('url', '')[:40]}"
        elif tool == "shell":
            return f"$ {params.get('command', '')[:40]}"
        return f"{tool}: {params}"

    def _get_color(self, class_id: int) -> tuple:
        if class_id not in self._colors:
            hue = (class_id * 47 + 15) % 180
            color_hsv = np.uint8([[[hue, 220, 230]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
            self._colors[class_id] = tuple(int(c) for c in color_bgr[0][0])
        return self._colors[class_id]
