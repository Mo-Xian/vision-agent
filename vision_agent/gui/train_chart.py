"""训练过程实时图表：Loss 和 Accuracy 曲线。"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import QWidget

from .styles import COLORS


class TrainChart(QWidget):
    """轻量级训练曲线图（无需 matplotlib）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMaximumHeight(220)
        self._loss_data: list[float] = []
        self._train_acc_data: list[float] = []
        self._val_acc_data: list[float] = []

    def clear(self):
        self._loss_data.clear()
        self._train_acc_data.clear()
        self._val_acc_data.clear()
        self.update()

    def add_point(self, loss: float, train_acc: float, val_acc: float):
        self._loss_data.append(loss)
        self._train_acc_data.append(train_acc)
        self._val_acc_data.append(val_acc)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        margin_left = 45
        margin_right = 10
        margin_top = 20
        margin_bottom = 25
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom

        # 背景
        painter.fillRect(0, 0, w, h, QColor(COLORS['bg_input']))

        # 绘图区域背景
        painter.setPen(QPen(QColor(COLORS['border']), 1))
        painter.drawRect(margin_left, margin_top, plot_w, plot_h)

        if not self._loss_data:
            painter.setPen(QColor(COLORS['text_dim']))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(0, 0, w, h, Qt.AlignCenter, "训练开始后显示曲线")
            painter.end()
            return

        n = len(self._loss_data)

        # 标题
        painter.setPen(QColor(COLORS['text_secondary']))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(margin_left, 2, plot_w, 16, Qt.AlignCenter, f"Epoch 1-{n}")

        # 网格线
        painter.setPen(QPen(QColor(COLORS['border']), 1, Qt.DashLine))
        for i in range(1, 4):
            y = margin_top + plot_h * i // 4
            painter.drawLine(margin_left, y, margin_left + plot_w, y)

        # Y 轴标签
        painter.setPen(QColor(COLORS['text_dim']))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(0, margin_top - 5, margin_left - 4, 12, Qt.AlignRight, "1.0")
        painter.drawText(0, margin_top + plot_h // 2 - 5, margin_left - 4, 12, Qt.AlignRight, "0.5")
        painter.drawText(0, margin_top + plot_h - 5, margin_left - 4, 12, Qt.AlignRight, "0.0")

        # X 轴标签
        painter.drawText(margin_left - 5, h - 12, 20, 12, Qt.AlignCenter, "1")
        painter.drawText(margin_left + plot_w - 10, h - 12, 20, 12, Qt.AlignCenter, str(n))

        def draw_line(data, color, max_val=None):
            if len(data) < 2:
                return
            if max_val is None:
                max_val = max(max(data), 0.01)
            pen = QPen(QColor(*color), 2)
            painter.setPen(pen)
            for i in range(1, len(data)):
                x1 = margin_left + int((i - 1) / max(n - 1, 1) * plot_w)
                x2 = margin_left + int(i / max(n - 1, 1) * plot_w)
                y1 = margin_top + plot_h - int(min(data[i - 1] / max_val, 1.0) * plot_h)
                y2 = margin_top + plot_h - int(min(data[i] / max_val, 1.0) * plot_h)
                painter.drawLine(x1, y1, x2, y2)

        # Loss 曲线（红色，归一化到自身最大值）
        max_loss = max(max(self._loss_data), 0.01)
        draw_line(self._loss_data, (231, 76, 60), max_loss)

        # Train Acc 曲线（蓝色，0-1 范围）
        draw_line(self._train_acc_data, (52, 152, 219), 1.0)

        # Val Acc 曲线（绿色，0-1 范围）
        draw_line(self._val_acc_data, (46, 204, 113), 1.0)

        # 图例
        legend_y = margin_top + 4
        painter.setFont(QFont("Segoe UI", 8))
        for i, (label, color) in enumerate([
            ("Loss", (231, 76, 60)),
            ("Train", (52, 152, 219)),
            ("Val", (46, 204, 113)),
        ]):
            lx = margin_left + 8 + i * 65
            painter.setPen(QPen(QColor(*color), 2))
            painter.drawLine(lx, legend_y + 5, lx + 14, legend_y + 5)
            painter.setPen(QColor(COLORS['text_secondary']))
            painter.drawText(lx + 18, legend_y, 40, 12, Qt.AlignLeft, label)

        # 当前值
        if self._loss_data:
            cur_text = (
                f"loss={self._loss_data[-1]:.4f}  "
                f"train={self._train_acc_data[-1]:.3f}  "
                f"val={self._val_acc_data[-1]:.3f}"
            )
            painter.setPen(QColor(COLORS['text']))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(margin_left, h - 12, plot_w, 12, Qt.AlignCenter, cur_text)

        painter.end()
