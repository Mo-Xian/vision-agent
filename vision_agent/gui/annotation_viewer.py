"""LLM 标注结果可视化回放对话框。

加载 JSONL 标注文件 + 对应视频，逐帧展示：
- YOLO 检测框
- LLM 决策动作 + 理由
- 动作分布统计
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QFileDialog, QGroupBox, QTextEdit, QSplitter,
    QWidget, QMessageBox, QLineEdit,
)

from .styles import DIALOG_STYLESHEET, COLORS

# 动作→颜色映射（循环使用）
_ACTION_COLORS = [
    (46, 204, 113),   # 绿
    (231, 76, 60),    # 红
    (52, 152, 219),   # 蓝
    (241, 196, 15),   # 黄
    (155, 89, 182),   # 紫
    (230, 126, 34),   # 橙
    (26, 188, 156),   # 青
    (236, 240, 241),  # 白
]


class AnnotationViewer(QDialog):
    """标注结果可视化回放。"""

    def __init__(self, parent=None, jsonl_path: str = "", video_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("标注结果可视化")
        self.setMinimumSize(960, 680)
        self.resize(1100, 750)
        self.setStyleSheet(DIALOG_STYLESHEET)

        self._samples: list[dict] = []
        self._video_path = ""
        self._cap: cv2.VideoCapture | None = None
        self._fps = 30.0
        self._total_video_frames = 0
        self._current_idx = 0
        self._action_color_map: dict[str, tuple] = {}
        self._playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        self._init_ui()

        if jsonl_path:
            self.jsonl_input.setText(jsonl_path)
        if video_path:
            self.video_input.setText(video_path)
        if jsonl_path:
            self._load_data()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # -- 文件选择 --
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("标注文件"))
        self.jsonl_input = QLineEdit()
        self.jsonl_input.setPlaceholderText("选择 JSONL 标注文件")
        file_row.addWidget(self.jsonl_input)
        browse_jsonl = QPushButton("浏览")
        browse_jsonl.setObjectName("browseBtn")
        browse_jsonl.clicked.connect(self._browse_jsonl)
        file_row.addWidget(browse_jsonl)

        file_row.addWidget(QLabel("视频"))
        self.video_input = QLineEdit()
        self.video_input.setPlaceholderText("对应的视频文件（可选）")
        file_row.addWidget(self.video_input)
        browse_video = QPushButton("浏览")
        browse_video.setObjectName("browseBtn")
        browse_video.clicked.connect(self._browse_video)
        file_row.addWidget(browse_video)

        load_btn = QPushButton("加载")
        load_btn.setObjectName("infoBtn")
        load_btn.clicked.connect(self._load_data)
        file_row.addWidget(load_btn)
        layout.addLayout(file_row)

        # -- 主体：左视频 + 右信息 --
        splitter = QSplitter(Qt.Horizontal)

        # 左：视频画面
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setMinimumSize(640, 400)
        self.frame_label.setStyleSheet(
            f"background-color: {COLORS['bg_deep']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 8px;"
        )
        left_layout.addWidget(self.frame_label, 1)

        # 播放控制
        ctrl_row = QHBoxLayout()
        self.prev_btn = QPushButton("< 上一帧")
        self.prev_btn.setObjectName("browseBtn")
        self.prev_btn.clicked.connect(self._prev_frame)
        ctrl_row.addWidget(self.prev_btn)

        self.play_btn = QPushButton("播放")
        self.play_btn.setObjectName("startBtn")
        self.play_btn.clicked.connect(self._toggle_play)
        ctrl_row.addWidget(self.play_btn)

        self.next_btn = QPushButton("下一帧 >")
        self.next_btn.setObjectName("browseBtn")
        self.next_btn.clicked.connect(self._next_frame)
        ctrl_row.addWidget(self.next_btn)
        left_layout.addLayout(ctrl_row)

        # 滑块
        slider_row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self.slider)
        self.frame_info_label = QLabel("0 / 0")
        self.frame_info_label.setMinimumWidth(120)
        self.frame_info_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 13px;")
        slider_row.addWidget(self.frame_info_label)
        left_layout.addLayout(slider_row)

        splitter.addWidget(left)

        # 右：决策信息 + 统计
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 当前帧决策
        decision_group = QGroupBox("LLM 决策")
        dg = QVBoxLayout(decision_group)
        self.action_label = QLabel("--")
        self.action_label.setStyleSheet(
            f"color: {COLORS['success']}; font-size: 22px; font-weight: bold;"
        )
        dg.addWidget(self.action_label)

        self.reason_label = QLabel("")
        self.reason_label.setWordWrap(True)
        self.reason_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 13px;")
        dg.addWidget(self.reason_label)

        self.detection_label = QLabel("")
        self.detection_label.setWordWrap(True)
        self.detection_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        dg.addWidget(self.detection_label)

        self.frame_meta_label = QLabel("")
        self.frame_meta_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        dg.addWidget(self.frame_meta_label)
        right_layout.addWidget(decision_group)

        # 动作分布统计
        stats_group = QGroupBox("动作分布")
        sg = QVBoxLayout(stats_group)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        sg.addWidget(self.stats_text)
        right_layout.addWidget(stats_group)

        right_layout.addStretch()
        splitter.addWidget(right)

        splitter.setSizes([700, 350])
        layout.addWidget(splitter, 1)

    # -- 文件浏览 --

    def _browse_jsonl(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择标注文件", "",
            "JSONL (*.jsonl);;JSON (*.json);;所有文件 (*)"
        )
        if path:
            self.jsonl_input.setText(path)

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*)"
        )
        if path:
            self.video_input.setText(path)

    # -- 数据加载 --

    @Slot()
    def _load_data(self):
        jsonl_path = self.jsonl_input.text().strip()
        if not jsonl_path or not Path(jsonl_path).exists():
            QMessageBox.warning(self, "提示", "请选择有效的 JSONL 文件")
            return

        self._samples.clear()
        self._action_color_map.clear()

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not self._samples:
            QMessageBox.warning(self, "提示", "标注文件为空或格式不正确")
            return

        # 构建动作颜色映射
        actions_seen = []
        for s in self._samples:
            action = self._get_action(s)
            if action and action not in actions_seen:
                actions_seen.append(action)
        for i, a in enumerate(actions_seen):
            self._action_color_map[a] = _ACTION_COLORS[i % len(_ACTION_COLORS)]

        # 尝试加载视频
        video_path = self.video_input.text().strip()
        if self._cap:
            self._cap.release()
            self._cap = None

        if video_path and Path(video_path).exists():
            self._cap = cv2.VideoCapture(video_path)
            if self._cap.isOpened():
                self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
                self._total_video_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._video_path = video_path

        # 更新 UI
        self.slider.setMaximum(max(0, len(self._samples) - 1))
        self._current_idx = 0
        self.slider.setValue(0)
        self._update_stats()
        self._show_sample(0)

    # -- 导航 --

    @Slot()
    def _prev_frame(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self.slider.setValue(self._current_idx)

    @Slot()
    def _next_frame(self):
        if self._current_idx < len(self._samples) - 1:
            self._current_idx += 1
            self.slider.setValue(self._current_idx)

    @Slot()
    def _toggle_play(self):
        if self._playing:
            self._playing = False
            self._play_timer.stop()
            self.play_btn.setText("播放")
            self.play_btn.setObjectName("startBtn")
            self.play_btn.setStyleSheet("")  # force re-style
            self.play_btn.style().unpolish(self.play_btn)
            self.play_btn.style().polish(self.play_btn)
        else:
            self._playing = True
            self.play_btn.setText("暂停")
            self.play_btn.setObjectName("stopBtn")
            self.play_btn.setStyleSheet("")
            self.play_btn.style().unpolish(self.play_btn)
            self.play_btn.style().polish(self.play_btn)
            interval = max(100, int(1000 / self._fps))
            self._play_timer.start(interval)

    @Slot()
    def _on_play_tick(self):
        if self._current_idx < len(self._samples) - 1:
            self._current_idx += 1
            self.slider.setValue(self._current_idx)
        else:
            self._toggle_play()

    @Slot(int)
    def _on_slider_changed(self, value: int):
        self._current_idx = value
        self._show_sample(value)

    # -- 渲染 --

    def _show_sample(self, idx: int):
        if idx < 0 or idx >= len(self._samples):
            return

        sample = self._samples[idx]
        action = self._get_action(sample)
        reason = self._get_reason(sample)
        detections = sample.get("detections", [])
        frame_id = sample.get("frame_id", 0)
        timestamp = sample.get("action_timestamp", sample.get("timestamp", 0))
        inference_ms = sample.get("inference_ms", 0)
        frame_size = sample.get("frame_size", [0, 0])

        # 更新信息面板
        self.frame_info_label.setText(f"{idx + 1} / {len(self._samples)}")

        color = self._action_color_map.get(action, (255, 255, 255))
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        self.action_label.setText(action or "--")
        self.action_label.setStyleSheet(
            f"color: {hex_color}; font-size: 22px; font-weight: bold;"
        )
        self.reason_label.setText(reason if reason else "")

        # 检测信息
        counts = sample.get("object_counts", {})
        if counts:
            det_parts = [f"{name} x{cnt}" for name, cnt in counts.items()]
            self.detection_label.setText(f"检测: {', '.join(det_parts)}  ({len(detections)} 个目标)")
        else:
            self.detection_label.setText(f"检测: {len(detections)} 个目标")

        self.frame_meta_label.setText(
            f"帧 #{frame_id}  |  时间 {timestamp:.1f}s  |  "
            f"推理 {inference_ms:.1f}ms  |  尺寸 {frame_size[0]}x{frame_size[1]}"
        )

        # 渲染画面
        frame = self._get_video_frame(frame_id, frame_size)
        self._draw_annotated_frame(frame, detections, action, reason, color)

    def _get_video_frame(self, frame_id: int, frame_size: list) -> np.ndarray:
        """从视频获取对应帧，或生成黑底占位帧。"""
        if self._cap and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id - 1))
            ret, frame = self._cap.read()
            if ret:
                return frame

        # 无视频：生成黑底帧
        w = frame_size[0] if frame_size[0] > 0 else 640
        h = frame_size[1] if frame_size[1] > 0 else 480
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (20, 25, 35)  # 深蓝底色
        # 提示文字
        cv2.putText(frame, "No video loaded", (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 120), 1, cv2.LINE_AA)
        return frame

    def _draw_annotated_frame(self, frame: np.ndarray, detections: list,
                               action: str, reason: str, action_color: tuple):
        """在帧上绘制检测框和 LLM 决策叠加层。"""
        h, w = frame.shape[:2]

        # 绘制检测框
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            bbox_norm = det.get("bbox_norm")
            class_name = det.get("class_name", "?")
            conf = det.get("confidence", 0)
            class_id = det.get("class_id", 0)

            # 优先用像素坐标，否则从归一化坐标计算
            if bbox[2] > 0:
                x1, y1, x2, y2 = [int(v) for v in bbox]
            elif bbox_norm:
                x1 = int(bbox_norm[0] * w)
                y1 = int(bbox_norm[1] * h)
                x2 = int(bbox_norm[2] * w)
                y2 = int(bbox_norm[3] * h)
            else:
                continue

            color = self._det_color(class_id)

            # 半透明填充
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

            # 边框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # 标签
            label = f"{class_name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - th - 8, 0)
            cv2.rectangle(frame, (x1, label_y), (x1 + tw + 8, label_y + th + 8), color, -1)
            cv2.putText(frame, label, (x1 + 4, label_y + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # -- LLM 决策叠加层（底部横幅） --
        if action:
            banner_h = 50
            banner_y = h - banner_h

            # 半透明黑底
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, banner_y), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # 动作标签（彩色圆点 + 文字）
            cv2.circle(frame, (20, banner_y + banner_h // 2), 8, action_color, -1, cv2.LINE_AA)
            cv2.putText(frame, f"ACTION: {action}", (36, banner_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2, cv2.LINE_AA)

            # 理由
            if reason:
                max_len = max(40, w // 12)
                display_reason = reason[:max_len] + "..." if len(reason) > max_len else reason
                cv2.putText(frame, display_reason, (36, banner_y + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 210), 1, cv2.LINE_AA)

        # 转为 QPixmap 显示
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.frame_label.setPixmap(scaled)

    def _update_stats(self):
        """更新动作分布统计。"""
        dist: dict[str, int] = {}
        for s in self._samples:
            action = self._get_action(s)
            if action:
                dist[action] = dist.get(action, 0) + 1

        total = sum(dist.values()) or 1
        lines = []
        for action, count in sorted(dist.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            color = self._action_color_map.get(action, (200, 200, 200))
            hex_c = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            bar = "#" * max(1, int(pct / 3))
            lines.append(
                f'<span style="color:{hex_c}; font-weight:bold;">{action}</span>'
                f'  {count} ({pct:.1f}%)  '
                f'<span style="color:{hex_c};">{bar}</span>'
            )

        self.stats_text.setHtml("<br>".join(lines) if lines else "无数据")

    # -- 辅助 --

    @staticmethod
    def _get_action(sample: dict) -> str:
        ha = sample.get("human_action", {})
        return ha.get("key", "") or ha.get("action", "")

    @staticmethod
    def _get_reason(sample: dict) -> str:
        return sample.get("human_action", {}).get("reason", "")

    @staticmethod
    def _det_color(class_id: int) -> tuple:
        hue = (class_id * 47 + 15) % 180
        color_hsv = np.uint8([[[hue, 220, 230]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
        return tuple(int(c) for c in color_bgr[0][0])

    def closeEvent(self, event):
        self._playing = False
        self._play_timer.stop()
        if self._cap:
            self._cap.release()
        event.accept()
