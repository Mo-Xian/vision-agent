"""LLM 标注结果可视化回放对话框。

功能：
- 加载 JSONL 标注文件 + 对应视频，逐帧展示检测框和 LLM 决策
- 编辑/纠正动作标注，删除坏样本，保存修正后的 JSONL
- 动作分布统计 + 数据集质量分析
- 键盘快捷键导航
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QFileDialog, QGroupBox, QTextEdit, QSplitter,
    QWidget, QMessageBox, QLineEdit, QComboBox,
)

from .styles import DIALOG_STYLESHEET, COLORS

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
    """标注结果可视化回放 + 纠错编辑。"""

    def __init__(self, parent=None, jsonl_path: str = "", video_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("标注结果可视化")
        self.setMinimumSize(960, 680)
        self.resize(1100, 750)
        self.setStyleSheet(DIALOG_STYLESHEET)

        self._samples: list[dict] = []
        self._jsonl_path = ""
        self._video_path = ""
        self._cap: cv2.VideoCapture | None = None
        self._fps = 30.0
        self._total_video_frames = 0
        self._current_idx = 0
        self._action_color_map: dict[str, tuple] = {}
        self._all_actions: list[str] = []
        self._playing = False
        self._modified = False
        self._deleted_indices: set[int] = set()
        self._compare_samples: list[dict] = []  # A/B 对比用
        self._compare_map: dict[int, str] = {}  # frame_id → compare action
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        self._init_ui()
        self._init_shortcuts()

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

        compare_btn = QPushButton("对比")
        compare_btn.setObjectName("purpleBtn")
        compare_btn.setToolTip("加载第二份标注文件进行 A/B 对比")
        compare_btn.clicked.connect(self._load_compare)
        file_row.addWidget(compare_btn)
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

        # 右：决策信息 + 编辑 + 统计
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

        # -- 编辑区域 --
        edit_group = QGroupBox("纠正标注")
        eg = QVBoxLayout(edit_group)

        action_edit_row = QHBoxLayout()
        action_edit_row.addWidget(QLabel("动作"))
        self.action_combo = QComboBox()
        self.action_combo.setEditable(True)
        self.action_combo.setMinimumWidth(120)
        action_edit_row.addWidget(self.action_combo)

        self.apply_btn = QPushButton("修改")
        self.apply_btn.setObjectName("infoBtn")
        self.apply_btn.clicked.connect(self._apply_edit)
        action_edit_row.addWidget(self.apply_btn)

        self.delete_btn = QPushButton("删除此帧")
        self.delete_btn.setObjectName("stopBtn")
        self.delete_btn.clicked.connect(self._delete_sample)
        action_edit_row.addWidget(self.delete_btn)
        eg.addLayout(action_edit_row)

        save_row = QHBoxLayout()
        self.save_btn = QPushButton("保存修正")
        self.save_btn.setObjectName("startBtn")
        self.save_btn.clicked.connect(self._save_corrections)
        self.save_btn.setEnabled(False)
        save_row.addWidget(self.save_btn)
        self.edit_status = QLabel("")
        self.edit_status.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        save_row.addWidget(self.edit_status)
        eg.addLayout(save_row)
        right_layout.addWidget(edit_group)

        # 动作分布统计
        stats_group = QGroupBox("动作分布")
        sg = QVBoxLayout(stats_group)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(160)
        sg.addWidget(self.stats_text)
        right_layout.addWidget(stats_group)

        # 数据集质量
        quality_group = QGroupBox("数据质量")
        qg = QVBoxLayout(quality_group)
        self.quality_label = QLabel("")
        self.quality_label.setWordWrap(True)
        self.quality_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        qg.addWidget(self.quality_label)
        right_layout.addWidget(quality_group)

        right_layout.addStretch()
        splitter.addWidget(right)

        splitter.setSizes([700, 350])
        layout.addWidget(splitter, 1)

        # 底部快捷键提示
        shortcut_label = QLabel(
            "快捷键:  ← → 切换帧  |  Space 播放/暂停  |  Del 删除  |  Ctrl+S 保存"
        )
        shortcut_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        shortcut_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(shortcut_label)

    def _init_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, self._prev_frame)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._next_frame)
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self._delete_sample)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_corrections)

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

    @Slot()
    def _load_compare(self):
        """加载第二份标注文件进行 A/B 对比。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择对比标注文件", "",
            "JSONL (*.jsonl);;所有文件 (*)"
        )
        if not path:
            return

        self._compare_samples.clear()
        self._compare_map.clear()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    self._compare_samples.append(sample)
                    fid = sample.get("frame_id", 0)
                    action = self._get_action(sample)
                    if action:
                        self._compare_map[fid] = action
                except json.JSONDecodeError:
                    continue

        if self._compare_map:
            # 统计差异
            diff_count = 0
            match_count = 0
            for s in self._samples:
                fid = s.get("frame_id", 0)
                if fid in self._compare_map:
                    if self._get_action(s) == self._compare_map[fid]:
                        match_count += 1
                    else:
                        diff_count += 1

            total = match_count + diff_count
            agreement = match_count / total * 100 if total > 0 else 0
            QMessageBox.information(
                self, "对比已加载",
                f"对比文件: {Path(path).name}\n"
                f"匹配帧: {total}\n"
                f"一致: {match_count} ({agreement:.1f}%)\n"
                f"不一致: {diff_count}\n\n"
                "不一致帧底部横幅将显示两个决策"
            )
            self._show_sample(self._current_idx)
        else:
            QMessageBox.warning(self, "提示", "对比文件为空")

    # -- 数据加载 --

    @Slot()
    def _load_data(self):
        jsonl_path = self.jsonl_input.text().strip()
        if not jsonl_path or not Path(jsonl_path).exists():
            QMessageBox.warning(self, "提示", "请选择有效的 JSONL 文件")
            return

        self._jsonl_path = jsonl_path
        self._samples.clear()
        self._action_color_map.clear()
        self._deleted_indices.clear()
        self._modified = False

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

        # 构建动作列表和颜色映射
        self._all_actions.clear()
        for s in self._samples:
            action = self._get_action(s)
            if action and action not in self._all_actions:
                self._all_actions.append(action)
        for i, a in enumerate(self._all_actions):
            self._action_color_map[a] = _ACTION_COLORS[i % len(_ACTION_COLORS)]

        # 更新动作下拉框
        self.action_combo.clear()
        self.action_combo.addItems(self._all_actions)

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
        self._update_quality()
        self._show_sample(0)
        self.save_btn.setEnabled(False)
        self.edit_status.setText("")

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
            self.play_btn.setStyleSheet("")
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

    # -- 编辑功能 --

    @Slot()
    def _apply_edit(self):
        """修改当前帧的动作标注。"""
        if not self._samples or self._current_idx >= len(self._samples):
            return
        new_action = self.action_combo.currentText().strip()
        if not new_action:
            return

        sample = self._samples[self._current_idx]
        ha = sample.setdefault("human_action", {})
        old_action = ha.get("key", "") or ha.get("action", "")
        if new_action == old_action:
            return

        ha["key"] = new_action
        if "action" in ha and ha["action"] != "press":
            ha["action"] = "press"

        # 记录新动作到列表
        if new_action not in self._all_actions:
            self._all_actions.append(new_action)
            self._action_color_map[new_action] = _ACTION_COLORS[
                len(self._all_actions) % len(_ACTION_COLORS)]
            self.action_combo.addItem(new_action)

        self._modified = True
        self.save_btn.setEnabled(True)
        self.edit_status.setText(f"已修改: {old_action} → {new_action}")
        self.edit_status.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px;")
        self._update_stats()
        self._update_quality()
        self._show_sample(self._current_idx)

    @Slot()
    def _delete_sample(self):
        """标记当前帧为删除。"""
        if not self._samples or self._current_idx >= len(self._samples):
            return
        self._deleted_indices.add(self._current_idx)
        self._modified = True
        self.save_btn.setEnabled(True)
        self.edit_status.setText(f"已标记删除 ({len(self._deleted_indices)} 帧待删除)")
        self.edit_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px;")
        self._update_stats()
        self._update_quality()
        self._show_sample(self._current_idx)

    @Slot()
    def _save_corrections(self):
        """保存修正后的 JSONL 文件。"""
        if not self._modified or not self._jsonl_path:
            return

        # 过滤掉删除的样本
        kept = [s for i, s in enumerate(self._samples) if i not in self._deleted_indices]
        deleted_count = len(self._deleted_indices)

        # 写回文件
        with open(self._jsonl_path, "w", encoding="utf-8") as f:
            for sample in kept:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # 更新内部状态
        self._samples = kept
        self._deleted_indices.clear()
        self._modified = False
        self.save_btn.setEnabled(False)

        # 重新调整滑块
        self.slider.setMaximum(max(0, len(self._samples) - 1))
        if self._current_idx >= len(self._samples):
            self._current_idx = max(0, len(self._samples) - 1)
        self.slider.setValue(self._current_idx)

        self.edit_status.setText(f"已保存 ({len(kept)} 条, 删除 {deleted_count} 条)")
        self.edit_status.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px;")
        self._update_stats()
        self._update_quality()
        self._show_sample(self._current_idx)

    # -- 渲染 --

    def _show_sample(self, idx: int):
        if idx < 0 or idx >= len(self._samples):
            return

        is_deleted = idx in self._deleted_indices
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

        if is_deleted:
            self.action_label.setText(f"{action} [已删除]")
            self.action_label.setStyleSheet(
                f"color: {COLORS['danger']}; font-size: 22px; font-weight: bold; "
                "text-decoration: line-through;"
            )
        else:
            self.action_label.setText(action or "--")
            self.action_label.setStyleSheet(
                f"color: {hex_color}; font-size: 22px; font-weight: bold;"
            )

        self.reason_label.setText(reason if reason else "")

        # 设置编辑框当前值
        combo_idx = self.action_combo.findText(action)
        if combo_idx >= 0:
            self.action_combo.setCurrentIndex(combo_idx)

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

        # 对比信息
        compare_action = self._compare_map.get(frame_id, "")

        # 渲染画面
        frame = self._get_video_frame(frame_id, frame_size)
        self._draw_annotated_frame(frame, detections, action, reason, color,
                                   is_deleted, compare_action)

    def _get_video_frame(self, frame_id: int, frame_size: list) -> np.ndarray:
        if self._cap and self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id - 1))
            ret, frame = self._cap.read()
            if ret:
                return frame

        w = frame_size[0] if frame_size[0] > 0 else 640
        h = frame_size[1] if frame_size[1] > 0 else 480
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (20, 25, 35)
        cv2.putText(frame, "No video loaded", (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 120), 1, cv2.LINE_AA)
        return frame

    def _draw_annotated_frame(self, frame: np.ndarray, detections: list,
                               action: str, reason: str, action_color: tuple,
                               is_deleted: bool = False,
                               compare_action: str = ""):
        h, w = frame.shape[:2]

        # 绘制检测框
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            bbox_norm = det.get("bbox_norm")
            class_name = det.get("class_name", "?")
            conf = det.get("confidence", 0)
            class_id = det.get("class_id", 0)

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

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

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

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, banner_y), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            if is_deleted:
                cv2.putText(frame, f"[DELETED] {action}", (36, banner_y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2, cv2.LINE_AA)
                cv2.line(frame, (36, banner_y + 16), (36 + 300, banner_y + 16),
                         (100, 100, 100), 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, (20, banner_y + banner_h // 2), 8,
                           action_color, -1, cv2.LINE_AA)
                cv2.putText(frame, f"ACTION: {action}", (36, banner_y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2, cv2.LINE_AA)

                if compare_action and compare_action != action:
                    # A/B 对比：显示差异
                    cv2.putText(frame, f"vs B: {compare_action}", (w // 2, banner_y + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (241, 196, 15), 2, cv2.LINE_AA)
                    if reason:
                        max_len = max(30, w // 16)
                        display_reason = reason[:max_len] + "..." if len(reason) > max_len else reason
                        cv2.putText(frame, display_reason, (36, banner_y + 42),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 210), 1, cv2.LINE_AA)
                elif reason:
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
        dist: dict[str, int] = {}
        for i, s in enumerate(self._samples):
            if i in self._deleted_indices:
                continue
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

    def _update_quality(self):
        """更新数据集质量分析。"""
        active = [s for i, s in enumerate(self._samples) if i not in self._deleted_indices]
        total = len(active)
        if total == 0:
            self.quality_label.setText("无数据")
            return

        # 动作分布
        dist: dict[str, int] = {}
        for s in active:
            action = self._get_action(s)
            if action:
                dist[action] = dist.get(action, 0) + 1

        # 均衡度分析
        counts = list(dist.values()) if dist else [0]
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / max(min_count, 1)

        # 评级
        issues = []
        if total < 50:
            issues.append(f"样本过少 ({total})，建议 > 100")
        if imbalance_ratio > 5:
            least = min(dist, key=dist.get)
            issues.append(f"严重不均衡: '{least}' 仅 {dist[least]} 条 (比例 1:{imbalance_ratio:.0f})")
        elif imbalance_ratio > 3:
            least = min(dist, key=dist.get)
            issues.append(f"轻度不均衡: '{least}' 仅 {dist[least]} 条")

        if len(dist) < 2:
            issues.append("只有 1 种动作，无法训练分类器")

        if self._deleted_indices:
            issues.append(f"{len(self._deleted_indices)} 帧待删除")

        lines = [f"有效样本: {total} 条  |  动作种类: {len(dist)}"]
        if issues:
            lines.append("")
            for issue in issues:
                lines.append(f"  ! {issue}")
        else:
            lines.append("  数据质量良好")

        self.quality_label.setText("\n".join(lines))

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
        if self._modified:
            reply = QMessageBox.question(
                self, "未保存的修改",
                "有未保存的修正，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self._save_corrections()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return

        self._playing = False
        self._play_timer.stop()
        if self._cap:
            self._cap.release()
        event.accept()
