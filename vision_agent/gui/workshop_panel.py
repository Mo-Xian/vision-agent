"""训练工坊面板：场景管理 → 视频输入 → LLM 分析 → 模型训练 一站式界面。"""

import json
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QLineEdit, QPushButton, QSpinBox,
    QDoubleSpinBox, QTextEdit, QFormLayout, QProgressBar,
    QScrollArea, QFrame, QFileDialog, QListWidget,
    QListWidgetItem, QCheckBox, QAbstractItemView,
    QInputDialog, QMessageBox,
)

from .styles import COLORS
from .widgets import CollapsibleSection


class WorkshopPanel(QWidget):
    """训练工坊面板 — 场景驱动的一站式视频学习流程。

    布局：场景管理 → 视频输入 → 学习控制 → 进度/见解 → 模型管理 → 高级设置
    """

    learn_requested = Signal()
    stop_requested = Signal()
    apply_model_requested = Signal(str)  # model_dir
    scene_changed = Signal(str)  # scene_name

    _VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts", ".m4v"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_scene = None
        self._scene_manager = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ━━━ 场景管理 ━━━
        scene_group = QGroupBox("场景")
        sg = QHBoxLayout(scene_group)
        sg.setSpacing(6)

        self.scene_combo = QComboBox()
        self.scene_combo.setMinimumWidth(160)
        self.scene_combo.setToolTip("选择或创建训练场景")
        self.scene_combo.currentTextChanged.connect(self._on_scene_selected)
        sg.addWidget(self.scene_combo, 1)

        self.new_scene_btn = QPushButton("+ 新建")
        self.new_scene_btn.setObjectName("startBtn")
        self.new_scene_btn.setCursor(Qt.PointingHandCursor)
        self.new_scene_btn.clicked.connect(self._new_scene)
        sg.addWidget(self.new_scene_btn)

        self.delete_scene_btn = QPushButton("删除")
        self.delete_scene_btn.setObjectName("stopBtn")
        self.delete_scene_btn.setCursor(Qt.PointingHandCursor)
        self.delete_scene_btn.clicked.connect(self._delete_scene)
        sg.addWidget(self.delete_scene_btn)

        # 场景状态标签
        self.scene_status = QLabel("")
        self.scene_status.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        sg.addWidget(self.scene_status)

        layout.addWidget(scene_group)

        # ━━━ 视频输入 ━━━
        video_group = QGroupBox("视频源")
        vg = QVBoxLayout(video_group)
        vg.setSpacing(6)

        # 输入方式选择
        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("来源"))
        self.input_type = QComboBox()
        self.input_type.addItems(["本地视频", "视频文件夹", "RTSP/流地址"])
        self.input_type.currentTextChanged.connect(self._on_input_type_changed)
        input_row.addWidget(self.input_type, 1)
        vg.addLayout(input_row)

        # 文件/文件夹路径
        path_row = QHBoxLayout()
        self.video_path_input = QLineEdit()
        self.video_path_input.setPlaceholderText("视频文件或文件夹路径")
        path_row.addWidget(self.video_path_input, 1)
        self.browse_file_btn = QPushButton("文件")
        self.browse_file_btn.setObjectName("browseBtn")
        self.browse_file_btn.setMaximumWidth(50)
        self.browse_file_btn.clicked.connect(self._browse_video_files)
        path_row.addWidget(self.browse_file_btn)
        self.browse_dir_btn = QPushButton("文件夹")
        self.browse_dir_btn.setObjectName("browseBtn")
        self.browse_dir_btn.setMaximumWidth(60)
        self.browse_dir_btn.clicked.connect(self._browse_video_dir)
        path_row.addWidget(self.browse_dir_btn)
        vg.addLayout(path_row)

        # 流地址
        self.stream_input = QLineEdit()
        self.stream_input.setPlaceholderText("rtsp://... 或 B站直播间号")
        self.stream_input.setVisible(False)
        vg.addWidget(self.stream_input)

        # 添加到场景按钮
        add_row = QHBoxLayout()
        self.add_videos_btn = QPushButton("添加到场景")
        self.add_videos_btn.setObjectName("purpleBtn")
        self.add_videos_btn.setCursor(Qt.PointingHandCursor)
        self.add_videos_btn.clicked.connect(self._add_videos_to_scene)
        add_row.addWidget(self.add_videos_btn)
        add_row.addStretch()
        vg.addLayout(add_row)

        # 场景视频列表
        self.video_list = QListWidget()
        self.video_list.setMaximumHeight(120)
        self.video_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.video_list.setStyleSheet(
            f"QListWidget {{ background: {COLORS['bg_input']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 6px; "
            f"color: {COLORS['text']}; font-size: 12px; padding: 4px; }}"
        )
        vg.addWidget(self.video_list)

        # 视频列表操作
        video_ops = QHBoxLayout()
        self.remove_video_btn = QPushButton("移除选中")
        self.remove_video_btn.setObjectName("browseBtn")
        self.remove_video_btn.clicked.connect(self._remove_selected_videos)
        video_ops.addWidget(self.remove_video_btn)
        self.video_count_label = QLabel("0 个视频")
        self.video_count_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        video_ops.addWidget(self.video_count_label)
        video_ops.addStretch()
        vg.addLayout(video_ops)

        # 场景描述
        desc_row = QHBoxLayout()
        desc_row.addWidget(QLabel("描述"))
        self.description_input = QLineEdit()
        self.description_input.setPlaceholderText("场景描述（可选，如：王者荣耀5v5团战）")
        desc_row.addWidget(self.description_input, 1)
        vg.addLayout(desc_row)

        layout.addWidget(video_group)

        # ━━━ 场景知识 ━━━
        knowledge_group = QGroupBox("场景知识（可选，提升 LLM 分析准确度）")
        kg = QVBoxLayout(knowledge_group)
        kg.setSpacing(4)

        kb_btn_row = QHBoxLayout()
        self.kb_import_btn = QPushButton("导入文件")
        self.kb_import_btn.setObjectName("browseBtn")
        self.kb_import_btn.setCursor(Qt.PointingHandCursor)
        self.kb_import_btn.clicked.connect(self._import_knowledge)
        kb_btn_row.addWidget(self.kb_import_btn)

        self.kb_clear_btn = QPushButton("清空")
        self.kb_clear_btn.setObjectName("browseBtn")
        self.kb_clear_btn.setCursor(Qt.PointingHandCursor)
        self.kb_clear_btn.clicked.connect(lambda: self.knowledge_input.clear())
        kb_btn_row.addWidget(self.kb_clear_btn)

        self.kb_char_label = QLabel("0 字")
        self.kb_char_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        kb_btn_row.addWidget(self.kb_char_label)
        kb_btn_row.addStretch()
        kg.addLayout(kb_btn_row)

        self.knowledge_input = QTextEdit()
        self.knowledge_input.setMaximumHeight(120)
        self.knowledge_input.setPlaceholderText(
            "输入或导入场景规则/教程，例如：\n"
            "• 这是王者荣耀5v5 MOBA 游戏\n"
            "• 核心操作：普攻、技能1/2/3、闪现、回城\n"
            "• 血量低于30%应回城或撤退\n"
            "支持导入 .txt / .md / .json / .yaml 文件"
        )
        self.knowledge_input.textChanged.connect(self._on_knowledge_changed)
        kg.addWidget(self.knowledge_input)

        layout.addWidget(knowledge_group)

        # ━━━ 核心操作 ━━━
        self.learn_btn = QPushButton("开始学习")
        self.learn_btn.setObjectName("startBtn")
        self.learn_btn.setCursor(Qt.PointingHandCursor)
        self.learn_btn.setMinimumHeight(42)
        self.learn_btn.setStyleSheet(
            self.learn_btn.styleSheet() +
            "QPushButton#startBtn { font-size: 15px; }"
        )
        self.learn_btn.clicked.connect(self.learn_requested)
        layout.addWidget(self.learn_btn)

        self.stop_learn_btn = QPushButton("停止学习")
        self.stop_learn_btn.setObjectName("stopBtn")
        self.stop_learn_btn.setCursor(Qt.PointingHandCursor)
        self.stop_learn_btn.setEnabled(False)
        self.stop_learn_btn.setVisible(False)
        self.stop_learn_btn.clicked.connect(self.stop_requested)
        layout.addWidget(self.stop_learn_btn)

        # ━━━ 学习进度 ━━━
        progress_group = QGroupBox("学习进度")
        pg = QVBoxLayout(progress_group)
        pg.setSpacing(6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("准备中")
        pg.addWidget(self.progress_bar)

        self.phase_label = QLabel("等待启动...")
        self.phase_label.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px;")
        pg.addWidget(self.phase_label)

        layout.addWidget(progress_group)

        # ━━━ LLM 见解 ━━━
        insight_group = QGroupBox("LLM 见解")
        ig = QVBoxLayout(insight_group)
        ig.setSpacing(4)

        self.insight_text = QTextEdit()
        self.insight_text.setReadOnly(True)
        self.insight_text.setMaximumHeight(120)
        self.insight_text.setPlaceholderText("分析后显示 LLM 对视频内容的见解...")
        ig.addWidget(self.insight_text)

        layout.addWidget(insight_group)

        # ━━━ 模型管理 ━━━
        model_group = QGroupBox("训练产出")
        mg_layout = QVBoxLayout(model_group)
        mg_layout.setSpacing(6)

        from .train_chart import TrainChart
        self.train_chart = TrainChart()
        self.train_chart.setVisible(False)
        mg_layout.addWidget(self.train_chart)

        self.model_status = QLabel("")
        self.model_status.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px;")
        mg_layout.addWidget(self.model_status)

        model_btn_row = QHBoxLayout()
        self.apply_model_btn = QPushButton("应用到 Agent")
        self.apply_model_btn.setObjectName("purpleBtn")
        self.apply_model_btn.setCursor(Qt.PointingHandCursor)
        self.apply_model_btn.setEnabled(False)
        model_btn_row.addWidget(self.apply_model_btn)

        self.view_models_btn = QPushButton("查看所有模型")
        self.view_models_btn.setObjectName("infoBtn")
        self.view_models_btn.setCursor(Qt.PointingHandCursor)
        model_btn_row.addWidget(self.view_models_btn)

        self.view_sessions_btn = QPushButton("训练历史")
        self.view_sessions_btn.setObjectName("infoBtn")
        self.view_sessions_btn.setCursor(Qt.PointingHandCursor)
        self.view_sessions_btn.clicked.connect(self._show_session_history)
        model_btn_row.addWidget(self.view_sessions_btn)

        mg_layout.addLayout(model_btn_row)

        layout.addWidget(model_group)

        # ━━━ 日志 ━━━
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)

        # ━━━ 高级设置（折叠） ━━━
        more = CollapsibleSection("高级设置")
        mg = more.content_layout()

        mg.addWidget(QLabel("LLM 配置使用全局 LLM 设置"))

        # 训练参数
        mg.addWidget(QLabel("── 训练参数 ──"))
        train_form = QFormLayout()
        train_form.setSpacing(4)

        self.sample_count_spin = QSpinBox()
        self.sample_count_spin.setRange(20, 5000)
        self.sample_count_spin.setValue(300)
        train_form.addRow("标注帧数", self.sample_count_spin)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["e2e（端到端）", "mlp", "rf"])
        train_form.addRow("模型类型", self.model_type_combo)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000)
        self.epochs_spin.setValue(100)
        train_form.addRow("训练轮数", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.0005)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.001)
        train_form.addRow("学习率", self.lr_spin)

        self.rl_steps_spin = QSpinBox()
        self.rl_steps_spin.setRange(0, 50000)
        self.rl_steps_spin.setValue(2000)
        self.rl_steps_spin.setSingleStep(500)
        train_form.addRow("RL 步数", self.rl_steps_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10)
        self.batch_size_spin.setValue(5)
        self.batch_size_spin.setToolTip("每次 LLM 调用标注的帧数（批量模式）")
        train_form.addRow("批量大小", self.batch_size_spin)

        self.send_image_check = QCheckBox("发送画面给 LLM（多模态，推荐开启）")
        self.send_image_check.setChecked(True)
        train_form.addRow("", self.send_image_check)

        mg.addLayout(train_form)

        # YOLO 模型
        mg.addWidget(QLabel("── YOLO 模型（可选） ──"))
        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.addItems(["（不使用 YOLO）", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
        self.yolo_model_combo.setEditable(True)
        self.yolo_model_combo.setToolTip("LLM 视觉分析为主，YOLO 为可选辅助")
        mg.addWidget(self.yolo_model_combo)

        yolo_btn_row = QHBoxLayout()
        self.load_yolo_btn = QPushButton("加载自定义")
        self.load_yolo_btn.setObjectName("browseBtn")
        yolo_btn_row.addWidget(self.load_yolo_btn)
        self.train_yolo_btn = QPushButton("训练 YOLO")
        self.train_yolo_btn.setObjectName("purpleBtn")
        self.train_yolo_btn.setCursor(Qt.PointingHandCursor)
        yolo_btn_row.addWidget(self.train_yolo_btn)
        mg.addLayout(yolo_btn_row)

        yolo_form = QFormLayout()
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.50)
        yolo_form.addRow("置信度", self.conf_spin)
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "416", "512", "640", "960", "1280"])
        self.imgsz_combo.setCurrentText("640")
        yolo_form.addRow("分辨率", self.imgsz_combo)
        mg.addLayout(yolo_form)

        # 手动工具
        mg.addWidget(QLabel("── 手动工具 ──"))
        manual_row = QHBoxLayout()
        self.auto_annotate_btn = QPushButton("手动标注")
        self.auto_annotate_btn.setObjectName("purpleBtn")
        self.auto_annotate_btn.setCursor(Qt.PointingHandCursor)
        manual_row.addWidget(self.auto_annotate_btn)
        self.view_annotation_btn = QPushButton("查看标注")
        self.view_annotation_btn.setObjectName("infoBtn")
        self.view_annotation_btn.setCursor(Qt.PointingHandCursor)
        manual_row.addWidget(self.view_annotation_btn)
        mg.addLayout(manual_row)

        # 手动训练
        mg.addWidget(QLabel("── 手动训练 ──"))
        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("数据"))
        self.dt_data_dir = QLineEdit("data/recordings")
        data_row.addWidget(self.dt_data_dir)
        self.dt_data_browse = QPushButton("...")
        self.dt_data_browse.setObjectName("browseBtn")
        self.dt_data_browse.setMaximumWidth(32)
        data_row.addWidget(self.dt_data_browse)
        mg.addLayout(data_row)

        manual_train_row = QHBoxLayout()
        self.dt_preview_btn = QPushButton("预览数据")
        self.dt_preview_btn.setObjectName("infoBtn")
        self.dt_preview_btn.setCursor(Qt.PointingHandCursor)
        manual_train_row.addWidget(self.dt_preview_btn)
        self.dt_train_btn = QPushButton("手动训练")
        self.dt_train_btn.setObjectName("startBtn")
        self.dt_train_btn.setCursor(Qt.PointingHandCursor)
        manual_train_row.addWidget(self.dt_train_btn)
        mg.addLayout(manual_train_row)

        self.dt_output_dir = QLineEdit("runs/decision/exp1")
        mg.addWidget(self.dt_output_dir)

        self.dt_progress = QProgressBar()
        self.dt_progress.setRange(0, 100)
        self.dt_progress.setVisible(False)
        mg.addWidget(self.dt_progress)

        self.dt_status_label = QLabel("")
        self.dt_status_label.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px;")
        mg.addWidget(self.dt_status_label)

        # Profile 管理
        mg.addWidget(QLabel("── Profile 管理 ──"))
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Profile"))
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("(无)")
        profile_row.addWidget(self.profile_combo, 1)
        self.refresh_profiles_btn = QPushButton("刷新")
        self.refresh_profiles_btn.setObjectName("browseBtn")
        self.refresh_profiles_btn.setCursor(Qt.PointingHandCursor)
        profile_row.addWidget(self.refresh_profiles_btn)
        mg.addLayout(profile_row)

        self.profile_info = QTextEdit()
        self.profile_info.setReadOnly(True)
        self.profile_info.setMaximumHeight(80)
        self.profile_info.setPlaceholderText("选择 Profile 查看详情")
        mg.addWidget(self.profile_info)

        ie_row = QHBoxLayout()
        self.export_btn = QPushButton("导出配置")
        self.export_btn.setObjectName("infoBtn")
        ie_row.addWidget(self.export_btn)
        self.import_btn = QPushButton("导入配置")
        self.import_btn.setObjectName("browseBtn")
        ie_row.addWidget(self.import_btn)
        mg.addLayout(ie_row)

        # 人工录制
        mg.addWidget(QLabel("── 人工录制 ──"))
        rec_dir_row = QHBoxLayout()
        rec_dir_row.addWidget(QLabel("目录"))
        self.rec_dir_input = QLineEdit("data/recordings")
        rec_dir_row.addWidget(self.rec_dir_input)
        self.rec_browse_btn = QPushButton("...")
        self.rec_browse_btn.setObjectName("browseBtn")
        self.rec_browse_btn.setMaximumWidth(32)
        rec_dir_row.addWidget(self.rec_browse_btn)
        mg.addLayout(rec_dir_row)
        self.rec_session_input = QLineEdit()
        self.rec_session_input.setPlaceholderText("会话名称（留空自动生成）")
        mg.addWidget(self.rec_session_input)
        self.rec_start_btn = QPushButton("开始录制")
        self.rec_start_btn.setObjectName("stopBtn")
        self.rec_start_btn.setCursor(Qt.PointingHandCursor)
        mg.addWidget(self.rec_start_btn)
        self.rec_status_label = QLabel("就绪")
        self.rec_status_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        mg.addWidget(self.rec_status_label)

        layout.addWidget(more)
        layout.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll)

    # ── 场景管理方法 ──

    def init_scene_manager(self):
        """初始化场景管理器并刷新列表。"""
        from ..workshop.scene import SceneManager
        self._scene_manager = SceneManager()
        self._refresh_scene_list()

    def _refresh_scene_list(self):
        """刷新场景下拉列表。"""
        if not self._scene_manager:
            return
        current = self.scene_combo.currentText()
        self.scene_combo.blockSignals(True)
        self.scene_combo.clear()
        self.scene_combo.addItem("（选择场景或新建）")
        scenes = self._scene_manager.list_scenes()
        for s in scenes:
            status_icon = {"ready": "✓", "idle": "○", "training": "◉"}.get(s.status, "○")
            label = f"{status_icon} {s.display_name}"
            if s.best_val_acc > 0:
                label += f" (acc: {s.best_val_acc:.1%})"
            self.scene_combo.addItem(label, s.name)
        # 恢复之前选择
        for i in range(self.scene_combo.count()):
            if self.scene_combo.itemData(i) == current or self.scene_combo.itemText(i) == current:
                self.scene_combo.setCurrentIndex(i)
                break
        self.scene_combo.blockSignals(False)

    def _on_scene_selected(self, text: str):
        """场景选择变更。"""
        if not self._scene_manager:
            return
        idx = self.scene_combo.currentIndex()
        name = self.scene_combo.itemData(idx)
        if not name:
            self._current_scene = None
            self._update_scene_ui(None)
            return
        scene = self._scene_manager.load(name)
        if scene:
            self._current_scene = scene
            self._update_scene_ui(scene)
            self.scene_changed.emit(name)

    def _update_scene_ui(self, scene):
        """根据场景数据更新 UI。"""
        if scene is None:
            self.video_list.clear()
            self.video_count_label.setText("0 个视频")
            self.description_input.clear()
            self.knowledge_input.clear()
            self.scene_status.setText("")
            self.insight_text.clear()
            self.model_status.setText("")
            self.apply_model_btn.setEnabled(False)
            return

        # 视频列表
        self.video_list.clear()
        for v in scene.video_sources:
            item = QListWidgetItem(Path(v).name)
            item.setData(Qt.UserRole, v)
            item.setToolTip(v)
            self.video_list.addItem(item)
        self.video_count_label.setText(f"{len(scene.video_sources)} 个视频")

        # 描述
        self.description_input.setText(scene.display_name)

        # 知识
        self.knowledge_input.blockSignals(True)
        self.knowledge_input.setPlainText(scene.knowledge or "")
        self.knowledge_input.blockSignals(False)

        # 状态
        status_map = {
            "idle": "空闲",
            "analyzing": "分析中...",
            "annotating": "标注中...",
            "training": "训练中...",
            "ready": "模型就绪",
        }
        self.scene_status.setText(status_map.get(scene.status, scene.status))

        # 见解
        if scene.analysis_summary:
            lines = []
            if scene.scene_type:
                lines.append(f"场景: {scene.scene_type}")
            if scene.actions:
                lines.append(f"动作: {', '.join(scene.actions)}")
            if scene.analysis_summary:
                lines.append(f"\n{scene.analysis_summary}")
            self.insight_text.setPlainText("\n".join(lines))

        # 模型状态
        if scene.best_model_dir:
            self.model_status.setText(
                f"最佳模型: acc={scene.best_val_acc:.1%} | "
                f"训练 {scene.session_count} 次 | 标注 {scene.total_annotated} 条"
            )
            self.apply_model_btn.setEnabled(True)
        else:
            self.model_status.setText(f"训练 {scene.session_count} 次 | 标注 {scene.total_annotated} 条")
            self.apply_model_btn.setEnabled(False)

    def _new_scene(self):
        """创建新场景。"""
        name, ok = QInputDialog.getText(
            self, "新建场景", "场景名称（如：王者荣耀5v5）:",
        )
        if not ok or not name.strip():
            return
        if not self._scene_manager:
            self.init_scene_manager()
        scene = self._scene_manager.create(name.strip(), display_name=name.strip())
        self._refresh_scene_list()
        # 选中新建的场景
        for i in range(self.scene_combo.count()):
            if self.scene_combo.itemData(i) == scene.name:
                self.scene_combo.setCurrentIndex(i)
                break

    def _delete_scene(self):
        """删除当前场景。"""
        if not self._current_scene or not self._scene_manager:
            return
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定删除场景「{self._current_scene.display_name}」？\n（训练数据将移至回收站）",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._scene_manager.delete(self._current_scene.name)
            self._current_scene = None
            self._refresh_scene_list()
            self.scene_combo.setCurrentIndex(0)

    def _add_videos_to_scene(self):
        """将当前路径中的视频添加到场景。"""
        if not self._current_scene:
            QMessageBox.information(self, "提示", "请先选择或创建一个场景")
            return
        paths = self._get_input_paths()
        if not paths:
            return
        self._current_scene.add_videos(paths)
        self._update_scene_ui(self._current_scene)

    def _remove_selected_videos(self):
        """从场景中移除选中的视频。"""
        if not self._current_scene:
            return
        selected = self.video_list.selectedItems()
        if not selected:
            return
        paths = [item.data(Qt.UserRole) for item in selected]
        self._current_scene.remove_videos(paths)
        self._update_scene_ui(self._current_scene)

    def _show_session_history(self):
        """显示训练历史。"""
        if not self._current_scene:
            QMessageBox.information(self, "提示", "请先选择场景")
            return
        sessions = self._current_scene.list_sessions()
        if not sessions:
            QMessageBox.information(self, "训练历史", "暂无训练记录")
            return
        lines = []
        for s in sessions[:10]:
            model_dir = s.get("model_dir", "")
            acc = s.get("metrics", {}).get("best_val_acc", 0)
            ann = s.get("annotated_count", 0)
            success = "✓" if s.get("success") else "✗"
            lines.append(f"{success} acc={acc:.1%} | 标注 {ann} 条 | {model_dir}")
        QMessageBox.information(
            self, f"训练历史 - {self._current_scene.display_name}",
            "\n".join(lines)
        )

    def _on_knowledge_changed(self):
        """知识输入变更时保存到场景。"""
        text = self.knowledge_input.toPlainText().strip()
        self.kb_char_label.setText(f"{len(text)} 字")
        if self._current_scene:
            self._current_scene.knowledge = text
            self._current_scene.save()

    def _import_knowledge(self):
        """从文件导入场景知识。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "导入场景知识",  "",
            "知识文件 (*.txt *.md *.json *.yaml *.yml);;所有文件 (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="gbk") as f:
                content = f.read()

        if not content.strip():
            QMessageBox.warning(self, "提示", "文件内容为空")
            return

        # 如果已有内容，询问追加还是替换
        existing = self.knowledge_input.toPlainText().strip()
        if existing:
            box = QMessageBox(self)
            box.setWindowTitle("导入方式")
            box.setText(f"当前已有 {len(existing)} 字知识内容，如何处理？")
            btn_replace = box.addButton("替换", QMessageBox.ButtonRole.YesRole)
            btn_append = box.addButton("追加", QMessageBox.ButtonRole.NoRole)
            box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked == btn_append:
                content = existing + "\n\n" + content
            elif clicked != btn_replace:
                return

        self.knowledge_input.setPlainText(content.strip())

    @property
    def current_scene(self):
        return self._current_scene

    # ── 视频输入辅助方法 ──

    def _on_input_type_changed(self, text: str):
        is_stream = text == "RTSP/流地址"
        self.stream_input.setVisible(is_stream)
        self.video_path_input.setVisible(not is_stream)
        self.browse_file_btn.setVisible(not is_stream)
        self.browse_dir_btn.setVisible(text == "视频文件夹")

    def _browse_video_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.webm *.flv *.ts *.m4v);;所有文件 (*)"
        )
        if files:
            self.video_path_input.setText(";".join(files))
            # 自动添加到场景
            if self._current_scene:
                self._current_scene.add_videos(files)
                self._update_scene_ui(self._current_scene)

    def _browse_video_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择视频文件夹")
        if folder:
            self.video_path_input.setText(folder)
            files = self._scan_folder(folder)
            if self._current_scene and files:
                self._current_scene.add_videos(files)
                self._update_scene_ui(self._current_scene)

    def _scan_folder(self, folder: str) -> list[str]:
        files = []
        for f in sorted(Path(folder).iterdir()):
            if f.is_file() and f.suffix.lower() in self._VIDEO_EXTS:
                files.append(str(f))
        return files

    def _get_input_paths(self) -> list[str]:
        """获取输入区域中的路径列表。"""
        input_type = self.input_type.currentText()
        if input_type == "RTSP/流地址":
            url = self.stream_input.text().strip()
            return [url] if url else []
        paths_text = self.video_path_input.text().strip()
        if not paths_text:
            return []
        raw_paths = [p.strip() for p in paths_text.split(";") if p.strip()]
        result = []
        for p in raw_paths:
            pp = Path(p)
            if pp.is_dir():
                result.extend(self._scan_folder(p))
            elif pp.is_file():
                result.append(str(pp))
        return result

    def get_video_paths(self) -> list[str]:
        """获取当前场景的视频路径列表。"""
        if self._current_scene and self._current_scene.video_sources:
            return list(self._current_scene.video_sources)
        return self._get_input_paths()

    def set_learning_state(self, running: bool):
        self.learn_btn.setEnabled(not running)
        self.learn_btn.setVisible(not running)
        self.stop_learn_btn.setEnabled(running)
        self.stop_learn_btn.setVisible(running)

    def update_insight(self, insight_dict: dict):
        lines = []
        if insight_dict.get("scene_type"):
            lines.append(f"场景: {insight_dict['scene_type']}")
        if insight_dict.get("scene_description"):
            lines.append(f"描述: {insight_dict['scene_description']}")
        if insight_dict.get("suggested_actions"):
            lines.append(f"动作: {', '.join(insight_dict['suggested_actions'])}")
        if insight_dict.get("analysis_summary"):
            lines.append(f"\n{insight_dict['analysis_summary']}")
        self.insight_text.setPlainText("\n".join(lines))
        # 同步到场景
        if self._current_scene:
            self._current_scene.update_from_insight(insight_dict)
