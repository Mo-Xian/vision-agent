"""主窗口：训练工坊 + Agent 执行 双模式界面。"""

import os
import json
from pathlib import Path
from PySide6.QtCore import Qt, Slot, QSettings, QMimeData
from PySide6.QtGui import QFont, QIcon, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGroupBox,
    QLabel, QComboBox, QLineEdit, QPushButton, QSpinBox,
    QDoubleSpinBox, QFileDialog, QTextEdit, QSplitter, QMessageBox,
    QScrollArea, QSizePolicy, QDialog, QTabWidget, QFormLayout,
    QProgressBar, QCheckBox, QStackedWidget, QButtonGroup, QFrame,
)

from ..sources import create_source
from ..core.detector import Detector, DetectionResult
from ..core.model_manager import ModelManager
from ..core.state import StateManager
from ..decision import RuleEngine, LLMEngine, TrainedEngine, HierarchicalEngine, RLEngine, PROVIDER_PRESETS, create_provider, Action
from ..profiles import ProfileManager, SceneProfile
from ..core.scene_classifier import SceneClassifier
from ..core.roi_extractor import ROIExtractor
from ..auto import AutoPilot
from ..data.recorder import DataRecorder
from ..tools import ToolRegistry
from ..tools.keyboard import KeyboardTool
from ..tools.mouse import MouseTool
from ..tools.api_call import ApiCallTool
from ..tools.shell import ShellTool
from ..agents.action_agent import ActionAgent
from .video_widget import VideoWidget
from .worker import DetectionWorker
from .decision_train_worker import DecisionTrainWorker
from .chat_panel import ChatPanel

from .styles import MAIN_STYLESHEET, COLORS

# 模式切换按钮额外样式
_MODE_BTN_STYLE = f"""
QPushButton#modeBtnActive {{
    background-color: {COLORS['accent']};
    color: white;
    font-size: 14px;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 0px;
    border: none;
}}
QPushButton#modeBtnInactive {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_secondary']};
    font-size: 14px;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 0px;
    border: 1px solid {COLORS['border']};
}}
QPushButton#modeBtnInactive:hover {{
    color: {COLORS['text']};
    border-color: {COLORS['border_hover']};
    background-color: {COLORS['bg_input']};
}}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Agent")

        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(1400, int(avail.width() * 0.85))
            h = min(820, int(avail.height() * 0.85))
            min_w = min(1100, int(avail.width() * 0.7))
            min_h = min(700, int(avail.height() * 0.7))
            self.setMinimumSize(min_w, min_h)
            self.resize(w, h)
            self.move(
                avail.x() + (avail.width() - w) // 2,
                avail.y() + (avail.height() - h) // 2,
            )
        else:
            self.setMinimumSize(800, 500)
            self.resize(1200, 720)
        self.setStyleSheet(MAIN_STYLESHEET + _MODE_BTN_STYLE)
        self.setAcceptDrops(True)

        self._worker: DetectionWorker | None = None
        self._frame_count = 0
        self._model_manager = ModelManager()
        self._settings = QSettings("VisionAgent", "VisionAgent")
        self._recorder: DataRecorder | None = None
        self._train_worker: DecisionTrainWorker | None = None
        self._profile_mgr = ProfileManager("profiles")
        self._scene_classifier = SceneClassifier()
        self._auto_pilot: AutoPilot | None = None
        self._current_mode = "agent"
        self._init_ui()
        self._scan_models()
        self._load_settings()

    # ================================================================
    #  UI 构建
    # ================================================================

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # ===== 左侧面板 =====
        left_panel = QWidget()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(440)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # -- 模式切换 --
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self.mode_train_btn = QPushButton("训练工坊")
        self.mode_agent_btn = QPushButton("Agent 执行")
        self.mode_train_btn.setCursor(Qt.PointingHandCursor)
        self.mode_agent_btn.setCursor(Qt.PointingHandCursor)
        self.mode_train_btn.clicked.connect(lambda: self._switch_mode("train"))
        self.mode_agent_btn.clicked.connect(lambda: self._switch_mode("agent"))
        mode_row.addWidget(self.mode_train_btn)
        mode_row.addWidget(self.mode_agent_btn)
        left_layout.addLayout(mode_row)

        # -- 模式内容 (Stacked) --
        self.mode_stack = QStackedWidget()

        # 训练模式 tabs
        self.train_tabs = QTabWidget()
        self._build_train_data_tab()
        self._build_train_annotate_tab()
        self._build_train_training_tab()
        self._build_train_model_tab()
        self.mode_stack.addWidget(self.train_tabs)  # index 0

        # Agent 模式 tabs
        self.agent_tabs = QTabWidget()
        self._build_agent_source_tab()
        self._build_agent_engine_tab()
        self._build_agent_scene_tab()
        self._build_agent_chat_tab()
        self.mode_stack.addWidget(self.agent_tabs)  # index 1

        left_layout.addWidget(self.mode_stack, 1)

        # -- Agent 控制区（仅 Agent 模式可见） --
        self.agent_controls = QWidget()
        ac_layout = QVBoxLayout(self.agent_controls)
        ac_layout.setContentsMargins(0, 0, 0, 0)
        ac_layout.setSpacing(6)

        self.dryrun_check = QCheckBox("仅观察（不控制键盘鼠标）")
        self.dryrun_check.setChecked(True)
        self.dryrun_check.setToolTip("勾选后只显示决策结果，不实际执行键盘/鼠标操作")
        ac_layout.addWidget(self.dryrun_check)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        self.start_btn = QPushButton("▶  启动检测")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._start_detection)
        btn_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("■  停止")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_detection)
        btn_layout.addWidget(self.stop_btn)
        ac_layout.addLayout(btn_layout)

        left_layout.addWidget(self.agent_controls)

        # -- 状态栏 --
        status_w = QWidget()
        status_w.setStyleSheet(
            f"background-color: {COLORS['bg_card']}; "
            f"border: 1px solid {COLORS['border']}; "
            "border-radius: 8px; padding: 4px;"
        )
        status_layout = QVBoxLayout(status_w)
        status_layout.setContentsMargins(10, 6, 10, 6)
        status_layout.setSpacing(3)
        self.fps_label = QLabel("FPS: --  |  推理: --ms")
        self.fps_label.setStyleSheet(
            f"font-size: 14px; color: {COLORS['success']}; font-weight: bold; padding: 0;"
        )
        status_layout.addWidget(self.fps_label)
        self.count_label = QLabel("检测: 0  |  帧: 0")
        self.count_label.setStyleSheet("font-size: 12px; padding: 0;")
        status_layout.addWidget(self.count_label)
        self.engine_status = QLabel("决策: 0  |  执行: 0  |  失败: 0")
        self.engine_status.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 12px; padding: 0;"
        )
        status_layout.addWidget(self.engine_status)
        self.scene_label = QLabel("场景: --")
        self.scene_label.setStyleSheet(
            f"color: {COLORS['purple']}; font-size: 12px; padding: 0;"
        )
        status_layout.addWidget(self.scene_label)
        left_layout.addWidget(status_w)

        # -- 日志区域 --
        self.log_tabs = QTabWidget()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_tabs.addTab(self.log_text, "日志")
        self.decision_log_text = QTextEdit()
        self.decision_log_text.setReadOnly(True)
        self.log_tabs.addTab(self.decision_log_text, "决策")
        left_layout.addWidget(self.log_tabs, 0)

        splitter.addWidget(left_panel)

        # ===== 右侧视频预览 =====
        self.video_widget = VideoWidget()
        splitter.addWidget(self.video_widget)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 1000])

        # 初始化模式
        self._switch_mode("agent")

    # ================================================================
    #  训练模式 Tabs
    # ================================================================

    def _build_train_data_tab(self):
        """训练 Tab 1: 数据获取"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── 自动学习（一键） ──
        auto_group = QGroupBox("一键自动学习")
        ag = QVBoxLayout(auto_group)
        ag.addWidget(QLabel("输入场景描述，自动完成全部流程：\n搜索下载 → YOLO检测 → LLM标注 → 训练 → RL强化 → 生成模型"))
        auto_learn_btn = QPushButton("打开自动学习")
        auto_learn_btn.setObjectName("startBtn")
        auto_learn_btn.setCursor(Qt.PointingHandCursor)
        auto_learn_btn.clicked.connect(self._open_auto_learn)
        ag.addWidget(auto_learn_btn)
        layout.addWidget(auto_group)

        # ── LLM 配置（标注/自动学习共用） ──
        llm_group = QGroupBox("LLM 配置（标注 / 自动学习共用）")
        lg = QFormLayout(llm_group)
        lg.setSpacing(6)

        self.llm_provider_combo = QComboBox()
        self.llm_provider_combo.addItems(list(PROVIDER_PRESETS.keys()))
        self.llm_provider_combo.currentTextChanged.connect(self._on_provider_changed)
        lg.addRow("供应商", self.llm_provider_combo)

        self.llm_model_combo = QComboBox()
        self.llm_model_combo.setEditable(True)
        lg.addRow("模型", self.llm_model_combo)

        self.llm_api_key = QLineEdit()
        self.llm_api_key.setEchoMode(QLineEdit.Password)
        self.llm_api_key.setPlaceholderText("API Key 或留空用环境变量")
        lg.addRow("API Key", self.llm_api_key)

        self.llm_base_url = QLineEdit()
        self.llm_base_url.setPlaceholderText("留空用默认地址")
        lg.addRow("Base URL", self.llm_base_url)

        self.llm_test_btn = QPushButton("测试连接")
        self.llm_test_btn.setObjectName("infoBtn")
        self.llm_test_btn.setCursor(Qt.PointingHandCursor)
        self.llm_test_btn.clicked.connect(self._test_llm_connection)
        lg.addRow("", self.llm_test_btn)

        layout.addWidget(llm_group)

        # 初始化供应商预设
        self._on_provider_changed(self.llm_provider_combo.currentText())

        # ── YOLO 检测模型 ──
        yolo_group = QGroupBox("YOLO 检测模型")
        yg = QVBoxLayout(yolo_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
        self.model_combo.setEditable(True)
        yg.addWidget(self.model_combo)

        yolo_btn_row = QHBoxLayout()
        self.load_model_btn = QPushButton("加载自定义")
        self.load_model_btn.setObjectName("browseBtn")
        self.load_model_btn.clicked.connect(self._load_custom_model)
        yolo_btn_row.addWidget(self.load_model_btn)
        self.train_btn_open = QPushButton("训练 YOLO")
        self.train_btn_open.setObjectName("purpleBtn")
        self.train_btn_open.setCursor(Qt.PointingHandCursor)
        self.train_btn_open.clicked.connect(self._open_train_dialog)
        yolo_btn_row.addWidget(self.train_btn_open)
        yg.addLayout(yolo_btn_row)

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
        yg.addLayout(yolo_form)

        layout.addWidget(yolo_group)

        layout.addStretch()
        scroll.setWidget(tab)
        self.train_tabs.addTab(scroll, "数据获取")

    def _build_train_annotate_tab(self):
        """训练 Tab 2: 标注"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── LLM 自动标注 ──
        anno_group = QGroupBox("LLM 自动标注")
        anno_g = QVBoxLayout(anno_group)
        anno_g.addWidget(QLabel("选择视频 → YOLO 检测目标 → LLM 判断动作 → 生成训练数据"))

        anno_btn_row = QHBoxLayout()
        self.auto_annotate_btn = QPushButton("开始 LLM 标注")
        self.auto_annotate_btn.setObjectName("purpleBtn")
        self.auto_annotate_btn.setCursor(Qt.PointingHandCursor)
        self.auto_annotate_btn.clicked.connect(self._open_annotate_dialog)
        anno_btn_row.addWidget(self.auto_annotate_btn)

        self.view_annotation_btn = QPushButton("查看标注结果")
        self.view_annotation_btn.setObjectName("infoBtn")
        self.view_annotation_btn.setCursor(Qt.PointingHandCursor)
        self.view_annotation_btn.clicked.connect(self._open_annotation_viewer)
        anno_btn_row.addWidget(self.view_annotation_btn)
        anno_g.addLayout(anno_btn_row)

        layout.addWidget(anno_group)

        # ── 人工录制 ──
        rec_group = QGroupBox("人工操作录制")
        rg = QVBoxLayout(rec_group)
        rg.setSpacing(6)
        rg.addWidget(QLabel("启动检测后录制键盘/鼠标操作，与 YOLO 检测结果一起保存"))

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("保存目录"))
        self.rec_dir_input = QLineEdit("data/recordings")
        dir_row.addWidget(self.rec_dir_input)
        rec_browse = QPushButton("...")
        rec_browse.setObjectName("browseBtn")
        rec_browse.setMaximumWidth(32)
        rec_browse.clicked.connect(lambda: self._browse_dir_to(self.rec_dir_input))
        dir_row.addWidget(rec_browse)
        rg.addLayout(dir_row)

        session_row = QHBoxLayout()
        session_row.addWidget(QLabel("会话名称"))
        self.rec_session_input = QLineEdit()
        self.rec_session_input.setPlaceholderText("留空自动生成时间戳")
        session_row.addWidget(self.rec_session_input)
        rg.addLayout(session_row)

        rec_btn_row = QHBoxLayout()
        self.rec_start_btn = QPushButton("●  开始录制")
        self.rec_start_btn.setObjectName("stopBtn")
        self.rec_start_btn.setCursor(Qt.PointingHandCursor)
        self.rec_start_btn.setToolTip("需要先在 Agent 模式启动检测")
        self.rec_start_btn.clicked.connect(self._toggle_recording)
        rec_btn_row.addWidget(self.rec_start_btn)
        rg.addLayout(rec_btn_row)

        self.rec_status_label = QLabel("就绪（需先在 Agent 模式启动检测）")
        self.rec_status_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        rg.addWidget(self.rec_status_label)

        layout.addWidget(rec_group)

        layout.addStretch()
        scroll.setWidget(tab)
        self.train_tabs.addTab(scroll, "标注")

    def _build_train_training_tab(self):
        """训练 Tab 3: 模型训练"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── 决策模型训练 ──
        train_group = QGroupBox("决策模型训练")
        tg = QVBoxLayout(train_group)
        tg.setSpacing(6)

        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("数据目录"))
        self.dt_data_dir = QLineEdit("data/recordings")
        data_row.addWidget(self.dt_data_dir)
        dt_browse = QPushButton("...")
        dt_browse.setObjectName("browseBtn")
        dt_browse.setMaximumWidth(32)
        dt_browse.clicked.connect(lambda: self._browse_dir_to(self.dt_data_dir))
        data_row.addWidget(dt_browse)
        tg.addLayout(data_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("输出目录"))
        self.dt_output_dir = QLineEdit("runs/decision/exp1")
        out_row.addWidget(self.dt_output_dir)
        tg.addLayout(out_row)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("模型"))
        self.dt_model_type = QComboBox()
        self.dt_model_type.addItems(["mlp", "rf"])
        param_row.addWidget(self.dt_model_type)
        param_row.addWidget(QLabel("轮数"))
        self.dt_epochs_spin = QSpinBox()
        self.dt_epochs_spin.setRange(10, 1000)
        self.dt_epochs_spin.setValue(100)
        param_row.addWidget(self.dt_epochs_spin)
        param_row.addWidget(QLabel("学习率"))
        self.dt_lr_spin = QDoubleSpinBox()
        self.dt_lr_spin.setRange(0.0001, 0.1)
        self.dt_lr_spin.setSingleStep(0.0005)
        self.dt_lr_spin.setDecimals(4)
        self.dt_lr_spin.setValue(0.001)
        param_row.addWidget(self.dt_lr_spin)
        tg.addLayout(param_row)

        dt_btn_row = QHBoxLayout()
        self.dt_preview_btn = QPushButton("预览数据")
        self.dt_preview_btn.setObjectName("infoBtn")
        self.dt_preview_btn.setCursor(Qt.PointingHandCursor)
        self.dt_preview_btn.clicked.connect(self._preview_data)
        dt_btn_row.addWidget(self.dt_preview_btn)

        self.dt_train_btn = QPushButton("▶  开始训练")
        self.dt_train_btn.setObjectName("startBtn")
        self.dt_train_btn.setCursor(Qt.PointingHandCursor)
        self.dt_train_btn.clicked.connect(self._start_decision_train)
        dt_btn_row.addWidget(self.dt_train_btn)

        self.dt_use_btn = QPushButton("应用到 Agent")
        self.dt_use_btn.setObjectName("purpleBtn")
        self.dt_use_btn.setCursor(Qt.PointingHandCursor)
        self.dt_use_btn.setToolTip("将训练好的模型设为 Agent 的决策引擎")
        self.dt_use_btn.setEnabled(False)
        self.dt_use_btn.clicked.connect(self._use_trained_model)
        dt_btn_row.addWidget(self.dt_use_btn)
        tg.addLayout(dt_btn_row)

        self.dt_progress = QProgressBar()
        self.dt_progress.setRange(0, 100)
        self.dt_progress.setValue(0)
        self.dt_progress.setVisible(False)
        tg.addWidget(self.dt_progress)

        self.dt_status_label = QLabel("")
        self.dt_status_label.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px;")
        tg.addWidget(self.dt_status_label)

        from .train_chart import TrainChart
        self.dt_chart = TrainChart()
        self.dt_chart.setVisible(False)
        tg.addWidget(self.dt_chart)

        layout.addWidget(train_group)

        layout.addStretch()
        scroll.setWidget(tab)
        self.train_tabs.addTab(scroll, "训练")

    def _build_train_model_tab(self):
        """训练 Tab 4: 模型管理"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Profile 管理 ──
        profile_group = QGroupBox("场景 Profile 管理")
        pg = QVBoxLayout(profile_group)

        pg.addWidget(QLabel("Profile"))
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("(无)")
        self._refresh_profiles()
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        pg.addWidget(self.profile_combo)

        self.profile_info = QTextEdit()
        self.profile_info.setReadOnly(True)
        self.profile_info.setMaximumHeight(120)
        self.profile_info.setPlaceholderText("选择 Profile 查看详情")
        pg.addWidget(self.profile_info)

        profile_btn_row = QHBoxLayout()
        refresh_btn = QPushButton("刷新")
        refresh_btn.setObjectName("browseBtn")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.clicked.connect(self._refresh_profiles)
        profile_btn_row.addWidget(refresh_btn)

        export_btn = QPushButton("导出配置")
        export_btn.setObjectName("infoBtn")
        export_btn.clicked.connect(self._export_config)
        profile_btn_row.addWidget(export_btn)

        import_btn = QPushButton("导入配置")
        import_btn.setObjectName("browseBtn")
        import_btn.clicked.connect(self._import_config)
        profile_btn_row.addWidget(import_btn)
        pg.addLayout(profile_btn_row)

        layout.addWidget(profile_group)

        layout.addStretch()
        scroll.setWidget(tab)
        self.train_tabs.addTab(scroll, "模型管理")

    # ================================================================
    #  Agent 模式 Tabs
    # ================================================================

    def _build_agent_source_tab(self):
        """Agent Tab 1: 输入源"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setSpacing(6)
        self.source_type = QComboBox()
        self.source_type.addItems(["screen", "camera", "video", "image", "stream"])
        self.source_type.currentTextChanged.connect(self._on_source_type_changed)
        form.addRow("类型", self.source_type)
        layout.addLayout(form)

        # 路径行
        self.path_label = QLabel("路径")
        layout.addWidget(self.path_label)
        path_row = QHBoxLayout()
        path_row.setSpacing(4)
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("文件路径 / rtsp:// / http://")
        path_row.addWidget(self.path_input)
        self.browse_btn = QPushButton("...")
        self.browse_btn.setObjectName("browseBtn")
        self.browse_btn.setMaximumWidth(32)
        self.browse_btn.clicked.connect(self._browse_file)
        path_row.addWidget(self.browse_btn)
        layout.addLayout(path_row)

        loop_row = QHBoxLayout()
        loop_row.addWidget(QLabel("循环"))
        self.loop_combo = QComboBox()
        self.loop_combo.addItems(["否", "是"])
        loop_row.addWidget(self.loop_combo)
        layout.addLayout(loop_row)

        # 摄像头
        self.camera_label = QLabel("设备号")
        layout.addWidget(self.camera_label)
        self.camera_device = QSpinBox()
        self.camera_device.setRange(0, 10)
        layout.addWidget(self.camera_device)

        # 屏幕
        self.monitor_label = QLabel("显示器")
        layout.addWidget(self.monitor_label)
        self.monitor_num = QSpinBox()
        self.monitor_num.setRange(1, 5)
        layout.addWidget(self.monitor_num)

        # 实时流
        self.stream_label = QLabel("流地址")
        layout.addWidget(self.stream_label)
        self.stream_input = QLineEdit()
        self.stream_input.setPlaceholderText(
            "rtsp://user:pass@ip:554/stream  或  B站直播间号(如 12345)"
        )
        layout.addWidget(self.stream_input)
        self.stream_hint = QLabel(
            "支持: RTSP / HTTP-FLV / HLS(m3u8) / B站直播间号"
        )
        self.stream_hint.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.stream_hint)

        layout.addStretch()
        self.agent_tabs.addTab(tab, "输入源")

        # 初始化可见性
        self.source_type.setCurrentText("video")
        self._on_source_type_changed("video")

    def _build_agent_engine_tab(self):
        """Agent Tab 2: 检测 + 决策"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── 决策引擎 ──
        engine_group = QGroupBox("决策引擎")
        eg = QVBoxLayout(engine_group)

        form_top = QFormLayout()
        form_top.setSpacing(6)
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["none", "rule", "trained", "llm", "hierarchical", "rl"])
        self.engine_combo.setCurrentText("rule")
        self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        form_top.addRow("引擎", self.engine_combo)
        eg.addLayout(form_top)

        # LLM 决策间隔
        self.llm_interval_widget = QWidget()
        interval_layout = QFormLayout(self.llm_interval_widget)
        interval_layout.setContentsMargins(0, 0, 0, 0)
        self.llm_interval = QDoubleSpinBox()
        self.llm_interval.setRange(0.1, 30.0)
        self.llm_interval.setSingleStep(0.5)
        self.llm_interval.setValue(1.0)
        interval_layout.addRow("决策间隔(秒)", self.llm_interval)
        self.llm_interval_widget.setVisible(False)
        eg.addWidget(self.llm_interval_widget)

        layout.addWidget(engine_group)

        # ── Trained 配置 ──
        self.trained_config_widget = QGroupBox("训练模型配置")
        trained_layout = QVBoxLayout(self.trained_config_widget)
        trained_layout.setSpacing(6)

        trained_dir_row = QHBoxLayout()
        trained_dir_row.addWidget(QLabel("模型目录"))
        self.trained_model_dir = QLineEdit()
        self.trained_model_dir.setPlaceholderText("runs/decision/exp1")
        self.trained_model_dir.setText("runs/decision/exp1")
        trained_dir_row.addWidget(self.trained_model_dir)
        trained_browse = QPushButton("...")
        trained_browse.setObjectName("browseBtn")
        trained_browse.setMaximumWidth(32)
        trained_browse.clicked.connect(self._browse_trained_dir)
        trained_dir_row.addWidget(trained_browse)
        trained_layout.addLayout(trained_dir_row)

        trained_form = QFormLayout()
        self.trained_conf_spin = QDoubleSpinBox()
        self.trained_conf_spin.setRange(0.05, 1.0)
        self.trained_conf_spin.setSingleStep(0.05)
        self.trained_conf_spin.setValue(0.30)
        trained_form.addRow("置信阈值", self.trained_conf_spin)
        trained_layout.addLayout(trained_form)

        trained_layout.addWidget(QLabel("动作映射 (语义名→按键):"))
        self.trained_action_map_edit = QTextEdit()
        self.trained_action_map_edit.setMaximumHeight(100)
        self.trained_action_map_edit.setPlaceholderText(
            '{\n  "attack": {"type": "key", "key": "a"},\n'
            '  "retreat": {"type": "key", "key": "s"}\n}'
        )
        default_map = {
            "attack": {"type": "key", "key": "a"},
            "retreat": {"type": "key", "key": "s"},
            "skill_1": {"type": "key", "key": "1"},
            "skill_2": {"type": "key", "key": "2"},
            "skill_3": {"type": "key", "key": "3"},
        }
        self.trained_action_map_edit.setPlainText(json.dumps(default_map, indent=2, ensure_ascii=False))
        trained_layout.addWidget(self.trained_action_map_edit)

        self.trained_config_widget.setVisible(False)
        layout.addWidget(self.trained_config_widget)

        # ── LLM 配置区域（Agent 模式独立，与训练模式的 LLM 配置分开展示但共享数据） ──
        self.llm_config_widget = QGroupBox("LLM 决策配置")
        self.llm_config_widget.setVisible(False)
        llm_hint = QVBoxLayout(self.llm_config_widget)
        llm_hint.addWidget(QLabel("使用训练工坊中配置的 LLM 供应商和模型"))
        layout.addWidget(self.llm_config_widget)

        layout.addStretch()
        scroll.setWidget(tab)
        self.agent_tabs.addTab(scroll, "检测决策")

    def _build_agent_scene_tab(self):
        """Agent Tab 3: 场景"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Profile 快速选择
        layout.addWidget(QLabel("场景 Profile（训练工坊中管理）"))
        self.agent_profile_combo = QComboBox()
        self.agent_profile_combo.addItem("(无)")
        self._refresh_agent_profiles()
        self.agent_profile_combo.currentTextChanged.connect(self._on_agent_profile_changed)
        layout.addWidget(self.agent_profile_combo)

        refresh_btn = QPushButton("刷新")
        refresh_btn.setObjectName("browseBtn")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.clicked.connect(self._refresh_agent_profiles)
        layout.addWidget(refresh_btn)

        # AutoPilot
        self.autopilot_check = QCheckBox("启用 AutoPilot（自动场景识别 + 自动切换）")
        self.autopilot_check.setToolTip(
            "开启后自动根据检测结果识别场景，\n"
            "切换到对应 Profile 并使用其决策模型"
        )
        layout.addWidget(self.autopilot_check)

        # 场景状态
        scene_group = QGroupBox("当前场景状态")
        sg = QVBoxLayout(scene_group)
        self.scene_status_text = QTextEdit()
        self.scene_status_text.setReadOnly(True)
        self.scene_status_text.setMaximumHeight(100)
        self.scene_status_text.setPlaceholderText("启动检测后显示场景信息")
        sg.addWidget(self.scene_status_text)
        layout.addWidget(scene_group)

        layout.addStretch()
        self.agent_tabs.addTab(tab, "场景")

    def _build_agent_chat_tab(self):
        """Agent Tab 4: LLM 对话控制"""
        self.chat_panel = ChatPanel()
        self.chat_panel._try_auto_init = self._init_chat_provider
        self.agent_tabs.addTab(self.chat_panel, "对话控制")
        self.agent_tabs.currentChanged.connect(self._on_agent_tab_changed)

    def _on_agent_tab_changed(self, index: int):
        """切换到对话控制 tab 时自动初始化 LLM。"""
        tab_text = self.agent_tabs.tabText(index)
        if tab_text == "对话控制" and self.chat_panel._provider is None:
            self._init_chat_provider()

    def _init_chat_provider(self):
        """根据当前 LLM 配置初始化对话面板的 Provider 和工具。"""
        api_key = self._get_llm_api_key()
        provider_name = self.llm_provider_combo.currentText()
        model = self.llm_model_combo.currentText()
        base_url = self.llm_base_url.text().strip()

        if not api_key and provider_name != "ollama":
            self.chat_panel.set_provider(None)
            return False

        try:
            provider = create_provider(provider_name, api_key or "ollama", model, base_url)
            self.chat_panel.set_provider(provider)
        except Exception as e:
            self._log(f"[对话] 创建 LLM Provider 失败: {e}")
            self.chat_panel.set_provider(None)
            return False

        # 工具注册（对话面板始终注册真实工具，由面板自身 dryrun 控制是否执行）
        registry = ToolRegistry()
        try:
            registry.register(KeyboardTool())
            registry.register(MouseTool())
        except ImportError:
            self._log("[对话] pynput 未安装，键盘鼠标工具不可用")
            return False

        self.chat_panel.set_tool_registry(registry)
        self._log(f"[对话] LLM 已就绪: {provider_name}/{model}")
        return True

    # ================================================================
    #  模式切换
    # ================================================================

    def _switch_mode(self, mode: str):
        self._current_mode = mode
        is_train = mode == "train"
        self.mode_stack.setCurrentIndex(0 if is_train else 1)
        self.agent_controls.setVisible(not is_train)

        # 更新按钮样式
        self.mode_train_btn.setObjectName("modeBtnActive" if is_train else "modeBtnInactive")
        self.mode_agent_btn.setObjectName("modeBtnInactive" if is_train else "modeBtnActive")
        self.mode_train_btn.setStyle(self.mode_train_btn.style())
        self.mode_agent_btn.setStyle(self.mode_agent_btn.style())

    # ================================================================
    #  Profile 管理
    # ================================================================

    def _refresh_profiles(self):
        current = self.profile_combo.currentText() if hasattr(self, 'profile_combo') else ""
        self.profile_combo.clear()
        self.profile_combo.addItem("(无)", "")
        profiles = self._profile_mgr.load_all()
        for name, p in profiles.items():
            self.profile_combo.addItem(f"{p.display_name} ({name})", name)
        if current:
            idx = self.profile_combo.findText(current)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)
        # 同步刷新 Agent 模式的 profile 列表
        if hasattr(self, 'agent_profile_combo'):
            self._refresh_agent_profiles()

    def _refresh_agent_profiles(self):
        if not hasattr(self, 'agent_profile_combo'):
            return
        current = self.agent_profile_combo.currentText()
        self.agent_profile_combo.clear()
        self.agent_profile_combo.addItem("(无)", "")
        profiles = self._profile_mgr.load_all()
        for name, p in profiles.items():
            self.agent_profile_combo.addItem(f"{p.display_name} ({name})", name)
        if current:
            idx = self.agent_profile_combo.findText(current)
            if idx >= 0:
                self.agent_profile_combo.setCurrentIndex(idx)

    @Slot(str)
    def _on_profile_changed(self, text: str):
        if not text or text == "(无)":
            self.profile_info.clear()
            return
        name = self.profile_combo.currentData() or ""
        profile = self._profile_mgr.get(name) if name else None
        if not profile:
            return
        info_lines = [
            f"名称: {profile.display_name}",
            f"YOLO 模型: {profile.yolo_model}",
            f"决策引擎: {profile.decision_engine}",
            f"动作: {', '.join(profile.actions)}",
            f"场景关键词: {', '.join(profile.scene_keywords)}",
            f"ROI 区域: {', '.join(profile.roi_regions.keys())}",
            f"自动训练: {'启用' if profile.auto_train.get('enabled') else '禁用'}",
        ]
        self.profile_info.setPlainText("\n".join(info_lines))

    @Slot(str)
    def _on_agent_profile_changed(self, text: str):
        """Agent 模式选择 Profile，自动填充相关配置。"""
        if not text or text == "(无)":
            return
        name = self.agent_profile_combo.currentData() or ""
        profile = self._profile_mgr.get(name) if name else None
        if not profile:
            return
        if profile.yolo_model:
            self.model_combo.setEditText(profile.yolo_model)
        if profile.decision_engine and profile.decision_engine != "rule":
            self.engine_combo.setCurrentText(profile.decision_engine)
        if profile.action_key_map:
            self.trained_action_map_edit.setPlainText(
                json.dumps(profile.action_key_map, indent=2, ensure_ascii=False)
            )
        if profile.decision_model_dir:
            self.trained_model_dir.setText(profile.decision_model_dir)

    # ================================================================
    #  训练模式事件
    # ================================================================

    @Slot()
    def _open_auto_learn(self):
        from .auto_learn_dialog import AutoLearnDialog
        dialog = AutoLearnDialog(self)
        dialog.exec()
        self._refresh_profiles()

    @Slot()
    def _open_annotate_dialog(self):
        from .annotate_dialog import AnnotateDialog
        default_video = self.path_input.text().strip() if self.source_type.currentText() == "video" else ""
        default_model = self.model_combo.currentText()
        dialog = AnnotateDialog(self, default_video=default_video, default_model=default_model)
        dialog.exec()
        save_path = dialog.get_save_path()
        if save_path:
            self.dt_data_dir.setText(str(Path(save_path).parent))
            self._log(f"[标注] 数据已保存到 {save_path}")

    @Slot()
    def _open_annotation_viewer(self):
        from .annotation_viewer import AnnotationViewer
        jsonl_path = ""
        video_path = ""
        data_dir = self.dt_data_dir.text().strip() if hasattr(self, 'dt_data_dir') else ""
        if data_dir and Path(data_dir).is_dir():
            jsonl_files = sorted(Path(data_dir).glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if jsonl_files:
                jsonl_path = str(jsonl_files[0])
        if self.source_type.currentText() == "video":
            video_path = self.path_input.text().strip()
        dialog = AnnotationViewer(self, jsonl_path=jsonl_path, video_path=video_path)
        dialog.exec()

    def _browse_dir_to(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "选择目录")
        if path:
            line_edit.setText(path)

    def _browse_trained_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if path:
            self.trained_model_dir.setText(path)

    @Slot()
    def _toggle_recording(self):
        if self._recorder and self._recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if not self._worker:
            QMessageBox.warning(self, "提示", "请先在 Agent 模式启动检测，再开始录制")
            return
        save_dir = self.rec_dir_input.text().strip() or "data/recordings"
        session = self.rec_session_input.text().strip() or None
        try:
            self._recorder = DataRecorder(save_dir=save_dir, session_name=session)
            self._recorder.on_start()
            self._worker.agents.append(self._recorder)
            self.rec_start_btn.setText("■  停止录制")
            self.rec_status_label.setText(f"● 录制中... → {self._recorder.file_path}")
            self.rec_status_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 12px; font-weight: bold;")
            self._log(f"[录制] 开始 → {self._recorder.file_path}")
            from PySide6.QtCore import QTimer
            self._rec_timer = QTimer(self)
            self._rec_timer.timeout.connect(self._update_rec_status)
            self._rec_timer.start(1000)
        except Exception as e:
            QMessageBox.critical(self, "录制失败", str(e))

    def _stop_recording(self):
        if not self._recorder:
            return
        if hasattr(self, '_rec_timer'):
            self._rec_timer.stop()
        if self._worker and self._recorder in self._worker.agents:
            self._worker.agents.remove(self._recorder)
        count = self._recorder.sample_count
        path = self._recorder.file_path
        self._recorder.on_stop()
        self.rec_start_btn.setText("●  开始录制")
        self.rec_status_label.setText(f"录制完成: {count} 条样本 → {path}")
        self.rec_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px;")
        self._log(f"[录制] 停止, {count} 条样本已保存")
        if path:
            self.dt_data_dir.setText(str(path.parent))
        self._recorder = None

    @Slot()
    def _update_rec_status(self):
        if self._recorder and self._recorder.is_recording:
            self.rec_status_label.setText(
                f"录制中... {self._recorder.sample_count} 条样本 → {self._recorder.file_path}"
            )

    @Slot()
    def _preview_data(self):
        data_dir = self.dt_data_dir.text().strip()
        if not data_dir:
            QMessageBox.warning(self, "提示", "请指定数据目录")
            return
        try:
            from ..data.dataset import ActionDataset
            dataset = ActionDataset(data_dir)
            total = dataset.load()
            if total == 0:
                QMessageBox.information(self, "数据预览", f"目录 {data_dir} 中没有找到录制数据(.jsonl)")
                return
            summary = dataset.summary()
            lines = [f"总样本数: {summary['total']}", ""]
            lines.append("动作分布:")
            dist = summary.get("action_distribution", {})
            for action, count in dist.items():
                pct = count / summary["total"] * 100
                lines.append(f"  {action}: {count} ({pct:.1f}%)")
            lines.append("")
            lines.append("检测类别:")
            for cls, count in summary.get("detection_classes", {}).items():
                lines.append(f"  {cls}: {count}")
            lines.append("")
            lines.append("── 质量分析 ──")
            issues = []
            if total < 50:
                issues.append(f"! 样本过少 ({total})，建议至少 100 条")
            if total < 10:
                issues.append("! 无法训练：少于 10 条有效样本")
            if dist:
                counts = list(dist.values())
                max_c, min_c = max(counts), min(counts)
                ratio = max_c / max(min_c, 1)
                least = min(dist, key=dist.get)
                if ratio > 5:
                    issues.append(f"! 严重不均衡：'{least}' 仅 {min_c} 条 (1:{ratio:.0f})")
                elif ratio > 3:
                    issues.append(f"! 轻度不均衡：'{least}' 仅 {min_c} 条")
                if len(dist) < 2:
                    issues.append("! 只有 1 种动作，无法训练分类器")
            if issues:
                for issue in issues:
                    lines.append(f"  {issue}")
            else:
                lines.append("  数据质量良好，可以开始训练")
            QMessageBox.information(self, "数据预览", "\n".join(lines))
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    @Slot()
    def _start_decision_train(self):
        data_dir = self.dt_data_dir.text().strip()
        output_dir = self.dt_output_dir.text().strip()
        if not data_dir:
            QMessageBox.warning(self, "提示", "请指定数据目录")
            return
        self.dt_train_btn.setEnabled(False)
        self.dt_use_btn.setEnabled(False)
        self.dt_progress.setVisible(True)
        self.dt_progress.setValue(0)
        self.dt_chart.clear()
        self.dt_chart.setVisible(True)
        self.dt_status_label.setText("准备中...")
        model_type = self.dt_model_type.currentText()
        self._train_worker = DecisionTrainWorker(
            data_dir=data_dir, output_dir=output_dir, model_type=model_type,
            epochs=self.dt_epochs_spin.value(), lr=self.dt_lr_spin.value(),
        )
        self._train_worker.log_message.connect(self._on_dt_log)
        self._train_worker.progress.connect(self._on_dt_progress)
        self._train_worker.finished_ok.connect(self._on_dt_finished)
        self._train_worker.finished_err.connect(self._on_dt_error)
        self._train_worker.start()
        self._log("[训练] 决策模型训练开始")

    @Slot(str)
    def _on_dt_log(self, msg: str):
        self._log(f"[训练] {msg}")

    @Slot(int, int, float, float, float)
    def _on_dt_progress(self, epoch, total, loss, train_acc, val_acc):
        pct = int(epoch / total * 100) if total > 0 else 0
        self.dt_progress.setValue(pct)
        self.dt_status_label.setText(
            f"Epoch {epoch}/{total}  loss={loss:.4f}  "
            f"train={train_acc:.3f}  val={val_acc:.3f}"
        )
        self.dt_chart.add_point(loss, train_acc, val_acc)

    @Slot(str, dict)
    def _on_dt_finished(self, model_dir, metrics):
        self.dt_train_btn.setEnabled(True)
        self.dt_use_btn.setEnabled(True)
        self.dt_progress.setValue(100)
        val_acc = metrics.get("best_val_acc", 0)
        train_acc = metrics.get("final_train_acc", 0)
        epochs = metrics.get("epochs_trained", "?")
        self.dt_status_label.setText(
            f"训练完成  val={val_acc:.3f}  train={train_acc:.3f}  ({epochs} epochs)"
        )
        self.dt_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px; font-weight: bold;")
        self._log(f"[训练] 完成 → {model_dir} (val_acc={val_acc:.4f})")
        self._train_worker = None

    @Slot(str)
    def _on_dt_error(self, error):
        self.dt_train_btn.setEnabled(True)
        self.dt_progress.setVisible(False)
        self.dt_status_label.setText(f"训练失败: {error}")
        self.dt_status_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 12px;")
        QMessageBox.critical(self, "训练失败", error)
        self._train_worker = None

    @Slot()
    def _use_trained_model(self):
        model_dir = self.dt_output_dir.text().strip()
        self._switch_mode("agent")
        self.engine_combo.setCurrentText("trained")
        self.trained_model_dir.setText(model_dir)
        self._log(f"[引擎] 已切换到 Agent 模式 - trained: {model_dir}")
        QMessageBox.information(
            self, "已应用",
            f"已切换到 Agent 模式\n决策引擎: trained\n模型: {model_dir}\n\n启动检测即可使用"
        )

    @Slot()
    def _export_config(self):
        import zipfile
        save_path, _ = QFileDialog.getSaveFileName(self, "导出配置", "vision_agent_config.zip", "ZIP (*.zip)")
        if not save_path:
            return
        try:
            with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as zf:
                config_path = Path("config.yaml")
                if config_path.exists():
                    zf.write(config_path, "config.yaml")
                profiles_dir = Path("profiles")
                if profiles_dir.is_dir():
                    for f in profiles_dir.glob("*.yaml"):
                        zf.write(f, f"profiles/{f.name}")
            self._log(f"[配置] 已导出到 {save_path}")
            QMessageBox.information(self, "导出成功", f"配置已导出到:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    @Slot()
    def _import_config(self):
        import zipfile
        path, _ = QFileDialog.getOpenFileName(self, "导入配置", "", "ZIP (*.zip)")
        if not path:
            return
        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                for name in names:
                    if ".." in name or name.startswith("/"):
                        raise ValueError(f"不安全的文件路径: {name}")
                existing = [n for n in names if Path(n).exists()]
                if existing:
                    reply = QMessageBox.question(
                        self, "确认覆盖",
                        f"以下 {len(existing)} 个文件将被覆盖:\n" +
                        "\n".join(existing[:10]) +
                        ("\n..." if len(existing) > 10 else ""),
                        QMessageBox.Yes | QMessageBox.Cancel,
                    )
                    if reply != QMessageBox.Yes:
                        return
                zf.extractall(".")
            self._refresh_profiles()
            self._log(f"[配置] 已从 {path} 导入")
            QMessageBox.information(self, "导入成功", f"配置已导入，共 {len(names)} 个文件")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

    # ================================================================
    #  Agent 模式事件
    # ================================================================

    def _on_source_type_changed(self, source_type: str):
        is_video = source_type == "video"
        is_camera = source_type == "camera"
        is_screen = source_type == "screen"
        is_image = source_type == "image"
        is_stream = source_type == "stream"

        self.path_label.setVisible(is_video or is_image)
        self.path_input.setVisible(is_video or is_image)
        self.browse_btn.setVisible(is_video or is_image)
        self.loop_combo.setVisible(is_video or is_image)

        if is_image:
            self.path_label.setText("图片路径 / 目录")
        else:
            self.path_label.setText("路径 / 流地址")

        self.camera_label.setVisible(is_camera)
        self.camera_device.setVisible(is_camera)
        self.monitor_label.setVisible(is_screen)
        self.monitor_num.setVisible(is_screen)
        self.stream_label.setVisible(is_stream)
        self.stream_input.setVisible(is_stream)
        self.stream_hint.setVisible(is_stream)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "",
            "视频/图片 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.jpg *.png *.bmp);;所有文件 (*)"
        )
        if path:
            self.path_input.setText(path)

    def _build_source_config(self) -> dict:
        source_type = self.source_type.currentText()
        config = {"type": source_type}
        if source_type == "video":
            config["video"] = {
                "path": self.path_input.text().strip(),
                "loop": self.loop_combo.currentIndex() == 1,
            }
        elif source_type == "camera":
            config["camera"] = {"device": self.camera_device.value()}
        elif source_type == "screen":
            config["screen"] = {"monitor": self.monitor_num.value()}
        elif source_type == "image":
            config["image"] = {
                "path": self.path_input.text().strip(),
                "loop": self.loop_combo.currentIndex() == 1,
            }
        elif source_type == "stream":
            raw = self.stream_input.text().strip()
            if raw.isdigit():
                url = f"bilibili://{raw}"
            else:
                url = raw
            config["stream"] = {"url": url}
        return config

    @Slot(str)
    def _on_engine_changed(self, engine_type: str):
        self.trained_config_widget.setVisible(engine_type == "trained")
        self.llm_config_widget.setVisible(engine_type == "llm")
        self.llm_interval_widget.setVisible(engine_type == "llm")

    @Slot(str)
    def _on_provider_changed(self, provider_name: str):
        preset = PROVIDER_PRESETS.get(provider_name, {})
        self.llm_model_combo.clear()
        self.llm_model_combo.addItems(preset.get("models", []))
        self.llm_base_url.setText(preset.get("base_url", ""))
        env_key = preset.get("api_key_env", "")
        if env_key and os.environ.get(env_key):
            self.llm_api_key.setPlaceholderText(f"已从 {env_key} 读取")
        else:
            self.llm_api_key.setPlaceholderText(f"API Key" + (f" 或设置 {env_key}" if env_key else ""))

    @Slot()
    def _test_llm_connection(self):
        provider_name = self.llm_provider_combo.currentText()
        api_key = self._get_llm_api_key()
        model = self.llm_model_combo.currentText()
        base_url = self.llm_base_url.text().strip()
        if not api_key and provider_name != "ollama":
            QMessageBox.warning(self, "提示", "请输入 API Key")
            return
        self.llm_test_btn.setEnabled(False)
        self.llm_test_btn.setText("测试中...")
        try:
            provider = create_provider(provider_name, api_key or "ollama", model, base_url)
            ok = provider.test_connection()
            if ok:
                QMessageBox.information(self, "成功", f"{provider_name} 连接成功!\n模型: {model}")
                self._log(f"LLM 连接测试成功: {provider_name}/{model}")
            else:
                QMessageBox.warning(self, "失败", "连接测试失败，请检查配置")
        except ImportError as e:
            pkg = "anthropic" if provider_name == "claude" else "openai"
            QMessageBox.critical(self, "缺少依赖", f"需要安装: pip install {pkg}\n\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"连接失败:\n{e}")
        finally:
            self.llm_test_btn.setEnabled(True)
            self.llm_test_btn.setText("测试连接")

    def _get_llm_api_key(self) -> str:
        key = self.llm_api_key.text().strip()
        if key:
            return key
        provider_name = self.llm_provider_combo.currentText()
        env_key = PROVIDER_PRESETS.get(provider_name, {}).get("api_key_env", "")
        return os.environ.get(env_key, "") if env_key else ""

    # ================================================================
    #  Agent 构建
    # ================================================================

    def _build_agents(self) -> list:
        engine_type = self.engine_combo.currentText()
        if engine_type == "none":
            return []

        registry = ToolRegistry()
        if self.dryrun_check.isChecked():
            from ..tools.base import BaseTool, ToolResult
            class _DryRunTool(BaseTool):
                def __init__(self, tool_name):
                    self._name = tool_name
                @property
                def name(self): return self._name
                @property
                def description(self): return f"DryRun {self._name}"
                @property
                def parameters_schema(self): return {"type": "object", "properties": {}}
                def execute(self, **kwargs): return ToolResult(success=True)
            for name in ("keyboard", "mouse", "api_call", "shell"):
                registry.register(_DryRunTool(name))
        else:
            registry.register(KeyboardTool())
            registry.register(MouseTool())
            registry.register(ApiCallTool())
            registry.register(ShellTool())

        if engine_type == "rule":
            engine = RuleEngine()
            def _rule_attack_on_detect(result, state):
                if not result.detections:
                    return None
                det = result.detections[0]
                return Action(
                    tool_name="keyboard",
                    parameters={"action": "press", "key": "a"},
                    reason=f"发现 {det.class_name} ({det.confidence:.0%})",
                    target_bbox=det.bbox,
                )
            engine.add_rule("attack_on_detect", _rule_attack_on_detect)
        elif engine_type == "trained":
            model_dir = self.trained_model_dir.text().strip() or "runs/decision/exp1"
            action_key_map = {}
            map_text = self.trained_action_map_edit.toPlainText().strip()
            if map_text:
                try:
                    action_key_map = json.loads(map_text)
                except json.JSONDecodeError as e:
                    self._log(f"[警告] 动作映射 JSON 格式错误: {e}，将使用默认")
            try:
                engine = TrainedEngine(
                    model_dir=model_dir,
                    confidence_threshold=self.trained_conf_spin.value(),
                    action_key_map=action_key_map,
                )
                self._log(f"Trained 引擎: {model_dir}")
                if action_key_map:
                    self._log(f"动作映射: {list(action_key_map.keys())}")
            except Exception as e:
                self._log(f"[错误] 加载 Trained 引擎失败: {e}")
                return []
        elif engine_type == "llm":
            api_key = self._get_llm_api_key()
            provider_name = self.llm_provider_combo.currentText()
            model = self.llm_model_combo.currentText()
            base_url = self.llm_base_url.text().strip()
            if not api_key and provider_name != "ollama":
                self._log("[警告] API Key 未设置，LLM 引擎不可用")
                return []
            try:
                provider = create_provider(provider_name, api_key or "ollama", model, base_url)
            except Exception as e:
                self._log(f"[错误] 创建 LLM Provider 失败: {e}")
                return []
            engine = LLMEngine(
                provider=provider,
                tools_schema=registry.to_claude_tools(),
                system_prompt="你是一个视觉AI助手，根据画面检测结果决定下一步动作。只在需要时才调用工具。",
                decision_interval=self.llm_interval.value(),
            )
            engine.set_log_callback(self._decision_log_callback)
            self._log(f"LLM 引擎: {provider_name}/{model}")
        elif engine_type == "hierarchical":
            name = self.agent_profile_combo.currentData() if hasattr(self, 'agent_profile_combo') else ""
            profile = self._profile_mgr.get(name) if name else None
            micro = RuleEngine()
            if profile and profile.actions:
                first_action = profile.actions[0]
                key_map = profile.action_key_map.get(first_action, {})
                key = key_map.get("key", first_action)
                def _make_rule(k):
                    def _rule(r, s):
                        if not r.detections:
                            return None
                        det = r.detections[0]
                        return Action(
                            tool_name="keyboard",
                            parameters={"action": "press", "key": k},
                            reason=f"发现 {det.class_name}",
                            target_bbox=det.bbox,
                        )
                    return _rule
                micro.add_rule("default_action", _make_rule(key))
                self._log(f"分层引擎: 默认规则 → {first_action} ({key})")
            else:
                micro.add_rule("detect_any", lambda r, s: Action(
                    tool_name="keyboard",
                    parameters={"action": "press", "key": "space"},
                    reason=f"发现 {r.detections[0].class_name}" if r.detections else "",
                    target_bbox=r.detections[0].bbox if r.detections else None,
                ) if r.detections else None)
                self._log("分层引擎: 默认规则 → space（未配置 Profile）")
            engine = HierarchicalEngine(micro=micro)
            engine.set_log_callback(self._decision_log_callback)
        elif engine_type == "rl":
            actions = ["idle", "attack", "retreat"]
            action_key_map = {}
            name = self.agent_profile_combo.currentData() if hasattr(self, 'agent_profile_combo') else ""
            profile = self._profile_mgr.get(name) if name else None
            if profile:
                actions = profile.actions or actions
                action_key_map = profile.action_key_map or {}
            engine = RLEngine(
                actions=actions, action_key_map=action_key_map, training=True,
            )
            engine.set_log_callback(self._decision_log_callback)
            self._log(f"RL 引擎: {len(actions)} 动作, training=True")
        else:
            return []

        state_mgr = StateManager()
        agent = ActionAgent(
            decision_engine=engine, tool_registry=registry,
            state_manager=state_mgr, on_log=self._decision_log_callback,
        )
        self._log(f"Agent 就绪 ({engine_type}, 工具: {registry.tool_names})")
        return [agent]

    # ================================================================
    #  检测控制
    # ================================================================

    @Slot()
    def _start_detection(self):
        source_config = self._build_source_config()
        if source_config["type"] in ("video", "image"):
            key = source_config["type"]
            path = source_config[key]["path"]
            if not path:
                QMessageBox.warning(self, "提示", "请输入路径")
                return
        if source_config["type"] == "stream":
            url = source_config["stream"].get("url", "")
            if not url:
                QMessageBox.warning(self, "提示", "请输入流地址或直播间号")
                return

        try:
            source = create_source(source_config)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建视频源失败:\n{e}")
            return
        try:
            detector = Detector(
                model=self.model_combo.currentText(),
                confidence=self.conf_spin.value(),
                imgsz=int(self.imgsz_combo.currentText()),
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载检测模型失败:\n{e}")
            return
        try:
            agents = self._build_agents()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建 Agent 失败:\n{e}")
            return

        self._frame_count = 0
        self.log_text.clear()
        self.decision_log_text.clear()

        self._auto_pilot = None
        if hasattr(self, 'autopilot_check') and self.autopilot_check.isChecked():
            try:
                self._auto_pilot = AutoPilot(
                    profile_manager=self._profile_mgr,
                    scene_classifier=self._scene_classifier,
                    on_log=self._decision_log_callback,
                )
                self._log("AutoPilot 已启用")
            except Exception as e:
                self._log(f"AutoPilot 初始化失败: {e}")

        self._log("启动检测...")
        self._current_agents = agents

        self._worker = DetectionWorker(source, detector, agents=agents,
                                       auto_pilot=self._auto_pilot)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.fps_updated.connect(self._on_fps_updated)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.agent_stats_updated.connect(self._on_agent_stats)
        self._worker.decision_log.connect(self._on_decision_log, Qt.QueuedConnection)
        self._worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    @Slot()
    def _stop_detection(self):
        if self._worker:
            self._log("正在停止...")
            self._worker.stop()

    # ================================================================
    #  信号槽
    # ================================================================

    @Slot(object, object)
    def _on_frame_ready(self, frame, result: DetectionResult):
        self._frame_count += 1
        for agent in getattr(self, '_current_agents', []):
            if hasattr(agent, 'pop_recent_actions'):
                for action in agent.pop_recent_actions():
                    self.video_widget.set_action(
                        action.tool_name, action.parameters,
                        action.reason, target_bbox=action.target_bbox,
                    )
        self.video_widget.update_frame(frame, result)
        if hasattr(self, 'chat_panel'):
            self.chat_panel.update_frame(frame)
        self.count_label.setText(
            f"检测: {len(result.detections)}  |  帧: {self._frame_count}"
        )
        if self._auto_pilot:
            scene = self._auto_pilot.current_scene
            self.scene_label.setText(f"场景: {scene}")
            engines = self._auto_pilot.available_engines
            if hasattr(self, 'scene_status_text'):
                self.scene_status_text.setPlainText(
                    f"当前场景: {scene}\n"
                    f"已加载引擎: {', '.join(engines) if engines else '无'}\n"
                    f"帧缓冲: {self._auto_pilot.buffer_size}"
                )
        if self._frame_count % 30 == 1 and result.detections:
            names = [f"{d.class_name}({d.confidence:.2f})" for d in result.detections[:5]]
            self._log(f"[F{self._frame_count}] {', '.join(names)}")

    @Slot(float, float)
    def _on_fps_updated(self, fps, inference_ms):
        self.fps_label.setText(f"FPS: {fps:.1f}  |  推理: {inference_ms:.1f}ms")

    @Slot(str)
    def _on_error(self, msg):
        self._log(f"[错误] {msg}")

    @Slot(dict)
    def _on_agent_stats(self, stats):
        self.engine_status.setText(
            f"决策: {stats.get('decisions', 0)}  |  "
            f"执行: {stats.get('actions_executed', 0)}  |  "
            f"失败: {stats.get('actions_failed', 0)}"
        )

    @Slot()
    def _on_worker_finished(self):
        if self._recorder and self._recorder.is_recording:
            self._stop_recording()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._log("检测已停止")
        self._worker = None

    def _decision_log_callback(self, msg):
        w = self._worker
        if w:
            w.decision_log.emit(msg)

    @Slot(str)
    def _on_decision_log(self, msg):
        import time
        ts = time.strftime("%H:%M:%S")
        self.decision_log_text.append(f"[{ts}] {msg}")
        scrollbar = self.decision_log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _log(self, text):
        self.log_text.append(text)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ================================================================
    #  模型管理
    # ================================================================

    def _scan_models(self):
        import glob
        existing = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
        for best in glob.glob("runs/**/best.pt", recursive=True):
            best = best.replace("\\", "/")
            if best not in existing:
                self.model_combo.addItem(f"{best}")
                existing.append(best)

    @Slot()
    def _load_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "PyTorch 模型 (*.pt);;ONNX 模型 (*.onnx);;所有文件 (*)"
        )
        if path:
            self.model_combo.setEditText(path)
            self._log(f"已加载: {path}")

    @Slot()
    def _open_train_dialog(self):
        from .train_dialog import TrainDialog
        dialog = TrainDialog(self)
        if dialog.exec() == QDialog.Accepted:
            model_path = dialog.get_model_path()
            if model_path:
                self.model_combo.setEditText(model_path)
                self._log(f"训练模型: {model_path}")

    # ================================================================
    #  设置保存 / 加载
    # ================================================================

    def _save_settings(self):
        s = self._settings
        s.setValue("app/mode", self._current_mode)
        s.setValue("source/type", self.source_type.currentText())
        s.setValue("source/path", self.path_input.text())
        s.setValue("source/loop", self.loop_combo.currentIndex())
        s.setValue("source/camera", self.camera_device.value())
        s.setValue("source/monitor", self.monitor_num.value())
        s.setValue("source/stream_url", self.stream_input.text())
        s.setValue("detector/model", self.model_combo.currentText())
        s.setValue("detector/confidence", self.conf_spin.value())
        s.setValue("detector/imgsz", self.imgsz_combo.currentText())
        s.setValue("decision/engine", self.engine_combo.currentText())
        s.setValue("decision/trained_dir", self.trained_model_dir.text())
        s.setValue("decision/trained_conf", self.trained_conf_spin.value())
        s.setValue("decision/trained_action_map", self.trained_action_map_edit.toPlainText())
        s.setValue("recorder/dir", self.rec_dir_input.text())
        s.setValue("train/data_dir", self.dt_data_dir.text())
        s.setValue("train/output_dir", self.dt_output_dir.text())
        s.setValue("train/model_type", self.dt_model_type.currentText())
        s.setValue("train/epochs", self.dt_epochs_spin.value())
        s.setValue("train/lr", self.dt_lr_spin.value())
        s.setValue("decision/provider", self.llm_provider_combo.currentText())
        s.setValue("decision/model", self.llm_model_combo.currentText())
        s.setValue("decision/base_url", self.llm_base_url.text())
        s.setValue("decision/interval", self.llm_interval.value())

    def _load_settings(self):
        s = self._settings
        if s.value("app/mode"):
            self._switch_mode(s.value("app/mode"))
        if s.value("source/type"):
            self.source_type.setCurrentText(s.value("source/type"))
        if s.value("source/path"):
            self.path_input.setText(s.value("source/path"))
        if s.value("source/loop") is not None:
            self.loop_combo.setCurrentIndex(int(s.value("source/loop", 0)))
        if s.value("source/camera") is not None:
            self.camera_device.setValue(int(s.value("source/camera", 0)))
        if s.value("source/monitor") is not None:
            self.monitor_num.setValue(int(s.value("source/monitor", 1)))
        if s.value("source/stream_url"):
            self.stream_input.setText(s.value("source/stream_url"))
        if s.value("detector/model"):
            self.model_combo.setEditText(s.value("detector/model"))
        if s.value("detector/confidence") is not None:
            self.conf_spin.setValue(float(s.value("detector/confidence", 0.5)))
        if s.value("detector/imgsz"):
            self.imgsz_combo.setCurrentText(s.value("detector/imgsz"))
        if s.value("decision/engine"):
            self.engine_combo.setCurrentText(s.value("decision/engine"))
        if s.value("decision/trained_dir"):
            self.trained_model_dir.setText(s.value("decision/trained_dir"))
        if s.value("decision/trained_conf") is not None:
            self.trained_conf_spin.setValue(float(s.value("decision/trained_conf", 0.3)))
        if s.value("decision/trained_action_map"):
            self.trained_action_map_edit.setPlainText(s.value("decision/trained_action_map"))
        if s.value("recorder/dir"):
            self.rec_dir_input.setText(s.value("recorder/dir"))
        if s.value("train/data_dir"):
            self.dt_data_dir.setText(s.value("train/data_dir"))
        if s.value("train/output_dir"):
            self.dt_output_dir.setText(s.value("train/output_dir"))
        if s.value("train/model_type"):
            self.dt_model_type.setCurrentText(s.value("train/model_type"))
        if s.value("train/epochs") is not None:
            self.dt_epochs_spin.setValue(int(s.value("train/epochs", 100)))
        if s.value("train/lr") is not None:
            self.dt_lr_spin.setValue(float(s.value("train/lr", 0.001)))
        if s.value("decision/provider"):
            self.llm_provider_combo.setCurrentText(s.value("decision/provider"))
        if s.value("decision/model"):
            self.llm_model_combo.setCurrentText(s.value("decision/model"))
        if s.value("decision/base_url"):
            self.llm_base_url.setText(s.value("decision/base_url"))
        if s.value("decision/interval") is not None:
            self.llm_interval.setValue(float(s.value("decision/interval", 1.0)))

    def closeEvent(self, event):
        self._save_settings()
        if self._worker:
            self._worker.stop()
        event.accept()

    # ================================================================
    #  拖放支持
    # ================================================================

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if not path:
            return
        ext = Path(path).suffix.lower()
        video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv'}
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        model_exts = {'.pt', '.onnx'}

        if ext in video_exts:
            self._switch_mode("agent")
            self.source_type.setCurrentText("video")
            self.path_input.setText(path)
            self._log(f"已加载视频: {Path(path).name}")
        elif ext in image_exts:
            self._switch_mode("agent")
            self.source_type.setCurrentText("image")
            self.path_input.setText(path)
            self._log(f"已加载图片: {Path(path).name}")
        elif ext in model_exts:
            self.model_combo.setEditText(path)
            self._log(f"已加载模型: {Path(path).name}")
        else:
            self._log(f"不支持的文件类型: {ext}")
