"""Agent 执行面板：输入源、检测决策、场景管理、对话控制。"""

import json
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QLineEdit, QPushButton, QSpinBox,
    QDoubleSpinBox, QTextEdit, QFormLayout, QCheckBox,
    QScrollArea, QFrame,
)

from .chat_panel import ChatPanel
from .styles import COLORS


class AgentPanel(QTabWidget):
    """Agent 执行面板，包含 4 个 Tab。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_source_tab()
        self._build_engine_tab()
        self._build_scene_tab()
        self._build_chat_tab()

    # ── Tab 1: 输入源 ──

    def _build_source_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setSpacing(6)
        self.source_type = QComboBox()
        self.source_type.addItems(["screen", "camera", "video", "image", "stream"])
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
        self.stream_hint = QLabel("支持: RTSP / HTTP-FLV / HLS(m3u8) / B站直播间号")
        self.stream_hint.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.stream_hint)

        layout.addStretch()
        self.addTab(tab, "输入源")

        # 初始化可见性
        self.source_type.setCurrentText("video")

    # ── Tab 2: 检测决策 ──

    def _build_engine_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 决策引擎
        engine_group = QGroupBox("决策引擎")
        eg = QVBoxLayout(engine_group)

        form_top = QFormLayout()
        form_top.setSpacing(6)
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["none", "rule", "trained", "llm", "hierarchical", "rl"])
        self.engine_combo.setCurrentText("rule")
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

        # Trained 配置
        self.trained_config_widget = QGroupBox("训练模型配置")
        trained_layout = QVBoxLayout(self.trained_config_widget)
        trained_layout.setSpacing(6)

        trained_dir_row = QHBoxLayout()
        trained_dir_row.addWidget(QLabel("模型目录"))
        self.trained_model_dir = QLineEdit()
        self.trained_model_dir.setPlaceholderText("runs/decision/exp1")
        self.trained_model_dir.setText("runs/decision/exp1")
        trained_dir_row.addWidget(self.trained_model_dir)
        self.trained_browse_btn = QPushButton("...")
        self.trained_browse_btn.setObjectName("browseBtn")
        self.trained_browse_btn.setMaximumWidth(32)
        trained_dir_row.addWidget(self.trained_browse_btn)
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

        # LLM 配置提示
        self.llm_config_widget = QGroupBox("LLM 决策配置")
        self.llm_config_widget.setVisible(False)
        llm_hint = QVBoxLayout(self.llm_config_widget)
        llm_hint.addWidget(QLabel("使用训练工坊中配置的 LLM 供应商和模型"))
        layout.addWidget(self.llm_config_widget)

        layout.addStretch()
        scroll.setWidget(tab)
        self.addTab(scroll, "检测决策")

    # ── Tab 3: 场景 ──

    def _build_scene_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Profile 快速选择
        layout.addWidget(QLabel("场景 Profile（训练工坊中管理）"))
        self.agent_profile_combo = QComboBox()
        self.agent_profile_combo.addItem("(无)")
        layout.addWidget(self.agent_profile_combo)

        self.profile_refresh_btn = QPushButton("刷新")
        self.profile_refresh_btn.setObjectName("browseBtn")
        self.profile_refresh_btn.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.profile_refresh_btn)

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
        self.addTab(tab, "场景")

    # ── Tab 4: 对话控制 ──

    def _build_chat_tab(self):
        self.chat_panel = ChatPanel()
        self.addTab(self.chat_panel, "对话控制")

    def update_source_visibility(self, source_type: str):
        """根据输入源类型更新控件可见性。"""
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

    def update_engine_visibility(self, engine_type: str):
        """根据引擎类型更新控件可见性。"""
        self.trained_config_widget.setVisible(engine_type == "trained")
        self.llm_config_widget.setVisible(engine_type == "llm")
        self.llm_interval_widget.setVisible(engine_type == "llm")
