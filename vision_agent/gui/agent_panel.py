"""Agent 执行面板：运行配置 + 对话控制（2 Tab 精简布局）。"""

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
from .widgets import CollapsibleSection


class AgentPanel(QTabWidget):
    """Agent 执行面板，2 个 Tab：运行配置、对话控制。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_config_tab()
        self._build_chat_tab()

    # ── Tab 1: 运行配置（输入源 + 决策引擎 + 场景 合并） ──

    def _build_config_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ▸ 输入源 + 决策引擎（核心设置，平铺）
        core_group = QGroupBox("基本配置")
        cg = QFormLayout(core_group)
        cg.setSpacing(6)

        self.source_type = QComboBox()
        self.source_type.addItems(["screen", "camera", "video", "image", "stream"])
        self.source_type.setCurrentText("video")
        cg.addRow("输入源", self.source_type)

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["none", "rule", "trained", "llm", "hierarchical", "rl"])
        self.engine_combo.setCurrentText("rule")
        cg.addRow("决策引擎", self.engine_combo)

        self.agent_profile_combo = QComboBox()
        self.agent_profile_combo.addItem("(无)")
        cg.addRow("场景 Profile", self.agent_profile_combo)

        layout.addWidget(core_group)

        # ▸ 输入源详细配置（根据类型动态显示）
        self._source_detail = QWidget()
        sd_layout = QVBoxLayout(self._source_detail)
        sd_layout.setContentsMargins(0, 0, 0, 0)
        sd_layout.setSpacing(4)

        # 路径行（video / image）
        path_row = QHBoxLayout()
        self.path_label = QLabel("文件路径")
        path_row.addWidget(self.path_label)
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("文件路径 / rtsp:// / http://")
        path_row.addWidget(self.path_input)
        self.browse_btn = QPushButton("...")
        self.browse_btn.setObjectName("browseBtn")
        self.browse_btn.setMaximumWidth(32)
        path_row.addWidget(self.browse_btn)
        sd_layout.addLayout(path_row)

        loop_row = QHBoxLayout()
        loop_row.addWidget(QLabel("循环"))
        self.loop_combo = QComboBox()
        self.loop_combo.addItems(["否", "是"])
        loop_row.addWidget(self.loop_combo)
        loop_row.addStretch()
        sd_layout.addLayout(loop_row)

        # 摄像头
        self.camera_label = QLabel("设备号")
        sd_layout.addWidget(self.camera_label)
        self.camera_device = QSpinBox()
        self.camera_device.setRange(0, 10)
        sd_layout.addWidget(self.camera_device)

        # 屏幕
        self.monitor_label = QLabel("显示器")
        sd_layout.addWidget(self.monitor_label)
        self.monitor_num = QSpinBox()
        self.monitor_num.setRange(1, 5)
        sd_layout.addWidget(self.monitor_num)

        # 实时流
        self.stream_label = QLabel("流地址")
        sd_layout.addWidget(self.stream_label)
        self.stream_input = QLineEdit()
        self.stream_input.setPlaceholderText(
            "rtsp://...  |  B站直播间号  |  http://.../live.flv"
        )
        sd_layout.addWidget(self.stream_input)
        self.stream_hint = QLabel("支持: RTSP / HTTP-FLV / HLS / B站直播间号")
        self.stream_hint.setStyleSheet("color: gray; font-size: 11px;")
        sd_layout.addWidget(self.stream_hint)

        layout.addWidget(self._source_detail)

        # ▸ 决策引擎配置（按引擎类型动态显示）

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
        layout.addWidget(self.llm_interval_widget)

        # LLM 配置提示
        self.llm_config_widget = QWidget()
        self.llm_config_widget.setVisible(False)
        llm_hint_layout = QVBoxLayout(self.llm_config_widget)
        llm_hint_layout.setContentsMargins(0, 0, 0, 0)
        llm_hint_layout.addWidget(QLabel("使用训练工坊中配置的 LLM"))
        layout.addWidget(self.llm_config_widget)

        # Trained 模型配置（折叠）
        self.trained_config_widget = CollapsibleSection("训练模型配置")
        self.trained_config_widget.setVisible(False)
        tc = self.trained_config_widget.content_layout()

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
        tc.addLayout(trained_dir_row)

        trained_form = QFormLayout()
        self.trained_conf_spin = QDoubleSpinBox()
        self.trained_conf_spin.setRange(0.05, 1.0)
        self.trained_conf_spin.setSingleStep(0.05)
        self.trained_conf_spin.setValue(0.30)
        trained_form.addRow("置信阈值", self.trained_conf_spin)
        tc.addLayout(trained_form)

        tc.addWidget(QLabel("动作映射 (语义→按键):"))
        self.trained_action_map_edit = QTextEdit()
        self.trained_action_map_edit.setMaximumHeight(80)
        self.trained_action_map_edit.setPlaceholderText(
            '{"attack": {"type": "key", "key": "a"}, ...}'
        )
        default_map = {
            "attack": {"type": "key", "key": "a"},
            "retreat": {"type": "key", "key": "s"},
            "skill_1": {"type": "key", "key": "1"},
            "skill_2": {"type": "key", "key": "2"},
            "skill_3": {"type": "key", "key": "3"},
        }
        self.trained_action_map_edit.setPlainText(json.dumps(default_map, indent=2, ensure_ascii=False))
        tc.addWidget(self.trained_action_map_edit)
        layout.addWidget(self.trained_config_widget)

        # ▸ 场景（折叠）
        scene_section = CollapsibleSection("场景与 AutoPilot")
        sg = scene_section.content_layout()

        self.profile_refresh_btn = QPushButton("刷新 Profile 列表")
        self.profile_refresh_btn.setObjectName("browseBtn")
        self.profile_refresh_btn.setCursor(Qt.PointingHandCursor)
        sg.addWidget(self.profile_refresh_btn)

        self.autopilot_check = QCheckBox("启用 AutoPilot（自动识别场景并切换）")
        self.autopilot_check.setToolTip("自动根据检测结果识别场景，切换 Profile")
        sg.addWidget(self.autopilot_check)

        self.scene_status_text = QTextEdit()
        self.scene_status_text.setReadOnly(True)
        self.scene_status_text.setMaximumHeight(80)
        self.scene_status_text.setPlaceholderText("启动检测后显示场景信息")
        sg.addWidget(self.scene_status_text)
        layout.addWidget(scene_section)

        layout.addStretch()
        scroll.setWidget(tab)
        self.addTab(scroll, "运行配置")

        # 初始化可见性
        self.update_source_visibility("video")

    # ── Tab 2: 对话控制 ──

    def _build_chat_tab(self):
        self.chat_panel = ChatPanel()
        self.addTab(self.chat_panel, "对话控制")

    # ── 动态可见性 ──

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
            self.path_label.setText("文件路径")

        self.camera_label.setVisible(is_camera)
        self.camera_device.setVisible(is_camera)
        self.monitor_label.setVisible(is_screen)
        self.monitor_num.setVisible(is_screen)
        self.stream_label.setVisible(is_stream)
        self.stream_input.setVisible(is_stream)
        self.stream_hint.setVisible(is_stream)

    def update_engine_visibility(self, engine_type: str):
        """根据引擎类型更新控件可见性。"""
        is_trained = engine_type == "trained"
        is_llm = engine_type == "llm"

        self.trained_config_widget.setVisible(is_trained)
        if is_trained:
            self.trained_config_widget.set_expanded(True)
        self.llm_config_widget.setVisible(is_llm)
        self.llm_interval_widget.setVisible(is_llm)
