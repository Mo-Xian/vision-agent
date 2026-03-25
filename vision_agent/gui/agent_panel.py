"""Agent 执行面板：极简布局，核心配置 + 对话控制。"""

import json
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QSpinBox,
    QDoubleSpinBox, QTextEdit, QFormLayout, QCheckBox,
    QScrollArea, QFrame,
)

from .chat_panel import ChatPanel
from .styles import COLORS
from .widgets import CollapsibleSection


class AgentPanel(QTabWidget):
    """Agent 执行面板，2 个 Tab：配置、对话。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_config_tab()
        self._build_chat_tab()

    # ── Tab 1: 配置 ──

    def _build_config_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ━━━ 核心 3 项：输入源 / 引擎 / Profile ━━━
        core = QFormLayout()
        core.setSpacing(6)

        self.source_type = QComboBox()
        self.source_type.addItems(["screen", "camera", "video", "image", "stream"])
        self.source_type.setCurrentText("video")
        core.addRow("输入源", self.source_type)

        # 文件路径（video/image 时显示）
        path_widget = QWidget()
        path_layout = QHBoxLayout(path_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(4)
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("视频/图片路径")
        path_layout.addWidget(self.path_input)
        self.browse_btn = QPushButton("...")
        self.browse_btn.setObjectName("browseBtn")
        self.browse_btn.setMaximumWidth(32)
        path_layout.addWidget(self.browse_btn)
        self.path_label = QLabel("文件")
        core.addRow(self.path_label, path_widget)

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["none", "rule", "trained", "llm", "hierarchical", "rl"])
        self.engine_combo.setCurrentText("rule")
        core.addRow("引擎", self.engine_combo)

        self.agent_profile_combo = QComboBox()
        self.agent_profile_combo.addItem("(无)")
        core.addRow("Profile", self.agent_profile_combo)

        layout.addLayout(core)

        # ━━━ 输入源附加参数（动态显示） ━━━

        # 循环播放
        self.loop_combo = QComboBox()
        self.loop_combo.addItems(["否", "是"])
        self._loop_row = QWidget()
        lr = QHBoxLayout(self._loop_row)
        lr.setContentsMargins(0, 0, 0, 0)
        lr.addWidget(QLabel("循环"))
        lr.addWidget(self.loop_combo)
        lr.addStretch()
        layout.addWidget(self._loop_row)

        # 摄像头设备号
        self.camera_label = QLabel("摄像头设备号")
        self.camera_device = QSpinBox()
        self.camera_device.setRange(0, 10)
        layout.addWidget(self.camera_label)
        layout.addWidget(self.camera_device)

        # 显示器编号
        self.monitor_label = QLabel("显示器编号")
        self.monitor_num = QSpinBox()
        self.monitor_num.setRange(1, 5)
        layout.addWidget(self.monitor_label)
        layout.addWidget(self.monitor_num)

        # 流地址
        self.stream_label = QLabel("流地址")
        self.stream_input = QLineEdit()
        self.stream_input.setPlaceholderText("rtsp://... | B站直播间号 | http://.../live.flv")
        self.stream_hint = QLabel("RTSP / HTTP-FLV / HLS / B站直播间号")
        self.stream_hint.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.stream_label)
        layout.addWidget(self.stream_input)
        layout.addWidget(self.stream_hint)

        # ━━━ 更多设置（折叠） ━━━
        more = CollapsibleSection("更多设置")
        mg = more.content_layout()

        # LLM 决策间隔
        self.llm_interval_widget = QWidget()
        interval_layout = QFormLayout(self.llm_interval_widget)
        interval_layout.setContentsMargins(0, 0, 0, 0)
        self.llm_interval = QDoubleSpinBox()
        self.llm_interval.setRange(0.1, 30.0)
        self.llm_interval.setSingleStep(0.5)
        self.llm_interval.setValue(1.0)
        interval_layout.addRow("LLM 决策间隔(秒)", self.llm_interval)
        self.llm_interval_widget.setVisible(False)
        mg.addWidget(self.llm_interval_widget)

        # LLM 提示
        self.llm_config_widget = QWidget()
        self.llm_config_widget.setVisible(False)
        llm_hint_layout = QVBoxLayout(self.llm_config_widget)
        llm_hint_layout.setContentsMargins(0, 0, 0, 0)
        llm_hint_layout.addWidget(QLabel("使用训练工坊中配置的 LLM"))
        mg.addWidget(self.llm_config_widget)

        # Trained 配置
        self.trained_config_widget = QWidget()
        self.trained_config_widget.setVisible(False)
        tc_layout = QVBoxLayout(self.trained_config_widget)
        tc_layout.setContentsMargins(0, 0, 0, 0)
        tc_layout.setSpacing(4)

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
        tc_layout.addLayout(trained_dir_row)

        trained_form = QFormLayout()
        self.trained_conf_spin = QDoubleSpinBox()
        self.trained_conf_spin.setRange(0.05, 1.0)
        self.trained_conf_spin.setSingleStep(0.05)
        self.trained_conf_spin.setValue(0.30)
        trained_form.addRow("置信阈值", self.trained_conf_spin)
        tc_layout.addLayout(trained_form)

        tc_layout.addWidget(QLabel("动作映射 (语义→按键):"))
        self.trained_action_map_edit = QTextEdit()
        self.trained_action_map_edit.setMaximumHeight(80)
        default_map = {
            "attack": {"type": "key", "key": "a"},
            "retreat": {"type": "key", "key": "s"},
            "skill_1": {"type": "key", "key": "1"},
            "skill_2": {"type": "key", "key": "2"},
            "skill_3": {"type": "key", "key": "3"},
        }
        self.trained_action_map_edit.setPlainText(
            json.dumps(default_map, indent=2, ensure_ascii=False)
        )
        tc_layout.addWidget(self.trained_action_map_edit)
        mg.addWidget(self.trained_config_widget)

        # AutoPilot
        self.autopilot_check = QCheckBox("AutoPilot（自动识别场景并切换）")
        self.autopilot_check.setToolTip("自动根据检测结果识别场景，切换 Profile")
        mg.addWidget(self.autopilot_check)

        self.profile_refresh_btn = QPushButton("刷新 Profile")
        self.profile_refresh_btn.setObjectName("browseBtn")
        self.profile_refresh_btn.setCursor(Qt.PointingHandCursor)
        mg.addWidget(self.profile_refresh_btn)

        self.scene_status_text = QTextEdit()
        self.scene_status_text.setReadOnly(True)
        self.scene_status_text.setMaximumHeight(60)
        self.scene_status_text.setPlaceholderText("场景状态")
        mg.addWidget(self.scene_status_text)

        layout.addWidget(more)

        layout.addStretch()
        scroll.setWidget(tab)
        self.addTab(scroll, "配置")

        # 初始化
        self.update_source_visibility("video")

    # ── Tab 2: 对话控制 ──

    def _build_chat_tab(self):
        self.chat_panel = ChatPanel()
        self.addTab(self.chat_panel, "对话")

    # ── 动态可见性 ──

    def update_source_visibility(self, source_type: str):
        """根据输入源类型更新控件可见性。"""
        is_video = source_type == "video"
        is_camera = source_type == "camera"
        is_screen = source_type == "screen"
        is_image = source_type == "image"
        is_stream = source_type == "stream"

        show_path = is_video or is_image
        self.path_label.setVisible(show_path)
        self.path_input.setVisible(show_path)
        self.browse_btn.setVisible(show_path)
        self.path_input.parentWidget().setVisible(show_path)
        self._loop_row.setVisible(is_video or is_image)

        if is_image:
            self.path_label.setText("图片")
        else:
            self.path_label.setText("文件")

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
