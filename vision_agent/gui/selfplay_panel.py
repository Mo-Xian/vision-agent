"""Agent 部署面板：设备连接 → 模型加载 → Agent 接管 → 实时画面。"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QLineEdit, QPushButton, QSpinBox,
    QTextEdit, QScrollArea, QFrame, QFileDialog, QSplitter,
)

from .styles import COLORS


class SelfPlayPanel(QWidget):
    """Agent 部署面板 — PC/手机 + 模型接管 + 实时画面。"""

    agent_start_requested = Signal()
    agent_stop_requested = Signal()
    log_signal = Signal(str)
    frame_signal = Signal(object)  # numpy array

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._connect_internal()

        # 帧刷新定时器
        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._request_frame)
        self._frame_timer.setInterval(200)  # 5 FPS display

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # 主分割：左侧控制 + 右侧画面
        splitter = QSplitter(Qt.Horizontal)

        # ── 左侧：控制面板 ──
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(6, 4, 6, 4)
        left_layout.setSpacing(6)

        # ━━━ 部署目标 ━━━
        target_group = QGroupBox("部署目标")
        tg = QVBoxLayout(target_group)
        tg.setSpacing(4)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("目标"))
        self.target_combo = QComboBox()
        self.target_combo.addItems([
            "PC（窗口捕获 + 键鼠）",
            "远程设备（PC / 手机客户端）",
        ])
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        target_row.addWidget(self.target_combo, 1)
        tg.addLayout(target_row)

        # -- PC 控件 --
        self.pc_target_widget = QWidget()
        pc_layout = QVBoxLayout(self.pc_target_widget)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        pc_layout.setSpacing(4)

        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("窗口"))
        self.window_combo = QComboBox()
        self.window_combo.setEditable(True)
        self.window_combo.addItem("（全屏）")
        self.window_combo.setToolTip("选择游戏窗口，Agent 将捕获此窗口并用键鼠控制")
        win_row.addWidget(self.window_combo, 1)
        self.refresh_windows_btn = QPushButton("刷新")
        self.refresh_windows_btn.setObjectName("browseBtn")
        self.refresh_windows_btn.setMaximumWidth(50)
        self.refresh_windows_btn.clicked.connect(self._refresh_window_list)
        win_row.addWidget(self.refresh_windows_btn)
        pc_layout.addLayout(win_row)

        pc_hint = QLabel("Agent 将通过 pynput 模拟键鼠操作控制游戏")
        pc_hint.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        pc_layout.addWidget(pc_hint)

        tg.addWidget(self.pc_target_widget)

        # -- 远程 PC 控件 --
        self.remote_target_widget = QWidget()
        rem_layout = QVBoxLayout(self.remote_target_widget)
        rem_layout.setContentsMargins(0, 0, 0, 0)
        rem_layout.setSpacing(4)

        # 连接方式
        rmode_row = QHBoxLayout()
        rmode_row.addWidget(QLabel("连接"))
        self.remote_agent_mode = QComboBox()
        self.remote_agent_mode.addItems(["局域网直连", "公网中继"])
        self.remote_agent_mode.currentIndexChanged.connect(self._on_agent_remote_mode_changed)
        rmode_row.addWidget(self.remote_agent_mode, 1)
        rem_layout.addLayout(rmode_row)

        # 直连
        self.agent_direct_widget = QWidget()
        ad_layout = QHBoxLayout(self.agent_direct_widget)
        ad_layout.setContentsMargins(0, 0, 0, 0)
        ad_layout.addWidget(QLabel("端口"))
        self.remote_agent_port = QSpinBox()
        self.remote_agent_port.setRange(1, 65535)
        self.remote_agent_port.setValue(9876)
        self.remote_agent_port.setMaximumWidth(70)
        ad_layout.addWidget(self.remote_agent_port)
        rem_layout.addWidget(self.agent_direct_widget)

        # 中继
        self.agent_relay_widget = QWidget()
        ar_layout = QVBoxLayout(self.agent_relay_widget)
        ar_layout.setContentsMargins(0, 0, 0, 0)
        ar_layout.setSpacing(4)
        ar1 = QHBoxLayout()
        ar1.addWidget(QLabel("中继"))
        self.agent_relay_url = QLineEdit()
        self.agent_relay_url.setPlaceholderText("ws://你的公网服务器:9877")
        ar1.addWidget(self.agent_relay_url, 1)
        ar_layout.addLayout(ar1)
        ar2 = QHBoxLayout()
        ar2.addWidget(QLabel("房间"))
        self.agent_relay_room = QLineEdit()
        self.agent_relay_room.setPlaceholderText("自动生成")
        self.agent_relay_room.setMaximumWidth(120)
        ar2.addWidget(self.agent_relay_room)
        ar2.addWidget(QLabel("Token"))
        self.agent_relay_token = QLineEdit()
        self.agent_relay_token.setPlaceholderText("可选")
        self.agent_relay_token.setMaximumWidth(100)
        ar2.addWidget(self.agent_relay_token)
        ar_layout.addLayout(ar2)
        rem_layout.addWidget(self.agent_relay_widget)
        self.agent_relay_widget.setVisible(False)

        # 按钮
        rhub_btn_row = QHBoxLayout()
        self.start_agent_hub_btn = QPushButton("启动中转服务")
        self.start_agent_hub_btn.setObjectName("startBtn")
        self.start_agent_hub_btn.clicked.connect(self._start_agent_hub)
        rhub_btn_row.addWidget(self.start_agent_hub_btn)
        self.stop_agent_hub_btn = QPushButton("停止")
        self.stop_agent_hub_btn.setObjectName("stopBtn")
        self.stop_agent_hub_btn.setMaximumWidth(50)
        self.stop_agent_hub_btn.setEnabled(False)
        self.stop_agent_hub_btn.clicked.connect(self._stop_agent_hub)
        rhub_btn_row.addWidget(self.stop_agent_hub_btn)
        rem_layout.addLayout(rhub_btn_row)

        self.remote_agent_status = QLabel(
            "启动后，远程设备连接即可，Agent 获取画面并发送操控指令"
        )
        self.remote_agent_status.setWordWrap(True)
        self.remote_agent_status.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 11px;"
        )
        rem_layout.addWidget(self.remote_agent_status)

        tg.addWidget(self.remote_target_widget)
        self.remote_target_widget.setVisible(False)

        left_layout.addWidget(target_group)

        # ━━━ 游戏预设 ━━━
        preset_group = QGroupBox("游戏预设")
        pg = QVBoxLayout(preset_group)
        pg.setSpacing(4)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("预设"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["wzry - 王者荣耀", "fps - FPS 射击", "generic - 通用"])
        preset_row.addWidget(self.preset_combo, 1)
        pg.addLayout(preset_row)

        custom_row = QHBoxLayout()
        custom_row.addWidget(QLabel("自定义"))
        self.custom_yaml = QLineEdit()
        self.custom_yaml.setPlaceholderText("YAML 配置文件路径（可选，覆盖预设）")
        custom_row.addWidget(self.custom_yaml, 1)
        self.browse_yaml_btn = QPushButton("...")
        self.browse_yaml_btn.setObjectName("browseBtn")
        self.browse_yaml_btn.setMaximumWidth(30)
        custom_row.addWidget(self.browse_yaml_btn)
        pg.addLayout(custom_row)

        left_layout.addWidget(preset_group)

        # ━━━ Agent 部署 ━━━
        agent_group = QGroupBox("Agent 部署")
        ag = QVBoxLayout(agent_group)
        ag.setSpacing(4)

        agent_info = QLabel(
            "选择训练好的模型，Agent 将根据模型决策自动操控游戏（自动检测模型类型）。"
        )
        agent_info.setWordWrap(True)
        agent_info.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        ag.addWidget(agent_info)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型"))
        self.agent_model_path = QLineEdit()
        self.agent_model_path.setPlaceholderText("模型目录（如 runs/selfplay/exp1/latest）")
        model_row.addWidget(self.agent_model_path, 1)
        self.browse_agent_model_btn = QPushButton("...")
        self.browse_agent_model_btn.setObjectName("browseBtn")
        self.browse_agent_model_btn.setMaximumWidth(30)
        model_row.addWidget(self.browse_agent_model_btn)
        ag.addLayout(model_row)

        agent_btn_row = QHBoxLayout()
        self.agent_start_btn = QPushButton("Agent 接管")
        self.agent_start_btn.setObjectName("purpleBtn")
        self.agent_start_btn.setCursor(Qt.PointingHandCursor)
        self.agent_start_btn.setMinimumHeight(32)
        agent_btn_row.addWidget(self.agent_start_btn)

        self.agent_stop_btn = QPushButton("停止")
        self.agent_stop_btn.setObjectName("stopBtn")
        self.agent_stop_btn.setCursor(Qt.PointingHandCursor)
        self.agent_stop_btn.setMinimumHeight(32)
        self.agent_stop_btn.setEnabled(False)
        agent_btn_row.addWidget(self.agent_stop_btn)
        ag.addLayout(agent_btn_row)

        self.agent_status = QLabel("")
        self.agent_status.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        ag.addWidget(self.agent_status)

        left_layout.addWidget(agent_group)

        # ━━━ 日志 ━━━
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setPlaceholderText("Agent 日志...")
        left_layout.addWidget(self.log_text)

        left_layout.addStretch()
        left_scroll.setWidget(left_content)

        # ── 右侧：实时画面 ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        screen_label = QLabel("实时画面")
        screen_label.setStyleSheet(
            f"color: {COLORS['text']}; font-size: 12px; font-weight: bold; padding: 2px;"
        )
        right_layout.addWidget(screen_label)

        self.screen_display = QLabel()
        self.screen_display.setAlignment(Qt.AlignCenter)
        self.screen_display.setMinimumSize(320, 180)
        self.screen_display.setStyleSheet(
            f"background-color: {COLORS['bg_deep']}; "
            f"border: 1px solid {COLORS['border']}; "
            f"border-radius: 8px;"
        )
        self.screen_display.setText("未连接设备")
        right_layout.addWidget(self.screen_display, 1)

        self.screen_info = QLabel("画面来自游戏窗口截取或远程客户端推送，Agent 接管后自动显示")
        self.screen_info.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        self.screen_info.setWordWrap(True)
        right_layout.addWidget(self.screen_info)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([420, 560])

        root.addWidget(splitter)

    def _connect_internal(self):
        self.browse_yaml_btn.clicked.connect(self._browse_yaml)
        self.browse_agent_model_btn.clicked.connect(self._browse_agent_model)
        self.agent_start_btn.clicked.connect(self.agent_start_requested)
        self.agent_stop_btn.clicked.connect(self.agent_stop_requested)
        self.log_signal.connect(self._append_log)
        self.frame_signal.connect(self._update_frame)

    # ── 部署目标切换 ──

    def _on_target_changed(self, index: int):
        self.pc_target_widget.setVisible(index == 0)
        self.remote_target_widget.setVisible(index == 1)
        if index == 0:
            self._refresh_window_list()

    def is_remote_target(self) -> bool:
        return self.target_combo.currentIndex() == 1

    def get_agent_hub(self):
        """返回当前 Agent 中转服务 RemoteHub 实例。"""
        return getattr(self, '_agent_hub', None)

    def get_agent_hub_port(self) -> int:
        return self.remote_agent_port.value()

    def _on_agent_remote_mode_changed(self, index: int):
        self.agent_direct_widget.setVisible(index == 0)
        self.agent_relay_widget.setVisible(index == 1)
        self.start_agent_hub_btn.setText("启动中转服务" if index == 0 else "连接中继")

    def _start_agent_hub(self):
        from ..data.remote_hub import RemoteHub
        is_relay = self.remote_agent_mode.currentIndex() == 1

        if is_relay:
            relay_url = self.agent_relay_url.text().strip()
            if not relay_url:
                self.remote_agent_status.setText("请输入中继服务器地址")
                self.remote_agent_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px;")
                return
            if not relay_url.startswith("ws://") and not relay_url.startswith("wss://"):
                relay_url = f"ws://{relay_url}"
            room_id = self.agent_relay_room.text().strip()
            token = self.agent_relay_token.text().strip()
            self._agent_hub = RemoteHub(
                relay_url=relay_url,
                room_id=room_id,
                relay_token=token,
                on_log=lambda msg: self.remote_agent_status.setText(msg.split("] ", 1)[-1]),
            )
            self._agent_hub.start()
            self.agent_relay_room.setText(self._agent_hub.room_id)
            self.remote_agent_status.setText(
                f"正在连接中继: {relay_url}\n"
                f"房间: {self._agent_hub.room_id}"
            )
        else:
            port = self.get_agent_hub_port()
            self._agent_hub = RemoteHub(
                port=port,
                on_log=lambda msg: self.remote_agent_status.setText(msg.split("] ", 1)[-1]),
            )
            self._agent_hub.start()
            local_ip = self._agent_hub.get_local_ip()
            self.remote_agent_status.setText(
                f"中转服务已启动: ws://{local_ip}:{port}\n"
                f"请在远程设备连接此地址"
            )

        self.remote_agent_status.setStyleSheet(
            f"color: {COLORS['success']}; font-size: 11px;"
        )
        self.start_agent_hub_btn.setEnabled(False)
        self.stop_agent_hub_btn.setEnabled(True)
        self.remote_agent_mode.setEnabled(False)

    def _stop_agent_hub(self):
        hub = getattr(self, '_agent_hub', None)
        if hub:
            hub.stop()
            self._agent_hub = None
        self.remote_agent_status.setText("已停止")
        self.remote_agent_status.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 11px;"
        )
        self.start_agent_hub_btn.setEnabled(True)
        self.stop_agent_hub_btn.setEnabled(False)
        self.remote_agent_mode.setEnabled(True)

    def get_window_title(self) -> str:
        text = self.window_combo.currentText()
        if text == "（全屏）" or not text.strip():
            return ""
        return text.strip()

    def _refresh_window_list(self):
        from ..data.game_recorder import GameRecorder
        titles = GameRecorder.list_windows()
        current = self.window_combo.currentText()
        self.window_combo.clear()
        self.window_combo.addItem("（全屏）")
        for t in titles:
            self.window_combo.addItem(t)
        idx = self.window_combo.findText(current)
        if idx >= 0:
            self.window_combo.setCurrentIndex(idx)

    def _browse_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择预设 YAML", "profiles",
            "YAML 文件 (*.yaml *.yml);;所有文件 (*)",
        )
        if path:
            self.custom_yaml.setText(path)

    def _browse_agent_model(self):
        from .workshop_panel import ModelBrowserDialog
        from PySide6.QtWidgets import QDialog
        dlg = ModelBrowserDialog(self, base_dir="runs")
        if dlg.exec() == QDialog.Accepted and dlg.selected_path():
            self.agent_model_path.setText(dlg.selected_path())

    # ── 公共接口 ──

    def get_preset_name(self) -> str:
        custom = self.custom_yaml.text().strip()
        if custom:
            return custom
        text = self.preset_combo.currentText()
        return text.split(" - ")[0].strip()

    def get_agent_model_dir(self) -> str:
        return self.agent_model_path.text().strip()

    def set_model_dir(self, path: str):
        """设置模型目录（供外部调用，如学习完成后自动填入）。"""
        self.agent_model_path.setText(path)

    def set_agent_running_state(self, running: bool):
        self.agent_start_btn.setEnabled(not running)
        self.agent_stop_btn.setEnabled(running)
        self.target_combo.setEnabled(not running)
        self.window_combo.setEnabled(not running)
        self.preset_combo.setEnabled(not running)
        if running:
            self.agent_status.setText("Agent 运行中...")
            self.agent_status.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px;")
            self._frame_timer.start()
        else:
            self.agent_status.setText("")
            self.agent_status.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
            self._frame_timer.stop()

    # ── 信号处理 ──

    def _append_log(self, msg: str):
        self.log_text.append(msg)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _update_frame(self, frame):
        """将 numpy BGR 帧显示到 QLabel。"""
        if frame is None:
            return
        try:
            import cv2
            import numpy as np
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            label_size = self.screen_display.size()
            pixmap = QPixmap.fromImage(qimg).scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self.screen_display.setPixmap(pixmap)
        except Exception:
            pass

    def _request_frame(self):
        pass
