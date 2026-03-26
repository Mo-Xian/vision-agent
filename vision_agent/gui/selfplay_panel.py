"""Agent 部署面板：手机连接 → 模型加载 → Agent 接管 → 实时画面。"""

import subprocess
import re
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QLineEdit, QPushButton,
    QTextEdit, QScrollArea, QFrame, QFileDialog, QSplitter,
)

from .styles import COLORS


class SelfPlayPanel(QWidget):
    """Agent 部署面板 — 手机连接 + 模型接管 + 实时画面。"""

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
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(10)

        # ━━━ 设备连接 ━━━
        device_group = QGroupBox("设备连接")
        dg = QVBoxLayout(device_group)
        dg.setSpacing(6)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("设备"))
        self.device_combo = QComboBox()
        self.device_combo.setEditable(True)
        self.device_combo.setPlaceholderText("自动检测或输入序列号")
        dev_row.addWidget(self.device_combo, 1)
        self.refresh_device_btn = QPushButton("刷新")
        self.refresh_device_btn.setObjectName("browseBtn")
        self.refresh_device_btn.setMaximumWidth(50)
        dev_row.addWidget(self.refresh_device_btn)
        dg.addLayout(dev_row)

        status_row = QHBoxLayout()
        self.device_status = QLabel("未连接")
        self.device_status.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        status_row.addWidget(self.device_status)
        status_row.addStretch()
        self.check_device_btn = QPushButton("测试连接")
        self.check_device_btn.setObjectName("infoBtn")
        self.check_device_btn.setMaximumWidth(80)
        status_row.addWidget(self.check_device_btn)
        dg.addLayout(status_row)

        left_layout.addWidget(device_group)

        # ━━━ 游戏预设 ━━━
        preset_group = QGroupBox("游戏预设")
        pg = QVBoxLayout(preset_group)
        pg.setSpacing(6)

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
        ag.setSpacing(6)

        agent_info = QLabel(
            "选择训练好的模型（BC 或 DQN），Agent 将根据模型决策自动操控游戏。"
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
        self.agent_start_btn.setMinimumHeight(42)
        agent_btn_row.addWidget(self.agent_start_btn)

        self.agent_stop_btn = QPushButton("停止")
        self.agent_stop_btn.setObjectName("stopBtn")
        self.agent_stop_btn.setCursor(Qt.PointingHandCursor)
        self.agent_stop_btn.setMinimumHeight(42)
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
        self.log_text.setMaximumHeight(200)
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
            f"color: {COLORS['text']}; font-size: 13px; font-weight: bold; padding: 4px;"
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

        screen_info = QLabel("画面来自 scrcpy 窗口截取，Agent 接管后自动显示")
        screen_info.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        screen_info.setWordWrap(True)
        right_layout.addWidget(screen_info)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        root.addWidget(splitter)

    def _connect_internal(self):
        self.refresh_device_btn.clicked.connect(self._refresh_devices)
        self.check_device_btn.clicked.connect(self._check_device)
        self.browse_yaml_btn.clicked.connect(self._browse_yaml)
        self.browse_agent_model_btn.clicked.connect(self._browse_agent_model)
        self.agent_start_btn.clicked.connect(self.agent_start_requested)
        self.agent_stop_btn.clicked.connect(self.agent_stop_requested)
        self.log_signal.connect(self._append_log)
        self.frame_signal.connect(self._update_frame)

    # ── 设备管理 ──

    @staticmethod
    def _find_adb() -> str | None:
        """查找 adb 可执行文件路径。"""
        import shutil
        path = shutil.which("adb")
        if path:
            return path
        candidates = [
            Path.home() / "AppData/Local/Android/Sdk/platform-tools/adb.exe",
            Path("C:/platform-tools/adb.exe"),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def _adb_cmd(self, *args) -> list[str]:
        """构造 adb 命令，自动查找 adb 路径。"""
        adb = self._find_adb()
        if not adb:
            raise FileNotFoundError("adb")
        cmd = [adb]
        serial = self.get_device_serial()
        if serial:
            cmd.extend(["-s", serial])
        cmd.extend(args)
        return cmd

    def _get_device_model(self, adb: str, serial: str) -> str:
        """获取设备型号和分辨率。"""
        try:
            r = subprocess.run(
                [adb, "-s", serial, "shell", "getprop", "ro.product.model"],
                capture_output=True, text=True, timeout=3,
            )
            model = r.stdout.strip() or "未知型号"
        except Exception:
            model = "未知型号"

        try:
            r = subprocess.run(
                [adb, "-s", serial, "shell", "wm", "size"],
                capture_output=True, text=True, timeout=3,
            )
            m = re.search(r"(\d+x\d+)", r.stdout)
            resolution = m.group(1) if m else ""
        except Exception:
            resolution = ""

        parts = [model]
        if resolution:
            parts.append(resolution)
        return " | ".join(parts)

    def _refresh_devices(self):
        """刷新 ADB 设备列表，显示序列号 + 型号 + 分辨率。"""
        self.device_combo.clear()
        try:
            adb = self._find_adb()
            if not adb:
                raise FileNotFoundError("adb")
            r = subprocess.run(
                [adb, "devices"], capture_output=True, text=True, timeout=5,
            )
            lines = [
                l for l in r.stdout.strip().split("\n")[1:]
                if l.strip() and "device" in l
            ]
            for line in lines:
                serial = line.split()[0]
                info = self._get_device_model(adb, serial)
                self.device_combo.addItem(f"{serial} ({info})", serial)
            if lines:
                self.device_status.setText(f"发现 {len(lines)} 个设备")
                self.device_status.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px;")
            else:
                self.device_status.setText("未发现设备，请连接手机并开启 USB 调试")
                self.device_status.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px;")
        except FileNotFoundError:
            self.device_status.setText(
                "未找到 adb，请运行: winget install Google.PlatformTools 然后重启程序"
            )
            self.device_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px;")
        except Exception as e:
            self.device_status.setText(f"错误: {e}")
            self.device_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px;")

    def _check_device(self):
        """测试设备连接。"""
        try:
            cmd = self._adb_cmd("shell", "wm", "size")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and "x" in r.stdout:
                self.device_status.setText(f"连接成功: {r.stdout.strip()}")
                self.device_status.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px;")
            else:
                self.device_status.setText("连接失败，请检查设备 USB 调试是否开启")
                self.device_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px;")
        except FileNotFoundError:
            self.device_status.setText(
                "未找到 adb，请运行: winget install Google.PlatformTools 然后重启程序"
            )
            self.device_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px;")
        except Exception as e:
            self.device_status.setText(f"连接失败: {e}")
            self.device_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px;")

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

    def get_device_serial(self) -> str:
        data = self.device_combo.currentData()
        if data:
            return data
        return self.device_combo.currentText().strip()

    def get_preset_name(self) -> str:
        custom = self.custom_yaml.text().strip()
        if custom:
            return custom
        text = self.preset_combo.currentText()
        return text.split(" - ")[0].strip()

    def get_agent_model_dir(self) -> str:
        return self.agent_model_path.text().strip()

    def set_agent_running_state(self, running: bool):
        self.agent_start_btn.setEnabled(not running)
        self.agent_stop_btn.setEnabled(running)
        self.device_combo.setEnabled(not running)
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
