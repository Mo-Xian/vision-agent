"""LLM 自动标注对话框。

选择视频 → 配置动作空间和 LLM → 自动标注 → 生成训练数据。
"""

import os
from pathlib import Path
from PySide6.QtCore import Qt, Slot, QSettings
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QFileDialog, QTextEdit, QProgressBar, QMessageBox, QFormLayout,
    QCheckBox,
)

from ..decision.llm_provider import PROVIDER_PRESETS, create_provider
from .annotate_worker import AnnotateWorker

DIALOG_STYLE = """
QDialog { background-color: #0f0f23; }
QGroupBox {
    background-color: #16213e; border: 1px solid #0f3460; border-radius: 8px;
    margin-top: 14px; padding: 12px; padding-top: 24px;
    color: #e0e0e0; font-weight: bold; font-size: 13px;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QLabel { color: #c0c0c0; font-size: 13px; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #1a1a2e; color: #e0e0e0;
    border: 1px solid #0f3460; border-radius: 4px;
    padding: 4px 8px; min-height: 24px; font-size: 13px;
}
QPushButton {
    border-radius: 6px; padding: 6px 12px;
    font-size: 13px; font-weight: bold; min-height: 28px;
}
QPushButton#startBtn { background-color: #00b894; color: white; }
QPushButton#startBtn:hover { background-color: #00a381; }
QPushButton#startBtn:disabled { background-color: #555; }
QPushButton#stopBtn { background-color: #e94560; color: white; }
QPushButton#stopBtn:disabled { background-color: #555; }
QPushButton#browseBtn {
    background-color: #0f3460; color: #e0e0e0;
    padding: 4px 10px; min-height: 22px; font-size: 12px;
}
QTextEdit {
    background-color: #1a1a2e; color: #a0e0a0;
    border: 1px solid #0f3460; border-radius: 6px;
    font-family: Consolas, monospace; font-size: 12px; padding: 4px;
}
QProgressBar {
    background-color: #1a1a2e; border: 1px solid #0f3460; border-radius: 4px;
    text-align: center; color: #e0e0e0; font-size: 13px; min-height: 22px;
}
QProgressBar::chunk { background-color: #00b894; border-radius: 3px; }
"""


class AnnotateDialog(QDialog):
    """LLM 自动标注对话框。"""

    def __init__(self, parent=None, default_video: str = "", default_model: str = ""):
        super().__init__(parent)
        self.setWindowTitle("LLM 自动标注")
        self.setMinimumSize(680, 720)
        self.resize(720, 760)
        self.setStyleSheet(DIALOG_STYLE)

        self._worker: AnnotateWorker | None = None
        self._save_path: str | None = None
        self._default_video = default_video
        self._default_model = default_model
        self._settings = QSettings("VisionAgent", "VisionAgent")
        self._init_ui()
        self._load_settings()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── 视频 & 检测 ──
        video_group = QGroupBox("视频与检测")
        vg = QVBoxLayout(video_group)
        vg.setSpacing(6)

        video_row = QHBoxLayout()
        video_row.addWidget(QLabel("视频文件"))
        self.video_input = QLineEdit(self._default_video)
        self.video_input.setPlaceholderText("选择要标注的视频文件")
        video_row.addWidget(self.video_input)
        video_browse = QPushButton("浏览")
        video_browse.setObjectName("browseBtn")
        video_browse.clicked.connect(self._browse_video)
        video_row.addWidget(video_browse)
        vg.addLayout(video_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("YOLO 模型"))
        self.yolo_model_input = QLineEdit(self._default_model)
        self.yolo_model_input.setPlaceholderText("YOLO 检测模型路径")
        model_row.addWidget(self.yolo_model_input)
        model_browse = QPushButton("浏览")
        model_browse.setObjectName("browseBtn")
        model_browse.clicked.connect(self._browse_model)
        model_row.addWidget(model_browse)
        vg.addLayout(model_row)

        sample_row = QHBoxLayout()
        form = QFormLayout()
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 300)
        self.sample_spin.setValue(30)
        form.addRow("采样间隔(帧)", self.sample_spin)

        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 100000)
        self.max_frames_spin.setValue(500)
        self.max_frames_spin.setSpecialValueText("不限制")
        form.addRow("最大帧数", self.max_frames_spin)

        self.det_conf_spin = QDoubleSpinBox()
        self.det_conf_spin.setRange(0.1, 1.0)
        self.det_conf_spin.setSingleStep(0.05)
        self.det_conf_spin.setValue(0.5)
        form.addRow("检测置信度", self.det_conf_spin)
        vg.addLayout(form)

        layout.addWidget(video_group)

        # ── 动作空间 ──
        action_group = QGroupBox("动作空间")
        ag = QVBoxLayout(action_group)
        ag.setSpacing(6)

        ag.addWidget(QLabel("可用动作（逗号分隔）"))
        self.actions_input = QLineEdit("attack, retreat, skill_1, skill_2, idle")
        self.actions_input.setPlaceholderText("attack, retreat, skill_1, idle")
        ag.addWidget(self.actions_input)

        self.send_image_check = QCheckBox("发送帧图像给 LLM（多模态，需模型支持 vision）")
        self.send_image_check.setStyleSheet("color: #c0c0c0; font-size: 13px; padding-left: 4px;")
        ag.addWidget(self.send_image_check)

        self.tool_calling_check = QCheckBox("使用 Tool Calling 规范化输出（推荐）")
        self.tool_calling_check.setStyleSheet("color: #c0c0c0; font-size: 13px; padding-left: 4px;")
        self.tool_calling_check.setChecked(True)
        self.tool_calling_check.setToolTip("通过函数调用强制 LLM 返回结构化结果，避免格式解析问题")
        ag.addWidget(self.tool_calling_check)

        layout.addWidget(action_group)

        # ── LLM 配置 ──
        llm_group = QGroupBox("LLM 配置")
        lg = QVBoxLayout(llm_group)
        lg.setSpacing(6)

        llm_form = QFormLayout()
        self.llm_provider = QComboBox()
        self.llm_provider.addItems(list(PROVIDER_PRESETS.keys()))
        self.llm_provider.currentTextChanged.connect(self._on_provider_changed)
        llm_form.addRow("供应商", self.llm_provider)

        self.llm_model = QComboBox()
        self.llm_model.setEditable(True)
        llm_form.addRow("模型", self.llm_model)

        self.llm_api_key = QLineEdit()
        self.llm_api_key.setEchoMode(QLineEdit.Password)
        self.llm_api_key.setPlaceholderText("API Key 或留空用环境变量")
        llm_form.addRow("API Key", self.llm_api_key)

        self.llm_base_url = QLineEdit()
        self.llm_base_url.setPlaceholderText("留空用默认地址")
        llm_form.addRow("Base URL", self.llm_base_url)
        lg.addLayout(llm_form)

        layout.addWidget(llm_group)

        # ── 输出 ──
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("输出文件"))
        self.output_input = QLineEdit("data/recordings/auto_annotated.jsonl")
        out_row.addWidget(self.output_input)
        layout.addLayout(out_row)

        # ── 按钮 ──
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始标注")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._start_annotate)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_annotate)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        # ── 进度 ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #74b9ff; font-size: 12px;")
        layout.addWidget(self.status_label)

        # ── 日志 ──
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        layout.addWidget(self.log_text)

    # ── 事件 ──

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv);;所有文件 (*)"
        )
        if path:
            self.video_input.setText(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", "",
            "PyTorch 模型 (*.pt);;所有文件 (*)"
        )
        if path:
            self.yolo_model_input.setText(path)

    @Slot(str)
    def _on_provider_changed(self, name: str):
        preset = PROVIDER_PRESETS.get(name, {})
        self.llm_model.clear()
        self.llm_model.addItems(preset.get("models", []))
        self.llm_base_url.setText(preset.get("base_url", ""))
        env_key = preset.get("api_key_env", "")
        if env_key and os.environ.get(env_key):
            self.llm_api_key.setPlaceholderText(f"已从 {env_key} 读取")
        else:
            self.llm_api_key.setPlaceholderText(f"API Key" + (f" 或设置 {env_key}" if env_key else ""))

    def _get_api_key(self) -> str:
        key = self.llm_api_key.text().strip()
        if key:
            return key
        name = self.llm_provider.currentText()
        env_key = PROVIDER_PRESETS.get(name, {}).get("api_key_env", "")
        return os.environ.get(env_key, "") if env_key else ""

    @Slot()
    def _start_annotate(self):
        video = self.video_input.text().strip()
        if not video:
            QMessageBox.warning(self, "提示", "请选择视频文件")
            return
        if not Path(video).exists():
            QMessageBox.warning(self, "提示", f"视频文件不存在: {video}")
            return

        yolo_model = self.yolo_model_input.text().strip() or "yolov8n.pt"
        actions_text = self.actions_input.text().strip()
        if not actions_text:
            QMessageBox.warning(self, "提示", "请输入至少一个动作")
            return
        actions = [a.strip() for a in actions_text.split(",") if a.strip()]

        api_key = self._get_api_key()
        provider_name = self.llm_provider.currentText()
        if not api_key and provider_name != "ollama":
            QMessageBox.warning(self, "提示", "请输入 API Key")
            return

        save_path = self.output_input.text().strip()
        if not save_path:
            QMessageBox.warning(self, "提示", "请指定输出文件路径")
            return

        self._save_settings()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        self._save_path = save_path

        self._worker = AnnotateWorker(
            video_path=video,
            model_path=yolo_model,
            provider_name=provider_name,
            api_key=api_key or "ollama",
            llm_model=self.llm_model.currentText(),
            base_url=self.llm_base_url.text().strip(),
            actions=actions,
            save_path=save_path,
            sample_interval=self.sample_spin.value(),
            max_frames=self.max_frames_spin.value(),
            confidence=self.det_conf_spin.value(),
            send_image=self.send_image_check.isChecked(),
            use_tool_calling=self.tool_calling_check.isChecked(),
        )
        self._worker.log_message.connect(self._on_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.finished_err.connect(self._on_error)
        self._worker.start()

    @Slot()
    def _stop_annotate(self):
        if self._worker:
            self._on_log("正在停止...")
            self._worker.stop()

    @Slot(str)
    def _on_log(self, msg: str):
        self.log_text.append(msg)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot(int, int, int)
    def _on_progress(self, current: int, total: int, annotated: int):
        if total > 0:
            pct = int(current / total * 100)
            self.progress_bar.setValue(pct)
        self.status_label.setText(
            f"帧 {current}/{total}  |  已标注: {annotated} 条"
        )

    @Slot(dict)
    def _on_finished(self, stats: dict):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText(
            f"完成! 标注 {stats['annotated']} 条 → {self._save_path}"
        )
        self.status_label.setStyleSheet("color: #00b894; font-size: 12px; font-weight: bold;")
        self._worker = None

    @Slot(str)
    def _on_error(self, error: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"失败: {error}")
        self.status_label.setStyleSheet("color: #d63031; font-size: 12px;")
        QMessageBox.critical(self, "标注失败", error)
        self._worker = None

    def get_save_path(self) -> str | None:
        return self._save_path

    # ── 设置持久化 ──

    def _save_settings(self):
        s = self._settings
        s.setValue("annotate/video", self.video_input.text())
        s.setValue("annotate/yolo_model", self.yolo_model_input.text())
        s.setValue("annotate/sample_interval", self.sample_spin.value())
        s.setValue("annotate/max_frames", self.max_frames_spin.value())
        s.setValue("annotate/det_conf", self.det_conf_spin.value())
        s.setValue("annotate/actions", self.actions_input.text())
        s.setValue("annotate/provider", self.llm_provider.currentText())
        s.setValue("annotate/model", self.llm_model.currentText())
        s.setValue("annotate/base_url", self.llm_base_url.text())
        s.setValue("annotate/output", self.output_input.text())
        s.setValue("annotate/send_image", self.send_image_check.isChecked())
        s.setValue("annotate/use_tool_calling", self.tool_calling_check.isChecked())

    def _load_settings(self):
        s = self._settings

        # 视频和模型：优先用传入的默认值，否则用上次保存的
        if self._default_video:
            self.video_input.setText(self._default_video)
        elif s.value("annotate/video"):
            self.video_input.setText(s.value("annotate/video"))

        if self._default_model:
            self.yolo_model_input.setText(self._default_model)
        elif s.value("annotate/yolo_model"):
            self.yolo_model_input.setText(s.value("annotate/yolo_model"))

        if s.value("annotate/sample_interval") is not None:
            self.sample_spin.setValue(int(s.value("annotate/sample_interval", 30)))
        if s.value("annotate/max_frames") is not None:
            self.max_frames_spin.setValue(int(s.value("annotate/max_frames", 500)))
        if s.value("annotate/det_conf") is not None:
            self.det_conf_spin.setValue(float(s.value("annotate/det_conf", 0.5)))
        if s.value("annotate/actions"):
            self.actions_input.setText(s.value("annotate/actions"))
        if s.value("annotate/output"):
            self.output_input.setText(s.value("annotate/output"))

        # LLM 配置：优先从标注自己的设置读，否则复用主窗口决策引擎的 LLM 配置
        provider = s.value("annotate/provider") or s.value("decision/provider")
        if provider:
            self.llm_provider.setCurrentText(provider)
        # 手动触发一次预设加载
        self._on_provider_changed(self.llm_provider.currentText())

        model = s.value("annotate/model") or s.value("decision/model")
        if model:
            self.llm_model.setCurrentText(model)

        base_url = s.value("annotate/base_url") or s.value("decision/base_url")
        if base_url:
            self.llm_base_url.setText(base_url)

        if s.value("annotate/send_image") is not None:
            self.send_image_check.setChecked(s.value("annotate/send_image") == "true"
                                             or s.value("annotate/send_image") is True)
        if s.value("annotate/use_tool_calling") is not None:
            val = s.value("annotate/use_tool_calling")
            self.tool_calling_check.setChecked(val == "true" or val is True)

    def closeEvent(self, event):
        self._save_settings()
        event.accept()
