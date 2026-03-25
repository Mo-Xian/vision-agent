"""自动学习对话框：输入兴趣描述 → 全自动搜索/标注/训练/强化。"""

import threading
from pathlib import Path
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QComboBox, QSpinBox, QProgressBar, QGroupBox, QFormLayout,
    QCheckBox, QFileDialog, QMessageBox,
)
from PySide6.QtGui import QFont

from .styles import COLORS


class AutoLearnDialog(QDialog):
    """全自动学习对话框。"""

    log_signal = Signal(str)
    progress_signal = Signal(str, float)
    done_signal = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("自动学习 - 从兴趣到模型")
        self.setMinimumSize(700, 600)
        self.resize(750, 650)
        self._worker_thread = None
        self._learner = None
        self._init_ui()
        self.log_signal.connect(self._append_log)
        self.progress_signal.connect(self._update_progress)
        self.done_signal.connect(self._on_done)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── 兴趣描述 ──
        desc_group = QGroupBox("场景描述")
        desc_layout = QVBoxLayout(desc_group)

        self.interest_edit = QLineEdit()
        self.interest_edit.setPlaceholderText("描述你感兴趣的场景，如：王者荣耀5v5团战、FPS射击游戏、交通路口监控...")
        self.interest_edit.setFont(QFont("Microsoft YaHei", 11))
        desc_layout.addWidget(self.interest_edit)

        layout.addWidget(desc_group)

        # ── 配置 ──
        config_group = QGroupBox("配置")
        config_layout = QFormLayout(config_group)

        # LLM
        llm_row = QHBoxLayout()
        self.llm_provider = QComboBox()
        self.llm_provider.addItems(["claude", "openai", "deepseek", "ollama"])
        llm_row.addWidget(QLabel("LLM:"))
        llm_row.addWidget(self.llm_provider)
        self.llm_model = QLineEdit("claude-sonnet-4-20250514")
        self.llm_model.setPlaceholderText("模型名")
        llm_row.addWidget(self.llm_model)
        config_layout.addRow("LLM 模型", llm_row)

        self.llm_api_key = QLineEdit()
        self.llm_api_key.setPlaceholderText("API Key（环境变量已设则可留空）")
        self.llm_api_key.setEchoMode(QLineEdit.Password)
        config_layout.addRow("API Key", self.llm_api_key)

        self.llm_base_url = QLineEdit()
        self.llm_base_url.setPlaceholderText("自定义 Base URL（可选）")
        config_layout.addRow("Base URL", self.llm_base_url)

        # B站Cookie
        self.bilibili_cookie = QLineEdit()
        self.bilibili_cookie.setPlaceholderText("B站Cookie（可选，提高下载成功率；或设置环境变量 BILIBILI_COOKIE）")
        self.bilibili_cookie.setEchoMode(QLineEdit.Password)
        config_layout.addRow("B站Cookie", self.bilibili_cookie)

        # 资源
        res_row = QHBoxLayout()
        self.resource_type = QComboBox()
        self.resource_type.addItems(["video", "image"])
        res_row.addWidget(self.resource_type)
        res_row.addWidget(QLabel("来源:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["bilibili", "url", "ytdlp"])
        self.source_combo.setToolTip("bilibili: B站搜索下载\nurl: 直接URL下载\nytdlp: yt-dlp多平台下载")
        res_row.addWidget(self.source_combo)
        res_row.addWidget(QLabel("数量:"))
        self.max_resources = QSpinBox()
        self.max_resources.setRange(1, 20)
        self.max_resources.setValue(3)
        res_row.addWidget(self.max_resources)
        config_layout.addRow("资源类型", res_row)

        # 本地视频
        local_row = QHBoxLayout()
        self.skip_fetch = QCheckBox("使用本地视频（跳过下载）")
        local_row.addWidget(self.skip_fetch)
        self.local_videos = QLineEdit()
        self.local_videos.setPlaceholderText("本地视频路径或文件夹，多个用分号分隔")
        local_row.addWidget(self.local_videos)
        browse_btn = QPushButton("文件")
        browse_btn.setMaximumWidth(40)
        browse_btn.setToolTip("选择视频文件")
        browse_btn.clicked.connect(self._browse_videos)
        local_row.addWidget(browse_btn)
        browse_dir_btn = QPushButton("文件夹")
        browse_dir_btn.setMaximumWidth(50)
        browse_dir_btn.setToolTip("选择文件夹（自动扫描其中的视频文件）")
        browse_dir_btn.clicked.connect(self._browse_video_dir)
        local_row.addWidget(browse_dir_btn)
        config_layout.addRow("", local_row)

        # 训练参数
        train_row = QHBoxLayout()
        train_row.addWidget(QLabel("标注帧数:"))
        self.sample_count = QSpinBox()
        self.sample_count.setRange(20, 5000)
        self.sample_count.setValue(300)
        train_row.addWidget(self.sample_count)
        train_row.addWidget(QLabel("RL 步数:"))
        self.rl_steps = QSpinBox()
        self.rl_steps.setRange(0, 50000)
        self.rl_steps.setValue(2000)
        self.rl_steps.setSingleStep(500)
        train_row.addWidget(self.rl_steps)
        config_layout.addRow("训练参数", train_row)

        # YOLO 模型
        self.yolo_model = QComboBox()
        self.yolo_model.setEditable(True)
        self.yolo_model.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
        config_layout.addRow("YOLO 模型", self.yolo_model)

        layout.addWidget(config_group)

        # ── 进度 ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - 准备中")
        layout.addWidget(self.progress_bar)

        # ── 日志 ──
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)

        # ── 按钮 ──
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始自动学习")
        self.start_btn.setStyleSheet(
            f"background-color: {COLORS['accent']}; color: white; "
            "font-size: 13px; font-weight: bold; padding: 8px 20px; border-radius: 6px;"
        )
        self.start_btn.clicked.connect(self._start)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)

    _VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".ts", ".m4v"}

    def _browse_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.webm *.flv *.ts *.m4v);;所有文件 (*)"
        )
        if files:
            self.local_videos.setText(";".join(files))
            self.skip_fetch.setChecked(True)

    def _browse_video_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择视频文件夹")
        if folder:
            # 直接填入文件夹路径，启动时再扫描
            existing = self.local_videos.text().strip()
            if existing:
                self.local_videos.setText(existing + ";" + folder)
            else:
                self.local_videos.setText(folder)
            self.skip_fetch.setChecked(True)

    @staticmethod
    def _expand_paths(paths: list[str]) -> list[str]:
        """展开路径列表：文件夹自动扫描其中的视频文件，普通文件直接保留。"""
        from pathlib import Path
        result = []
        for p in paths:
            pp = Path(p)
            if pp.is_dir():
                for f in sorted(pp.iterdir()):
                    if f.is_file() and f.suffix.lower() in AutoLearnDialog._VIDEO_EXTS:
                        result.append(str(f))
            elif pp.is_file():
                result.append(str(pp))
        return result

    def _start(self):
        interest = self.interest_edit.text().strip()
        if not interest:
            QMessageBox.warning(self, "提示", "请输入场景描述")
            return

        api_key = self.llm_api_key.text().strip()
        provider = self.llm_provider.currentText()
        if not api_key and provider not in ("ollama",):
            # 尝试从环境变量获取
            import os
            env_keys = {
                "claude": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
            }
            env_key = env_keys.get(provider, "")
            api_key = os.environ.get(env_key, "")
            if not api_key:
                QMessageBox.warning(self, "提示", f"请输入 API Key 或设置环境变量 {env_key}")
                return

        self.log_text.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% - 启动中...")

        # 获取本地视频列表（支持文件和文件夹混合输入）
        video_paths = None
        if self.skip_fetch.isChecked():
            paths_text = self.local_videos.text().strip()
            if paths_text:
                raw_paths = [p.strip() for p in paths_text.split(";") if p.strip()]
                video_paths = self._expand_paths(raw_paths)
                if not video_paths:
                    QMessageBox.warning(self, "提示", "指定的路径中未找到视频文件")
                    self.start_btn.setEnabled(True)
                    self.stop_btn.setEnabled(False)
                    return

        from ..auto.auto_learn import AutoLearn

        self._learner = AutoLearn(
            llm_provider_name=provider,
            llm_api_key=api_key,
            llm_model=self.llm_model.text().strip(),
            llm_base_url=self.llm_base_url.text().strip(),
            yolo_model=self.yolo_model.currentText(),
            on_log=lambda msg: self.log_signal.emit(msg),
            on_progress=lambda phase, pct: self.progress_signal.emit(phase, pct),
            bilibili_cookie=self.bilibili_cookie.text().strip(),
        )

        kwargs = {
            "interest": interest,
            "resource_type": self.resource_type.currentText(),
            "max_resources": self.max_resources.value(),
            "sample_count": self.sample_count.value(),
            "rl_steps": self.rl_steps.value(),
            "skip_fetch": self.skip_fetch.isChecked(),
            "video_paths": video_paths,
            "source": self.source_combo.currentText(),
        }

        self._worker_thread = threading.Thread(
            target=self._run_worker, args=(kwargs,), daemon=True
        )
        self._worker_thread.start()

    def _run_worker(self, kwargs):
        try:
            result = self._learner.run(**kwargs)
            self.done_signal.emit(result)
        except Exception as e:
            self.log_signal.emit(f"[错误] {e}")
            self.done_signal.emit({})

    def _stop(self):
        if self._learner:
            self._learner.stop()
        self.stop_btn.setEnabled(False)

    @Slot(str)
    def _append_log(self, msg: str):
        self.log_text.append(msg)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot(str, float)
    def _update_progress(self, phase: str, pct: float):
        phase_labels = {
            "plan": "规划搜索策略",
            "fetch": "搜索下载资源",
            "searching": "搜索中",
            "downloading": "下载中",
            "annotate": "LLM 标注",
            "train": "监督学习训练",
            "rl": "RL 强化学习",
            "export": "导出 Profile",
            "done": "完成",
        }
        label = phase_labels.get(phase, phase)
        self.progress_bar.setValue(int(pct * 100))
        self.progress_bar.setFormat(f"%p% - {label}")

    @Slot(dict)
    def _on_done(self, result: dict):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if result.get("model_dir"):
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("100% - 完成")
        else:
            self.progress_bar.setFormat("%p% - 失败或中止")

        if result.get("model_dir"):
            QMessageBox.information(
                self, "自动学习完成",
                f"决策模型: {result['model_dir']}\n"
                f"RL 模型: {result.get('rl_dir', 'N/A')}\n"
                f"Profile: {result.get('profile_path', 'N/A')}\n\n"
                "模型已自动添加到 profiles 目录，重启后可选择使用。"
            )
