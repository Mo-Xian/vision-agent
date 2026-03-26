"""主窗口：训练工坊 + LLM 配置。"""

import os
import json
import threading
from pathlib import Path
from PySide6.QtCore import Qt, Slot, Signal, QSettings
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFileDialog, QTextEdit, QSplitter,
    QMessageBox, QStackedWidget, QTabWidget,
)

from ..decision import PROVIDER_PRESETS, create_provider
from .workshop_panel import WorkshopPanel
from .llm_panel import LLMPanel
from .styles import MAIN_STYLESHEET, COLORS

# 模式切换按钮样式
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
    _learn_log = Signal(str)
    _learn_progress = Signal(str, float)
    _learn_done = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Agent")

        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(1200, int(avail.width() * 0.75))
            h = min(820, int(avail.height() * 0.85))
            self.setMinimumSize(min(900, int(avail.width() * 0.6)),
                                min(600, int(avail.height() * 0.6)))
            self.resize(w, h)
            self.move(avail.x() + (avail.width() - w) // 2,
                      avail.y() + (avail.height() - h) // 2)
        else:
            self.setMinimumSize(700, 500)
            self.resize(1000, 720)

        self.setStyleSheet(MAIN_STYLESHEET + _MODE_BTN_STYLE)
        self.setAcceptDrops(True)

        self._settings = QSettings("VisionAgent", "VisionAgent")
        self._current_mode = "workshop"
        self._learner = None
        self._last_model_dir = ""

        self._init_ui()
        self._connect_signals()
        self._load_settings()
        self.workshop_panel.init_scene_manager()

    # ================================================================
    #  UI 构建
    # ================================================================

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # 模式切换
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self.mode_workshop_btn = QPushButton("训练工坊")
        self.mode_llm_btn = QPushButton("LLM 设置")
        for btn, mode in [
            (self.mode_workshop_btn, "workshop"),
            (self.mode_llm_btn, "llm"),
        ]:
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, m=mode: self._switch_mode(m))
            mode_row.addWidget(btn)
        mode_row.addStretch()
        root_layout.addLayout(mode_row)

        # 模式面板栈
        self.mode_stack = QStackedWidget()
        self.workshop_panel = WorkshopPanel()
        self.llm_panel = LLMPanel()
        self.mode_stack.addWidget(self.workshop_panel)  # 0
        self.mode_stack.addWidget(self.llm_panel)       # 1
        root_layout.addWidget(self.mode_stack, 1)

        # 底部日志
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setPlaceholderText("系统日志...")
        root_layout.addWidget(self.log_text)

        self._switch_mode("workshop")

    # ================================================================
    #  信号连接
    # ================================================================

    def _connect_signals(self):
        wp = self.workshop_panel
        lp = self.llm_panel

        # 学习管线信号
        self._learn_log.connect(self._on_learn_log)
        self._learn_progress.connect(self._on_learn_progress)
        self._learn_done.connect(self._on_learn_done)

        # LLM 面板
        lp.llm_provider_combo.addItems(list(PROVIDER_PRESETS.keys()))
        lp.llm_provider_combo.currentTextChanged.connect(self._on_provider_changed)
        lp.llm_test_btn.clicked.connect(self._test_llm_connection)
        lp.llm_save_btn.clicked.connect(self._save_llm_settings)

        # 训练工坊
        wp.learn_requested.connect(self._start_learning)
        wp.stop_requested.connect(self._stop_learning)
        wp.view_models_btn.clicked.connect(self._view_models)

        self._on_provider_changed(lp.llm_provider_combo.currentText())

    # ================================================================
    #  模式切换
    # ================================================================

    def _switch_mode(self, mode: str):
        self._current_mode = mode
        index = {"workshop": 0, "llm": 1}.get(mode, 0)
        self.mode_stack.setCurrentIndex(index)

        for btn, m in [
            (self.mode_workshop_btn, "workshop"),
            (self.mode_llm_btn, "llm"),
        ]:
            btn.setObjectName("modeBtnActive" if m == mode else "modeBtnInactive")
            btn.setStyle(btn.style())

    # ================================================================
    #  训练工坊 - 视频学习
    # ================================================================

    @Slot()
    def _start_learning(self):
        wp = self.workshop_panel
        video_paths = wp.get_video_paths()
        if not video_paths:
            QMessageBox.warning(self, "提示", "请选择视频文件或输入流地址")
            return

        api_key = self._get_llm_api_key()
        provider_name = self.llm_panel.llm_provider_combo.currentText()
        if not api_key and provider_name != "ollama":
            QMessageBox.warning(self, "提示", "请先在 LLM 设置中配置 API Key")
            self._switch_mode("llm")
            return

        wp.set_learning_state(True)
        wp.log_text.clear()
        wp.insight_text.clear()
        wp.progress_bar.setValue(0)
        wp.train_chart.clear()
        wp.train_chart.setVisible(True)

        from ..workshop.learning_pipeline import LearningPipeline

        output_dir = "runs/workshop"
        if wp.current_scene:
            output_dir = str(Path(wp.current_scene.scene_dir) / "sessions")
            wp.current_scene.session_count += 1
            wp.current_scene.status = "training"
            wp.current_scene.save()

        self._learner = LearningPipeline(
            llm_provider_name=provider_name,
            llm_api_key=api_key,
            llm_model=self.llm_panel.llm_model_combo.currentText(),
            llm_base_url=self.llm_panel.llm_base_url.text().strip(),
            output_dir=output_dir,
            on_log=lambda msg: self._learn_log.emit(msg),
            on_progress=lambda phase, pct: self._learn_progress.emit(phase, pct),
        )

        kwargs = {
            "video_paths": video_paths,
            "description": wp.description_input.text().strip(),
            "sample_count": wp.sample_count_spin.value(),
            "epochs": wp.epochs_spin.value(),
            "rl_steps": wp.rl_steps_spin.value(),
            "send_image": wp.send_image_check.isChecked(),
            "batch_size": wp.batch_size_spin.value(),
            "knowledge": wp.knowledge_input.toPlainText().strip(),
        }

        def _run():
            try:
                result = self._learner.learn_from_videos(**kwargs)
                self._learn_done.emit(result.to_dict())
            except Exception as e:
                self._learn_log.emit(f"[错误] {e}")
                self._learn_done.emit({})

        threading.Thread(target=_run, daemon=True).start()
        self._log("[工坊] 学习开始...")

    @Slot()
    def _stop_learning(self):
        if self._learner:
            self._learner.stop()
        self.workshop_panel.stop_learn_btn.setEnabled(False)

    @Slot(str)
    def _on_learn_log(self, msg: str):
        self.workshop_panel.log_text.append(msg)
        sb = self.workshop_panel.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    @Slot(str, float)
    def _on_learn_progress(self, phase: str, pct: float):
        wp = self.workshop_panel
        phase_labels = {
            "analyze": "分析视频", "annotate": "LLM 标注",
            "encode": "视觉编码", "train": "训练模型",
            "rl": "强化学习", "done": "完成",
        }
        label = phase_labels.get(phase, phase)
        wp.progress_bar.setValue(int(pct * 100))
        wp.progress_bar.setFormat(f"%p% - {label}")
        wp.phase_label.setText(f"阶段: {label} ({pct*100:.0f}%)")

    @Slot(dict)
    def _on_learn_done(self, result: dict):
        wp = self.workshop_panel
        wp.set_learning_state(False)
        self._learner = None

        scene = wp.current_scene

        if result.get("model_dir"):
            wp.progress_bar.setValue(100)
            wp.progress_bar.setFormat("100% - 学习完成")
            wp.model_status.setText(f"模型已生成: {result['model_dir']}")
            self._last_model_dir = result["model_dir"]

            insight = result.get("insight")
            if insight:
                wp.update_insight(insight)

            if scene:
                val_acc = result.get("metrics", {}).get("best_val_acc", 0)
                scene.total_annotated += result.get("annotated_count", 0)
                scene.update_best_model(
                    result["model_dir"], val_acc,
                    result.get("profile_path", ""),
                )
                wp._update_scene_ui(scene)
                wp._refresh_scene_list()

            self._log(f"[工坊] 学习完成 → {result['model_dir']}")

            QMessageBox.information(
                self, "学习完成",
                f"模型: {result['model_dir']}\n"
                f"Profile: {result.get('profile_path', 'N/A')}\n\n"
                f"使用 eval_model.py 评估模型效果。"
            )
        else:
            wp.progress_bar.setFormat("失败或中止")
            if scene:
                scene.status = "idle"
                scene.save()
            self._log("[工坊] 学习失败或中止")

    @Slot()
    def _view_models(self):
        from ..workshop.model_registry import ModelRegistry
        registry = ModelRegistry()
        count = registry.scan()
        models = registry.list_models()
        if not models:
            QMessageBox.information(self, "模型列表", "未找到训练产出的模型")
            return
        lines = [f"共找到 {count} 个模型:\n"]
        for m in models[:20]:
            lines.append(
                f"  {m.name}  |  val_acc={m.val_acc:.3f}  |  "
                f"type={m.model_type}  |  samples={m.train_samples}  |  {m.trained_at}"
            )
        QMessageBox.information(self, "模型列表", "\n".join(lines))

    # ================================================================
    #  LLM 设置
    # ================================================================

    @Slot(str)
    def _on_provider_changed(self, provider_name: str):
        lp = self.llm_panel
        preset = PROVIDER_PRESETS.get(provider_name, {})
        lp.llm_model_combo.clear()
        lp.llm_model_combo.addItems(preset.get("models", []))
        lp.llm_base_url.setText(preset.get("base_url", ""))
        env_key = preset.get("api_key_env", "")
        if env_key and os.environ.get(env_key):
            lp.llm_api_key.setPlaceholderText(f"已从 {env_key} 读取")
        else:
            lp.llm_api_key.setPlaceholderText(f"API Key" + (f" 或设置 {env_key}" if env_key else ""))

    @Slot()
    def _test_llm_connection(self):
        lp = self.llm_panel
        provider_name = lp.llm_provider_combo.currentText()
        api_key = self._get_llm_api_key()
        model = lp.llm_model_combo.currentText()
        base_url = lp.llm_base_url.text().strip()
        if not api_key and provider_name != "ollama":
            QMessageBox.warning(self, "提示", "请输入 API Key")
            return
        lp.llm_test_btn.setEnabled(False)
        lp.llm_test_btn.setText("测试中...")
        try:
            provider = create_provider(provider_name, api_key or "ollama", model, base_url)
            ok = provider.test_connection()
            if ok:
                QMessageBox.information(self, "成功", f"{provider_name} 连接成功!\n模型: {model}")
            else:
                QMessageBox.warning(self, "失败", "连接测试失败")
        except ImportError as e:
            pkg = "anthropic" if provider_name == "claude" else "openai"
            QMessageBox.critical(self, "缺少依赖", f"需要安装: pip install {pkg}\n\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"连接失败:\n{e}")
        finally:
            lp.llm_test_btn.setEnabled(True)
            lp.llm_test_btn.setText("测试连接")

    @Slot()
    def _save_llm_settings(self):
        lp = self.llm_panel
        s = self._settings
        s.setValue("decision/provider", lp.llm_provider_combo.currentText())
        s.setValue("decision/model", lp.llm_model_combo.currentText())
        s.setValue("decision/base_url", lp.llm_base_url.text())
        s.sync()
        lp.save_hint.setText("已保存")
        self._log("[LLM] 配置已保存")
        from PySide6.QtCore import QTimer
        QTimer.singleShot(3000, lambda: lp.save_hint.setText(""))

    def _get_llm_api_key(self) -> str:
        lp = self.llm_panel
        key = lp.llm_api_key.text().strip()
        if key:
            return key
        provider_name = lp.llm_provider_combo.currentText()
        env_key = PROVIDER_PRESETS.get(provider_name, {}).get("api_key_env", "")
        return os.environ.get(env_key, "") if env_key else ""

    # ================================================================
    #  日志
    # ================================================================

    def _log(self, text):
        self.log_text.append(text)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ================================================================
    #  设置保存 / 加载
    # ================================================================

    def _save_settings(self):
        s = self._settings
        lp = self.llm_panel
        wp = self.workshop_panel
        s.setValue("app/mode", self._current_mode)
        s.setValue("decision/provider", lp.llm_provider_combo.currentText())
        s.setValue("decision/model", lp.llm_model_combo.currentText())
        s.setValue("decision/base_url", lp.llm_base_url.text())
        s.setValue("train/epochs", wp.epochs_spin.value())
        s.setValue("train/lr", wp.lr_spin.value())
        s.setValue("train/sample_count", wp.sample_count_spin.value())
        s.setValue("train/rl_steps", wp.rl_steps_spin.value())
        s.setValue("train/batch_size", wp.batch_size_spin.value())

    def _load_settings(self):
        s = self._settings
        lp = self.llm_panel
        wp = self.workshop_panel
        if s.value("app/mode"):
            self._switch_mode(s.value("app/mode"))
        if s.value("decision/provider"):
            lp.llm_provider_combo.setCurrentText(s.value("decision/provider"))
        if s.value("decision/model"):
            lp.llm_model_combo.setCurrentText(s.value("decision/model"))
        if s.value("decision/base_url"):
            lp.llm_base_url.setText(s.value("decision/base_url"))
        if s.value("train/epochs") is not None:
            try: wp.epochs_spin.setValue(int(s.value("train/epochs", 100)))
            except (TypeError, ValueError): pass
        if s.value("train/lr") is not None:
            try: wp.lr_spin.setValue(float(s.value("train/lr", 0.001)))
            except (TypeError, ValueError): pass
        if s.value("train/sample_count") is not None:
            try: wp.sample_count_spin.setValue(int(s.value("train/sample_count", 300)))
            except (TypeError, ValueError): pass
        if s.value("train/rl_steps") is not None:
            try: wp.rl_steps_spin.setValue(int(s.value("train/rl_steps", 2000)))
            except (TypeError, ValueError): pass
        if s.value("train/batch_size") is not None:
            try: wp.batch_size_spin.setValue(int(s.value("train/batch_size", 5)))
            except (TypeError, ValueError): pass

    def closeEvent(self, event):
        self._save_settings()
        if self._learner:
            self._learner.stop()
        event.accept()

    # ================================================================
    #  拖放
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

        if ext in video_exts:
            wp = self.workshop_panel
            wp.video_path_input.setText(path)
            if wp.current_scene:
                wp.current_scene.add_videos([path])
                wp._update_scene_ui(wp.current_scene)
            self._log(f"已加载视频: {Path(path).name}")
        elif Path(path).is_dir():
            wp = self.workshop_panel
            wp.input_type.setCurrentText("视频文件夹")
            wp.video_path_input.setText(path)
            files = wp._scan_folder(path)
            if wp.current_scene and files:
                wp.current_scene.add_videos(files)
                wp._update_scene_ui(wp.current_scene)
            self._log(f"已加载文件夹: {Path(path).name}")
