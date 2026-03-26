"""主窗口：训练工坊 + Agent 执行 双模式界面。"""

import os
import json
import threading
from pathlib import Path
from PySide6.QtCore import Qt, Slot, Signal, QSettings
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFileDialog, QTextEdit, QSplitter,
    QMessageBox, QDialog, QStackedWidget, QCheckBox, QTabWidget,
)

from ..sources import create_source
from ..core.detector import Detector, DetectionResult
from ..core.model_manager import ModelManager
from ..core.state import StateManager
from ..decision import (
    RuleEngine, LLMEngine, TrainedEngine, HierarchicalEngine, RLEngine,
    PROVIDER_PRESETS, create_provider, Action,
)
from ..profiles import ProfileManager
from ..core.scene_classifier import SceneClassifier
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
from .workshop_panel import WorkshopPanel
from .agent_panel import AgentPanel
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
    # 学习管线信号（线程安全）
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
            w = min(1400, int(avail.width() * 0.85))
            h = min(820, int(avail.height() * 0.85))
            self.setMinimumSize(min(1100, int(avail.width() * 0.7)),
                                min(700, int(avail.height() * 0.7)))
            self.resize(w, h)
            self.move(avail.x() + (avail.width() - w) // 2,
                      avail.y() + (avail.height() - h) // 2)
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
        self._current_mode = "workshop"
        self._learner = None
        self._last_model_dir = ""

        self._init_ui()
        self._connect_signals()
        self._scan_models()
        self._load_settings()
        self.workshop_panel.init_scene_manager()

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

        # 模式切换（两个核心模式 + LLM 设置）
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self.mode_workshop_btn = QPushButton("训练工坊")
        self.mode_agent_btn = QPushButton("Agent")
        self.mode_llm_btn = QPushButton("LLM")
        for btn, mode in [
            (self.mode_workshop_btn, "workshop"),
            (self.mode_agent_btn, "agent"),
            (self.mode_llm_btn, "llm"),
        ]:
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, m=mode: self._switch_mode(m))
            mode_row.addWidget(btn)
        left_layout.addLayout(mode_row)

        # 模式面板栈
        self.mode_stack = QStackedWidget()
        self.workshop_panel = WorkshopPanel()
        self.agent_panel = AgentPanel()
        self.llm_panel = LLMPanel()
        self.mode_stack.addWidget(self.workshop_panel)  # 0
        self.mode_stack.addWidget(self.agent_panel)      # 1
        self.mode_stack.addWidget(self.llm_panel)        # 2
        left_layout.addWidget(self.mode_stack, 1)

        # Agent 控制区（嵌入 Agent 配置 Tab）
        ap = self.agent_panel
        config_scroll = ap.widget(0)
        config_widget = config_scroll.widget()
        config_layout = config_widget.layout()

        self.dryrun_check = QCheckBox("仅观察（不控制键盘鼠标）")
        self.dryrun_check.setChecked(True)
        config_layout.insertWidget(0, self.dryrun_check)

        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)
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
        config_layout.insertWidget(1, btn_widget)

        # 状态栏 + 日志
        config_stretch = None
        for i in range(config_layout.count()):
            item = config_layout.itemAt(i)
            if item and item.spacerItem():
                config_stretch = i
                break
        if config_stretch is not None:
            config_layout.takeAt(config_stretch)

        self._build_status_bar(config_layout)

        self.log_tabs = QTabWidget()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_tabs.addTab(self.log_text, "日志")
        self.decision_log_text = QTextEdit()
        self.decision_log_text.setReadOnly(True)
        self.log_tabs.addTab(self.decision_log_text, "决策")
        config_layout.addWidget(self.log_tabs)

        splitter.addWidget(left_panel)

        # ===== 右侧视频预览 =====
        self.video_widget = VideoWidget()
        splitter.addWidget(self.video_widget)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 1000])

        self._switch_mode("workshop")

    def _build_status_bar(self, parent_layout):
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
        parent_layout.addWidget(status_w)

    # ================================================================
    #  信号连接
    # ================================================================

    def _connect_signals(self):
        wp = self.workshop_panel
        ap = self.agent_panel
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
        wp.apply_model_btn.clicked.connect(self._apply_model_to_agent)
        wp.view_models_btn.clicked.connect(self._view_models)
        wp.auto_annotate_btn.clicked.connect(self._open_annotate_dialog)
        wp.view_annotation_btn.clicked.connect(self._open_annotation_viewer)
        wp.load_yolo_btn.clicked.connect(self._load_custom_model)
        wp.train_yolo_btn.clicked.connect(self._open_train_dialog)
        wp.dt_data_browse.clicked.connect(lambda: self._browse_dir_to(wp.dt_data_dir))
        wp.dt_preview_btn.clicked.connect(self._preview_data)
        wp.dt_train_btn.clicked.connect(self._start_decision_train)
        wp.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        wp.refresh_profiles_btn.clicked.connect(self._refresh_profiles)
        wp.export_btn.clicked.connect(self._export_config)
        wp.import_btn.clicked.connect(self._import_config)
        wp.rec_browse_btn.clicked.connect(lambda: self._browse_dir_to(wp.rec_dir_input))
        wp.rec_start_btn.clicked.connect(self._toggle_recording)

        # Agent 面板
        ap.source_type.currentTextChanged.connect(self._on_source_type_changed)
        ap.browse_btn.clicked.connect(self._browse_file)
        ap.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        ap.trained_browse_btn.clicked.connect(self._browse_trained_dir)
        ap.agent_profile_combo.currentTextChanged.connect(self._on_agent_profile_changed)
        ap.profile_refresh_btn.clicked.connect(self._refresh_agent_profiles)
        ap.currentChanged.connect(self._on_agent_tab_changed)
        ap.chat_panel._try_auto_init = self._init_chat_provider
        ap.chat_panel.set_log_callback(self._log)

        self._on_provider_changed(lp.llm_provider_combo.currentText())
        self._on_source_type_changed("video")
        self._refresh_profiles()

    # ================================================================
    #  模式切换
    # ================================================================

    def _switch_mode(self, mode: str):
        self._current_mode = mode
        index = {"workshop": 0, "agent": 1, "llm": 2}.get(mode, 0)
        self.mode_stack.setCurrentIndex(index)
        self.video_widget.setVisible(mode == "agent")

        for btn, m in [
            (self.mode_workshop_btn, "workshop"),
            (self.mode_agent_btn, "agent"),
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
            QMessageBox.warning(self, "提示", "请先在 LLM 面板配置 API Key")
            self._switch_mode("llm")
            return

        wp.set_learning_state(True)
        wp.log_text.clear()
        wp.insight_text.clear()
        wp.progress_bar.setValue(0)
        wp.train_chart.clear()
        wp.train_chart.setVisible(True)

        from ..workshop.learning_pipeline import LearningPipeline

        # 如果有场景，使用场景的 sessions 目录作为输出根
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
            yolo_model="" if "不使用" in wp.yolo_model_combo.currentText() else wp.yolo_model_combo.currentText(),
            output_dir=output_dir,
            on_log=lambda msg: self._learn_log.emit(msg),
            on_progress=lambda phase, pct: self._learn_progress.emit(phase, pct),
        )

        model_type_text = wp.model_type_combo.currentText()
        is_e2e = "e2e" in model_type_text or "端到端" in model_type_text

        kwargs = {
            "video_paths": video_paths,
            "description": wp.description_input.text().strip(),
            "sample_count": wp.sample_count_spin.value(),
            "model_type": "mlp" if is_e2e else model_type_text,
            "epochs": wp.epochs_spin.value(),
            "rl_steps": wp.rl_steps_spin.value(),
            "send_image": wp.send_image_check.isChecked(),
            "batch_size": wp.batch_size_spin.value(),
            "knowledge": wp.knowledge_input.toPlainText().strip(),
            "e2e": is_e2e,
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
            "train": "训练模型", "rl": "强化学习",
            "done": "完成",
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

        # 更新场景状态
        scene = wp.current_scene

        if result.get("model_dir"):
            wp.progress_bar.setValue(100)
            wp.progress_bar.setFormat("100% - 学习完成")
            wp.model_status.setText(f"模型已生成: {result['model_dir']}")
            wp.apply_model_btn.setEnabled(True)
            self._last_model_dir = result["model_dir"]

            # 更新见解
            insight = result.get("insight")
            if insight:
                wp.update_insight(insight)

            # 同步到场景
            if scene:
                val_acc = result.get("metrics", {}).get("best_val_acc", 0)
                scene.total_annotated += result.get("annotated_count", 0)
                scene.update_best_model(
                    result["model_dir"], val_acc,
                    result.get("profile_path", ""),
                )
                wp._update_scene_ui(scene)
                wp._refresh_scene_list()

            self._refresh_profiles()
            self._log(f"[工坊] 学习完成 → {result['model_dir']}")

            QMessageBox.information(
                self, "学习完成",
                f"模型: {result['model_dir']}\n"
                f"Profile: {result.get('profile_path', 'N/A')}\n\n"
                "点击「应用到 Agent」即可使用。"
            )
        else:
            wp.progress_bar.setFormat("失败或中止")
            if scene:
                scene.status = "idle"
                scene.save()
            self._log("[工坊] 学习失败或中止")

    @Slot()
    def _apply_model_to_agent(self):
        model_dir = self._last_model_dir
        if not model_dir:
            QMessageBox.warning(self, "提示", "没有可用的训练模型")
            return
        self._switch_mode("agent")
        self.agent_panel.engine_combo.setCurrentText("trained")
        self.agent_panel.trained_model_dir.setText(model_dir)
        self._log(f"[Agent] 已应用模型: {model_dir}")

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
    #  训练工坊 - 手动工具
    # ================================================================

    @Slot()
    def _open_annotate_dialog(self):
        from .annotate_dialog import AnnotateDialog
        ap = self.agent_panel
        default_video = ap.path_input.text().strip() if ap.source_type.currentText() == "video" else ""
        default_model = self.workshop_panel.yolo_model_combo.currentText()
        dialog = AnnotateDialog(self, default_video=default_video, default_model=default_model)
        dialog.exec()
        save_path = dialog.get_save_path()
        if save_path:
            self.workshop_panel.dt_data_dir.setText(str(Path(save_path).parent))
            self._log(f"[标注] 数据已保存到 {save_path}")

    @Slot()
    def _open_annotation_viewer(self):
        from .annotation_viewer import AnnotationViewer
        jsonl_path = ""
        video_path = ""
        data_dir = self.workshop_panel.dt_data_dir.text().strip()
        if data_dir and Path(data_dir).is_dir():
            jsonl_files = sorted(Path(data_dir).glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if jsonl_files:
                jsonl_path = str(jsonl_files[0])
        if self.agent_panel.source_type.currentText() == "video":
            video_path = self.agent_panel.path_input.text().strip()
        dialog = AnnotationViewer(self, jsonl_path=jsonl_path, video_path=video_path)
        dialog.exec()

    def _browse_dir_to(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "选择目录")
        if path:
            line_edit.setText(path)

    @Slot()
    def _preview_data(self):
        data_dir = self.workshop_panel.dt_data_dir.text().strip()
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
            for action, count in summary.get("action_distribution", {}).items():
                pct = count / summary["total"] * 100
                lines.append(f"  {action}: {count} ({pct:.1f}%)")
            lines.append("")
            lines.append("检测类别:")
            for cls, count in summary.get("detection_classes", {}).items():
                lines.append(f"  {cls}: {count}")
            QMessageBox.information(self, "数据预览", "\n".join(lines))
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    @Slot()
    def _start_decision_train(self):
        wp = self.workshop_panel
        data_dir = wp.dt_data_dir.text().strip()
        output_dir = wp.dt_output_dir.text().strip()
        if not data_dir:
            QMessageBox.warning(self, "提示", "请指定数据目录")
            return
        wp.dt_train_btn.setEnabled(False)
        wp.dt_progress.setVisible(True)
        wp.dt_progress.setValue(0)
        wp.train_chart.clear()
        wp.train_chart.setVisible(True)
        wp.dt_status_label.setText("准备中...")
        self._train_worker = DecisionTrainWorker(
            data_dir=data_dir, output_dir=output_dir,
            model_type=wp.model_type_combo.currentText(),
            epochs=wp.epochs_spin.value(), lr=wp.lr_spin.value(),
        )
        self._train_worker.log_message.connect(lambda msg: self._log(f"[训练] {msg}"))
        self._train_worker.progress.connect(self._on_dt_progress)
        self._train_worker.finished_ok.connect(self._on_dt_finished)
        self._train_worker.finished_err.connect(self._on_dt_error)
        self._train_worker.start()
        self._log("[训练] 决策模型训练开始")

    @Slot(int, int, float, float, float)
    def _on_dt_progress(self, epoch, total, loss, train_acc, val_acc):
        wp = self.workshop_panel
        pct = int(epoch / total * 100) if total > 0 else 0
        wp.dt_progress.setValue(pct)
        wp.dt_status_label.setText(
            f"Epoch {epoch}/{total}  loss={loss:.4f}  "
            f"train={train_acc:.3f}  val={val_acc:.3f}"
        )
        wp.train_chart.add_point(loss, train_acc, val_acc)

    @Slot(str, dict)
    def _on_dt_finished(self, model_dir, metrics):
        wp = self.workshop_panel
        wp.dt_train_btn.setEnabled(True)
        wp.apply_model_btn.setEnabled(True)
        wp.dt_progress.setValue(100)
        val_acc = metrics.get("best_val_acc", 0)
        wp.dt_status_label.setText(
            f"训练完成  val={val_acc:.3f}  ({metrics.get('epochs_trained', '?')} epochs)"
        )
        wp.dt_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px; font-weight: bold;")
        self._last_model_dir = model_dir
        self._log(f"[训练] 完成 → {model_dir}")
        self._train_worker = None

    @Slot(str)
    def _on_dt_error(self, error):
        wp = self.workshop_panel
        wp.dt_train_btn.setEnabled(True)
        wp.dt_progress.setVisible(False)
        wp.dt_status_label.setText(f"训练失败: {error}")
        wp.dt_status_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 12px;")
        QMessageBox.critical(self, "训练失败", error)
        self._train_worker = None

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
        wp = self.workshop_panel
        save_dir = wp.rec_dir_input.text().strip() or "data/recordings"
        session = wp.rec_session_input.text().strip() or None
        try:
            self._recorder = DataRecorder(save_dir=save_dir, session_name=session)
            self._recorder.on_start()
            self._worker.agents.append(self._recorder)
            wp.rec_start_btn.setText("■  停止录制")
            wp.rec_status_label.setText(f"● 录制中... → {self._recorder.file_path}")
            wp.rec_status_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 12px; font-weight: bold;")
            self._log(f"[录制] 开始 → {self._recorder.file_path}")
        except Exception as e:
            QMessageBox.critical(self, "录制失败", str(e))

    def _stop_recording(self):
        if not self._recorder:
            return
        wp = self.workshop_panel
        if self._worker and self._recorder in self._worker.agents:
            self._worker.agents.remove(self._recorder)
        count = self._recorder.sample_count
        path = self._recorder.file_path
        self._recorder.on_stop()
        wp.rec_start_btn.setText("●  开始录制")
        wp.rec_status_label.setText(f"录制完成: {count} 条样本")
        wp.rec_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px;")
        self._log(f"[录制] 停止, {count} 条样本已保存")
        if path:
            wp.dt_data_dir.setText(str(path.parent))
        self._recorder = None

    # ================================================================
    #  Profile 管理
    # ================================================================

    def _refresh_profiles(self):
        combo = self.workshop_panel.profile_combo
        current = combo.currentText()
        combo.clear()
        combo.addItem("(无)", "")
        profiles = self._profile_mgr.load_all()
        for name, p in profiles.items():
            combo.addItem(f"{p.display_name} ({name})", name)
        if current:
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self._refresh_agent_profiles()

    def _refresh_agent_profiles(self):
        combo = self.agent_panel.agent_profile_combo
        current = combo.currentText()
        combo.clear()
        combo.addItem("(无)", "")
        profiles = self._profile_mgr.load_all()
        for name, p in profiles.items():
            combo.addItem(f"{p.display_name} ({name})", name)
        if current:
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    @Slot(str)
    def _on_profile_changed(self, text: str):
        if not text or text == "(无)":
            self.workshop_panel.profile_info.clear()
            return
        name = self.workshop_panel.profile_combo.currentData() or ""
        profile = self._profile_mgr.get(name) if name else None
        if not profile:
            return
        info_lines = [
            f"名称: {profile.display_name}",
            f"YOLO 模型: {profile.yolo_model}",
            f"决策引擎: {profile.decision_engine}",
            f"动作: {', '.join(profile.actions)}",
            f"场景关键词: {', '.join(profile.scene_keywords)}",
        ]
        self.workshop_panel.profile_info.setPlainText("\n".join(info_lines))

    @Slot(str)
    def _on_agent_profile_changed(self, text: str):
        if not text or text == "(无)":
            return
        name = self.agent_panel.agent_profile_combo.currentData() or ""
        profile = self._profile_mgr.get(name) if name else None
        if not profile:
            return
        ap = self.agent_panel
        if profile.yolo_model:
            self.workshop_panel.yolo_model_combo.setEditText(profile.yolo_model)
        if profile.decision_engine and profile.decision_engine != "rule":
            ap.engine_combo.setCurrentText(profile.decision_engine)
        if profile.action_key_map:
            ap.trained_action_map_edit.setPlainText(
                json.dumps(profile.action_key_map, indent=2, ensure_ascii=False)
            )
        if profile.decision_model_dir:
            ap.trained_model_dir.setText(profile.decision_model_dir)

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
                zf.extractall(".")
            self._refresh_profiles()
            self._log(f"[配置] 已导入")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

    # ================================================================
    #  Agent 模式
    # ================================================================

    def _on_source_type_changed(self, source_type: str):
        self.agent_panel.update_source_visibility(source_type)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "",
            "视频/图片 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.jpg *.png *.bmp);;所有文件 (*)"
        )
        if path:
            self.agent_panel.path_input.setText(path)

    def _browse_trained_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if path:
            self.agent_panel.trained_model_dir.setText(path)

    @Slot(str)
    def _on_engine_changed(self, engine_type: str):
        self.agent_panel.update_engine_visibility(engine_type)

    def _build_source_config(self) -> dict:
        ap = self.agent_panel
        source_type = ap.source_type.currentText()
        config = {"type": source_type}
        if source_type == "video":
            config["video"] = {
                "path": ap.path_input.text().strip(),
                "loop": ap.loop_combo.currentIndex() == 1,
            }
        elif source_type == "camera":
            config["camera"] = {"device": ap.camera_device.value()}
        elif source_type == "screen":
            config["screen"] = {"monitor": ap.monitor_num.value()}
        elif source_type == "image":
            config["image"] = {
                "path": ap.path_input.text().strip(),
                "loop": ap.loop_combo.currentIndex() == 1,
            }
        elif source_type == "stream":
            raw = ap.stream_input.text().strip()
            url = f"bilibili://{raw}" if raw.isdigit() else raw
            config["stream"] = {"url": url}
        return config

    def _build_agents(self) -> list:
        ap = self.agent_panel
        engine_type = ap.engine_combo.currentText()
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

        engine = None
        if engine_type == "rule":
            engine = RuleEngine()
            def _rule_attack(result, state):
                if not result.detections:
                    return None
                det = result.detections[0]
                return Action(
                    tool_name="keyboard",
                    parameters={"action": "press", "key": "a"},
                    reason=f"发现 {det.class_name} ({det.confidence:.0%})",
                    target_bbox=det.bbox,
                )
            engine.add_rule("attack_on_detect", _rule_attack)
        elif engine_type == "trained":
            model_dir = ap.trained_model_dir.text().strip() or "runs/decision/exp1"
            action_key_map = {}
            map_text = ap.trained_action_map_edit.toPlainText().strip()
            if map_text:
                try:
                    action_key_map = json.loads(map_text)
                except json.JSONDecodeError as e:
                    self._log(f"[警告] 动作映射 JSON 错误: {e}")
            try:
                engine = TrainedEngine(
                    model_dir=model_dir,
                    confidence_threshold=ap.trained_conf_spin.value(),
                    action_key_map=action_key_map,
                )
                self._log(f"Trained 引擎: {model_dir}")
            except Exception as e:
                self._log(f"[错误] 加载 Trained 引擎失败: {e}")
                return []
        elif engine_type == "llm":
            api_key = self._get_llm_api_key()
            provider_name = self.llm_panel.llm_provider_combo.currentText()
            model = self.llm_panel.llm_model_combo.currentText()
            base_url = self.llm_panel.llm_base_url.text().strip()
            if not api_key and provider_name != "ollama":
                self._log("[警告] API Key 未设置")
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
                decision_interval=ap.llm_interval.value(),
            )
            engine.set_log_callback(self._decision_log_callback)
            self._log(f"LLM 引擎: {provider_name}/{model}")
        elif engine_type == "hierarchical":
            micro = RuleEngine()
            micro.add_rule("detect_any", lambda r, s: Action(
                tool_name="keyboard",
                parameters={"action": "press", "key": "space"},
                reason=f"发现 {r.detections[0].class_name}" if r.detections else "",
                target_bbox=r.detections[0].bbox if r.detections else None,
            ) if r.detections else None)
            engine = HierarchicalEngine(micro=micro)
            engine.set_log_callback(self._decision_log_callback)
        elif engine_type == "rl":
            actions = ["idle", "attack", "retreat"]
            action_key_map = {}
            name = ap.agent_profile_combo.currentData() or ""
            profile = self._profile_mgr.get(name) if name else None
            if profile:
                actions = profile.actions or actions
                action_key_map = profile.action_key_map or {}
            engine = RLEngine(actions=actions, action_key_map=action_key_map, training=True)
            engine.set_log_callback(self._decision_log_callback)
        else:
            return []

        state_mgr = StateManager()
        agent = ActionAgent(
            decision_engine=engine, tool_registry=registry,
            state_manager=state_mgr, on_log=self._decision_log_callback,
        )
        self._log(f"Agent 就绪 ({engine_type})")
        return [agent]

    # ================================================================
    #  检测控制
    # ================================================================

    @Slot()
    def _start_detection(self):
        source_config = self._build_source_config()
        if source_config["type"] in ("video", "image"):
            key = source_config["type"]
            if not source_config[key].get("path"):
                QMessageBox.warning(self, "提示", "请输入路径")
                return
        if source_config["type"] == "stream" and not source_config["stream"].get("url"):
            QMessageBox.warning(self, "提示", "请输入流地址")
            return

        try:
            source = create_source(source_config)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建视频源失败:\n{e}")
            return
        try:
            detector = Detector(
                model=self.workshop_panel.yolo_model_combo.currentText(),
                confidence=self.workshop_panel.conf_spin.value(),
                imgsz=int(self.workshop_panel.imgsz_combo.currentText()),
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
        if self.agent_panel.autopilot_check.isChecked():
            try:
                self._auto_pilot = AutoPilot(
                    profile_manager=self._profile_mgr,
                    scene_classifier=self._scene_classifier,
                    on_log=self._decision_log_callback,
                )
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
        self.agent_panel.chat_panel.update_frame(frame)
        self.count_label.setText(f"检测: {len(result.detections)}  |  帧: {self._frame_count}")
        if self._auto_pilot:
            self.scene_label.setText(f"场景: {self._auto_pilot.current_scene}")
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
        sb = self.decision_log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

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

    def _on_agent_tab_changed(self, index: int):
        if self.agent_panel.tabText(index) == "对话" and self.agent_panel.chat_panel._provider is None:
            self._init_chat_provider()

    def _init_chat_provider(self):
        lp = self.llm_panel
        api_key = self._get_llm_api_key()
        provider_name = lp.llm_provider_combo.currentText()
        model = lp.llm_model_combo.currentText()
        base_url = lp.llm_base_url.text().strip()
        if not api_key and provider_name != "ollama":
            self.agent_panel.chat_panel.set_provider(None)
            return False
        try:
            provider = create_provider(provider_name, api_key or "ollama", model, base_url)
            self.agent_panel.chat_panel.set_provider(provider)
        except Exception as e:
            self._log(f"[对话] 创建 LLM Provider 失败: {e}")
            self.agent_panel.chat_panel.set_provider(None)
            return False

        reg = ToolRegistry()
        try:
            reg.register(KeyboardTool())
            reg.register(MouseTool())
            self.agent_panel.chat_panel.set_tool_registry(reg)
        except ImportError:
            pass
        return True

    # ================================================================
    #  模型管理
    # ================================================================

    def _scan_models(self):
        import glob
        combo = self.workshop_panel.yolo_model_combo
        existing = [combo.itemText(i) for i in range(combo.count())]
        for best in glob.glob("runs/**/best.pt", recursive=True):
            best = best.replace("\\", "/")
            if best not in existing:
                combo.addItem(best)

    @Slot()
    def _load_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "PyTorch 模型 (*.pt);;ONNX 模型 (*.onnx);;所有文件 (*)"
        )
        if path:
            self.workshop_panel.yolo_model_combo.setEditText(path)
            self._log(f"已加载: {path}")

    @Slot()
    def _open_train_dialog(self):
        from .train_dialog import TrainDialog
        dialog = TrainDialog(self)
        if dialog.exec() == QDialog.Accepted:
            model_path = dialog.get_model_path()
            if model_path:
                self.workshop_panel.yolo_model_combo.setEditText(model_path)

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
        wp = self.workshop_panel
        ap = self.agent_panel
        lp = self.llm_panel
        s.setValue("app/mode", self._current_mode)
        s.setValue("source/type", ap.source_type.currentText())
        s.setValue("source/path", ap.path_input.text())
        s.setValue("source/loop", ap.loop_combo.currentIndex())
        s.setValue("source/camera", ap.camera_device.value())
        s.setValue("source/monitor", ap.monitor_num.value())
        s.setValue("source/stream_url", ap.stream_input.text())
        s.setValue("detector/model", wp.yolo_model_combo.currentText())
        s.setValue("detector/confidence", wp.conf_spin.value())
        s.setValue("detector/imgsz", wp.imgsz_combo.currentText())
        s.setValue("decision/engine", ap.engine_combo.currentText())
        s.setValue("decision/trained_dir", ap.trained_model_dir.text())
        s.setValue("decision/trained_conf", ap.trained_conf_spin.value())
        s.setValue("decision/trained_action_map", ap.trained_action_map_edit.toPlainText())
        s.setValue("recorder/dir", wp.rec_dir_input.text())
        s.setValue("train/data_dir", wp.dt_data_dir.text())
        s.setValue("train/output_dir", wp.dt_output_dir.text())
        s.setValue("train/model_type", wp.model_type_combo.currentText())
        s.setValue("train/epochs", wp.epochs_spin.value())
        s.setValue("train/lr", wp.lr_spin.value())
        s.setValue("decision/provider", lp.llm_provider_combo.currentText())
        s.setValue("decision/model", lp.llm_model_combo.currentText())
        s.setValue("decision/base_url", lp.llm_base_url.text())
        s.setValue("decision/interval", ap.llm_interval.value())

    def _load_settings(self):
        s = self._settings
        wp = self.workshop_panel
        ap = self.agent_panel
        lp = self.llm_panel
        if s.value("app/mode"):
            self._switch_mode(s.value("app/mode"))
        if s.value("source/type"):
            ap.source_type.setCurrentText(s.value("source/type"))
        if s.value("source/path"):
            ap.path_input.setText(s.value("source/path"))
        if s.value("source/loop") is not None:
            ap.loop_combo.setCurrentIndex(int(s.value("source/loop", 0)))
        if s.value("source/camera") is not None:
            ap.camera_device.setValue(int(s.value("source/camera", 0)))
        if s.value("source/monitor") is not None:
            ap.monitor_num.setValue(int(s.value("source/monitor", 1)))
        if s.value("source/stream_url"):
            ap.stream_input.setText(s.value("source/stream_url"))
        if s.value("detector/model"):
            wp.yolo_model_combo.setEditText(s.value("detector/model"))
        if s.value("detector/confidence") is not None:
            wp.conf_spin.setValue(float(s.value("detector/confidence", 0.5)))
        if s.value("detector/imgsz"):
            wp.imgsz_combo.setCurrentText(s.value("detector/imgsz"))
        if s.value("decision/engine"):
            ap.engine_combo.setCurrentText(s.value("decision/engine"))
        if s.value("decision/trained_dir"):
            ap.trained_model_dir.setText(s.value("decision/trained_dir"))
        if s.value("decision/trained_conf") is not None:
            ap.trained_conf_spin.setValue(float(s.value("decision/trained_conf", 0.3)))
        if s.value("decision/trained_action_map"):
            ap.trained_action_map_edit.setPlainText(s.value("decision/trained_action_map"))
        if s.value("recorder/dir"):
            wp.rec_dir_input.setText(s.value("recorder/dir"))
        if s.value("train/data_dir"):
            wp.dt_data_dir.setText(s.value("train/data_dir"))
        if s.value("train/output_dir"):
            wp.dt_output_dir.setText(s.value("train/output_dir"))
        if s.value("train/model_type"):
            wp.model_type_combo.setCurrentText(s.value("train/model_type"))
        if s.value("train/epochs") is not None:
            wp.epochs_spin.setValue(int(s.value("train/epochs", 100)))
        if s.value("train/lr") is not None:
            wp.lr_spin.setValue(float(s.value("train/lr", 0.001)))
        if s.value("decision/provider"):
            lp.llm_provider_combo.setCurrentText(s.value("decision/provider"))
        if s.value("decision/model"):
            lp.llm_model_combo.setCurrentText(s.value("decision/model"))
        if s.value("decision/base_url"):
            lp.llm_base_url.setText(s.value("decision/base_url"))
        if s.value("decision/interval") is not None:
            ap.llm_interval.setValue(float(s.value("decision/interval", 1.0)))

    def closeEvent(self, event):
        self._save_settings()
        if self._worker:
            self._worker.stop()
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
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        model_exts = {'.pt', '.onnx'}

        if ext in video_exts:
            if self._current_mode == "workshop":
                wp = self.workshop_panel
                wp.video_path_input.setText(path)
                if wp.current_scene:
                    wp.current_scene.add_videos([path])
                    wp._update_scene_ui(wp.current_scene)
                self._log(f"已加载视频: {Path(path).name}")
            else:
                self._switch_mode("agent")
                self.agent_panel.source_type.setCurrentText("video")
                self.agent_panel.path_input.setText(path)
                self._log(f"已加载视频: {Path(path).name}")
        elif ext in image_exts:
            self._switch_mode("agent")
            self.agent_panel.source_type.setCurrentText("image")
            self.agent_panel.path_input.setText(path)
            self._log(f"已加载图片: {Path(path).name}")
        elif ext in model_exts:
            self.workshop_panel.yolo_model_combo.setEditText(path)
            self._log(f"已加载模型: {Path(path).name}")
        elif Path(path).is_dir():
            self._switch_mode("workshop")
            wp = self.workshop_panel
            wp.input_type.setCurrentText("视频文件夹")
            wp.video_path_input.setText(path)
            files = wp._scan_folder(path)
            if wp.current_scene and files:
                wp.current_scene.add_videos(files)
                wp._update_scene_ui(wp.current_scene)
            self._log(f"已加载文件夹: {Path(path).name}")
