"""主窗口：训练工坊 + Agent 执行 双模式界面。"""

import os
import json
from pathlib import Path
from PySide6.QtCore import Qt, Slot, QSettings, QMimeData
from PySide6.QtGui import QFont, QIcon, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton,
    QFileDialog, QTextEdit, QSplitter, QMessageBox,
    QDialog, QTabWidget, QCheckBox, QStackedWidget, QFrame,
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
from .train_panel import TrainPanel
from .agent_panel import AgentPanel
from .llm_panel import LLMPanel

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
        self._connect_signals()
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
        mode_row.setSpacing(4)
        self.mode_train_btn = QPushButton("训练")
        self.mode_agent_btn = QPushButton("Agent")
        self.mode_llm_btn = QPushButton("LLM")
        for btn, mode in [
            (self.mode_train_btn, "train"),
            (self.mode_agent_btn, "agent"),
            (self.mode_llm_btn, "llm"),
        ]:
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, m=mode: self._switch_mode(m))
            mode_row.addWidget(btn)
        left_layout.addLayout(mode_row)

        # -- 模式面板 (Stacked) --
        self.mode_stack = QStackedWidget()
        self.train_panel = TrainPanel()
        self.agent_panel = AgentPanel()
        self.llm_panel = LLMPanel()
        self.mode_stack.addWidget(self.train_panel)   # index 0
        self.mode_stack.addWidget(self.agent_panel)   # index 1
        self.mode_stack.addWidget(self.llm_panel)     # index 2
        left_layout.addWidget(self.mode_stack, 1)

        # -- Agent 控制区 --
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
        self._build_status_bar(left_layout)

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
    #  信号连接（面板 → 主窗口）
    # ================================================================

    def _connect_signals(self):
        tp = self.train_panel
        ap = self.agent_panel
        lp = self.llm_panel

        # LLM 面板
        lp.llm_provider_combo.addItems(list(PROVIDER_PRESETS.keys()))
        lp.llm_provider_combo.currentTextChanged.connect(self._on_provider_changed)
        lp.llm_test_btn.clicked.connect(self._test_llm_connection)
        lp.llm_save_btn.clicked.connect(self._save_llm_settings)

        # 训练面板
        tp.auto_learn_btn.clicked.connect(self._open_auto_learn)
        tp.load_model_btn.clicked.connect(self._load_custom_model)
        tp.train_btn_open.clicked.connect(self._open_train_dialog)
        tp.auto_annotate_btn.clicked.connect(self._open_annotate_dialog)
        tp.view_annotation_btn.clicked.connect(self._open_annotation_viewer)
        tp.rec_browse_btn.clicked.connect(lambda: self._browse_dir_to(tp.rec_dir_input))
        tp.rec_start_btn.clicked.connect(self._toggle_recording)
        tp.dt_data_browse.clicked.connect(lambda: self._browse_dir_to(tp.dt_data_dir))
        tp.dt_preview_btn.clicked.connect(self._preview_data)
        tp.dt_train_btn.clicked.connect(self._start_decision_train)
        tp.dt_use_btn.clicked.connect(self._use_trained_model)
        tp.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        tp.refresh_btn.clicked.connect(self._refresh_profiles)
        tp.export_btn.clicked.connect(self._export_config)
        tp.import_btn.clicked.connect(self._import_config)

        # 初始化供应商预设
        self._on_provider_changed(lp.llm_provider_combo.currentText())

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

        # 初始化可见性
        self._on_source_type_changed("video")

        # 初始化 profiles
        self._refresh_profiles()

    # ================================================================
    #  模式切换
    # ================================================================

    def _switch_mode(self, mode: str):
        self._current_mode = mode
        index = {"train": 0, "agent": 1, "llm": 2}.get(mode, 1)
        self.mode_stack.setCurrentIndex(index)
        self.agent_controls.setVisible(mode == "agent")

        for btn, m in [
            (self.mode_train_btn, "train"),
            (self.mode_agent_btn, "agent"),
            (self.mode_llm_btn, "llm"),
        ]:
            btn.setObjectName("modeBtnActive" if m == mode else "modeBtnInactive")
            btn.setStyle(btn.style())

    # ================================================================
    #  Profile 管理
    # ================================================================

    def _refresh_profiles(self):
        combo = self.train_panel.profile_combo
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
            self.train_panel.profile_info.clear()
            return
        name = self.train_panel.profile_combo.currentData() or ""
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
        self.train_panel.profile_info.setPlainText("\n".join(info_lines))

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
            self.train_panel.model_combo.setEditText(profile.yolo_model)
        if profile.decision_engine and profile.decision_engine != "rule":
            ap.engine_combo.setCurrentText(profile.decision_engine)
        if profile.action_key_map:
            ap.trained_action_map_edit.setPlainText(
                json.dumps(profile.action_key_map, indent=2, ensure_ascii=False)
            )
        if profile.decision_model_dir:
            ap.trained_model_dir.setText(profile.decision_model_dir)

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
        ap = self.agent_panel
        default_video = ap.path_input.text().strip() if ap.source_type.currentText() == "video" else ""
        default_model = self.train_panel.model_combo.currentText()
        dialog = AnnotateDialog(self, default_video=default_video, default_model=default_model)
        dialog.exec()
        save_path = dialog.get_save_path()
        if save_path:
            self.train_panel.dt_data_dir.setText(str(Path(save_path).parent))
            self._log(f"[标注] 数据已保存到 {save_path}")

    @Slot()
    def _open_annotation_viewer(self):
        from .annotation_viewer import AnnotationViewer
        jsonl_path = ""
        video_path = ""
        data_dir = self.train_panel.dt_data_dir.text().strip()
        if data_dir and Path(data_dir).is_dir():
            jsonl_files = sorted(Path(data_dir).glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if jsonl_files:
                jsonl_path = str(jsonl_files[0])
        if self.agent_panel.source_type.currentText() == "video":
            video_path = self.agent_panel.path_input.text().strip()
        dialog = AnnotationViewer(self, jsonl_path=jsonl_path, video_path=video_path)
        dialog.exec()

    def _browse_dir_to(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "选择目录")
        if path:
            line_edit.setText(path)

    def _browse_trained_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if path:
            self.agent_panel.trained_model_dir.setText(path)

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
        tp = self.train_panel
        save_dir = tp.rec_dir_input.text().strip() or "data/recordings"
        session = tp.rec_session_input.text().strip() or None
        try:
            self._recorder = DataRecorder(save_dir=save_dir, session_name=session)
            self._recorder.on_start()
            self._worker.agents.append(self._recorder)
            tp.rec_start_btn.setText("■  停止录制")
            tp.rec_status_label.setText(f"● 录制中... → {self._recorder.file_path}")
            tp.rec_status_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 12px; font-weight: bold;")
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
        tp = self.train_panel
        if hasattr(self, '_rec_timer'):
            self._rec_timer.stop()
        if self._worker and self._recorder in self._worker.agents:
            self._worker.agents.remove(self._recorder)
        count = self._recorder.sample_count
        path = self._recorder.file_path
        self._recorder.on_stop()
        tp.rec_start_btn.setText("●  开始录制")
        tp.rec_status_label.setText(f"录制完成: {count} 条样本 → {path}")
        tp.rec_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px;")
        self._log(f"[录制] 停止, {count} 条样本已保存")
        if path:
            tp.dt_data_dir.setText(str(path.parent))
        self._recorder = None

    @Slot()
    def _update_rec_status(self):
        if self._recorder and self._recorder.is_recording:
            self.train_panel.rec_status_label.setText(
                f"录制中... {self._recorder.sample_count} 条样本 → {self._recorder.file_path}"
            )

    @Slot()
    def _preview_data(self):
        data_dir = self.train_panel.dt_data_dir.text().strip()
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
        tp = self.train_panel
        data_dir = tp.dt_data_dir.text().strip()
        output_dir = tp.dt_output_dir.text().strip()
        if not data_dir:
            QMessageBox.warning(self, "提示", "请指定数据目录")
            return
        tp.dt_train_btn.setEnabled(False)
        tp.dt_use_btn.setEnabled(False)
        tp.dt_progress.setVisible(True)
        tp.dt_progress.setValue(0)
        tp.dt_chart.clear()
        tp.dt_chart.setVisible(True)
        tp.dt_status_label.setText("准备中...")
        model_type = tp.dt_model_type.currentText()
        self._train_worker = DecisionTrainWorker(
            data_dir=data_dir, output_dir=output_dir, model_type=model_type,
            epochs=tp.dt_epochs_spin.value(), lr=tp.dt_lr_spin.value(),
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
        tp = self.train_panel
        pct = int(epoch / total * 100) if total > 0 else 0
        tp.dt_progress.setValue(pct)
        tp.dt_status_label.setText(
            f"Epoch {epoch}/{total}  loss={loss:.4f}  "
            f"train={train_acc:.3f}  val={val_acc:.3f}"
        )
        tp.dt_chart.add_point(loss, train_acc, val_acc)

    @Slot(str, dict)
    def _on_dt_finished(self, model_dir, metrics):
        tp = self.train_panel
        tp.dt_train_btn.setEnabled(True)
        tp.dt_use_btn.setEnabled(True)
        tp.dt_progress.setValue(100)
        val_acc = metrics.get("best_val_acc", 0)
        train_acc = metrics.get("final_train_acc", 0)
        epochs = metrics.get("epochs_trained", "?")
        tp.dt_status_label.setText(
            f"训练完成  val={val_acc:.3f}  train={train_acc:.3f}  ({epochs} epochs)"
        )
        tp.dt_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px; font-weight: bold;")
        self._log(f"[训练] 完成 → {model_dir} (val_acc={val_acc:.4f})")
        self._train_worker = None

    @Slot(str)
    def _on_dt_error(self, error):
        tp = self.train_panel
        tp.dt_train_btn.setEnabled(True)
        tp.dt_progress.setVisible(False)
        tp.dt_status_label.setText(f"训练失败: {error}")
        tp.dt_status_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 12px;")
        QMessageBox.critical(self, "训练失败", error)
        self._train_worker = None

    @Slot()
    def _use_trained_model(self):
        model_dir = self.train_panel.dt_output_dir.text().strip()
        self._switch_mode("agent")
        self.agent_panel.engine_combo.setCurrentText("trained")
        self.agent_panel.trained_model_dir.setText(model_dir)
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
        self.agent_panel.update_source_visibility(source_type)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "",
            "视频/图片 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.jpg *.png *.bmp);;所有文件 (*)"
        )
        if path:
            self.agent_panel.path_input.setText(path)

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
            if raw.isdigit():
                url = f"bilibili://{raw}"
            else:
                url = raw
            config["stream"] = {"url": url}
        return config

    @Slot(str)
    def _on_engine_changed(self, engine_type: str):
        self.agent_panel.update_engine_visibility(engine_type)

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
                self._log(f"LLM 连接测试成功: {provider_name}/{model}")
            else:
                QMessageBox.warning(self, "失败", "连接测试失败，请检查配置")
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
        tab_text = self.agent_panel.tabText(index)
        if tab_text == "对话" and self.agent_panel.chat_panel._provider is None:
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

        registry = ToolRegistry()
        try:
            registry.register(KeyboardTool())
            registry.register(MouseTool())
        except ImportError:
            self._log("[对话] pynput 未安装，键盘鼠标工具不可用")
            return False

        self.agent_panel.chat_panel.set_tool_registry(registry)
        self._log(f"[对话] LLM 已就绪: {provider_name}/{model}")
        return True

    # ================================================================
    #  Agent 构建
    # ================================================================

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
            model_dir = ap.trained_model_dir.text().strip() or "runs/decision/exp1"
            action_key_map = {}
            map_text = ap.trained_action_map_edit.toPlainText().strip()
            if map_text:
                try:
                    action_key_map = json.loads(map_text)
                except json.JSONDecodeError as e:
                    self._log(f"[警告] 动作映射 JSON 格式错误: {e}，将使用默认")
            try:
                engine = TrainedEngine(
                    model_dir=model_dir,
                    confidence_threshold=ap.trained_conf_spin.value(),
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
            provider_name = self.llm_panel.llm_provider_combo.currentText()
            model = self.llm_panel.llm_model_combo.currentText()
            base_url = self.llm_panel.llm_base_url.text().strip()
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
                decision_interval=ap.llm_interval.value(),
            )
            engine.set_log_callback(self._decision_log_callback)
            self._log(f"LLM 引擎: {provider_name}/{model}")
        elif engine_type == "hierarchical":
            name = ap.agent_profile_combo.currentData() or ""
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
            name = ap.agent_profile_combo.currentData() or ""
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
                model=self.train_panel.model_combo.currentText(),
                confidence=self.train_panel.conf_spin.value(),
                imgsz=int(self.train_panel.imgsz_combo.currentText()),
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
        self.agent_panel.chat_panel.update_frame(frame)
        self.count_label.setText(
            f"检测: {len(result.detections)}  |  帧: {self._frame_count}"
        )
        if self._auto_pilot:
            scene = self._auto_pilot.current_scene
            self.scene_label.setText(f"场景: {scene}")
            engines = self._auto_pilot.available_engines
            self.agent_panel.scene_status_text.setPlainText(
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
        combo = self.train_panel.model_combo
        existing = [combo.itemText(i) for i in range(combo.count())]
        for best in glob.glob("runs/**/best.pt", recursive=True):
            best = best.replace("\\", "/")
            if best not in existing:
                combo.addItem(f"{best}")
                existing.append(best)

    @Slot()
    def _load_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "PyTorch 模型 (*.pt);;ONNX 模型 (*.onnx);;所有文件 (*)"
        )
        if path:
            self.train_panel.model_combo.setEditText(path)
            self._log(f"已加载: {path}")

    @Slot()
    def _open_train_dialog(self):
        from .train_dialog import TrainDialog
        dialog = TrainDialog(self)
        if dialog.exec() == QDialog.Accepted:
            model_path = dialog.get_model_path()
            if model_path:
                self.train_panel.model_combo.setEditText(model_path)
                self._log(f"训练模型: {model_path}")

    # ================================================================
    #  设置保存 / 加载
    # ================================================================

    def _save_settings(self):
        s = self._settings
        tp = self.train_panel
        ap = self.agent_panel
        s.setValue("app/mode", self._current_mode)
        s.setValue("source/type", ap.source_type.currentText())
        s.setValue("source/path", ap.path_input.text())
        s.setValue("source/loop", ap.loop_combo.currentIndex())
        s.setValue("source/camera", ap.camera_device.value())
        s.setValue("source/monitor", ap.monitor_num.value())
        s.setValue("source/stream_url", ap.stream_input.text())
        s.setValue("detector/model", tp.model_combo.currentText())
        s.setValue("detector/confidence", tp.conf_spin.value())
        s.setValue("detector/imgsz", tp.imgsz_combo.currentText())
        s.setValue("decision/engine", ap.engine_combo.currentText())
        s.setValue("decision/trained_dir", ap.trained_model_dir.text())
        s.setValue("decision/trained_conf", ap.trained_conf_spin.value())
        s.setValue("decision/trained_action_map", ap.trained_action_map_edit.toPlainText())
        s.setValue("recorder/dir", tp.rec_dir_input.text())
        s.setValue("train/data_dir", tp.dt_data_dir.text())
        s.setValue("train/output_dir", tp.dt_output_dir.text())
        s.setValue("train/model_type", tp.dt_model_type.currentText())
        s.setValue("train/epochs", tp.dt_epochs_spin.value())
        s.setValue("train/lr", tp.dt_lr_spin.value())
        lp = self.llm_panel
        s.setValue("decision/provider", lp.llm_provider_combo.currentText())
        s.setValue("decision/model", lp.llm_model_combo.currentText())
        s.setValue("decision/base_url", lp.llm_base_url.text())
        s.setValue("decision/interval", ap.llm_interval.value())

    def _load_settings(self):
        s = self._settings
        tp = self.train_panel
        ap = self.agent_panel
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
            tp.model_combo.setEditText(s.value("detector/model"))
        if s.value("detector/confidence") is not None:
            tp.conf_spin.setValue(float(s.value("detector/confidence", 0.5)))
        if s.value("detector/imgsz"):
            tp.imgsz_combo.setCurrentText(s.value("detector/imgsz"))
        if s.value("decision/engine"):
            ap.engine_combo.setCurrentText(s.value("decision/engine"))
        if s.value("decision/trained_dir"):
            ap.trained_model_dir.setText(s.value("decision/trained_dir"))
        if s.value("decision/trained_conf") is not None:
            ap.trained_conf_spin.setValue(float(s.value("decision/trained_conf", 0.3)))
        if s.value("decision/trained_action_map"):
            ap.trained_action_map_edit.setPlainText(s.value("decision/trained_action_map"))
        if s.value("recorder/dir"):
            tp.rec_dir_input.setText(s.value("recorder/dir"))
        if s.value("train/data_dir"):
            tp.dt_data_dir.setText(s.value("train/data_dir"))
        if s.value("train/output_dir"):
            tp.dt_output_dir.setText(s.value("train/output_dir"))
        if s.value("train/model_type"):
            tp.dt_model_type.setCurrentText(s.value("train/model_type"))
        if s.value("train/epochs") is not None:
            tp.dt_epochs_spin.setValue(int(s.value("train/epochs", 100)))
        if s.value("train/lr") is not None:
            tp.dt_lr_spin.setValue(float(s.value("train/lr", 0.001)))
        lp = self.llm_panel
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
            self.agent_panel.source_type.setCurrentText("video")
            self.agent_panel.path_input.setText(path)
            self._log(f"已加载视频: {Path(path).name}")
        elif ext in image_exts:
            self._switch_mode("agent")
            self.agent_panel.source_type.setCurrentText("image")
            self.agent_panel.path_input.setText(path)
            self._log(f"已加载图片: {Path(path).name}")
        elif ext in model_exts:
            self.train_panel.model_combo.setEditText(path)
            self._log(f"已加载模型: {Path(path).name}")
        else:
            self._log(f"不支持的文件类型: {ext}")
