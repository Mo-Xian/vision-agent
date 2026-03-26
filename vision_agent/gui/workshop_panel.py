"""训练工坊面板：场景管理 → 录制操作 → 行为克隆训练。"""

import json
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QLineEdit, QPushButton, QSpinBox,
    QDoubleSpinBox, QTextEdit, QFormLayout, QProgressBar,
    QScrollArea, QFrame, QFileDialog, QListWidget,
    QListWidgetItem, QAbstractItemView,
    QInputDialog, QMessageBox, QDialog, QTreeWidget,
    QTreeWidgetItem, QHeaderView, QDialogButtonBox,
)

from .styles import COLORS
from .widgets import CollapsibleSection


class ModelBrowserDialog(QDialog):
    """扫描 runs/ 目录，列出所有可用模型供选择。"""

    def __init__(self, parent=None, base_dir: str = "runs"):
        super().__init__(parent)
        self.setWindowTitle("选择模型")
        self.setMinimumSize(620, 380)
        self._selected_path = ""

        layout = QVBoxLayout(self)

        hint = QLabel("扫描所有训练产出，选择模型用于热启动或部署：")
        hint.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        layout.addWidget(hint)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["路径", "类型", "动作数", "准确率", "样本", "训练时间"])
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 6):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(False)
        self.tree.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.tree)

        btn_row = QHBoxLayout()
        self.browse_btn = QPushButton("手动选择目录...")
        self.browse_btn.clicked.connect(self._manual_browse)
        btn_row.addWidget(self.browse_btn)
        btn_row.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_row.addWidget(buttons)
        layout.addLayout(btn_row)

        self._scan(base_dir)

    def _scan(self, base_dir: str):
        """扫描 base_dir 下所有 model.meta.json。"""
        base = Path(base_dir)
        if not base.exists():
            return
        meta_files = sorted(base.rglob("model.meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for mf in meta_files:
            try:
                meta = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:
                continue
            model_dir = str(mf.parent)
            # 显示相对路径
            try:
                rel = str(mf.parent.relative_to(Path.cwd()))
            except ValueError:
                rel = model_dir

            model_type = meta.get("model_type", "?")
            type_label = {"e2e_mlp": "BC (E2E)", "dqn": "DQN (RL)", "mlp": "BC (MLP)"}.get(model_type, model_type)
            num_actions = str(meta.get("num_actions", meta.get("num_classes", "?")))
            acc = meta.get("best_val_acc", meta.get("metrics", {}).get("best_val_acc"))
            acc_str = f"{acc:.1%}" if acc is not None else "-"
            samples = str(meta.get("train_samples", "-"))
            trained_at = meta.get("trained_at", "-")

            item = QTreeWidgetItem([rel, type_label, num_actions, acc_str, samples, trained_at])
            item.setData(0, Qt.UserRole, model_dir)
            item.setToolTip(0, model_dir)
            self.tree.addTopLevelItem(item)

    def _manual_browse(self):
        folder = QFileDialog.getExistingDirectory(self, "选择模型目录", "runs")
        if folder:
            self._selected_path = folder
            self.accept()

    def accept(self):
        if not self._selected_path:
            cur = self.tree.currentItem()
            if cur:
                self._selected_path = cur.data(0, Qt.UserRole)
        super().accept()

    def selected_path(self) -> str:
        return self._selected_path


class WorkshopPanel(QWidget):
    """训练工坊面板 — 录制操作 → 行为克隆训练。

    布局：场景管理 → 录制操作 → 学习控制 → 进度/见解 → 模型管理 → 高级设置
    """

    learn_from_recording_requested = Signal()
    stop_requested = Signal()
    recording_started = Signal()
    recording_stopped = Signal()
    scene_changed = Signal(str)
    selfplay_start_requested = Signal()
    selfplay_stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_scene = None
        self._scene_manager = None
        self._build_ui()
        self.sp_start_btn.clicked.connect(self.selfplay_start_requested)
        self.sp_stop_btn.clicked.connect(self.selfplay_stop_requested)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ━━━ 场景管理 ━━━
        scene_group = QGroupBox("场景管理")
        sg_layout = QVBoxLayout(scene_group)
        sg_layout.setSpacing(6)

        scene_row = QHBoxLayout()
        self.scene_combo = QComboBox()
        self.scene_combo.setMinimumWidth(160)
        self.scene_combo.setToolTip("选择或创建训练场景")
        self.scene_combo.currentTextChanged.connect(self._on_scene_selected)
        scene_row.addWidget(self.scene_combo, 1)

        self.new_scene_btn = QPushButton("+ 新建")
        self.new_scene_btn.setObjectName("startBtn")
        self.new_scene_btn.setCursor(Qt.PointingHandCursor)
        self.new_scene_btn.clicked.connect(self._new_scene)
        scene_row.addWidget(self.new_scene_btn)

        self.delete_scene_btn = QPushButton("删除")
        self.delete_scene_btn.setObjectName("stopBtn")
        self.delete_scene_btn.setCursor(Qt.PointingHandCursor)
        self.delete_scene_btn.clicked.connect(self._delete_scene)
        scene_row.addWidget(self.delete_scene_btn)
        sg_layout.addLayout(scene_row)

        # 场景描述（移入场景分组）
        desc_row = QHBoxLayout()
        desc_row.addWidget(QLabel("描述"))
        self.description_input = QLineEdit()
        self.description_input.setPlaceholderText("场景描述（可选，如：王者荣耀5v5团战）")
        desc_row.addWidget(self.description_input, 1)
        sg_layout.addLayout(desc_row)

        self.scene_status = QLabel("")
        self.scene_status.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        sg_layout.addWidget(self.scene_status)

        layout.addWidget(scene_group)

        # ━━━ 录制操作 ━━━
        source_group = QGroupBox("录制操作")
        src_layout = QVBoxLayout(source_group)
        src_layout.setSpacing(6)

        rec_layout = src_layout

        rec_info = QLabel(
            "录制你的游戏操作（屏幕 + 键鼠），自动生成训练数据。\n"
            "F9 暂停/恢复录制。支持窗口捕获和鼠标操作录制。"
        )
        rec_info.setWordWrap(True)
        rec_info.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        rec_layout.addWidget(rec_info)

        # 窗口选择
        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("窗口"))
        self.window_combo = QComboBox()
        self.window_combo.setEditable(True)
        self.window_combo.addItem("（全屏）")
        self.window_combo.setToolTip("选择游戏窗口（模糊匹配），或留空全屏捕获")
        win_row.addWidget(self.window_combo, 1)
        self.refresh_windows_btn = QPushButton("刷新")
        self.refresh_windows_btn.setObjectName("browseBtn")
        self.refresh_windows_btn.setMaximumWidth(50)
        self.refresh_windows_btn.clicked.connect(self._refresh_window_list)
        win_row.addWidget(self.refresh_windows_btn)
        rec_layout.addLayout(win_row)

        rec_btn_row = QHBoxLayout()
        self.record_start_btn = QPushButton("开始录制")
        self.record_start_btn.setObjectName("startBtn")
        self.record_start_btn.setCursor(Qt.PointingHandCursor)
        self.record_start_btn.setMinimumHeight(36)
        self.record_start_btn.clicked.connect(self.recording_started)
        rec_btn_row.addWidget(self.record_start_btn)

        self.record_stop_btn = QPushButton("停止录制")
        self.record_stop_btn.setObjectName("stopBtn")
        self.record_stop_btn.setCursor(Qt.PointingHandCursor)
        self.record_stop_btn.setMinimumHeight(36)
        self.record_stop_btn.setEnabled(False)
        self.record_stop_btn.clicked.connect(self.recording_stopped)
        rec_btn_row.addWidget(self.record_stop_btn)
        rec_layout.addLayout(rec_btn_row)

        self.record_status = QLabel("")
        self.record_status.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px;")
        rec_layout.addWidget(self.record_status)

        # 录制列表
        self.recording_list = QListWidget()
        self.recording_list.setMaximumHeight(100)
        self.recording_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.recording_list.setStyleSheet(
            f"QListWidget {{ background: {COLORS['bg_input']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 6px; "
            f"color: {COLORS['text']}; font-size: 12px; padding: 4px; }}"
        )
        rec_layout.addWidget(self.recording_list)

        rec_ops = QHBoxLayout()
        self.remove_recording_btn = QPushButton("移除选中")
        self.remove_recording_btn.setObjectName("browseBtn")
        self.remove_recording_btn.clicked.connect(self._remove_selected_recordings)
        rec_ops.addWidget(self.remove_recording_btn)
        self.recording_count_label = QLabel("0 个录制")
        self.recording_count_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        rec_ops.addWidget(self.recording_count_label)

        self.import_recording_btn = QPushButton("导入录制")
        self.import_recording_btn.setObjectName("browseBtn")
        self.import_recording_btn.clicked.connect(self._import_recording)
        rec_ops.addWidget(self.import_recording_btn)

        rec_ops.addStretch()
        rec_layout.addLayout(rec_ops)

        layout.addWidget(source_group)

        # ━━━ 场景知识（折叠） ━━━
        knowledge_section = CollapsibleSection("场景知识（可选，提升 LLM 分析准确度）")
        kg = knowledge_section.content_layout()

        kb_btn_row = QHBoxLayout()
        self.kb_import_btn = QPushButton("导入文件")
        self.kb_import_btn.setObjectName("browseBtn")
        self.kb_import_btn.setCursor(Qt.PointingHandCursor)
        self.kb_import_btn.clicked.connect(self._import_knowledge)
        kb_btn_row.addWidget(self.kb_import_btn)

        self.kb_clear_btn = QPushButton("清空")
        self.kb_clear_btn.setObjectName("browseBtn")
        self.kb_clear_btn.setCursor(Qt.PointingHandCursor)
        self.kb_clear_btn.clicked.connect(lambda: self.knowledge_input.clear())
        kb_btn_row.addWidget(self.kb_clear_btn)

        self.kb_char_label = QLabel("0 字")
        self.kb_char_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        kb_btn_row.addWidget(self.kb_char_label)
        kb_btn_row.addStretch()
        kg.addLayout(kb_btn_row)

        self.knowledge_input = QTextEdit()
        self.knowledge_input.setMaximumHeight(100)
        self.knowledge_input.setPlaceholderText(
            "输入或导入场景规则/教程，例如：\n"
            "• 这是王者荣耀5v5 MOBA 游戏\n"
            "• 核心操作：普攻、技能1/2/3、闪现、回城\n"
            "支持导入 .txt / .md / .json / .yaml 文件"
        )
        self.knowledge_input.textChanged.connect(self._on_knowledge_changed)
        kg.addWidget(self.knowledge_input)

        layout.addWidget(knowledge_section)

        # ━━━ 核心操作 ━━━
        self.learn_btn = QPushButton("开始学习")
        self.learn_btn.setObjectName("startBtn")
        self.learn_btn.setCursor(Qt.PointingHandCursor)
        self.learn_btn.setMinimumHeight(42)
        self.learn_btn.setStyleSheet(
            self.learn_btn.styleSheet() +
            "QPushButton#startBtn { font-size: 15px; }"
        )
        self.learn_btn.clicked.connect(self._on_learn_clicked)
        layout.addWidget(self.learn_btn)

        self.stop_learn_btn = QPushButton("停止学习")
        self.stop_learn_btn.setObjectName("stopBtn")
        self.stop_learn_btn.setCursor(Qt.PointingHandCursor)
        self.stop_learn_btn.setEnabled(False)
        self.stop_learn_btn.setVisible(False)
        self.stop_learn_btn.clicked.connect(self.stop_requested)
        layout.addWidget(self.stop_learn_btn)

        # ━━━ 学习进度 + 见解 ━━━
        progress_group = QGroupBox("学习进度")
        pg = QVBoxLayout(progress_group)
        pg.setSpacing(6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("准备中")
        pg.addWidget(self.progress_bar)

        self.phase_label = QLabel("等待启动...")
        self.phase_label.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px;")
        pg.addWidget(self.phase_label)

        self.insight_text = QTextEdit()
        self.insight_text.setReadOnly(True)
        self.insight_text.setMaximumHeight(100)
        self.insight_text.setPlaceholderText("LLM 分析见解...")
        pg.addWidget(self.insight_text)

        layout.addWidget(progress_group)

        # ━━━ 模型管理 ━━━
        model_group = QGroupBox("训练产出")
        mg_layout = QVBoxLayout(model_group)
        mg_layout.setSpacing(6)

        from .train_chart import TrainChart
        self.train_chart = TrainChart()
        self.train_chart.setVisible(False)
        mg_layout.addWidget(self.train_chart)

        self.model_status = QLabel("")
        self.model_status.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px;")
        mg_layout.addWidget(self.model_status)

        model_btn_row = QHBoxLayout()
        self.view_models_btn = QPushButton("查看所有模型")
        self.view_models_btn.setObjectName("infoBtn")
        self.view_models_btn.setCursor(Qt.PointingHandCursor)
        model_btn_row.addWidget(self.view_models_btn)

        self.view_sessions_btn = QPushButton("训练历史")
        self.view_sessions_btn.setObjectName("infoBtn")
        self.view_sessions_btn.setCursor(Qt.PointingHandCursor)
        self.view_sessions_btn.clicked.connect(self._show_session_history)
        model_btn_row.addWidget(self.view_sessions_btn)

        mg_layout.addLayout(model_btn_row)

        layout.addWidget(model_group)

        # ━━━ 自对弈训练（手机 RL） ━━━
        selfplay_section = CollapsibleSection("自对弈训练（手机端 RL）")
        sp_layout = selfplay_section.content_layout()

        sp_info = QLabel(
            "连接手机后，通过 DQN 强化学习自动训练。\n"
            "可用上方 BC 模型热启动加速收敛。需先在「Agent 部署」中配置设备。"
        )
        sp_info.setWordWrap(True)
        sp_info.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        sp_layout.addWidget(sp_info)

        sp_form = QFormLayout()
        sp_form.setSpacing(4)

        self.sp_preset_combo = QComboBox()
        self.sp_preset_combo.addItems(["wzry - 王者荣耀", "fps - FPS 射击", "generic - 通用"])
        sp_form.addRow("游戏预设", self.sp_preset_combo)

        sp_bc_row = QHBoxLayout()
        self.sp_bc_model = QLineEdit()
        self.sp_bc_model.setPlaceholderText("BC 模型目录（可选，热启动）")
        sp_bc_row.addWidget(self.sp_bc_model, 1)
        self.sp_bc_browse_btn = QPushButton("...")
        self.sp_bc_browse_btn.setObjectName("browseBtn")
        self.sp_bc_browse_btn.setMaximumWidth(30)
        self.sp_bc_browse_btn.clicked.connect(self._browse_sp_bc_model)
        sp_bc_row.addWidget(self.sp_bc_browse_btn)
        sp_form.addRow("热启动模型", sp_bc_row)

        self.sp_fps_spin = QSpinBox()
        self.sp_fps_spin.setRange(1, 30)
        self.sp_fps_spin.setValue(5)
        sp_form.addRow("帧率 (FPS)", self.sp_fps_spin)

        self.sp_max_episodes = QSpinBox()
        self.sp_max_episodes.setRange(0, 99999)
        self.sp_max_episodes.setValue(0)
        self.sp_max_episodes.setSpecialValueText("无限制")
        sp_form.addRow("最大对局数", self.sp_max_episodes)

        sp_layout.addLayout(sp_form)

        sp_btn_row = QHBoxLayout()
        self.sp_start_btn = QPushButton("启动自对弈")
        self.sp_start_btn.setObjectName("startBtn")
        self.sp_start_btn.setCursor(Qt.PointingHandCursor)
        self.sp_start_btn.setMinimumHeight(36)
        sp_btn_row.addWidget(self.sp_start_btn)

        self.sp_stop_btn = QPushButton("停止")
        self.sp_stop_btn.setObjectName("stopBtn")
        self.sp_stop_btn.setCursor(Qt.PointingHandCursor)
        self.sp_stop_btn.setMinimumHeight(36)
        self.sp_stop_btn.setEnabled(False)
        sp_btn_row.addWidget(self.sp_stop_btn)
        sp_layout.addLayout(sp_btn_row)

        # 训练统计
        self.sp_stats_label = QLabel("")
        self.sp_stats_label.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px;")
        sp_layout.addWidget(self.sp_stats_label)

        layout.addWidget(selfplay_section)

        # ━━━ 日志 ━━━
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)

        # ━━━ 高级设置（折叠） ━━━
        more = CollapsibleSection("高级设置")
        mg = more.content_layout()

        mg.addWidget(QLabel("LLM 配置使用全局 LLM 设置"))

        train_form = QFormLayout()
        train_form.setSpacing(4)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000)
        self.epochs_spin.setValue(100)
        train_form.addRow("训练轮数", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.0005)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.001)
        train_form.addRow("学习率", self.lr_spin)

        self.rl_steps_spin = QSpinBox()
        self.rl_steps_spin.setRange(0, 50000)
        self.rl_steps_spin.setValue(2000)
        self.rl_steps_spin.setSingleStep(500)
        train_form.addRow("RL 步数", self.rl_steps_spin)

        mg.addLayout(train_form)

        layout.addWidget(more)
        layout.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll)

    def _on_learn_clicked(self):
        self.learn_from_recording_requested.emit()

    # ── 自对弈训练 ──

    def _browse_sp_bc_model(self):
        dlg = ModelBrowserDialog(self, base_dir="runs")
        if dlg.exec() == QDialog.Accepted and dlg.selected_path():
            self.sp_bc_model.setText(dlg.selected_path())

    def get_selfplay_config(self) -> dict:
        """获取自对弈训练配置。"""
        text = self.sp_preset_combo.currentText()
        preset_name = text.split(" - ")[0].strip()
        return {
            "preset": preset_name,
            "bc_model_dir": self.sp_bc_model.text().strip(),
            "fps": self.sp_fps_spin.value(),
            "max_episodes": self.sp_max_episodes.value(),
        }

    def set_selfplay_state(self, running: bool):
        self.sp_start_btn.setEnabled(not running)
        self.sp_stop_btn.setEnabled(running)
        if not running:
            self.sp_stats_label.setText("")

    def update_selfplay_stats(self, stats: dict):
        self.sp_stats_label.setText(
            f"对局: {stats.get('episodes', 0)} | "
            f"步数: {stats.get('total_steps', 0)} | "
            f"ε: {stats.get('epsilon', 1.0):.3f} | "
            f"奖励: {stats.get('avg_reward_10ep', 0):.1f} | "
            f"loss: {stats.get('avg_loss_100', 0):.6f}"
        )

    # ── 窗口管理 ──

    def _refresh_window_list(self):
        """刷新可见窗口列表。"""
        from ..data.game_recorder import GameRecorder
        titles = GameRecorder.list_windows()
        current = self.window_combo.currentText()
        self.window_combo.clear()
        self.window_combo.addItem("（全屏）")
        for t in titles:
            self.window_combo.addItem(t)
        # 恢复之前的选择
        idx = self.window_combo.findText(current)
        if idx >= 0:
            self.window_combo.setCurrentIndex(idx)

    def get_window_title(self) -> str:
        """获取选择的窗口标题，空字符串表示全屏。"""
        text = self.window_combo.currentText()
        if text == "（全屏）" or not text.strip():
            return ""
        return text.strip()

    # ── 录制管理 ──

    def set_recording_state(self, recording: bool):
        self.record_start_btn.setEnabled(not recording)
        self.record_stop_btn.setEnabled(recording)
        if recording:
            self.record_status.setText("录制中...")
        else:
            self.record_status.setText("")

    def add_recording(self, rec_dir: str, stats: dict | None = None):
        """添加一条录制到列表。"""
        name = Path(rec_dir).name
        label = name
        if stats:
            frames = stats.get("total_frames", 0)
            dur = stats.get("duration_sec", 0)
            label = f"{name} ({frames} 帧, {dur:.0f}s)"
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, rec_dir)
        item.setToolTip(rec_dir)
        self.recording_list.addItem(item)
        count = self.recording_list.count()
        self.recording_count_label.setText(f"{count} 个录制")

    def get_recording_dirs(self) -> list[str]:
        """获取所有录制目录。"""
        dirs = []
        for i in range(self.recording_list.count()):
            item = self.recording_list.item(i)
            dirs.append(item.data(Qt.UserRole))
        return dirs

    def _remove_selected_recordings(self):
        for item in self.recording_list.selectedItems():
            row = self.recording_list.row(item)
            self.recording_list.takeItem(row)
        count = self.recording_list.count()
        self.recording_count_label.setText(f"{count} 个录制")

    def _import_recording(self):
        folder = QFileDialog.getExistingDirectory(self, "选择录制目录")
        if not folder:
            return
        p = Path(folder)
        if not (p / "recording.mp4").exists() or not (p / "actions.jsonl").exists():
            QMessageBox.warning(self, "提示", "目录缺少 recording.mp4 或 actions.jsonl")
            return
        meta_path = p / "meta.json"
        stats = None
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
            except Exception:
                pass
        self.add_recording(folder, stats)

    # ── 场景管理方法 ──

    def init_scene_manager(self):
        from ..workshop.scene import SceneManager
        self._scene_manager = SceneManager()
        self._refresh_scene_list()

    def _refresh_scene_list(self):
        if not self._scene_manager:
            return
        current = self.scene_combo.currentText()
        self.scene_combo.blockSignals(True)
        self.scene_combo.clear()
        self.scene_combo.addItem("（选择场景或新建）")
        scenes = self._scene_manager.list_scenes()
        for s in scenes:
            status_icon = {"ready": "✓", "idle": "○", "training": "◉"}.get(s.status, "○")
            label = f"{status_icon} {s.display_name}"
            if s.best_val_acc > 0:
                label += f" (acc: {s.best_val_acc:.1%})"
            self.scene_combo.addItem(label, s.name)
        for i in range(self.scene_combo.count()):
            if self.scene_combo.itemData(i) == current or self.scene_combo.itemText(i) == current:
                self.scene_combo.setCurrentIndex(i)
                break
        self.scene_combo.blockSignals(False)

    def _on_scene_selected(self, text: str):
        if not self._scene_manager:
            return
        idx = self.scene_combo.currentIndex()
        name = self.scene_combo.itemData(idx)
        if not name:
            self._current_scene = None
            self._update_scene_ui(None)
            return
        scene = self._scene_manager.load(name)
        if scene:
            self._current_scene = scene
            self._update_scene_ui(scene)
            self.scene_changed.emit(name)

    def _update_scene_ui(self, scene):
        if scene is None:
            self.description_input.clear()
            self.knowledge_input.clear()
            self.scene_status.setText("")
            self.insight_text.clear()
            self.model_status.setText("")
            return

        self.description_input.setText(scene.display_name)

        self.knowledge_input.blockSignals(True)
        self.knowledge_input.setPlainText(scene.knowledge or "")
        self.knowledge_input.blockSignals(False)

        status_map = {
            "idle": "空闲", "analyzing": "分析中...",
            "annotating": "标注中...", "training": "训练中...",
            "ready": "模型就绪",
        }
        self.scene_status.setText(status_map.get(scene.status, scene.status))

        if scene.analysis_summary:
            lines = []
            if scene.scene_type:
                lines.append(f"场景: {scene.scene_type}")
            if scene.actions:
                lines.append(f"动作: {', '.join(scene.actions)}")
            if scene.analysis_summary:
                lines.append(f"\n{scene.analysis_summary}")
            self.insight_text.setPlainText("\n".join(lines))

        if scene.best_model_dir:
            self.model_status.setText(
                f"最佳模型: acc={scene.best_val_acc:.1%} | "
                f"训练 {scene.session_count} 次 | 标注 {scene.total_annotated} 条"
            )
        else:
            self.model_status.setText(f"训练 {scene.session_count} 次 | 标注 {scene.total_annotated} 条")

    def _new_scene(self):
        name, ok = QInputDialog.getText(
            self, "新建场景", "场景名称（如：王者荣耀5v5）:",
        )
        if not ok or not name.strip():
            return
        if not self._scene_manager:
            self.init_scene_manager()
        scene = self._scene_manager.create(name.strip(), display_name=name.strip())
        self._refresh_scene_list()
        for i in range(self.scene_combo.count()):
            if self.scene_combo.itemData(i) == scene.name:
                self.scene_combo.setCurrentIndex(i)
                break

    def _delete_scene(self):
        if not self._current_scene or not self._scene_manager:
            return
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定删除场景「{self._current_scene.display_name}」？\n（训练数据将移至回收站）",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._scene_manager.delete(self._current_scene.name)
            self._current_scene = None
            self._refresh_scene_list()
            self.scene_combo.setCurrentIndex(0)

    def _show_session_history(self):
        if not self._current_scene:
            QMessageBox.information(self, "提示", "请先选择场景")
            return
        sessions = self._current_scene.list_sessions()
        if not sessions:
            QMessageBox.information(self, "训练历史", "暂无训练记录")
            return
        lines = []
        for s in sessions[:10]:
            model_dir = s.get("model_dir", "")
            acc = s.get("metrics", {}).get("best_val_acc", 0)
            ann = s.get("annotated_count", 0)
            success = "✓" if s.get("success") else "✗"
            lines.append(f"{success} acc={acc:.1%} | 标注 {ann} 条 | {model_dir}")
        QMessageBox.information(
            self, f"训练历史 - {self._current_scene.display_name}",
            "\n".join(lines)
        )

    def _on_knowledge_changed(self):
        text = self.knowledge_input.toPlainText().strip()
        self.kb_char_label.setText(f"{len(text)} 字")
        if self._current_scene:
            self._current_scene.knowledge = text
            self._current_scene.save()

    def _import_knowledge(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入场景知识", "",
            "知识文件 (*.txt *.md *.json *.yaml *.yml);;所有文件 (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="gbk") as f:
                content = f.read()

        if not content.strip():
            QMessageBox.warning(self, "提示", "文件内容为空")
            return

        existing = self.knowledge_input.toPlainText().strip()
        if existing:
            box = QMessageBox(self)
            box.setWindowTitle("导入方式")
            box.setText(f"当前已有 {len(existing)} 字知识内容，如何处理？")
            btn_replace = box.addButton("替换", QMessageBox.ButtonRole.YesRole)
            btn_append = box.addButton("追加", QMessageBox.ButtonRole.NoRole)
            box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked == btn_append:
                content = existing + "\n\n" + content
            elif clicked != btn_replace:
                return

        self.knowledge_input.setPlainText(content.strip())

    @property
    def current_scene(self):
        return self._current_scene

    def set_learning_state(self, running: bool):
        self.learn_btn.setEnabled(not running)
        self.learn_btn.setVisible(not running)
        self.stop_learn_btn.setEnabled(running)
        self.stop_learn_btn.setVisible(running)

    def update_insight(self, insight_dict: dict):
        lines = []
        if insight_dict.get("scene_type"):
            lines.append(f"场景: {insight_dict['scene_type']}")
        if insight_dict.get("scene_description"):
            lines.append(f"描述: {insight_dict['scene_description']}")
        if insight_dict.get("suggested_actions"):
            lines.append(f"动作: {', '.join(insight_dict['suggested_actions'])}")
        if insight_dict.get("analysis_summary"):
            lines.append(f"\n{insight_dict['analysis_summary']}")
        self.insight_text.setPlainText("\n".join(lines))
        if self._current_scene:
            self._current_scene.update_from_insight(insight_dict)
