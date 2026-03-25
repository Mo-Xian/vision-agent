"""训练工坊面板：极简布局，核心操作一目了然。"""

import json
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QLineEdit, QPushButton, QSpinBox,
    QDoubleSpinBox, QTextEdit, QFormLayout, QProgressBar,
    QScrollArea, QFrame,
)

from .styles import COLORS
from .widgets import CollapsibleSection


class TrainPanel(QWidget):
    """训练工坊面板 — 无 Tab，单页滚动布局。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

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

        # ━━━ 核心操作区 ━━━

        # 自动学习 — 最显眼
        self.auto_learn_btn = QPushButton("一键自动学习")
        self.auto_learn_btn.setObjectName("startBtn")
        self.auto_learn_btn.setCursor(Qt.PointingHandCursor)
        self.auto_learn_btn.setMinimumHeight(42)
        self.auto_learn_btn.setStyleSheet(
            self.auto_learn_btn.styleSheet() +
            "QPushButton#startBtn { font-size: 15px; }"
        )
        layout.addWidget(self.auto_learn_btn)

        # 标注 + 训练 — 并排
        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self.auto_annotate_btn = QPushButton("手动标注")
        self.auto_annotate_btn.setObjectName("purpleBtn")
        self.auto_annotate_btn.setCursor(Qt.PointingHandCursor)
        action_row.addWidget(self.auto_annotate_btn)
        self.view_annotation_btn = QPushButton("查看结果")
        self.view_annotation_btn.setObjectName("infoBtn")
        self.view_annotation_btn.setCursor(Qt.PointingHandCursor)
        action_row.addWidget(self.view_annotation_btn)
        layout.addLayout(action_row)

        # ━━━ 训练 ━━━
        train_group = QGroupBox("训练")
        tg = QVBoxLayout(train_group)
        tg.setSpacing(6)

        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("数据"))
        self.dt_data_dir = QLineEdit("data/recordings")
        data_row.addWidget(self.dt_data_dir)
        self.dt_data_browse = QPushButton("...")
        self.dt_data_browse.setObjectName("browseBtn")
        self.dt_data_browse.setMaximumWidth(32)
        data_row.addWidget(self.dt_data_browse)
        tg.addLayout(data_row)

        train_btn_row = QHBoxLayout()
        self.dt_preview_btn = QPushButton("预览")
        self.dt_preview_btn.setObjectName("infoBtn")
        self.dt_preview_btn.setCursor(Qt.PointingHandCursor)
        train_btn_row.addWidget(self.dt_preview_btn)
        self.dt_train_btn = QPushButton("开始训练")
        self.dt_train_btn.setObjectName("startBtn")
        self.dt_train_btn.setCursor(Qt.PointingHandCursor)
        train_btn_row.addWidget(self.dt_train_btn)
        self.dt_use_btn = QPushButton("应用到 Agent")
        self.dt_use_btn.setObjectName("purpleBtn")
        self.dt_use_btn.setCursor(Qt.PointingHandCursor)
        self.dt_use_btn.setToolTip("将训练好的模型设为 Agent 的决策引擎")
        self.dt_use_btn.setEnabled(False)
        train_btn_row.addWidget(self.dt_use_btn)
        tg.addLayout(train_btn_row)

        self.dt_progress = QProgressBar()
        self.dt_progress.setRange(0, 100)
        self.dt_progress.setValue(0)
        self.dt_progress.setVisible(False)
        tg.addWidget(self.dt_progress)

        self.dt_status_label = QLabel("")
        self.dt_status_label.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px;")
        tg.addWidget(self.dt_status_label)

        from .train_chart import TrainChart
        self.dt_chart = TrainChart()
        self.dt_chart.setVisible(False)
        tg.addWidget(self.dt_chart)

        layout.addWidget(train_group)

        # ━━━ Profile ━━━
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Profile"))
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("(无)")
        profile_row.addWidget(self.profile_combo, 1)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setObjectName("browseBtn")
        self.refresh_btn.setCursor(Qt.PointingHandCursor)
        profile_row.addWidget(self.refresh_btn)
        layout.addLayout(profile_row)

        self.profile_info = QTextEdit()
        self.profile_info.setReadOnly(True)
        self.profile_info.setMaximumHeight(80)
        self.profile_info.setPlaceholderText("选择 Profile 查看详情")
        layout.addWidget(self.profile_info)

        # ━━━ 更多设置（折叠） ━━━
        more = CollapsibleSection("更多设置")
        mg = more.content_layout()

        # YOLO 模型
        mg.addWidget(QLabel("── YOLO 模型 ──"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
        self.model_combo.setEditable(True)
        mg.addWidget(self.model_combo)
        yolo_btn_row = QHBoxLayout()
        self.load_model_btn = QPushButton("加载自定义")
        self.load_model_btn.setObjectName("browseBtn")
        yolo_btn_row.addWidget(self.load_model_btn)
        self.train_btn_open = QPushButton("训练 YOLO")
        self.train_btn_open.setObjectName("purpleBtn")
        self.train_btn_open.setCursor(Qt.PointingHandCursor)
        yolo_btn_row.addWidget(self.train_btn_open)
        mg.addLayout(yolo_btn_row)
        yolo_form = QFormLayout()
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.50)
        yolo_form.addRow("置信度", self.conf_spin)
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "416", "512", "640", "960", "1280"])
        self.imgsz_combo.setCurrentText("640")
        yolo_form.addRow("分辨率", self.imgsz_combo)
        mg.addLayout(yolo_form)

        # 训练高级参数
        mg.addWidget(QLabel("── 训练参数 ──"))
        train_adv = QFormLayout()
        train_adv.setSpacing(4)
        self.dt_output_dir = QLineEdit("runs/decision/exp1")
        train_adv.addRow("输出目录", self.dt_output_dir)
        self.dt_model_type = QComboBox()
        self.dt_model_type.addItems(["mlp", "rf"])
        train_adv.addRow("模型类型", self.dt_model_type)
        self.dt_epochs_spin = QSpinBox()
        self.dt_epochs_spin.setRange(10, 1000)
        self.dt_epochs_spin.setValue(100)
        train_adv.addRow("训练轮数", self.dt_epochs_spin)
        self.dt_lr_spin = QDoubleSpinBox()
        self.dt_lr_spin.setRange(0.0001, 0.1)
        self.dt_lr_spin.setSingleStep(0.0005)
        self.dt_lr_spin.setDecimals(4)
        self.dt_lr_spin.setValue(0.001)
        train_adv.addRow("学习率", self.dt_lr_spin)
        mg.addLayout(train_adv)

        # 人工录制
        mg.addWidget(QLabel("── 人工录制 ──"))
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("目录"))
        self.rec_dir_input = QLineEdit("data/recordings")
        dir_row.addWidget(self.rec_dir_input)
        self.rec_browse_btn = QPushButton("...")
        self.rec_browse_btn.setObjectName("browseBtn")
        self.rec_browse_btn.setMaximumWidth(32)
        dir_row.addWidget(self.rec_browse_btn)
        mg.addLayout(dir_row)
        self.rec_session_input = QLineEdit()
        self.rec_session_input.setPlaceholderText("会话名称（留空自动生成）")
        mg.addWidget(self.rec_session_input)
        self.rec_start_btn = QPushButton("开始录制")
        self.rec_start_btn.setObjectName("stopBtn")
        self.rec_start_btn.setCursor(Qt.PointingHandCursor)
        mg.addWidget(self.rec_start_btn)
        self.rec_status_label = QLabel("就绪")
        self.rec_status_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        mg.addWidget(self.rec_status_label)

        # Profile 导入导出
        mg.addWidget(QLabel("── Profile 导入导出 ──"))
        ie_row = QHBoxLayout()
        self.export_btn = QPushButton("导出")
        self.export_btn.setObjectName("infoBtn")
        ie_row.addWidget(self.export_btn)
        self.import_btn = QPushButton("导入")
        self.import_btn.setObjectName("browseBtn")
        ie_row.addWidget(self.import_btn)
        mg.addLayout(ie_row)

        layout.addWidget(more)
        layout.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll)
