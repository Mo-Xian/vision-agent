"""训练工坊面板：数据与标注 + 训练与模型（2 Tab 精简布局）。"""

import json
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QLineEdit, QPushButton, QSpinBox,
    QDoubleSpinBox, QTextEdit, QFormLayout, QProgressBar,
    QScrollArea, QFrame,
)

from .styles import COLORS
from .widgets import CollapsibleSection


class TrainPanel(QTabWidget):
    """训练工坊面板，2 个 Tab：数据与标注、训练与模型。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_data_tab()
        self._build_train_tab()

    # ── Tab 1: 数据与标注 ──

    def _build_data_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ▸ 一键自动学习（核心功能，最显眼）
        auto_group = QGroupBox("一键自动学习")
        ag = QVBoxLayout(auto_group)
        ag.addWidget(QLabel(
            "输入场景描述，自动完成：搜索 → 检测 → 标注 → 训练 → 强化"
        ))
        self.auto_learn_btn = QPushButton("打开自动学习")
        self.auto_learn_btn.setObjectName("startBtn")
        self.auto_learn_btn.setCursor(Qt.PointingHandCursor)
        ag.addWidget(self.auto_learn_btn)
        layout.addWidget(auto_group)

        # ▸ LLM 标注
        anno_group = QGroupBox("LLM 标注")
        anno_layout = QVBoxLayout(anno_group)
        anno_layout.setSpacing(6)
        anno_layout.addWidget(QLabel("视频 → YOLO 检测 → LLM 判断动作 → 训练数据"))

        anno_btn_row = QHBoxLayout()
        self.auto_annotate_btn = QPushButton("开始 LLM 标注")
        self.auto_annotate_btn.setObjectName("purpleBtn")
        self.auto_annotate_btn.setCursor(Qt.PointingHandCursor)
        anno_btn_row.addWidget(self.auto_annotate_btn)
        self.view_annotation_btn = QPushButton("查看结果")
        self.view_annotation_btn.setObjectName("infoBtn")
        self.view_annotation_btn.setCursor(Qt.PointingHandCursor)
        anno_btn_row.addWidget(self.view_annotation_btn)
        anno_layout.addLayout(anno_btn_row)
        layout.addWidget(anno_group)

        # ▸ LLM 配置（折叠）
        llm_section = CollapsibleSection("LLM 配置（标注 / 自动学习 / 对话共用）")
        lg = llm_section.content_layout()

        llm_form = QFormLayout()
        llm_form.setSpacing(6)
        self.llm_provider_combo = QComboBox()
        llm_form.addRow("供应商", self.llm_provider_combo)
        self.llm_model_combo = QComboBox()
        self.llm_model_combo.setEditable(True)
        llm_form.addRow("模型", self.llm_model_combo)
        self.llm_api_key = QLineEdit()
        self.llm_api_key.setEchoMode(QLineEdit.Password)
        self.llm_api_key.setPlaceholderText("API Key 或留空用环境变量")
        llm_form.addRow("API Key", self.llm_api_key)
        self.llm_base_url = QLineEdit()
        self.llm_base_url.setPlaceholderText("留空用默认地址")
        llm_form.addRow("Base URL", self.llm_base_url)
        lg.addLayout(llm_form)

        self.llm_test_btn = QPushButton("测试连接")
        self.llm_test_btn.setObjectName("infoBtn")
        self.llm_test_btn.setCursor(Qt.PointingHandCursor)
        lg.addWidget(self.llm_test_btn)
        layout.addWidget(llm_section)

        # ▸ YOLO 模型（折叠）
        yolo_section = CollapsibleSection("YOLO 检测模型")
        yg = yolo_section.content_layout()

        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
        self.model_combo.setEditable(True)
        yg.addWidget(self.model_combo)

        yolo_btn_row = QHBoxLayout()
        self.load_model_btn = QPushButton("加载自定义")
        self.load_model_btn.setObjectName("browseBtn")
        yolo_btn_row.addWidget(self.load_model_btn)
        self.train_btn_open = QPushButton("训练 YOLO")
        self.train_btn_open.setObjectName("purpleBtn")
        self.train_btn_open.setCursor(Qt.PointingHandCursor)
        yolo_btn_row.addWidget(self.train_btn_open)
        yg.addLayout(yolo_btn_row)

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
        yg.addLayout(yolo_form)
        layout.addWidget(yolo_section)

        # ▸ 人工录制（折叠）
        rec_section = CollapsibleSection("人工操作录制")
        rg = rec_section.content_layout()
        rg.addWidget(QLabel("启动检测后录制键鼠操作 + YOLO 检测结果"))

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("保存目录"))
        self.rec_dir_input = QLineEdit("data/recordings")
        dir_row.addWidget(self.rec_dir_input)
        self.rec_browse_btn = QPushButton("...")
        self.rec_browse_btn.setObjectName("browseBtn")
        self.rec_browse_btn.setMaximumWidth(32)
        dir_row.addWidget(self.rec_browse_btn)
        rg.addLayout(dir_row)

        session_row = QHBoxLayout()
        session_row.addWidget(QLabel("会话名称"))
        self.rec_session_input = QLineEdit()
        self.rec_session_input.setPlaceholderText("留空自动生成")
        session_row.addWidget(self.rec_session_input)
        rg.addLayout(session_row)

        self.rec_start_btn = QPushButton("开始录制")
        self.rec_start_btn.setObjectName("stopBtn")
        self.rec_start_btn.setCursor(Qt.PointingHandCursor)
        self.rec_start_btn.setToolTip("需先在 Agent 模式启动检测")
        rg.addWidget(self.rec_start_btn)

        self.rec_status_label = QLabel("就绪（需先启动检测）")
        self.rec_status_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        rg.addWidget(self.rec_status_label)
        layout.addWidget(rec_section)

        layout.addStretch()
        scroll.setWidget(tab)
        self.addTab(scroll, "数据与标注")

    # ── Tab 2: 训练与模型 ──

    def _build_train_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ▸ 决策模型训练
        train_group = QGroupBox("决策模型训练")
        tg = QVBoxLayout(train_group)
        tg.setSpacing(6)

        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("数据目录"))
        self.dt_data_dir = QLineEdit("data/recordings")
        data_row.addWidget(self.dt_data_dir)
        self.dt_data_browse = QPushButton("...")
        self.dt_data_browse.setObjectName("browseBtn")
        self.dt_data_browse.setMaximumWidth(32)
        data_row.addWidget(self.dt_data_browse)
        tg.addLayout(data_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("输出目录"))
        self.dt_output_dir = QLineEdit("runs/decision/exp1")
        out_row.addWidget(self.dt_output_dir)
        tg.addLayout(out_row)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("模型"))
        self.dt_model_type = QComboBox()
        self.dt_model_type.addItems(["mlp", "rf"])
        param_row.addWidget(self.dt_model_type)
        param_row.addWidget(QLabel("轮数"))
        self.dt_epochs_spin = QSpinBox()
        self.dt_epochs_spin.setRange(10, 1000)
        self.dt_epochs_spin.setValue(100)
        param_row.addWidget(self.dt_epochs_spin)
        tg.addLayout(param_row)

        # 学习率放入折叠
        adv_section = CollapsibleSection("高级参数")
        adv_layout = adv_section.content_layout()
        lr_row = QHBoxLayout()
        lr_row.addWidget(QLabel("学习率"))
        self.dt_lr_spin = QDoubleSpinBox()
        self.dt_lr_spin.setRange(0.0001, 0.1)
        self.dt_lr_spin.setSingleStep(0.0005)
        self.dt_lr_spin.setDecimals(4)
        self.dt_lr_spin.setValue(0.001)
        lr_row.addWidget(self.dt_lr_spin)
        lr_row.addStretch()
        adv_layout.addLayout(lr_row)
        tg.addWidget(adv_section)

        dt_btn_row = QHBoxLayout()
        self.dt_preview_btn = QPushButton("预览数据")
        self.dt_preview_btn.setObjectName("infoBtn")
        self.dt_preview_btn.setCursor(Qt.PointingHandCursor)
        dt_btn_row.addWidget(self.dt_preview_btn)
        self.dt_train_btn = QPushButton("开始训练")
        self.dt_train_btn.setObjectName("startBtn")
        self.dt_train_btn.setCursor(Qt.PointingHandCursor)
        dt_btn_row.addWidget(self.dt_train_btn)
        self.dt_use_btn = QPushButton("应用到 Agent")
        self.dt_use_btn.setObjectName("purpleBtn")
        self.dt_use_btn.setCursor(Qt.PointingHandCursor)
        self.dt_use_btn.setToolTip("将训练好的模型设为 Agent 的决策引擎")
        self.dt_use_btn.setEnabled(False)
        dt_btn_row.addWidget(self.dt_use_btn)
        tg.addLayout(dt_btn_row)

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

        # ▸ 场景 Profile 管理
        profile_group = QGroupBox("场景 Profile")
        pg = QVBoxLayout(profile_group)
        pg.setSpacing(6)

        self.profile_combo = QComboBox()
        self.profile_combo.addItem("(无)")
        pg.addWidget(self.profile_combo)

        self.profile_info = QTextEdit()
        self.profile_info.setReadOnly(True)
        self.profile_info.setMaximumHeight(100)
        self.profile_info.setPlaceholderText("选择 Profile 查看详情")
        pg.addWidget(self.profile_info)

        profile_btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setObjectName("browseBtn")
        self.refresh_btn.setCursor(Qt.PointingHandCursor)
        profile_btn_row.addWidget(self.refresh_btn)
        self.export_btn = QPushButton("导出")
        self.export_btn.setObjectName("infoBtn")
        profile_btn_row.addWidget(self.export_btn)
        self.import_btn = QPushButton("导入")
        self.import_btn.setObjectName("browseBtn")
        profile_btn_row.addWidget(self.import_btn)
        pg.addLayout(profile_btn_row)

        layout.addWidget(profile_group)

        layout.addStretch()
        scroll.setWidget(tab)
        self.addTab(scroll, "训练与模型")
