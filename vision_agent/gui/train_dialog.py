"""训练对话框：配置数据集和训练参数，一键训练。"""

from pathlib import Path
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QFileDialog, QTextEdit, QProgressBar, QMessageBox, QWidget,
)

from ..core.trainer import TrainConfig
from .train_worker import TrainWorker

from .styles import DIALOG_STYLESHEET


class TrainDialog(QDialog):
    """训练配置和执行对话框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("训练自定义模型")
        self.setMinimumSize(650, 700)
        self.resize(700, 750)
        self.setStyleSheet(DIALOG_STYLESHEET)

        self._worker: TrainWorker | None = None
        self._best_model_path: str | None = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # === 数据集配置 ===
        data_group = QGroupBox("数据集")
        dg = QVBoxLayout(data_group)
        dg.setSpacing(10)

        # 方式一：直接选择 data.yaml
        dg.addWidget(QLabel("数据集 YAML 配置文件"))
        yaml_row = QHBoxLayout()
        self.yaml_input = QLineEdit()
        self.yaml_input.setPlaceholderText("选择 data.yaml（YOLO 格式）")
        yaml_row.addWidget(self.yaml_input)
        yaml_browse = QPushButton("浏览")
        yaml_browse.setObjectName("browseBtn")
        yaml_browse.clicked.connect(self._browse_yaml)
        yaml_row.addWidget(yaml_browse)
        dg.addLayout(yaml_row)

        # 分隔
        sep_label = QLabel("—— 或手动指定目录 ——")
        sep_label.setAlignment(Qt.AlignCenter)
        sep_label.setStyleSheet("color: #666; font-size: 12px;")
        dg.addWidget(sep_label)

        # 方式二：手动配置
        train_row = QHBoxLayout()
        train_row.addWidget(QLabel("训练集目录"))
        self.train_dir_input = QLineEdit()
        self.train_dir_input.setPlaceholderText("包含 images/ 和 labels/ 的目录")
        train_row.addWidget(self.train_dir_input)
        train_browse = QPushButton("浏览")
        train_browse.setObjectName("browseBtn")
        train_browse.clicked.connect(lambda: self._browse_dir(self.train_dir_input))
        train_row.addWidget(train_browse)
        dg.addLayout(train_row)

        val_row = QHBoxLayout()
        val_row.addWidget(QLabel("验证集目录"))
        self.val_dir_input = QLineEdit()
        self.val_dir_input.setPlaceholderText("包含 images/ 和 labels/ 的目录")
        val_row.addWidget(self.val_dir_input)
        val_browse = QPushButton("浏览")
        val_browse.setObjectName("browseBtn")
        val_browse.clicked.connect(lambda: self._browse_dir(self.val_dir_input))
        val_row.addWidget(val_browse)
        dg.addLayout(val_row)

        cls_row = QHBoxLayout()
        cls_row.addWidget(QLabel("类别名称"))
        self.classes_input = QLineEdit()
        self.classes_input.setPlaceholderText("逗号分隔，如: person,car,dog")
        cls_row.addWidget(self.classes_input)
        dg.addLayout(cls_row)

        layout.addWidget(data_group)

        # === 训练参数 ===
        param_group = QGroupBox("训练参数")
        pg = QVBoxLayout(param_group)
        pg.setSpacing(10)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("基础模型"))
        self.base_model_combo = QComboBox()
        self.base_model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
        self.base_model_combo.setEditable(True)
        row1.addWidget(self.base_model_combo)
        pg.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("训练轮数"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        row2.addWidget(self.epochs_spin)

        row2.addWidget(QLabel("批大小"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        row2.addWidget(self.batch_spin)
        pg.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("推理分辨率"))
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "416", "512", "640", "960"])
        self.imgsz_combo.setCurrentText("640")
        row3.addWidget(self.imgsz_combo)

        row3.addWidget(QLabel("学习率"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.01)
        row3.addWidget(self.lr_spin)
        pg.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("设备"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["自动", "cpu", "0", "0,1"])
        row4.addWidget(self.device_combo)

        row4.addWidget(QLabel("早停耐心"))
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 100)
        self.patience_spin.setValue(10)
        row4.addWidget(self.patience_spin)
        pg.addLayout(row4)

        layout.addWidget(param_group)

        # === 操作按钮 ===
        btn_row = QHBoxLayout()
        self.train_btn = QPushButton("▶ 开始训练")
        self.train_btn.setObjectName("startBtn")
        self.train_btn.clicked.connect(self._start_train)
        btn_row.addWidget(self.train_btn)

        self.use_model_btn = QPushButton("使用训练好的模型")
        self.use_model_btn.setObjectName("purpleBtn")
        self.use_model_btn.setEnabled(False)
        self.use_model_btn.clicked.connect(self._use_model)
        btn_row.addWidget(self.use_model_btn)
        layout.addLayout(btn_row)

        # === 进度和日志 ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        log_group = QGroupBox("训练日志")
        lg = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        lg.addWidget(self.log_text)
        layout.addWidget(log_group)

    def _browse_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据集 YAML", "",
            "YAML 文件 (*.yaml *.yml);;所有文件 (*)"
        )
        if path:
            self.yaml_input.setText(path)

    def _browse_dir(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "选择目录")
        if path:
            line_edit.setText(path)

    def _build_data_yaml(self) -> str | None:
        """构建或返回 data.yaml 路径。"""
        yaml_path = self.yaml_input.text().strip()
        if yaml_path:
            if not Path(yaml_path).exists():
                QMessageBox.warning(self, "提示", f"YAML 文件不存在: {yaml_path}")
                return None
            return yaml_path

        # 手动配置模式
        train_dir = self.train_dir_input.text().strip()
        val_dir = self.val_dir_input.text().strip()
        classes_text = self.classes_input.text().strip()

        if not train_dir or not val_dir or not classes_text:
            QMessageBox.warning(self, "提示", "请选择数据集 YAML，或填写训练集/验证集目录和类别名称")
            return None

        if not Path(train_dir).exists():
            QMessageBox.warning(self, "提示", f"训练集目录不存在: {train_dir}")
            return None
        if not Path(val_dir).exists():
            QMessageBox.warning(self, "提示", f"验证集目录不存在: {val_dir}")
            return None

        class_names = [c.strip() for c in classes_text.split(",") if c.strip()]
        if not class_names:
            QMessageBox.warning(self, "提示", "请输入至少一个类别名称")
            return None

        # 生成 data.yaml
        from ..core.trainer import Trainer
        save_path = str(Path(train_dir).parent / "data.yaml")
        Trainer.create_data_yaml(save_path, train_dir, val_dir, class_names)
        self._log(f"已生成数据集配置: {save_path}")
        return save_path

    @Slot()
    def _start_train(self):
        data_yaml = self._build_data_yaml()
        if not data_yaml:
            return

        device = self.device_combo.currentText()
        if device == "自动":
            device = None

        config = TrainConfig(
            data_yaml=data_yaml,
            base_model=self.base_model_combo.currentText(),
            epochs=self.epochs_spin.value(),
            imgsz=int(self.imgsz_combo.currentText()),
            batch=self.batch_spin.value(),
            device=device,
            patience=self.patience_spin.value(),
            lr0=self.lr_spin.value(),
        )

        self.log_text.clear()
        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)

        self._worker = TrainWorker(config)
        self._worker.log_message.connect(self._log)
        self._worker.finished_ok.connect(self._on_train_ok)
        self._worker.finished_err.connect(self._on_train_err)
        self._worker.start()

    @Slot(str)
    def _on_train_ok(self, best_path: str):
        self._best_model_path = best_path
        self.train_btn.setEnabled(True)
        self.use_model_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._log(f"\n模型已保存到: {best_path}")
        self._log("可以点击「使用训练好的模型」将其应用到检测中")

    @Slot(str)
    def _on_train_err(self, error: str):
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "训练失败", error)

    @Slot()
    def _use_model(self):
        if self._best_model_path:
            self.accept()

    def get_model_path(self) -> str | None:
        """对话框关闭后获取训练好的模型路径。"""
        return self._best_model_path

    def _log(self, text: str):
        self.log_text.append(text)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
