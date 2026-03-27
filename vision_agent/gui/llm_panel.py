"""LLM 模型配置面板。"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QLineEdit, QPushButton,
    QFormLayout, QScrollArea, QFrame,
)

from .styles import COLORS


class LLMPanel(QWidget):
    """LLM 配置面板 — 供应商、模型、Key、Base URL。"""

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
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        # 说明
        hint = QLabel("标注、自动学习、对话控制共用此配置")
        hint.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        layout.addWidget(hint)

        # 配置表单
        group = QGroupBox("LLM 连接")
        form = QFormLayout(group)
        form.setSpacing(8)

        self.llm_provider_combo = QComboBox()
        form.addRow("供应商", self.llm_provider_combo)

        self.llm_model_combo = QComboBox()
        self.llm_model_combo.setEditable(True)
        form.addRow("模型", self.llm_model_combo)

        self.llm_api_key = QLineEdit()
        self.llm_api_key.setEchoMode(QLineEdit.Password)
        self.llm_api_key.setPlaceholderText("留空用环境变量")
        form.addRow("API Key", self.llm_api_key)

        self.llm_base_url = QLineEdit()
        self.llm_base_url.setPlaceholderText("留空用默认地址")
        form.addRow("Base URL", self.llm_base_url)

        layout.addWidget(group)

        # 按钮行
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.llm_test_btn = QPushButton("测试连接")
        self.llm_test_btn.setObjectName("infoBtn")
        self.llm_test_btn.setCursor(Qt.PointingHandCursor)
        self.llm_test_btn.setMinimumHeight(30)
        btn_row.addWidget(self.llm_test_btn)

        self.llm_save_btn = QPushButton("保存配置")
        self.llm_save_btn.setObjectName("startBtn")
        self.llm_save_btn.setCursor(Qt.PointingHandCursor)
        self.llm_save_btn.setMinimumHeight(30)
        btn_row.addWidget(self.llm_save_btn)

        layout.addLayout(btn_row)

        # 保存状态提示
        self.save_hint = QLabel("")
        self.save_hint.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px;")
        layout.addWidget(self.save_hint)

        layout.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll)
