"""可复用 GUI 小组件。"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton

from .styles import COLORS


class CollapsibleSection(QWidget):
    """可折叠区域：点击标题行展开/收起内容。"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 2)
        root.setSpacing(0)

        self._toggle = QPushButton(f"\u25b6  {title}")
        self._toggle.setCheckable(True)
        self._toggle.setCursor(Qt.PointingHandCursor)
        self._toggle.setStyleSheet(f"""
            QPushButton {{
                text-align: left;
                background: transparent;
                color: {COLORS['text_secondary']};
                border: none;
                border-bottom: 1px solid {COLORS['border']};
                padding: 6px 4px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                color: {COLORS['text']};
            }}
            QPushButton:checked {{
                color: {COLORS['accent']};
                border-bottom-color: {COLORS['accent']};
            }}
        """)
        self._toggle.toggled.connect(self._on_toggle)
        root.addWidget(self._toggle)

        self._content = QWidget()
        self._content.setVisible(False)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(4, 6, 0, 6)
        self._content_layout.setSpacing(6)
        root.addWidget(self._content)

    def _on_toggle(self, checked: bool):
        self._content.setVisible(checked)
        arrow = "\u25bc" if checked else "\u25b6"
        self._toggle.setText(f"{arrow}  {self._title}")

    def content_layout(self) -> QVBoxLayout:
        """返回内容区布局，向其中添加子控件。"""
        return self._content_layout

    def set_expanded(self, expanded: bool):
        """程序化设置展开/收起。"""
        self._toggle.setChecked(expanded)
