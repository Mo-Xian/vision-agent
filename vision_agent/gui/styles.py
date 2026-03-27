"""统一 GUI 样式定义。

设计语言：
- 深色科技感主题，以深蓝/靛蓝为主色
- 强调层次感：背景 → 卡片 → 控件 三层递进
- 交互状态明确：hover/focus/disabled 都有视觉反馈
- 圆角统一：卡片 10px，控件 6px，按钮 6px
"""

# 色板
COLORS = {
    "bg_deep":      "#0a0e1a",      # 最深背景
    "bg_base":      "#0f1525",      # 主背景
    "bg_card":      "#151d30",      # 卡片/分组背景
    "bg_input":     "#1a2338",      # 输入框背景
    "bg_input_focus": "#1e2a42",    # 输入框聚焦背景
    "border":       "#253352",      # 默认边框
    "border_focus": "#4a7dff",      # 聚焦边框
    "border_hover": "#3a5a8a",      # 悬停边框
    "text":         "#e8ecf4",      # 主文字
    "text_secondary": "#8892a8",    # 次要文字
    "text_dim":     "#5a6478",      # 暗淡文字
    "accent":       "#4a7dff",      # 主强调色（蓝）
    "accent_hover": "#3a6de8",      # 强调色悬停
    "success":      "#2ecc71",      # 成功/启动（绿）
    "success_hover": "#27ae60",
    "danger":       "#e74c3c",      # 危险/停止（红）
    "danger_hover": "#c0392b",
    "warning":      "#f39c12",      # 警告（橙）
    "purple":       "#9b59b6",      # 紫色（训练/标注）
    "purple_hover": "#8e44ad",
    "info":         "#3498db",      # 信息（蓝）
    "info_hover":   "#2980b9",
    "disabled":     "#3a4252",      # 禁用
    "log_green":    "#6bcf7f",      # 日志文字
    "splitter":     "#253352",      # 分割线
    "tab_active":   "#151d30",      # 活跃 tab 背景
    "tab_inactive": "#0f1525",      # 非活跃 tab 背景
    "scrollbar":    "#253352",      # 滚动条
    "scrollbar_hover": "#3a5a8a",
}

# 主窗口样式
MAIN_STYLESHEET = f"""
/* ── 全局 ── */
QMainWindow {{ background-color: {COLORS['bg_base']}; }}
QWidget {{ font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif; background-color: transparent; }}
QScrollArea {{ background-color: {COLORS['bg_base']}; }}
QScrollArea > QWidget > QWidget {{ background-color: {COLORS['bg_base']}; }}

/* ── 分组框 ── */
QGroupBox {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 10px;
    padding: 8px 10px;
    padding-top: 24px;
    color: {COLORS['text']};
    font-weight: 600;
    font-size: 13px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 2px;
    padding: 2px 8px;
    font-size: 12px;
    font-weight: 700;
    color: {COLORS['accent']};
    background-color: {COLORS['bg_card']};
    border: none;
    border-radius: 3px;
}}

/* ── 标签 ── */
QLabel {{
    color: {COLORS['text_secondary']};
    font-size: 12px;
    padding: 0px;
}}

/* ── 输入控件 ── */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    padding: 3px 8px;
    min-height: 22px;
    font-size: 12px;
    selection-background-color: {COLORS['accent']};
}}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {COLORS['border_hover']};
}}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS['border_focus']};
    background-color: {COLORS['bg_input_focus']};
}}
QComboBox::drop-down {{
    border: none;
    width: 26px;
    subcontrol-position: center right;
    padding-right: 6px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {COLORS['text_secondary']};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    selection-background-color: {COLORS['accent']};
    padding: 4px;
    outline: none;
}}

/* ── 按钮通用 ── */
QPushButton {{
    border-radius: 5px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
    min-height: 24px;
    border: 1px solid transparent;
}}
QPushButton:disabled {{
    background-color: {COLORS['disabled']};
    color: {COLORS['text_dim']};
    border-color: transparent;
}}

/* 启动按钮 */
QPushButton#startBtn {{
    background-color: {COLORS['success']};
    color: white;
}}
QPushButton#startBtn:hover {{
    background-color: {COLORS['success_hover']};
}}

/* 停止按钮 */
QPushButton#stopBtn {{
    background-color: {COLORS['danger']};
    color: white;
}}
QPushButton#stopBtn:hover {{
    background-color: {COLORS['danger_hover']};
}}

/* 浏览按钮 */
QPushButton#browseBtn {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text_secondary']};
    border: 1px solid {COLORS['border']};
    padding: 3px 8px;
    min-height: 20px;
    font-size: 11px;
    font-weight: normal;
}}
QPushButton#browseBtn:hover {{
    border-color: {COLORS['border_hover']};
    color: {COLORS['text']};
    background-color: {COLORS['bg_input_focus']};
}}

/* 功能按钮（紫色） */
QPushButton#purpleBtn {{
    background-color: {COLORS['purple']};
    color: white;
}}
QPushButton#purpleBtn:hover {{
    background-color: {COLORS['purple_hover']};
}}

/* 信息按钮（蓝色） */
QPushButton#infoBtn {{
    background-color: {COLORS['info']};
    color: white;
}}
QPushButton#infoBtn:hover {{
    background-color: {COLORS['info_hover']};
}}

/* ── 文本编辑框 / 日志 ── */
QTextEdit {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['log_green']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
    font-size: 11px;
    padding: 6px;
    selection-background-color: {COLORS['accent']};
}}

/* ── 分割线 ── */
QSplitter::handle {{
    background-color: {COLORS['splitter']};
    width: 2px;
}}
QSplitter::handle:hover {{
    background-color: {COLORS['border_focus']};
}}

/* ── Tab ── */
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    border-top: none;
    background-color: {COLORS['bg_card']};
    border-radius: 0 0 8px 8px;
}}
QTabBar::tab {{
    background: {COLORS['tab_inactive']};
    color: {COLORS['text_secondary']};
    padding: 7px 16px;
    border: 1px solid {COLORS['border']};
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    font-size: 12px;
    font-weight: 500;
    margin-right: 2px;
    min-width: 50px;
}}
QTabBar::tab:hover {{
    color: {COLORS['text']};
    background: {COLORS['bg_card']};
}}
QTabBar::tab:selected {{
    background: {COLORS['tab_active']};
    color: {COLORS['text']};
    border-bottom: 2px solid {COLORS['accent']};
}}

/* ── 进度条 ── */
QProgressBar {{
    background-color: {COLORS['bg_input']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    text-align: center;
    color: {COLORS['text']};
    font-size: 11px;
    min-height: 18px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS['accent']}, stop:1 {COLORS['success']});
    border-radius: 5px;
}}

/* ── 滚动条 ── */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['scrollbar']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['scrollbar_hover']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {COLORS['scrollbar']};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {COLORS['scrollbar_hover']};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── 复选框 ── */
QCheckBox {{
    color: {COLORS['text_secondary']};
    font-size: 13px;
    spacing: 8px;
    padding: 4px 0;
}}
QCheckBox:hover {{
    color: {COLORS['text']};
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid {COLORS['border']};
    background-color: {COLORS['bg_input']};
}}
QCheckBox::indicator:checked {{
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
}}
QCheckBox::indicator:hover {{
    border-color: {COLORS['border_hover']};
}}

/* ── Tooltip ── */
QToolTip {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}

/* ── FormLayout ── */
QFormLayout {{ margin: 0; }}
"""

# 对话框样式（继承主样式，额外定义对话框背景）
DIALOG_STYLESHEET = MAIN_STYLESHEET + f"""
QDialog {{ background-color: {COLORS['bg_base']}; }}
"""
