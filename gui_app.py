"""Vision Agent GUI 启动入口。"""

import sys
import os
import traceback


def _is_frozen():
    return getattr(sys, 'frozen', False)


def _get_exe_dir():
    """EXE 所在目录（可写，用于模型下载、日志等）。"""
    if _is_frozen():
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _get_data_dir():
    """数据文件目录（config.yaml, profiles/ 等）。"""
    if _is_frozen():
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def _show_error(title, msg):
    """尝试用 Qt 对话框显示错误，失败则写日志文件。"""
    log_path = os.path.join(_get_exe_dir(), "crash.log")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(msg)
    except Exception:
        pass

    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        box = QMessageBox()
        box.setWindowTitle(title)
        box.setText(msg[:2000])
        box.setIcon(QMessageBox.Icon.Critical)
        box.exec()
    except Exception:
        print(msg, file=sys.stderr)


def main():
    data_dir = _get_data_dir()
    exe_dir = _get_exe_dir()

    # 工作目录设为数据目录（config.yaml / profiles 在这里）
    os.chdir(data_dir)

    # 设置环境变量让 YOLO 模型下载到 EXE 所在目录（可写）
    if _is_frozen():
        os.environ.setdefault("YOLO_CONFIG_DIR", exe_dir)

    from PySide6.QtWidgets import QApplication
    from vision_agent.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Vision Agent")

    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _show_error("Vision Agent 启动失败", traceback.format_exc())
        sys.exit(1)
