"""Vision Agent GUI 启动入口。"""

import sys
import os
import traceback


def _show_error(title, msg):
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash.log")
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
