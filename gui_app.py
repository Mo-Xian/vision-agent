"""Vision Agent GUI 启动入口。"""

import sys
from PySide6.QtWidgets import QApplication
from vision_agent.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Vision Agent")

    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
