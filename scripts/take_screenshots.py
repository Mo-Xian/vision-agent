"""自动截图脚本：启动 GUI 并截取各模式的界面截图。"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer


def main():
    app = QApplication(sys.argv)

    from vision_agent.gui.main_window import MainWindow
    win = MainWindow()
    win.resize(1200, 750)
    win.show()

    screenshot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    step = [0]

    def take_next():
        s = step[0]
        if s == 0:
            # 先切到训练模式
            win.mode_train_btn.click()
            step[0] = 1
            QTimer.singleShot(500, take_next)
        elif s == 1:
            win.grab().save(os.path.join(screenshot_dir, "mode_train.png"))
            print("Saved: mode_train.png")
            win.mode_agent_btn.click()
            step[0] = 2
            QTimer.singleShot(500, take_next)
        elif s == 2:
            win.grab().save(os.path.join(screenshot_dir, "mode_agent.png"))
            print("Saved: mode_agent.png")
            win.mode_llm_btn.click()
            step[0] = 3
            QTimer.singleShot(500, take_next)
        elif s == 3:
            win.grab().save(os.path.join(screenshot_dir, "mode_llm.png"))
            print("Saved: mode_llm.png")
            win.mode_agent_btn.click()
            step[0] = 4
            QTimer.singleShot(500, take_next)
        elif s == 4:
            agent_panel = win.agent_panel
            if agent_panel.count() > 1:
                agent_panel.setCurrentIndex(1)
            step[0] = 5
            QTimer.singleShot(500, take_next)
        elif s == 5:
            win.grab().save(os.path.join(screenshot_dir, "mode_chat.png"))
            print("Saved: mode_chat.png")
            print("All screenshots saved!")
            app.quit()

    QTimer.singleShot(1000, take_next)
    app.exec()


if __name__ == "__main__":
    main()
