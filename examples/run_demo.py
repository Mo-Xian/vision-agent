"""快速启动示例：屏幕捕获 + YOLO 检测 + DemoAgent。

用法:
    python examples/run_demo.py
"""

import logging
from vision_agent.sources import ScreenSource
from vision_agent.core import Detector, Visualizer
from vision_agent.core.pipeline import Pipeline
from vision_agent.server import WebSocketServer
from vision_agent.agents import DemoAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    source = ScreenSource(monitor=1, fps=30)
    detector = Detector(model="yolov8n.pt", confidence=0.5)
    visualizer = Visualizer(show_fps=True)
    ws_server = WebSocketServer(port=8765)
    agent = DemoAgent(track_class="person")

    pipeline = Pipeline(
        source=source,
        detector=detector,
        ws_server=ws_server,
        visualizer=visualizer,
        agents=[agent],
    )
    pipeline.run()


if __name__ == "__main__":
    main()
