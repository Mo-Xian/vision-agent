"""主流程管线：视频采集 → 检测 → 状态更新 → 决策 → 执行 → 可视化。"""

import logging
import time
from ..sources.base import BaseSource
from ..agents.base import BaseAgent
from .detector import Detector
from .model_manager import ModelManager
from .visualizer import Visualizer
from .state import StateManager
from ..server.ws_server import WebSocketServer

logger = logging.getLogger(__name__)


class Pipeline:
    """将视频源、检测器、状态管理、Agent、WebSocket、可视化串联的主流程。"""

    def __init__(self, source: BaseSource, detector: Detector,
                 ws_server: WebSocketServer | None = None,
                 visualizer: Visualizer | None = None,
                 agents: list[BaseAgent] | None = None,
                 state_manager: StateManager | None = None,
                 model_manager: ModelManager | None = None):
        self.source = source
        self.detector = detector
        self.ws_server = ws_server
        self.visualizer = visualizer
        self.agents = agents or []
        self.state_manager = state_manager
        self.model_manager = model_manager
        self._running = False

    def run(self):
        """启动主循环。"""
        self._running = True

        # 启动各组件
        self.source.start()
        if self.ws_server:
            self.ws_server.start()
        for agent in self.agents:
            agent.on_start()

        logger.info("Pipeline 启动")

        try:
            while self._running:
                frame = self.source.read()
                if frame is None:
                    logger.info("视频源结束")
                    break

                # 如果有 ModelManager，使用其当前活跃模型
                detector = self.detector
                if self.model_manager and self.model_manager.current:
                    detector = self.model_manager.current

                # 检测
                result = detector.detect(frame)

                # 状态更新
                if self.state_manager:
                    self.state_manager.update(result)

                # WebSocket 推送
                if self.ws_server:
                    self.ws_server.broadcast(result)

                # Agent 处理
                for agent in self.agents:
                    agent.on_detection(result)

                # 可视化
                if self.visualizer:
                    display = self.visualizer.draw(frame, result)
                    if not self.visualizer.show(display):
                        logger.info("用户关闭窗口")
                        break

        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            self.stop()

    def switch_model(self, name: str) -> None:
        """运行时切换检测模型。"""
        if not self.model_manager:
            raise RuntimeError("未配置 ModelManager，无法切换模型")
        self.model_manager.switch(name)
        logger.info(f"Pipeline 模型切换为: {name}")

    def stop(self):
        """停止所有组件。"""
        self._running = False
        for agent in self.agents:
            agent.on_stop()
        self.source.stop()
        if self.ws_server:
            self.ws_server.stop()
        if self.visualizer:
            self.visualizer.close()
        logger.info("Pipeline 停止")
