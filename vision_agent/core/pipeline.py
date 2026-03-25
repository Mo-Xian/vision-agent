"""主流程管线：视频采集 → 检测 → 状态更新 → Agent → 可视化。"""

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
    """将视频源、检测器、Agent、WebSocket、可视化串联的主流程。"""

    def __init__(self, source: BaseSource, detector: Detector,
                 ws_server: WebSocketServer | None = None,
                 visualizer: Visualizer | None = None,
                 agents: list[BaseAgent] | None = None,
                 state_manager: StateManager | None = None,
                 model_manager: ModelManager | None = None,
                 auto_pilot=None,
                 roi_extractor=None,
                 on_frame_callback=None,
                 target_fps: float = 0):
        self.source = source
        self.detector = detector
        self.ws_server = ws_server
        self.visualizer = visualizer
        self.agents = agents or []
        self.state_manager = state_manager
        self.model_manager = model_manager
        self.auto_pilot = auto_pilot
        self.roi_extractor = roi_extractor
        self.on_frame_callback = on_frame_callback
        self.target_fps = target_fps
        self._running = False

    def run(self):
        """启动主循环。"""
        self._running = True

        self.source.start()
        if self.ws_server:
            self.ws_server.start()
        for agent in self.agents:
            agent.on_start()
        if self.auto_pilot:
            self.auto_pilot.start()

        logger.info("Pipeline 启动")

        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0

        try:
            while self._running:
                frame_start = time.monotonic()

                frame = self.source.read()
                if frame is None:
                    logger.info("视频源结束")
                    break

                detector = self.detector
                if self.model_manager and self.model_manager.current:
                    detector = self.model_manager.current

                result = detector.detect(frame)

                # ROI 特征提取
                roi_features = None
                if self.roi_extractor:
                    roi_features = self.roi_extractor.extract(frame)

                # AutoPilot：场景分类 + 自动训练触发
                if self.auto_pilot:
                    self.auto_pilot.on_frame(frame, result)

                # WebSocket 推送检测结果
                if self.ws_server:
                    self.ws_server.broadcast(result)

                # Agent 处理（决策通过 ActionAgent 统一执行）
                for agent in self.agents:
                    agent.on_detection(result)

                # 帧回调（GUI 用）
                if self.on_frame_callback:
                    self.on_frame_callback(frame, result)

                # 可视化
                if self.visualizer:
                    display = self.visualizer.draw(frame, result)
                    if not self.visualizer.show(display):
                        logger.info("用户关闭窗口")
                        break

                # 帧率限制
                if frame_interval > 0:
                    elapsed = time.monotonic() - frame_start
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)

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
        if self.auto_pilot:
            self.auto_pilot.stop()
        logger.info("Pipeline 停止")
