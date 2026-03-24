"""主流程管线：视频采集 → 检测 → 状态更新 → 决策 → 执行 → 可视化。"""

import logging
from ..sources.base import BaseSource
from ..agents.base import BaseAgent
from .detector import Detector
from .model_manager import ModelManager
from .visualizer import Visualizer
from .state import StateManager
from ..server.ws_server import WebSocketServer

logger = logging.getLogger(__name__)


class Pipeline:
    """将视频源、检测器、状态管理、Agent、WebSocket、可视化串联的主流程。

    支持运行时热切换决策引擎和 AutoPilot 集成。
    """

    def __init__(self, source: BaseSource, detector: Detector,
                 ws_server: WebSocketServer | None = None,
                 visualizer: Visualizer | None = None,
                 agents: list[BaseAgent] | None = None,
                 state_manager: StateManager | None = None,
                 model_manager: ModelManager | None = None,
                 auto_pilot=None,
                 roi_extractor=None,
                 on_frame_callback=None):
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
        self._running = False
        self._decision_engine = None

    @property
    def decision_engine(self):
        return self._decision_engine

    @decision_engine.setter
    def decision_engine(self, engine):
        """运行时热切换决策引擎。"""
        old = self._decision_engine
        if old:
            try:
                old.on_stop()
            except Exception:
                pass
        self._decision_engine = engine
        if engine:
            try:
                engine.on_start()
            except Exception:
                pass
        logger.info(f"决策引擎切换: {type(old).__name__ if old else 'None'} → {type(engine).__name__ if engine else 'None'}")

    def run(self):
        """启动主循环。"""
        self._running = True

        # 启动各组件
        self.source.start()
        if self.ws_server:
            self.ws_server.start()
        for agent in self.agents:
            agent.on_start()
        if self.auto_pilot:
            self.auto_pilot.start()
        if self._decision_engine:
            self._decision_engine.on_start()

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

                # ROI 特征提取
                roi_features = None
                if self.roi_extractor:
                    roi_features = self.roi_extractor.extract(frame)

                # 状态更新（优先使用增强状态）
                state = None
                if self.state_manager:
                    scene_name = ""
                    if self.auto_pilot:
                        scene_name = self.auto_pilot.current_scene
                    if roi_features or scene_name:
                        state = self.state_manager.update_enhanced(
                            result, roi_features=roi_features, scene_name=scene_name)
                    else:
                        state = self.state_manager.update(result)

                # AutoPilot：场景分类 + 自动训练触发
                if self.auto_pilot:
                    self.auto_pilot.on_frame(frame, result)
                    # 如果 AutoPilot 有对应场景的引擎，自动使用
                    ap_engine = self.auto_pilot.get_engine()
                    if ap_engine and ap_engine is not self._decision_engine:
                        self.decision_engine = ap_engine

                # WebSocket 推送
                if self.ws_server:
                    self.ws_server.broadcast(result)

                # 独立决策引擎（AutoPilot 热加载的引擎走这里）
                if self._decision_engine and state:
                    try:
                        actions = self._decision_engine.decide(result, state)
                        if actions:
                            self._execute_standalone_actions(actions)
                    except Exception as e:
                        logger.error(f"决策引擎异常: {e}")

                # Agent 处理
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

        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            self.stop()

    def _execute_standalone_actions(self, actions):
        """执行独立决策引擎产生的动作（通过已注册的 Agent 中的 ActionAgent）。"""
        from ..agents.action_agent import ActionAgent
        for agent in self.agents:
            if isinstance(agent, ActionAgent):
                agent._execute_actions(actions)
                return
        # 没有 ActionAgent 时仅记录
        action_names = [a.tool_name for a in actions]
        logger.warning(f"决策引擎产生动作 {action_names}，但无 ActionAgent 可执行")

    def switch_model(self, name: str) -> None:
        """运行时切换检测模型。"""
        if not self.model_manager:
            raise RuntimeError("未配置 ModelManager，无法切换模型")
        self.model_manager.switch(name)
        logger.info(f"Pipeline 模型切换为: {name}")

    def stop(self):
        """停止所有组件。"""
        self._running = False
        if self._decision_engine:
            try:
                self._decision_engine.on_stop()
            except Exception:
                pass
            self._decision_engine = None
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
