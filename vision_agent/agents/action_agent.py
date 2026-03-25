"""带决策和工具调用能力的 Agent。"""

import logging
import threading
import time
from ..core.detector import DetectionResult
from ..core.state import StateManager, SceneState
from ..decision.base import DecisionEngine, Action, LoggingMixin
from ..tools.base import ToolRegistry, ToolResult
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ActionAgent(LoggingMixin, BaseAgent):
    """智能 Agent：感知 → 决策 → 执行 全链路。

    决策在独立线程中异步执行，不阻塞主检测循环。
    """

    def __init__(
        self,
        decision_engine: DecisionEngine,
        tool_registry: ToolRegistry,
        state_manager: StateManager | None = None,
        on_log: "callable | None" = None,
        on_action: "callable | None" = None,
    ):
        self._engine = decision_engine
        self._tools = tool_registry
        self._state = state_manager or StateManager()
        self._decision_thread: threading.Thread | None = None
        self._running = False
        self._pending_state = None
        self._stats = {"decisions": 0, "actions_executed": 0, "actions_failed": 0}
        self._on_log = on_log
        self._on_action = on_action

    def on_start(self):
        self._running = True
        self._engine.on_start()
        self._decision_thread = threading.Thread(target=self._decision_loop, daemon=True)
        self._decision_thread.start()
        logger.info("ActionAgent 启动")

    def on_stop(self):
        self._running = False
        self._engine.on_stop()
        if self._decision_thread:
            self._decision_thread.join(timeout=3)
        logger.info(f"ActionAgent 停止 | 统计: {self._stats}")

    def on_detection(self, result: DetectionResult):
        """接收检测结果，更新状态，触发异步决策。"""
        state = self._state.update(result)
        self._pending_state = (result, state)

    def _decision_loop(self):
        """决策线程：从状态中获取最新检测结果，调用决策引擎，执行动作。"""
        while self._running:
            state_data = getattr(self, '_pending_state', None)
            if state_data is None:
                time.sleep(0.01)
                continue

            result, state = state_data
            self._pending_state = None

            try:
                actions = self._engine.decide(result, state)
                if actions:
                    self._stats["decisions"] += 1
                    action_names = [a.tool_name for a in actions]
                    self._emit_log(f"[决策] {action_names} (reason: {actions[0].reason})")
                    self._execute_actions(actions)
            except Exception as e:
                logger.error(f"决策异常: {e}")
                self._emit_log(f"[错误] 决策异常: {e}")

    def _execute_actions(self, actions: list[Action]):
        """按优先级执行动作列表。"""
        sorted_actions = sorted(actions, key=lambda a: -a.priority)
        for action in sorted_actions:
            try:
                result = self._tools.execute(action.tool_name, **action.parameters)
                if result.success:
                    self._stats["actions_executed"] += 1
                    self._emit_log(f"[执行] {action.tool_name} -> 成功")
                    if self._on_action:
                        try:
                            self._on_action(action.tool_name, action.parameters, action.reason, action.target_bbox)
                        except Exception:
                            pass
                else:
                    self._stats["actions_failed"] += 1
                    self._emit_log(f"[失败] {action.tool_name} -> {result.error}")
            except Exception as e:
                self._stats["actions_failed"] += 1
                self._emit_log(f"[异常] {action.tool_name} -> {e}")

    @property
    def stats(self) -> dict:
        return self._stats.copy()
