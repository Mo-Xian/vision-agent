import time
from .base import DecisionEngine, Action, LoggingMixin
from ..core.detector import DetectionResult
from ..core.state import SceneState


class HierarchicalEngine(LoggingMixin, DecisionEngine):
    """分层决策引擎：战略层(N秒) → 战术层(1秒) → 操作层(每帧)。"""

    def __init__(
        self,
        strategy: DecisionEngine | None = None,
        tactic: DecisionEngine | None = None,
        micro: DecisionEngine | None = None,
        strategy_interval: float = 5.0,
        tactic_interval: float = 1.0,
    ):
        self._strategy = strategy
        self._tactic = tactic
        self._micro = micro
        self._strategy_interval = strategy_interval
        self._tactic_interval = tactic_interval

        self._current_strategy: str = "idle"
        self._current_tactic: str = "idle"
        self._last_strategy_time: float = 0
        self._last_tactic_time: float = 0

    def decide(self, result: DetectionResult, state: SceneState) -> list[Action]:
        now = time.time()

        if self._strategy and (now - self._last_strategy_time >= self._strategy_interval):
            self._last_strategy_time = now
            state.custom_data["layer"] = "strategy"
            strategy_actions = self._strategy.decide(result, state)
            if strategy_actions:
                self._current_strategy = strategy_actions[0].reason or strategy_actions[0].tool_name
                self._emit_log(f"[战略] {self._current_strategy}")

        if self._tactic and (now - self._last_tactic_time >= self._tactic_interval):
            self._last_tactic_time = now
            state.custom_data["layer"] = "tactic"
            state.custom_data["strategy_goal"] = self._current_strategy
            tactic_actions = self._tactic.decide(result, state)
            if tactic_actions:
                self._current_tactic = tactic_actions[0].reason or tactic_actions[0].tool_name
                self._emit_log(f"[战术] {self._current_tactic} (战略: {self._current_strategy})")

        state.custom_data["layer"] = "micro"
        state.custom_data["strategy_goal"] = self._current_strategy
        state.custom_data["tactic_plan"] = self._current_tactic

        if self._micro:
            return self._micro.decide(result, state)
        return []

    def configure(self, **kwargs):
        for key in ("strategy_interval", "tactic_interval", "strategy", "tactic", "micro"):
            if key in kwargs:
                setattr(self, f"_{key}", kwargs[key])

    def on_start(self):
        for engine in [self._strategy, self._tactic, self._micro]:
            if engine:
                engine.on_start()

    def on_stop(self):
        for engine in [self._strategy, self._tactic, self._micro]:
            if engine:
                engine.on_stop()
