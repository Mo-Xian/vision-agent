import logging
import time
from .base import DecisionEngine, Action
from ..core.detector import DetectionResult
from ..core.state import SceneState

logger = logging.getLogger(__name__)


class HierarchicalEngine(DecisionEngine):
    """分层决策引擎。

    三层结构：
    - 战略层 (strategy): 每 N 秒决策一次，决定高层目标（推塔/打团/刷野）
    - 战术层 (tactic): 每秒决策，在战略目标下选择战术（进攻/撤退/绕后）
    - 操作层 (micro): 每帧决策，执行具体操作（技能/走位/普攻）

    每层可以是任意 DecisionEngine 实例（或 None 跳过该层）。
    上层的决策结果作为下层的 context 传入（通过 state.custom_data）。
    """

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
        self._on_log = None

    def set_log_callback(self, callback):
        self._on_log = callback

    def decide(self, result: DetectionResult, state: SceneState) -> list[Action]:
        now = time.time()

        # 战略层（低频）
        if self._strategy and (now - self._last_strategy_time >= self._strategy_interval):
            self._last_strategy_time = now
            state.custom_data["layer"] = "strategy"
            strategy_actions = self._strategy.decide(result, state)
            if strategy_actions:
                self._current_strategy = strategy_actions[0].reason or strategy_actions[0].tool_name
                self._log(f"[战略] {self._current_strategy}")

        # 战术层（中频）
        if self._tactic and (now - self._last_tactic_time >= self._tactic_interval):
            self._last_tactic_time = now
            state.custom_data["layer"] = "tactic"
            state.custom_data["strategy_goal"] = self._current_strategy
            tactic_actions = self._tactic.decide(result, state)
            if tactic_actions:
                self._current_tactic = tactic_actions[0].reason or tactic_actions[0].tool_name
                self._log(f"[战术] {self._current_tactic} (战略: {self._current_strategy})")

        # 操作层（每帧）
        state.custom_data["layer"] = "micro"
        state.custom_data["strategy_goal"] = self._current_strategy
        state.custom_data["tactic_plan"] = self._current_tactic

        if self._micro:
            return self._micro.decide(result, state)

        return []

    def configure(self, **kwargs):
        if "strategy_interval" in kwargs:
            self._strategy_interval = kwargs["strategy_interval"]
        if "tactic_interval" in kwargs:
            self._tactic_interval = kwargs["tactic_interval"]
        # 支持运行时替换各层引擎
        if "strategy" in kwargs:
            self._strategy = kwargs["strategy"]
        if "tactic" in kwargs:
            self._tactic = kwargs["tactic"]
        if "micro" in kwargs:
            self._micro = kwargs["micro"]

    def on_start(self):
        for engine in [self._strategy, self._tactic, self._micro]:
            if engine:
                engine.on_start()

    def on_stop(self):
        for engine in [self._strategy, self._tactic, self._micro]:
            if engine:
                engine.on_stop()

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass
