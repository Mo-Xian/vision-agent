"""基于规则的决策引擎，零延迟，适合实时场景。"""

import logging
from typing import Callable
from ..core.detector import DetectionResult
from ..core.state import SceneState
from .base import Action, DecisionEngine

logger = logging.getLogger(__name__)

# 规则函数签名: (result, state) -> list[Action] | Action | None
RuleFunc = Callable[[DetectionResult, SceneState], list[Action] | Action | None]


class RuleEngine(DecisionEngine):
    """规则引擎：按优先级顺序执行规则，返回第一个匹配的动作。"""

    def __init__(self, first_match: bool = True):
        """
        Args:
            first_match: True=返回第一个匹配的规则结果, False=收集所有规则结果
        """
        self._rules: list[tuple[str, RuleFunc]] = []
        self._first_match = first_match

    def add_rule(self, name: str, rule: RuleFunc) -> None:
        """添加一条规则。先添加的优先级更高。"""
        self._rules.append((name, rule))
        logger.info(f"添加规则: {name}")

    def decide(self, result: DetectionResult, state: SceneState) -> list[Action]:
        all_actions = []
        for name, rule in self._rules:
            try:
                output = rule(result, state)
                if output is None:
                    continue
                actions = output if isinstance(output, list) else [output]
                for a in actions:
                    a.reason = a.reason or f"rule:{name}"
                if self._first_match:
                    return actions
                all_actions.extend(actions)
            except Exception as e:
                logger.error(f"规则 '{name}' 执行异常: {e}")
        return all_actions
