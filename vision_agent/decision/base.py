"""决策引擎基类和动作数据结构。"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Action:
    """决策引擎输出的动作指令。"""
    name: str                   # 动作名（如 attack, defend, idle）
    parameters: dict = field(default_factory=dict)
    reason: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "reason": self.reason,
            "confidence": self.confidence,
        }


class LoggingMixin:
    """统一的日志回调 mixin。"""

    _on_log: "callable | None" = None

    def set_log_callback(self, callback) -> None:
        self._on_log = callback

    def _emit_log(self, msg: str):
        logging.getLogger(type(self).__module__).info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass


class DecisionEngine(ABC):
    """决策引擎基类。接收视觉嵌入，输出动作。"""

    @abstractmethod
    def decide(self, embedding, **context) -> list[Action]:
        """根据视觉嵌入做出决策。"""

    def configure(self, **kwargs) -> None:
        """运行时调整配置。"""

    def on_start(self) -> None:
        """引擎启动时调用。"""

    def on_stop(self) -> None:
        """引擎停止时调用。"""
