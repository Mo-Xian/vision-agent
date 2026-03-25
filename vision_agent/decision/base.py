"""决策引擎基类和动作数据结构。"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from ..core.detector import DetectionResult
from ..core.state import SceneState


@dataclass
class Action:
    """决策引擎输出的动作指令。"""
    tool_name: str              # 要调用的工具名
    parameters: dict = field(default_factory=dict)  # 工具参数
    reason: str = ""            # 决策理由（调试用）
    priority: int = 0           # 优先级，数值越大越优先
    confidence: float = 1.0     # 决策置信度
    target_bbox: tuple | None = None  # 触发此动作的目标位置 (x1,y1,x2,y2) 像素坐标

    def to_dict(self) -> dict:
        d = {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "reason": self.reason,
            "priority": self.priority,
            "confidence": self.confidence,
        }
        if self.target_bbox:
            d["target_bbox"] = list(self.target_bbox)
        return d


class LoggingMixin:
    """统一的日志回调 mixin，消除各模块重复的 _emit_log/_log 实现。"""

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
    """决策引擎基类。接收检测结果和场景状态，输出动作指令。"""

    @abstractmethod
    def decide(self, result: DetectionResult, state: SceneState) -> list[Action]:
        """根据检测结果和场景状态做出决策。

        返回动作列表，空列表表示不执行任何动作。
        """

    def configure(self, **kwargs) -> None:
        """运行时调整配置。子类可覆盖。"""

    def on_start(self) -> None:
        """引擎启动时调用。"""

    def on_stop(self) -> None:
        """引擎停止时调用。"""
