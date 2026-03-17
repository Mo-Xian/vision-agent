"""Agent 基类。"""

from abc import ABC, abstractmethod
from ..core.detector import DetectionResult


class BaseAgent(ABC):
    """所有 Agent 的抽象基类。

    Agent 接收检测结果，分析后执行动作。
    """

    @abstractmethod
    def on_detection(self, result: DetectionResult):
        """接收一帧的检测结果并处理。

        Args:
            result: 当前帧的检测结果
        """

    def on_start(self):
        """Agent 启动时调用，可选重写。"""

    def on_stop(self):
        """Agent 停止时调用，可选重写。"""
