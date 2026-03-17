"""跨帧场景状态管理。"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from .detector import DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class SceneState:
    """当前场景的完整状态快照。"""
    current_result: DetectionResult
    history: list[DetectionResult]
    object_counts: dict[str, int]           # 当前帧各类别计数
    scene_summary: str                       # 文字描述（供 LLM 消费）
    custom_data: dict = field(default_factory=dict)
    timestamp: float = 0.0


class StateManager:
    """维护跨帧的场景上下文，生成结构化状态供决策层使用。"""

    def __init__(self, history_size: int = 30):
        self._history: deque[DetectionResult] = deque(maxlen=history_size)
        self._custom_data: dict = {}

    def update(self, result: DetectionResult) -> SceneState:
        """用新的检测结果更新状态，返回当前场景快照。"""
        self._history.append(result)

        counts = {}
        for det in result.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1

        summary = self._build_summary(result, counts)

        return SceneState(
            current_result=result,
            history=list(self._history),
            object_counts=counts,
            scene_summary=summary,
            custom_data=self._custom_data.copy(),
            timestamp=time.time(),
        )

    def set_custom(self, key: str, value) -> None:
        """设置自定义场景数据。"""
        self._custom_data[key] = value

    def _build_summary(self, result: DetectionResult, counts: dict[str, int]) -> str:
        """生成当前场景的文字摘要。"""
        if not result.detections:
            return "当前画面未检测到任何目标。"

        parts = []
        for name, count in sorted(counts.items(), key=lambda x: -x[1]):
            parts.append(f"{name}×{count}" if count > 1 else name)

        detail_lines = []
        for det in result.detections[:10]:
            cx = det.bbox_norm[0] + det.bbox_norm[2] / 2
            cy = det.bbox_norm[1] + det.bbox_norm[3] / 2
            pos = self._describe_position(cx, cy)
            detail_lines.append(f"  - {det.class_name} (conf={det.confidence:.2f}) 位于{pos}")

        summary = f"检测到 {len(result.detections)} 个目标: {', '.join(parts)}\n"
        summary += "\n".join(detail_lines)
        if len(result.detections) > 10:
            summary += f"\n  ... 等共 {len(result.detections)} 个目标"

        return summary

    @staticmethod
    def _describe_position(cx: float, cy: float) -> str:
        """将归一化坐标转换为方位描述。"""
        v = "上方" if cy < 0.33 else ("中部" if cy < 0.66 else "下方")
        h = "左侧" if cx < 0.33 else ("中间" if cx < 0.66 else "右侧")
        return f"{v}{h}"
