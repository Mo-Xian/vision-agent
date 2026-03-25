"""示例 Agent：检测到特定目标时打印信息并记录统计。"""

import logging
import time
from collections import defaultdict
from ..core.detector import DetectionResult
from ..core.state import describe_position
from .base import BaseAgent

logger = logging.getLogger(__name__)


class DemoAgent(BaseAgent):
    """演示用 Agent，展示如何消费检测结果。

    功能：
    - 统计各类别目标出现次数
    - 当检测到 "person" 时打印位置信息
    - 每 5 秒输出一次统计摘要
    """

    def __init__(self, track_class: str = "person", summary_interval: float = 5.0):
        self.track_class = track_class
        self.summary_interval = summary_interval
        self._counts: dict[str, int] = defaultdict(int)
        self._last_summary = 0.0
        self._total_frames = 0

    def on_start(self):
        logger.info(f"DemoAgent 启动，追踪目标: {self.track_class}")
        self._last_summary = time.time()

    def on_detection(self, result: DetectionResult):
        self._total_frames += 1

        for det in result.detections:
            self._counts[det.class_name] += 1

            if det.class_name == self.track_class:
                cx = (det.bbox_norm[0] + det.bbox_norm[2]) / 2
                cy = (det.bbox_norm[1] + det.bbox_norm[3]) / 2
                position = describe_position(cx, cy)
                logger.info(
                    f"[Frame {result.frame_id}] 检测到 {self.track_class} "
                    f"@ {position} (置信度: {det.confidence:.2f})"
                )

        now = time.time()
        if now - self._last_summary >= self.summary_interval:
            self._print_summary()
            self._last_summary = now

    def on_stop(self):
        self._print_summary()
        logger.info("DemoAgent 停止")

    def _print_summary(self):
        if not self._counts:
            return
        summary = ", ".join(f"{k}: {v}" for k, v in sorted(self._counts.items()))
        logger.info(f"[统计] 已处理 {self._total_frames} 帧 | 累计检测: {summary}")

