"""检测工作线程，在后台运行视频采集和推理，通过信号将结果发回 GUI。"""

import time
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..sources.base import BaseSource
from ..core.detector import Detector, DetectionResult
from ..agents.base import BaseAgent


class DetectionWorker(QThread):
    """后台检测线程。"""

    frame_ready = Signal(np.ndarray, DetectionResult)  # 每帧检测完成
    error = Signal(str)                                 # 错误信息
    fps_updated = Signal(float, float)                  # (fps, inference_ms)
    agent_stats_updated = Signal(dict)                  # Agent 统计更新
    decision_log = Signal(str)                          # 决策日志

    def __init__(self, source: BaseSource, detector: Detector,
                 agents: list[BaseAgent] | None = None,
                 auto_pilot=None, parent=None):
        super().__init__(parent)
        self.source = source
        self.detector = detector
        self.agents = agents or []
        self.auto_pilot = auto_pilot
        self._running = False
        self._fps_alpha = 0.9
        self._fps = 0.0

    def run(self):
        self._running = True
        try:
            self.source.start()
        except Exception as e:
            self.error.emit(f"视频源启动失败: {e}")
            return

        for agent in self.agents:
            agent.on_start()
        if self.auto_pilot:
            self.auto_pilot.start()

        last_time = time.perf_counter()
        frame_count = 0

        while self._running:
            frame = self.source.read()
            if frame is None:
                self.error.emit("视频源已结束")
                break

            result = self.detector.detect(frame)

            # AutoPilot 场景分类
            if self.auto_pilot:
                self.auto_pilot.on_frame(frame, result)

            # Agent 处理
            for agent in self.agents:
                agent.on_detection(result)

            # 计算 FPS
            now = time.perf_counter()
            dt = now - last_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self._fps = self._fps * self._fps_alpha + instant_fps * (1 - self._fps_alpha)
            last_time = now

            self.frame_ready.emit(frame, result)
            self.fps_updated.emit(self._fps, result.inference_ms)

            # 每 30 帧发送 Agent 统计
            frame_count += 1
            if frame_count % 30 == 0:
                for agent in self.agents:
                    if hasattr(agent, 'stats'):
                        self.agent_stats_updated.emit(agent.stats)

        for agent in self.agents:
            agent.on_stop()
        if self.auto_pilot:
            self.auto_pilot.stop()
        self.source.stop()

    def stop(self):
        self._running = False
        self.wait(5000)
