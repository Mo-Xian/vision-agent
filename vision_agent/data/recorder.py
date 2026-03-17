"""数据录制模块：同步记录 YOLO 检测结果和人类操作，生成训练数据。

录制模式下，DataRecorder 作为 Agent 插入 Pipeline：
  - 接收每帧的 DetectionResult
  - 同时通过 pynput 监听人类的键盘/鼠标操作
  - 将人类操作与最近一帧的检测结果配对，生成 (state, action) 样本
  - 以 JSONL 格式保存到磁盘

输出格式 (每行一个 JSON):
{
    "timestamp": 1234567890.123,
    "frame_id": 42,
    "detections": [...],
    "object_counts": {"enemy": 2},
    "frame_size": [1920, 1080],
    "human_action": {
        "type": "keyboard",
        "action": "press",
        "key": "space"
    }
}
"""

import json
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from ..agents.base import BaseAgent
from ..core.detector import DetectionResult

logger = logging.getLogger(__name__)

try:
    import pynput.keyboard as kb
    import pynput.mouse as ms
    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False


@dataclass
class HumanAction:
    """一次人类操作。"""
    timestamp: float
    type: str           # "keyboard" | "mouse"
    action: str         # "press" | "release" | "click" | "scroll" | "move"
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "action": self.action,
            **self.detail,
        }


class DataRecorder(BaseAgent):
    """录制检测结果 + 人类操作的训练数据。

    用法:
        recorder = DataRecorder(save_dir="data/recordings")
        pipeline = Pipeline(..., agents=[recorder])
        pipeline.run()  # 录制期间正常操作，停止后数据自动保存
    """

    def __init__(
        self,
        save_dir: str = "data/recordings",
        session_name: str | None = None,
        record_mouse_move: bool = False,
        mouse_move_interval: float = 0.1,
        max_detection_age: float = 0.5,
    ):
        """
        Args:
            save_dir: 数据保存目录
            session_name: 会话名称，None 则自动生成时间戳名称
            record_mouse_move: 是否录制鼠标移动事件（数据量大，默认关闭）
            mouse_move_interval: 鼠标移动事件最小间隔(秒)，避免数据爆炸
            max_detection_age: 检测结果最大年龄(秒)，超时则不关联
        """
        if not _PYNPUT_AVAILABLE:
            raise ImportError("需要安装 pynput: pip install pynput")

        self._save_dir = Path(save_dir)
        self._session_name = session_name or time.strftime("%Y%m%d_%H%M%S")
        self._record_mouse_move = record_mouse_move
        self._mouse_move_interval = mouse_move_interval
        self._max_detection_age = max_detection_age

        # 最近的检测结果（线程安全访问）
        self._latest_result: DetectionResult | None = None
        self._result_lock = threading.Lock()

        # 录制缓冲区
        self._samples: deque = deque()
        self._sample_count = 0
        self._last_mouse_move_time = 0.0

        # pynput 监听器
        self._kb_listener: "kb.Listener | None" = None
        self._ms_listener: "ms.Listener | None" = None
        self._recording = False

        # 输出文件
        self._output_file = None
        self._file_path: Path | None = None

    def on_start(self):
        """启动录制：创建输出文件 + 启动输入监听。"""
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self._save_dir / f"{self._session_name}.jsonl"
        self._output_file = open(self._file_path, "a", encoding="utf-8")
        self._recording = True

        # 启动键盘监听
        self._kb_listener = kb.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._kb_listener.start()

        # 启动鼠标监听
        mouse_kwargs = {
            "on_click": self._on_mouse_click,
            "on_scroll": self._on_mouse_scroll,
        }
        if self._record_mouse_move:
            mouse_kwargs["on_move"] = self._on_mouse_move
        self._ms_listener = ms.Listener(**mouse_kwargs)
        self._ms_listener.start()

        logger.info(f"DataRecorder 启动 | 输出: {self._file_path}")

    def on_stop(self):
        """停止录制：关闭监听器和文件。"""
        self._recording = False

        if self._kb_listener:
            self._kb_listener.stop()
        if self._ms_listener:
            self._ms_listener.stop()
        if self._output_file:
            self._output_file.flush()
            self._output_file.close()

        logger.info(
            f"DataRecorder 停止 | 录制样本数: {self._sample_count} | 文件: {self._file_path}"
        )

    def on_detection(self, result: DetectionResult):
        """接收检测结果，更新最新帧。"""
        with self._result_lock:
            self._latest_result = result

    # ── 内部方法 ──

    def _get_current_state(self) -> dict | None:
        """获取当前检测状态快照，超时则返回 None。"""
        with self._result_lock:
            result = self._latest_result
        if result is None:
            return None
        if time.time() - result.timestamp > self._max_detection_age:
            return None

        counts = {}
        for det in result.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1

        return {
            "frame_id": result.frame_id,
            "timestamp": round(result.timestamp, 3),
            "frame_size": [result.frame_width, result.frame_height],
            "inference_ms": round(result.inference_ms, 1),
            "detections": [d.to_dict() for d in result.detections],
            "object_counts": counts,
        }

    def _record_sample(self, human_action: HumanAction):
        """将一条 (检测状态, 人类操作) 样本写入文件。"""
        if not self._recording:
            return

        state = self._get_current_state()
        if state is None:
            return

        sample = {
            **state,
            "human_action": human_action.to_dict(),
            "action_timestamp": round(human_action.timestamp, 3),
        }

        line = json.dumps(sample, ensure_ascii=False)
        self._output_file.write(line + "\n")
        self._sample_count += 1

        if self._sample_count % 100 == 0:
            self._output_file.flush()
            logger.debug(f"已录制 {self._sample_count} 条样本")

    # ── 键盘回调 ──

    def _key_to_str(self, key) -> str:
        """将 pynput 按键转为可读字符串。"""
        if isinstance(key, kb.KeyCode):
            return key.char if key.char else str(key)
        if isinstance(key, kb.Key):
            return key.name
        return str(key)

    def _on_key_press(self, key):
        if not self._recording:
            return
        self._record_sample(HumanAction(
            timestamp=time.time(),
            type="keyboard",
            action="press",
            detail={"key": self._key_to_str(key)},
        ))

    def _on_key_release(self, key):
        if not self._recording:
            return
        self._record_sample(HumanAction(
            timestamp=time.time(),
            type="keyboard",
            action="release",
            detail={"key": self._key_to_str(key)},
        ))

    # ── 鼠标回调 ──

    def _on_mouse_click(self, x, y, button, pressed):
        if not self._recording:
            return
        self._record_sample(HumanAction(
            timestamp=time.time(),
            type="mouse",
            action="click" if pressed else "release",
            detail={
                "x": x,
                "y": y,
                "button": button.name,
            },
        ))

    def _on_mouse_scroll(self, x, y, dx, dy):
        if not self._recording:
            return
        self._record_sample(HumanAction(
            timestamp=time.time(),
            type="mouse",
            action="scroll",
            detail={"x": x, "y": y, "dx": dx, "dy": dy},
        ))

    def _on_mouse_move(self, x, y):
        if not self._recording:
            return
        now = time.time()
        if now - self._last_mouse_move_time < self._mouse_move_interval:
            return
        self._last_mouse_move_time = now
        self._record_sample(HumanAction(
            timestamp=time.time(),
            type="mouse",
            action="move",
            detail={"x": x, "y": y},
        ))

    # ── 属性 ──

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def file_path(self) -> Path | None:
        return self._file_path

    @property
    def is_recording(self) -> bool:
        return self._recording
