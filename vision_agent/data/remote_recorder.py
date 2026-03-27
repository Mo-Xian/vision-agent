"""远程录制器 — 通过 RemoteHub 录制远程客户端画面为标准格式。

使用 RemoteHub 接收远程客户端推送的画面和键鼠事件，
保存为标准 recording.mp4 + actions.jsonl 格式，与训练管线完全兼容。

用法:
    hub = RemoteHub(port=9876)
    hub.start()
    recorder = RemoteRecorder(hub=hub, output_dir="recordings/remote1")
    recorder.start()
    # ... 远程客户端上的用户操作游戏 ...
    recorder.stop()
    # 产出: recording.mp4 + actions.jsonl + events.jsonl + meta.json
"""

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RemoteRecordingStats:
    """远程录制统计。"""
    total_frames: int = 0
    total_events: int = 0
    duration_sec: float = 0.0
    action_dist: dict = field(default_factory=dict)
    fps_actual: float = 0.0
    output_dir: str = ""
    video_path: str = ""
    actions_path: str = ""
    remote_host: str = ""


class RemoteRecorder:
    """通过 RemoteHub 录制远程客户端画面为标准训练数据格式。

    接口与 GameRecorder 一致：start() / stop() / toggle_pause()。
    """

    def __init__(
        self,
        hub,
        output_dir: str = "recordings",
        action_map: dict[str, str] | None = None,
        on_log=None,
        on_stats=None,
        on_frame=None,
    ):
        """
        Args:
            hub: RemoteHub 实例（已启动，等待客户端连接）
            output_dir: 本地录制输出目录
            action_map: 按键→动作名映射
            on_log: 日志回调
            on_stats: 录制完成统计回调
            on_frame: 实时帧回调（用于画面预览）
        """
        self._hub = hub
        self._output_dir = Path(output_dir)
        self._action_map = action_map or {}
        self._on_log = on_log
        self._on_stats = on_stats
        self._on_frame = on_frame

        self._recording = False
        self._paused = False
        self._stop_event = threading.Event()
        self._record_thread = None

        self._video_writer = None
        self._events_file = None
        self._frame_timestamps: list[float] = []
        self._events: list[dict] = []
        self._frame_count = 0
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_connected(self) -> bool:
        return self._hub.is_client_connected

    def start(self):
        """开始录制（从 hub 获取帧和事件）。"""
        if self._recording:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._frame_timestamps.clear()
        self._events.clear()
        self._stop_event.clear()
        self._paused = False
        self._frame_count = 0
        self._recording = True

        self._events_file = open(
            self._output_dir / "events.jsonl", "w", encoding="utf-8"
        )

        self._record_thread = threading.Thread(
            target=self._record_loop, daemon=True
        )
        self._record_thread.start()
        self._log("[远程录制] 开始录制（等待远程客户端画面）...")

    def stop(self) -> RemoteRecordingStats:
        """停止录制并保存数据。"""
        if not self._recording:
            return RemoteRecordingStats()

        self._recording = False
        self._stop_event.set()

        if self._record_thread:
            self._record_thread.join(timeout=5)

        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        if self._events_file:
            self._events_file.close()
            self._events_file = None

        self._log(f"[远程录制] 停止 | {self._frame_count} 帧, {len(self._events)} 事件")
        return self._save_actions()

    def toggle_pause(self):
        """切换暂停/恢复。"""
        self._paused = not self._paused
        state = "暂停" if self._paused else "恢复"
        self._log(f"[远程录制] {state}")

    # ── 录制线程 ──

    def _record_loop(self):
        """轮询 hub 获取帧和事件。"""
        last_frame = None
        fps = 10
        interval = 1.0 / fps

        while not self._stop_event.is_set():
            if self._paused:
                time.sleep(0.1)
                continue

            # 获取事件
            events = self._hub.get_events()
            for evt in events:
                self._handle_event(evt)

            # 获取帧
            frame_data = self._hub.get_frame_with_ts()
            if frame_data is not None and frame_data is not last_frame:
                last_frame = frame_data
                ts, frame = frame_data
                self._handle_frame(ts, frame)

                # 从 meta 更新 fps
                meta = self._hub.client_meta
                if meta.get("fps"):
                    new_fps = meta["fps"]
                    if new_fps != fps:
                        fps = new_fps
                        interval = 1.0 / fps

            time.sleep(interval * 0.5)  # 轮询频率 = 2x FPS

    def _handle_frame(self, ts: float, frame: np.ndarray):
        """处理一帧画面。"""
        if self._video_writer is None:
            h, w = frame.shape[:2]
            video_path = str(self._output_dir / "recording.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            meta = self._hub.client_meta
            fps = meta.get("fps", 10)
            self._video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        self._video_writer.write(frame)

        with self._lock:
            self._frame_count += 1
            self._frame_timestamps.append(ts)

        if self._on_frame:
            try:
                self._on_frame(frame)
            except Exception:
                pass

    def _handle_event(self, data: dict):
        """处理键鼠事件。"""
        with self._lock:
            self._events.append(data)
        if self._events_file and not self._events_file.closed:
            self._events_file.write(json.dumps(data, ensure_ascii=False) + "\n")
            self._events_file.flush()

    # ── 保存动作标注（与 GameRecorder 格式一致）──

    def _save_actions(self) -> RemoteRecordingStats:
        if not self._frame_timestamps:
            self._log("[远程录制] 无帧数据")
            return RemoteRecordingStats(output_dir=str(self._output_dir))

        duration = self._frame_timestamps[-1]
        actual_fps = len(self._frame_timestamps) / max(duration, 0.1)

        sorted_events = sorted(self._events, key=lambda e: e.get("time", 0))
        key_events = [
            e for e in sorted_events if e["type"] in ("key_down", "key_up")
        ]
        mouse_events = [
            e for e in sorted_events
            if e["type"] in ("mouse_down", "mouse_up", "mouse_scroll")
        ]

        actions_path = str(self._output_dir / "actions.jsonl")
        action_dist = defaultdict(int)

        active_keys: set[str] = set()
        active_mouse: set[str] = set()
        last_mouse_pos = (0, 0)
        key_idx = 0
        mouse_idx = 0

        with open(actions_path, "w", encoding="utf-8") as f:
            for frame_idx, ts in enumerate(self._frame_timestamps, 1):
                while key_idx < len(key_events) and key_events[key_idx]["time"] <= ts:
                    e = key_events[key_idx]
                    if e["type"] == "key_down":
                        active_keys.add(e["key"])
                    else:
                        active_keys.discard(e["key"])
                    key_idx += 1

                while mouse_idx < len(mouse_events) and mouse_events[mouse_idx]["time"] <= ts:
                    e = mouse_events[mouse_idx]
                    if e["type"] == "mouse_down":
                        active_mouse.add(e["button"])
                    elif e["type"] == "mouse_up":
                        active_mouse.discard(e["button"])
                    if "x" in e:
                        last_mouse_pos = (e["x"], e["y"])
                    mouse_idx += 1

                keys_sorted = sorted(active_keys)
                mouse_sorted = sorted(active_mouse)

                if keys_sorted:
                    primary_key = keys_sorted[0]
                    action_name = self._action_map.get(primary_key, primary_key)
                elif mouse_sorted:
                    action_name = f"mouse_{mouse_sorted[0]}"
                else:
                    action_name = "idle"

                sample = {
                    "frame_id": frame_idx,
                    "timestamp": round(ts, 3),
                    "human_action": {
                        "type": "recorded",
                        "key": action_name,
                        "raw_keys": keys_sorted,
                        "mouse_buttons": mouse_sorted,
                        "mouse_pos": list(last_mouse_pos),
                    },
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                action_dist[action_name] += 1

        self._log(f"[远程录制] 动作标注已保存: {actions_path}")
        self._log(f"[远程录制] 动作分布: {dict(action_dist)}")

        # 保存元数据
        video_path = str(self._output_dir / "recording.mp4")
        meta_path = self._output_dir / "meta.json"
        client_addr = self._hub.client_addr
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_frames": self._frame_count,
                    "total_events": len(self._events),
                    "duration_sec": round(duration, 1),
                    "fps_target": self._hub.client_meta.get("fps", 10),
                    "fps_actual": round(actual_fps, 1),
                    "action_dist": dict(action_dist),
                    "action_map": self._action_map,
                    "remote_client": client_addr,
                    "record_source": "remote_pc",
                    "video_path": video_path,
                    "actions_path": actions_path,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        stats = RemoteRecordingStats(
            total_frames=self._frame_count,
            total_events=len(self._events),
            duration_sec=round(duration, 1),
            action_dist=dict(action_dist),
            fps_actual=round(actual_fps, 1),
            output_dir=str(self._output_dir),
            video_path=video_path,
            actions_path=actions_path,
            remote_host=client_addr,
        )

        if self._on_stats:
            self._on_stats(stats)

        return stats

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass
