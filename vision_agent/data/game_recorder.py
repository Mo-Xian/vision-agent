"""游戏操作录制器：屏幕截图 + 键鼠事件 → 行为克隆训练数据。

核心特性：
    - 窗口捕获：自动检测游戏窗口，只截取游戏画面
    - 键鼠录制：键盘按键 + 鼠标点击/位置 全部记录
    - 流式保存：帧直写磁盘，不缓存在内存（支持长时间录制）
    - 快捷键控制：F9 切换录制状态（可自定义）
    - 多键支持：每帧记录所有活跃按键和鼠标状态

用法：
    recorder = GameRecorder(output_dir="recordings/session1")
    recorder.start()
    # 用户玩游戏... 按 F9 暂停/恢复
    recorder.stop()
    # 产出: recording.mp4 + actions.jsonl + events.jsonl + meta.json
"""

import json
import logging
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RecordingStats:
    """录制统计。"""
    total_frames: int = 0
    total_events: int = 0
    duration_sec: float = 0.0
    action_dist: dict = field(default_factory=dict)
    fps_actual: float = 0.0
    output_dir: str = ""
    video_path: str = ""
    actions_path: str = ""
    window_title: str = ""


class GameRecorder:
    """录制人类游戏操作，生成行为克隆训练数据。

    截屏使用 mss（高性能屏幕捕获），键鼠使用 pynput。
    支持窗口捕获、鼠标录制、流式保存、快捷键控制。
    """

    def __init__(
        self,
        output_dir: str = "recordings",
        fps: int = 10,
        window_title: str = "",
        screen_region: dict | None = None,
        record_mouse: bool = True,
        hotkey: str = "f9",
        action_map: dict[str, str] | None = None,
        on_log=None,
        on_stats=None,
        on_state_change=None,
    ):
        """
        Args:
            output_dir: 录制输出目录
            fps: 目标截屏帧率
            window_title: 游戏窗口标题（模糊匹配），空=全屏
            screen_region: 截屏区域（手动指定，优先于 window_title）
            record_mouse: 是否录制鼠标操作
            hotkey: 暂停/恢复快捷键（默认 F9）
            action_map: 按键→动作名映射
            on_log: 日志回调
            on_stats: 状态更新回调
            on_state_change: 录制状态变化回调 (paused: bool)
        """
        self._output_dir = Path(output_dir)
        self._fps = fps
        self._frame_interval = 1.0 / fps
        self._window_title = window_title
        self._screen_region = screen_region
        self._record_mouse = record_mouse
        self._hotkey = hotkey
        self._action_map = action_map or {}
        self._on_log = on_log
        self._on_stats = on_stats
        self._on_state_change = on_state_change

        self._recording = False
        self._paused = False
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._kb_listener_thread = None
        self._mouse_listener_thread = None

        # 运行时状态（流式，不缓存帧数据）
        self._frame_timestamps: list[float] = []
        self._events: list[dict] = []
        self._active_keys: set[str] = set()
        self._mouse_buttons: set[str] = set()  # 当前按下的鼠标键
        self._mouse_pos: tuple[int, int] = (0, 0)
        self._start_time = 0.0
        self._lock = threading.Lock()
        self._frame_count = 0

        # 流式写入
        self._video_writer = None
        self._events_file = None
        self._resolved_region = None  # 实际捕获区域

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self):
        """开始录制。"""
        if self._recording:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._frame_timestamps.clear()
        self._events.clear()
        self._active_keys.clear()
        self._mouse_buttons.clear()
        self._stop_event.clear()
        self._paused = False
        self._frame_count = 0
        self._start_time = time.time()
        self._recording = True

        # 打开事件流式文件
        self._events_file = open(
            self._output_dir / "events.jsonl", "w", encoding="utf-8"
        )

        # 解析捕获区域
        self._resolved_region = self._resolve_capture_region()

        # 启动线程
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self._kb_listener_thread = threading.Thread(
            target=self._keyboard_loop, daemon=True
        )
        self._kb_listener_thread.start()

        if self._record_mouse:
            self._mouse_listener_thread = threading.Thread(
                target=self._mouse_loop, daemon=True
            )
            self._mouse_listener_thread.start()

        region_info = ""
        if self._window_title and self._resolved_region:
            region_info = f" | 窗口: {self._window_title}"
        elif self._screen_region:
            region_info = f" | 区域: {self._screen_region}"
        else:
            region_info = " | 全屏"

        mouse_info = " + 鼠标" if self._record_mouse else ""
        self._log(
            f"[录制] 开始 | FPS={self._fps}{region_info} | "
            f"键盘{mouse_info} | 快捷键={self._hotkey.upper()}"
        )

    def stop(self) -> RecordingStats:
        """停止录制并保存数据。"""
        if not self._recording:
            return RecordingStats()

        self._recording = False
        self._stop_event.set()

        if self._capture_thread:
            self._capture_thread.join(timeout=5)
        if self._kb_listener_thread:
            self._kb_listener_thread.join(timeout=2)
        if self._mouse_listener_thread:
            self._mouse_listener_thread.join(timeout=2)

        # 关闭流式文件
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        if self._events_file:
            self._events_file.close()
            self._events_file = None

        self._log(f"[录制] 停止 | {self._frame_count} 帧, {len(self._events)} 事件")
        return self._save_actions()

    def toggle_pause(self):
        """切换暂停/恢复状态。"""
        self._paused = not self._paused
        state = "暂停" if self._paused else "恢复"
        self._log(f"[录制] {state}")
        if self._on_state_change:
            try:
                self._on_state_change(self._paused)
            except Exception:
                pass

    # ── 捕获区域 ──

    def _resolve_capture_region(self) -> dict | None:
        """解析捕获区域：手动指定 > 窗口标题 > 全屏。"""
        if self._screen_region:
            return self._screen_region

        if self._window_title:
            rect = self._find_window(self._window_title)
            if rect:
                self._log(f"[录制] 已定位窗口: {self._window_title} → {rect}")
                return rect
            else:
                self._log(f"[录制] 未找到窗口 '{self._window_title}'，使用全屏")

        return None

    @staticmethod
    def _find_window(title: str) -> dict | None:
        """通过窗口标题查找窗口位置（Windows）。"""
        if sys.platform != "win32":
            return None
        try:
            import ctypes
            user32 = ctypes.windll.user32

            # 枚举窗口查找匹配标题
            result = [None]
            title_lower = title.lower()

            def enum_callback(hwnd, _):
                if not user32.IsWindowVisible(hwnd):
                    return True
                buf = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, buf, 256)
                if title_lower in buf.value.lower():
                    rect = ctypes.wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    result[0] = {
                        "left": rect.left,
                        "top": rect.top,
                        "width": rect.right - rect.left,
                        "height": rect.bottom - rect.top,
                    }
                    return False  # 找到了，停止枚举
                return True

            import ctypes.wintypes
            WNDENUMPROC = ctypes.WINFUNCTYPE(
                ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
            )
            user32.EnumWindows(WNDENUMPROC(enum_callback), 0)
            return result[0]
        except Exception:
            return None

    @staticmethod
    def list_windows() -> list[str]:
        """列出所有可见窗口标题（用于 GUI 选择）。"""
        if sys.platform != "win32":
            return []
        try:
            import ctypes
            import ctypes.wintypes
            user32 = ctypes.windll.user32
            titles = []

            def enum_callback(hwnd, _):
                if not user32.IsWindowVisible(hwnd):
                    return True
                buf = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, buf, 256)
                if buf.value.strip():
                    titles.append(buf.value)
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(
                ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
            )
            user32.EnumWindows(WNDENUMPROC(enum_callback), 0)
            return titles
        except Exception:
            return []

    # ── 截屏线程 ──

    def _capture_loop(self):
        """截屏线程：流式写入视频文件。"""
        try:
            import mss
        except ImportError:
            self._log("[录制] 需要安装 mss: pip install mss")
            self._recording = False
            return

        with mss.mss() as sct:
            monitor = self._resolved_region or sct.monitors[1]

            while not self._stop_event.is_set():
                t0 = time.time()

                # 暂停时跳过截屏
                if self._paused:
                    self._stop_event.wait(0.1)
                    continue

                ts = t0 - self._start_time

                try:
                    img = sct.grab(monitor)
                    frame = np.array(img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # 首帧初始化 VideoWriter
                    if self._video_writer is None:
                        h, w = frame.shape[:2]
                        video_path = str(self._output_dir / "recording.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        self._video_writer = cv2.VideoWriter(
                            video_path, fourcc, self._fps, (w, h)
                        )

                    # 直写磁盘
                    self._video_writer.write(frame)

                    with self._lock:
                        self._frame_count += 1
                        self._frame_timestamps.append(ts)

                except Exception as e:
                    logger.debug(f"截屏失败: {e}")

                elapsed = time.time() - t0
                sleep_time = self._frame_interval - elapsed
                if sleep_time > 0:
                    self._stop_event.wait(sleep_time)

    # ── 键盘监听 ──

    def _keyboard_loop(self):
        """键盘事件监听线程（含快捷键）。"""
        try:
            from pynput import keyboard
        except ImportError:
            self._log("[录制] 需要安装 pynput: pip install pynput")
            return

        hotkey_name = self._hotkey.lower()

        def on_press(key):
            if self._stop_event.is_set():
                return False
            key_name = self._key_to_str(key)

            # 快捷键检测
            if key_name == hotkey_name:
                self.toggle_pause()
                return

            if self._paused:
                return

            ts = time.time() - self._start_time
            with self._lock:
                self._active_keys.add(key_name)
                evt = {
                    "time": round(ts, 4),
                    "type": "key_down",
                    "key": key_name,
                }
                self._events.append(evt)
                self._write_event(evt)

        def on_release(key):
            if self._stop_event.is_set():
                return False
            key_name = self._key_to_str(key)
            if key_name == hotkey_name or self._paused:
                return

            ts = time.time() - self._start_time
            with self._lock:
                self._active_keys.discard(key_name)
                evt = {
                    "time": round(ts, 4),
                    "type": "key_up",
                    "key": key_name,
                }
                self._events.append(evt)
                self._write_event(evt)

        with keyboard.Listener(
            on_press=on_press, on_release=on_release
        ) as listener:
            self._stop_event.wait()
            listener.stop()

    # ── 鼠标监听 ──

    def _mouse_loop(self):
        """鼠标事件监听线程。"""
        try:
            from pynput import mouse
        except ImportError:
            self._log("[录制] 鼠标录制需要 pynput")
            return

        def on_click(x, y, button, pressed):
            if self._stop_event.is_set():
                return False
            if self._paused:
                return

            ts = time.time() - self._start_time
            btn_name = button.name  # "left", "right", "middle"
            with self._lock:
                if pressed:
                    self._mouse_buttons.add(btn_name)
                else:
                    self._mouse_buttons.discard(btn_name)
                self._mouse_pos = (x, y)
                evt = {
                    "time": round(ts, 4),
                    "type": "mouse_down" if pressed else "mouse_up",
                    "button": btn_name,
                    "x": x,
                    "y": y,
                }
                self._events.append(evt)
                self._write_event(evt)

        def on_move(x, y):
            if self._stop_event.is_set():
                return False
            if self._paused:
                return
            with self._lock:
                self._mouse_pos = (x, y)

        def on_scroll(x, y, dx, dy):
            if self._stop_event.is_set():
                return False
            if self._paused:
                return

            ts = time.time() - self._start_time
            with self._lock:
                evt = {
                    "time": round(ts, 4),
                    "type": "mouse_scroll",
                    "x": x,
                    "y": y,
                    "dx": dx,
                    "dy": dy,
                }
                self._events.append(evt)
                self._write_event(evt)

        with mouse.Listener(
            on_click=on_click, on_move=on_move, on_scroll=on_scroll
        ) as listener:
            self._stop_event.wait()
            listener.stop()

    # ── 工具方法 ──

    @staticmethod
    def _key_to_str(key) -> str:
        """将 pynput key 转为字符串。"""
        if hasattr(key, "char") and key.char:
            return key.char.lower()
        return key.name if hasattr(key, "name") else str(key)

    def _write_event(self, evt: dict):
        """流式写入事件到 JSONL 文件。"""
        if self._events_file and not self._events_file.closed:
            self._events_file.write(json.dumps(evt, ensure_ascii=False) + "\n")
            self._events_file.flush()

    def _save_actions(self) -> RecordingStats:
        """从事件时间线生成帧-动作配对文件。"""
        if not self._frame_timestamps:
            self._log("[录制] 无帧数据可保存")
            return RecordingStats(output_dir=str(self._output_dir))

        duration = self._frame_timestamps[-1] if self._frame_timestamps else 0
        actual_fps = len(self._frame_timestamps) / max(duration, 0.1)

        # ── 构建高效的事件时间线索引 ──
        sorted_events = sorted(self._events, key=lambda e: e["time"])
        # 预计算：把事件分成 key 事件和 mouse 事件
        key_events = [
            e for e in sorted_events if e["type"] in ("key_down", "key_up")
        ]
        mouse_events = [
            e
            for e in sorted_events
            if e["type"] in ("mouse_down", "mouse_up", "mouse_scroll")
        ]

        # ── 生成帧-动作配对 ──
        actions_path = str(self._output_dir / "actions.jsonl")
        action_dist = defaultdict(int)

        # 用增量方式追踪状态，避免每帧都遍历全部事件
        active_keys: set[str] = set()
        active_mouse: set[str] = set()
        last_mouse_pos = (0, 0)
        key_idx = 0
        mouse_idx = 0

        with open(actions_path, "w", encoding="utf-8") as f:
            for frame_idx, ts in enumerate(self._frame_timestamps, 1):
                # 推进 key 事件到当前时间
                while key_idx < len(key_events) and key_events[key_idx]["time"] <= ts:
                    e = key_events[key_idx]
                    if e["type"] == "key_down":
                        active_keys.add(e["key"])
                    else:
                        active_keys.discard(e["key"])
                    key_idx += 1

                # 推进 mouse 事件到当前时间
                while (
                    mouse_idx < len(mouse_events)
                    and mouse_events[mouse_idx]["time"] <= ts
                ):
                    e = mouse_events[mouse_idx]
                    if e["type"] == "mouse_down":
                        active_mouse.add(e["button"])
                    elif e["type"] == "mouse_up":
                        active_mouse.discard(e["button"])
                    if "x" in e:
                        last_mouse_pos = (e["x"], e["y"])
                    mouse_idx += 1

                # 构建动作信息
                keys_sorted = sorted(active_keys)
                mouse_sorted = sorted(active_mouse)

                # 主动作：键盘优先，鼠标次之
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

        self._log(f"[录制] 动作标注已保存: {actions_path}")
        self._log(f"[录制] 动作分布: {dict(action_dist)}")

        video_path = str(self._output_dir / "recording.mp4")
        stats = RecordingStats(
            total_frames=self._frame_count,
            total_events=len(self._events),
            duration_sec=round(duration, 1),
            action_dist=dict(action_dist),
            fps_actual=round(actual_fps, 1),
            output_dir=str(self._output_dir),
            video_path=video_path,
            actions_path=actions_path,
            window_title=self._window_title,
        )

        # 保存元数据
        meta_path = self._output_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_frames": stats.total_frames,
                    "total_events": stats.total_events,
                    "duration_sec": stats.duration_sec,
                    "fps_target": self._fps,
                    "fps_actual": stats.fps_actual,
                    "action_dist": stats.action_dist,
                    "action_map": self._action_map,
                    "window_title": self._window_title,
                    "record_mouse": self._record_mouse,
                    "video_path": video_path,
                    "actions_path": actions_path,
                },
                f,
                ensure_ascii=False,
                indent=2,
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
