"""远程 PC 录制器 — 连接远程采集服务，录制为标准格式。

连接 RemoteCaptureServer 的 WebSocket 流，接收画面和键鼠事件，
保存为标准 recording.mp4 + actions.jsonl 格式，与训练管线完全兼容。

用法:
    recorder = RemoteRecorder(host="192.168.1.100", port=9876, output_dir="recordings/remote1")
    recorder.start()
    # ... 远程 PC 上的用户操作游戏 ...
    recorder.stop()
    # 产出: recording.mp4 + actions.jsonl + events.jsonl + meta.json
"""

import json
import logging
import struct
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
    """连接远程 PC 采集服务，录制为标准训练数据格式。

    接口与 GameRecorder 一致：start() / stop() / toggle_pause()。
    """

    def __init__(
        self,
        host: str = "192.168.1.100",
        port: int = 9876,
        output_dir: str = "recordings",
        action_map: dict[str, str] | None = None,
        on_log=None,
        on_stats=None,
        on_frame=None,
    ):
        """
        Args:
            host: 远程采集服务 IP
            port: 远程采集服务端口
            output_dir: 本地录制输出目录
            action_map: 按键→动作名映射
            on_log: 日志回调
            on_stats: 录制完成统计回调
            on_frame: 实时帧回调（用于画面预览）
        """
        self._host = host
        self._port = port
        self._output_dir = Path(output_dir)
        self._action_map = action_map or {}
        self._on_log = on_log
        self._on_stats = on_stats
        self._on_frame = on_frame

        self._recording = False
        self._paused = False
        self._connected = False
        self._stop_event = threading.Event()
        self._receive_thread = None

        self._video_writer = None
        self._events_file = None
        self._frame_timestamps: list[float] = []
        self._events: list[dict] = []
        self._frame_count = 0
        self._lock = threading.Lock()
        self._remote_meta: dict = {}

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
        return self._connected

    def start(self):
        """开始远程录制。"""
        if self._recording:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._frame_timestamps.clear()
        self._events.clear()
        self._stop_event.clear()
        self._paused = False
        self._frame_count = 0
        self._recording = True
        self._connected = False

        self._events_file = open(
            self._output_dir / "events.jsonl", "w", encoding="utf-8"
        )

        self._receive_thread = threading.Thread(
            target=self._receive_loop, daemon=True
        )
        self._receive_thread.start()
        self._log(f"[远程录制] 连接 {self._host}:{self._port}...")

    def stop(self) -> RemoteRecordingStats:
        """停止录制并保存数据。"""
        if not self._recording:
            return RemoteRecordingStats()

        self._recording = False
        self._stop_event.set()

        if self._receive_thread:
            self._receive_thread.join(timeout=5)

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

    # ── 接收线程 ──

    def _receive_loop(self):
        try:
            from websockets.sync.client import connect
        except ImportError:
            self._log("[远程录制] 需要安装 websockets: pip install websockets")
            self._recording = False
            return

        uri = f"ws://{self._host}:{self._port}"
        try:
            with connect(uri, max_size=10_000_000, open_timeout=10) as ws:
                self._connected = True
                self._log(f"[远程录制] 已连接 {uri}")

                while not self._stop_event.is_set():
                    try:
                        msg = ws.recv(timeout=1)
                    except TimeoutError:
                        continue
                    except Exception:
                        break

                    if isinstance(msg, str):
                        data = json.loads(msg)
                        if data.get("type") == "meta":
                            self._remote_meta = data
                            self._log(
                                f"[远程录制] 远程画面: "
                                f"{data.get('width')}x{data.get('height')} "
                                f"FPS={data.get('fps')} "
                                f"窗口={data.get('window') or '全屏'}"
                            )
                        elif data.get("type") == "error":
                            self._log(f"[远程录制] 服务端拒绝: {data.get('msg')}")
                            break
                        else:
                            if not self._paused:
                                self._handle_event(data)
                    else:
                        if not self._paused:
                            self._handle_frame(msg)

        except ConnectionRefusedError:
            self._log(f"[远程录制] 连接被拒绝 — 请确认远程采集服务已启动")
        except OSError as e:
            self._log(f"[远程录制] 网络错误: {e}")
        except Exception as e:
            self._log(f"[远程录制] 连接异常: {e}")
        finally:
            self._connected = False
            if not self._stop_event.is_set():
                self._log("[远程录制] 连接已断开")

    def _handle_frame(self, msg: bytes):
        """处理二进制帧消息：8字节时间戳 + JPEG。"""
        if len(msg) < 9:
            return

        ts = struct.unpack(">d", msg[:8])[0]
        jpeg_bytes = msg[8:]

        frame = cv2.imdecode(
            np.frombuffer(jpeg_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is None:
            return

        if self._video_writer is None:
            h, w = frame.shape[:2]
            video_path = str(self._output_dir / "recording.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self._remote_meta.get("fps", 10)
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
        # 流式写入事件文件
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
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_frames": self._frame_count,
                    "total_events": len(self._events),
                    "duration_sec": round(duration, 1),
                    "fps_target": self._remote_meta.get("fps", 10),
                    "fps_actual": round(actual_fps, 1),
                    "action_dist": dict(action_dist),
                    "action_map": self._action_map,
                    "remote_host": f"{self._host}:{self._port}",
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
            remote_host=f"{self._host}:{self._port}",
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

    # ── 连接测试（静态方法，GUI 用）──

    @staticmethod
    def test_connection(host: str, port: int, timeout: float = 5) -> tuple[bool, str]:
        """测试远程采集服务是否可连接。

        Returns:
            (success, message)
        """
        try:
            from websockets.sync.client import connect
        except ImportError:
            return False, "websockets 未安装"

        uri = f"ws://{host}:{port}"
        try:
            with connect(uri, open_timeout=timeout, close_timeout=2) as ws:
                msg = ws.recv(timeout=timeout)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("type") == "meta":
                        w, h = data.get("width", 0), data.get("height", 0)
                        fps = data.get("fps", 0)
                        return True, f"已连接 | {w}x{h} FPS={fps}"
                    if data.get("type") == "error":
                        return False, data.get("msg", "服务端拒绝")
                return False, "未收到元数据"
        except ConnectionRefusedError:
            return False, "连接被拒绝 — 服务未启动"
        except TimeoutError:
            return False, "连接超时"
        except OSError as e:
            return False, f"网络错误: {e}"
        except Exception as e:
            return False, str(e)


class RemoteAgentConnection:
    """Agent 远程控制连接 — 接收画面 + 发送操作指令。

    用于 Agent 部署模式：连接远程采集服务，获取游戏画面用于决策，
    并将决策结果（键鼠操作）发回远程 PC 执行。

    用法:
        conn = RemoteAgentConnection("192.168.1.100", 9876)
        conn.start()
        frame = conn.get_frame()        # 获取最新画面
        conn.send_key_tap("a")          # 远程按键
        conn.send_mouse_click(100, 200) # 远程点击
        conn.stop()
    """

    def __init__(self, host: str, port: int, on_log=None):
        self._host = host
        self._port = port
        self._on_log = on_log
        self._ws = None
        self._connected = False
        self._running = False
        self._stop_event = threading.Event()
        self._receive_thread = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._remote_meta: dict = {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._remote_meta.get("width", 0), self._remote_meta.get("height", 0))

    def start(self):
        """连接远程采集服务。"""
        self._running = True
        self._stop_event.clear()
        self._receive_thread = threading.Thread(
            target=self._receive_loop, daemon=True
        )
        self._receive_thread.start()

    def stop(self):
        """断开连接。"""
        self._running = False
        self._stop_event.set()
        if self._receive_thread:
            self._receive_thread.join(timeout=5)
        self._connected = False

    def get_frame(self) -> np.ndarray | None:
        """获取最新一帧画面（BGR numpy array）。"""
        with self._frame_lock:
            return self._latest_frame

    def send_key_tap(self, key: str):
        """远程按键（按下+释放）。"""
        self._send_cmd({"cmd": "key_tap", "key": key})

    def send_key_press(self, key: str):
        """远程按下键。"""
        self._send_cmd({"cmd": "key_press", "key": key})

    def send_key_release(self, key: str):
        """远程释放键。"""
        self._send_cmd({"cmd": "key_release", "key": key})

    def send_mouse_click(self, x: int, y: int, button: str = "left"):
        """远程鼠标点击。"""
        self._send_cmd({"cmd": "mouse_click", "x": x, "y": y, "button": button})

    def send_mouse_move(self, x: int, y: int):
        """远程鼠标移动。"""
        self._send_cmd({"cmd": "mouse_move", "x": x, "y": y})

    def _send_cmd(self, data: dict):
        if self._ws and self._connected:
            try:
                self._ws.send(json.dumps(data))
            except Exception as e:
                logger.debug(f"发送控制指令失败: {e}")

    def _receive_loop(self):
        try:
            from websockets.sync.client import connect
        except ImportError:
            self._log("[远程Agent] websockets 未安装")
            return

        uri = f"ws://{self._host}:{self._port}"
        try:
            with connect(uri, max_size=10_000_000, open_timeout=10) as ws:
                self._ws = ws
                self._connected = True
                self._log(f"[远程Agent] 已连接 {uri}")

                while not self._stop_event.is_set():
                    try:
                        msg = ws.recv(timeout=1)
                    except TimeoutError:
                        continue
                    except Exception:
                        break

                    if isinstance(msg, str):
                        data = json.loads(msg)
                        if data.get("type") == "meta":
                            self._remote_meta = data
                            self._log(
                                f"[远程Agent] 画面: "
                                f"{data.get('width')}x{data.get('height')} "
                                f"FPS={data.get('fps')}"
                            )
                    else:
                        if len(msg) < 9:
                            continue
                        jpeg_bytes = msg[8:]
                        frame = cv2.imdecode(
                            np.frombuffer(jpeg_bytes, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if frame is not None:
                            with self._frame_lock:
                                self._latest_frame = frame

        except ConnectionRefusedError:
            self._log(f"[远程Agent] 连接被拒绝 — 请确认远程服务已启动")
        except Exception as e:
            self._log(f"[远程Agent] 连接异常: {e}")
        finally:
            self._ws = None
            self._connected = False

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass
