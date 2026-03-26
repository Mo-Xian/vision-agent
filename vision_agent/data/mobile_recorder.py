"""手机游戏录制器：scrcpy 投屏 + ADB 触摸采集。

核心流程：
    1. scrcpy --no-control 投屏到电脑（只看不控）
    2. 捕获 scrcpy 窗口画面（mss）
    3. adb shell getevent 采集触摸事件
    4. 触摸位置 → 触摸区域 → 动作名 配对到帧
    5. 输出标准格式，兼容 E2E 训练管线

前提条件：
    - 手机已开启 USB 调试，通过 USB/WiFi 连接
    - 已安装 adb 和 scrcpy（PATH 中可用）

用法：
    recorder = MobileRecorder(
        output_dir="recordings/mobile_session1",
        touch_zones={
            "skill_1": {"x": 0.85, "y": 0.55, "r": 0.06},
            "skill_2": {"x": 0.92, "y": 0.45, "r": 0.06},
            "attack":  {"x": 0.92, "y": 0.65, "r": 0.07},
            "move":    {"x": 0.15, "y": 0.70, "r": 0.12},
        },
    )
    recorder.start()
    # 用户在手机上玩游戏...
    recorder.stop()
"""

import json
import logging
import math
import re
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TouchZone:
    """触摸区域定义（归一化坐标 0~1）。

    圆形区域：中心 (x, y) + 半径 r，全部相对于屏幕宽高。
    """
    name: str
    x: float      # 中心 x（0~1，左→右）
    y: float      # 中心 y（0~1，上→下）
    r: float      # 半径（相对于屏幕宽度）


@dataclass
class MobileRecordingStats:
    """手机录制统计。"""
    total_frames: int = 0
    total_touch_events: int = 0
    duration_sec: float = 0.0
    action_dist: dict = field(default_factory=dict)
    fps_actual: float = 0.0
    output_dir: str = ""
    video_path: str = ""
    actions_path: str = ""
    device: str = ""
    resolution: tuple[int, int] = (0, 0)


class MobileRecorder:
    """手机游戏录制器：scrcpy 投屏 + ADB 触摸事件采集。

    两个数据源并行：
        - 画面：scrcpy 投屏窗口 → mss 截屏 → 视频
        - 操作：adb getevent → 触摸坐标 → 区域匹配 → 动作
    """

    def __init__(
        self,
        output_dir: str = "recordings/mobile",
        fps: int = 10,
        touch_zones: dict[str, dict] | None = None,
        scrcpy_args: list[str] | None = None,
        adb_device: str = "",
        on_log=None,
        on_stats=None,
    ):
        """
        Args:
            output_dir: 录制输出目录
            fps: 截屏帧率
            touch_zones: 触摸区域映射，格式:
                {"action_name": {"x": 0.9, "y": 0.5, "r": 0.06}, ...}
                坐标为归一化值 (0~1)，r 为半径（相对屏幕宽度）
            scrcpy_args: 额外 scrcpy 参数
            adb_device: ADB 设备序列号（多设备时指定）
            on_log: 日志回调
            on_stats: 状态更新回调
        """
        self._output_dir = Path(output_dir)
        self._fps = fps
        self._frame_interval = 1.0 / fps
        self._adb_device = adb_device
        self._scrcpy_args = scrcpy_args or []
        self._on_log = on_log
        self._on_stats = on_stats

        # 解析触摸区域
        self._touch_zones: list[TouchZone] = []
        if touch_zones:
            for name, zone in touch_zones.items():
                self._touch_zones.append(TouchZone(
                    name=name,
                    x=float(zone.get("x", 0)),
                    y=float(zone.get("y", 0)),
                    r=float(zone.get("r", 0.05)),
                ))

        # 运行时状态
        self._recording = False
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._touch_thread = None
        self._scrcpy_proc = None

        self._frame_timestamps: list[float] = []
        self._touch_events: list[dict] = []
        self._active_touches: dict[int, dict] = {}  # slot → {x, y, time}
        self._start_time = 0.0
        self._lock = threading.Lock()
        self._frame_count = 0

        # 设备信息
        self._screen_w = 0
        self._screen_h = 0
        self._touch_max_x = 0
        self._touch_max_y = 0
        self._input_device = ""

        # 流式写入
        self._video_writer = None
        self._events_file = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self):
        """开始录制：启动 scrcpy + ADB 触摸采集。"""
        if self._recording:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._frame_timestamps.clear()
        self._touch_events.clear()
        self._active_touches.clear()
        self._stop_event.clear()
        self._frame_count = 0

        # 检查 ADB 连接
        if not self._check_adb():
            return

        # 获取设备信息
        self._get_device_info()

        # 启动 scrcpy
        if not self._start_scrcpy():
            return

        # 等待 scrcpy 窗口出现
        self._log("[手机录制] 等待 scrcpy 窗口...")
        time.sleep(2)

        self._start_time = time.time()
        self._recording = True

        # 打开事件流式文件
        self._events_file = open(
            self._output_dir / "events.jsonl", "w", encoding="utf-8"
        )

        # 启动截屏线程
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._capture_thread.start()

        # 启动触摸采集线程
        self._touch_thread = threading.Thread(
            target=self._touch_loop, daemon=True
        )
        self._touch_thread.start()

        zones_info = f", {len(self._touch_zones)} 个触摸区域" if self._touch_zones else ""
        self._log(
            f"[手机录制] 开始 | FPS={self._fps} | "
            f"设备={self._adb_device or 'default'} | "
            f"屏幕={self._screen_w}x{self._screen_h}{zones_info}"
        )

    def stop(self) -> MobileRecordingStats:
        """停止录制。"""
        if not self._recording:
            return MobileRecordingStats()

        self._recording = False
        self._stop_event.set()

        if self._capture_thread:
            self._capture_thread.join(timeout=5)
        if self._touch_thread:
            self._touch_thread.join(timeout=3)

        # 关闭 scrcpy
        if self._scrcpy_proc:
            self._scrcpy_proc.terminate()
            try:
                self._scrcpy_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._scrcpy_proc.kill()
            self._scrcpy_proc = None

        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        if self._events_file:
            self._events_file.close()
            self._events_file = None

        self._log(
            f"[手机录制] 停止 | {self._frame_count} 帧, "
            f"{len(self._touch_events)} 触摸事件"
        )
        return self._save_actions()

    # ── ADB / 设备 ──

    def _adb_cmd(self, *args) -> list[str]:
        """构建 ADB 命令。"""
        cmd = ["adb"]
        if self._adb_device:
            cmd.extend(["-s", self._adb_device])
        cmd.extend(args)
        return cmd

    def _check_adb(self) -> bool:
        """检查 ADB 连接。"""
        try:
            r = subprocess.run(
                self._adb_cmd("devices"),
                capture_output=True, text=True, timeout=5,
            )
            lines = [
                l for l in r.stdout.strip().split("\n")[1:]
                if l.strip() and "device" in l
            ]
            if not lines:
                self._log("[手机录制] 未检测到 ADB 设备，请检查 USB 连接和调试模式")
                return False
            self._log(f"[手机录制] ADB 设备: {lines[0].split()[0]}")
            if not self._adb_device:
                self._adb_device = lines[0].split()[0]
            return True
        except FileNotFoundError:
            self._log("[手机录制] 未找到 adb 命令，请安装 Android SDK Platform Tools")
            return False
        except Exception as e:
            self._log(f"[手机录制] ADB 检查失败: {e}")
            return False

    def _get_device_info(self):
        """获取设备屏幕分辨率和触摸设备信息。"""
        # 屏幕分辨率
        try:
            r = subprocess.run(
                self._adb_cmd("shell", "wm", "size"),
                capture_output=True, text=True, timeout=5,
            )
            match = re.search(r"(\d+)x(\d+)", r.stdout)
            if match:
                self._screen_w = int(match.group(1))
                self._screen_h = int(match.group(2))
                self._log(f"[手机录制] 屏幕分辨率: {self._screen_w}x{self._screen_h}")
        except Exception as e:
            self._log(f"[手机录制] 获取分辨率失败: {e}")

        # 触摸设备和坐标范围
        try:
            r = subprocess.run(
                self._adb_cmd("shell", "getevent", "-lp"),
                capture_output=True, text=True, timeout=5,
            )
            self._parse_input_devices(r.stdout)
        except Exception as e:
            self._log(f"[手机录制] 获取输入设备信息失败: {e}")

    def _parse_input_devices(self, info: str):
        """解析 getevent -lp 输出，找到触摸设备和坐标范围。"""
        current_device = ""
        is_touch = False

        for line in info.split("\n"):
            if line.startswith("add device"):
                current_device = line.split(":")[-1].strip()
                is_touch = False
            elif "ABS_MT_POSITION_X" in line:
                is_touch = True
                match = re.search(r"max\s+(\d+)", line)
                if match:
                    self._touch_max_x = int(match.group(1))
            elif "ABS_MT_POSITION_Y" in line and is_touch:
                match = re.search(r"max\s+(\d+)", line)
                if match:
                    self._touch_max_y = int(match.group(1))

        if is_touch and current_device:
            self._input_device = current_device
            self._log(
                f"[手机录制] 触摸设备: {current_device} | "
                f"坐标范围: {self._touch_max_x}x{self._touch_max_y}"
            )

        # 回退
        if not self._touch_max_x:
            self._touch_max_x = self._screen_w or 1080
        if not self._touch_max_y:
            self._touch_max_y = self._screen_h or 2400

    # ── scrcpy ──

    def _start_scrcpy(self) -> bool:
        """启动 scrcpy 投屏（只看不控）。"""
        cmd = ["scrcpy", "--no-control", "--no-audio", "--window-title", "scrcpy-mirror"]
        if self._adb_device:
            cmd.extend(["-s", self._adb_device])
        cmd.extend(self._scrcpy_args)

        try:
            self._scrcpy_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._log(f"[手机录制] scrcpy 已启动 (PID={self._scrcpy_proc.pid})")
            return True
        except FileNotFoundError:
            self._log("[手机录制] 未找到 scrcpy，请安装: https://github.com/Genymobile/scrcpy")
            return False
        except Exception as e:
            self._log(f"[手机录制] scrcpy 启动失败: {e}")
            return False

    # ── 截屏线程 ──

    def _capture_loop(self):
        """截屏线程：捕获 scrcpy 窗口。"""
        try:
            import mss
        except ImportError:
            self._log("[手机录制] 需要安装 mss: pip install mss")
            self._recording = False
            return

        from .game_recorder import GameRecorder

        # 等待并查找 scrcpy 窗口
        scrcpy_region = None
        for _ in range(20):
            scrcpy_region = GameRecorder._find_window("scrcpy-mirror")
            if scrcpy_region:
                break
            time.sleep(0.5)

        if not scrcpy_region:
            self._log("[手机录制] 未找到 scrcpy 窗口")
            self._recording = False
            return

        self._log(f"[手机录制] scrcpy 窗口: {scrcpy_region}")

        with mss.mss() as sct:
            while not self._stop_event.is_set():
                t0 = time.time()
                ts = t0 - self._start_time

                try:
                    # 刷新窗口位置（scrcpy 可能被移动）
                    if self._frame_count % 100 == 0 and self._frame_count > 0:
                        new_region = GameRecorder._find_window("scrcpy-mirror")
                        if new_region:
                            scrcpy_region = new_region

                    img = sct.grab(scrcpy_region)
                    frame = np.array(img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    if self._video_writer is None:
                        h, w = frame.shape[:2]
                        video_path = str(self._output_dir / "recording.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        self._video_writer = cv2.VideoWriter(
                            video_path, fourcc, self._fps, (w, h)
                        )

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

    # ── ADB 触摸采集线程 ──

    def _touch_loop(self):
        """通过 adb getevent 采集触摸事件。"""
        cmd = self._adb_cmd("shell", "getevent", "-lt")

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except Exception as e:
            self._log(f"[手机录制] getevent 启动失败: {e}")
            return

        # 多点触控状态追踪
        current_slot = 0
        touch_x = {}  # slot → raw_x
        touch_y = {}  # slot → raw_y
        tracking_ids = {}  # slot → tracking_id

        try:
            while not self._stop_event.is_set():
                line = proc.stdout.readline()
                if not line:
                    break

                parsed = self._parse_getevent_line(line.strip())
                if not parsed:
                    continue

                evt_type, evt_code, evt_value = parsed
                ts = time.time() - self._start_time

                if evt_code == "ABS_MT_SLOT":
                    current_slot = evt_value

                elif evt_code == "ABS_MT_TRACKING_ID":
                    if evt_value == -1:
                        # 手指离开
                        if current_slot in touch_x:
                            nx = touch_x[current_slot] / max(self._touch_max_x, 1)
                            ny = touch_y[current_slot] / max(self._touch_max_y, 1)
                            action = self._match_zone(nx, ny)
                            evt = {
                                "time": round(ts, 4),
                                "type": "touch_up",
                                "slot": current_slot,
                                "x": round(nx, 4),
                                "y": round(ny, 4),
                                "action": action,
                            }
                            with self._lock:
                                self._touch_events.append(evt)
                                self._active_touches.pop(current_slot, None)
                            self._write_event(evt)
                            touch_x.pop(current_slot, None)
                            touch_y.pop(current_slot, None)
                        tracking_ids.pop(current_slot, None)
                    else:
                        tracking_ids[current_slot] = evt_value

                elif evt_code == "ABS_MT_POSITION_X":
                    touch_x[current_slot] = evt_value

                elif evt_code == "ABS_MT_POSITION_Y":
                    touch_y[current_slot] = evt_value

                elif evt_type == "EV_SYN" and evt_code == "SYN_REPORT":
                    # 一帧触摸数据完整，更新活跃触摸
                    for slot in list(tracking_ids.keys()):
                        if slot in touch_x and slot in touch_y:
                            nx = touch_x[slot] / max(self._touch_max_x, 1)
                            ny = touch_y[slot] / max(self._touch_max_y, 1)
                            action = self._match_zone(nx, ny)

                            was_active = slot in self._active_touches
                            with self._lock:
                                self._active_touches[slot] = {
                                    "x": nx, "y": ny,
                                    "time": ts, "action": action,
                                }

                            if not was_active:
                                evt = {
                                    "time": round(ts, 4),
                                    "type": "touch_down",
                                    "slot": slot,
                                    "x": round(nx, 4),
                                    "y": round(ny, 4),
                                    "action": action,
                                }
                                self._touch_events.append(evt)
                                self._write_event(evt)

        except Exception as e:
            logger.debug(f"getevent 异常: {e}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    @staticmethod
    def _parse_getevent_line(line: str) -> tuple[str, str, int] | None:
        """解析 getevent -lt 输出行。

        格式: [  timestamp] /dev/input/event2: EV_ABS ABS_MT_POSITION_X 000002a8
        """
        # 跳过设备名之前的部分，找到事件数据
        match = re.search(
            r"(EV_\w+)\s+([\w]+)\s+([0-9a-fA-F]+)\s*$", line
        )
        if not match:
            return None

        evt_type = match.group(1)
        evt_code = match.group(2)
        try:
            evt_value = int(match.group(3), 16)
            # 处理有符号值（如 tracking_id = ffffffff → -1）
            if evt_value > 0x7FFFFFFF:
                evt_value -= 0x100000000
        except ValueError:
            return None

        return evt_type, evt_code, evt_value

    def _match_zone(self, nx: float, ny: float) -> str:
        """将归一化触摸坐标匹配到最近的触摸区域。"""
        if not self._touch_zones:
            return f"touch_{nx:.2f}_{ny:.2f}"

        best_zone = None
        best_dist = float("inf")

        for zone in self._touch_zones:
            dx = nx - zone.x
            dy = ny - zone.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= zone.r and dist < best_dist:
                best_zone = zone
                best_dist = dist

        if best_zone:
            return best_zone.name

        # 未匹配到任何区域 → 通用触摸
        return "touch"

    # ── 保存 ──

    def _write_event(self, evt: dict):
        """流式写入事件。"""
        if self._events_file and not self._events_file.closed:
            self._events_file.write(json.dumps(evt, ensure_ascii=False) + "\n")
            self._events_file.flush()

    def _save_actions(self) -> MobileRecordingStats:
        """生成帧-动作配对文件。"""
        if not self._frame_timestamps:
            self._log("[手机录制] 无帧数据")
            return MobileRecordingStats(output_dir=str(self._output_dir))

        duration = self._frame_timestamps[-1]
        actual_fps = len(self._frame_timestamps) / max(duration, 0.1)

        # 按时间排序触摸事件
        sorted_events = sorted(self._touch_events, key=lambda e: e["time"])
        down_events = [e for e in sorted_events if e["type"] == "touch_down"]
        up_events = [e for e in sorted_events if e["type"] == "touch_up"]

        # 增量追踪活跃触摸
        active: dict[int, str] = {}  # slot → action
        down_idx = 0
        up_idx = 0

        actions_path = str(self._output_dir / "actions.jsonl")
        action_dist = defaultdict(int)

        with open(actions_path, "w", encoding="utf-8") as f:
            for frame_idx, ts in enumerate(self._frame_timestamps, 1):
                # 推进 touch_down
                while down_idx < len(down_events) and down_events[down_idx]["time"] <= ts:
                    e = down_events[down_idx]
                    active[e["slot"]] = e["action"]
                    down_idx += 1

                # 推进 touch_up
                while up_idx < len(up_events) and up_events[up_idx]["time"] <= ts:
                    e = up_events[up_idx]
                    active.pop(e["slot"], None)
                    up_idx += 1

                # 确定主动作
                active_actions = list(active.values())
                if active_actions:
                    # 优先非 "touch" 的具名动作
                    named = [a for a in active_actions if a != "touch"]
                    action_name = named[0] if named else active_actions[0]
                else:
                    action_name = "idle"

                sample = {
                    "frame_id": frame_idx,
                    "timestamp": round(ts, 3),
                    "human_action": {
                        "type": "mobile_touch",
                        "key": action_name,
                        "raw_keys": sorted(set(active_actions)),
                        "touch_count": len(active),
                    },
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                action_dist[action_name] += 1

        self._log(f"[手机录制] 动作标注: {actions_path}")
        self._log(f"[手机录制] 动作分布: {dict(action_dist)}")

        video_path = str(self._output_dir / "recording.mp4")
        stats = MobileRecordingStats(
            total_frames=self._frame_count,
            total_touch_events=len(self._touch_events),
            duration_sec=round(duration, 1),
            action_dist=dict(action_dist),
            fps_actual=round(actual_fps, 1),
            output_dir=str(self._output_dir),
            video_path=video_path,
            actions_path=actions_path,
            device=self._adb_device,
            resolution=(self._screen_w, self._screen_h),
        )

        # 保存元数据
        meta = {
            "total_frames": stats.total_frames,
            "total_touch_events": stats.total_touch_events,
            "duration_sec": stats.duration_sec,
            "fps_target": self._fps,
            "fps_actual": stats.fps_actual,
            "action_dist": stats.action_dist,
            "device": self._adb_device,
            "screen_resolution": [self._screen_w, self._screen_h],
            "touch_max": [self._touch_max_x, self._touch_max_y],
            "touch_zones": {
                z.name: {"x": z.x, "y": z.y, "r": z.r}
                for z in self._touch_zones
            },
            "video_path": video_path,
            "actions_path": actions_path,
            "source": "mobile",
        }
        with open(self._output_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 保存触摸区域配置（方便复用）
        if self._touch_zones:
            zones_path = self._output_dir / "touch_zones.json"
            with open(zones_path, "w", encoding="utf-8") as f:
                json.dump(meta["touch_zones"], f, ensure_ascii=False, indent=2)

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

    # ── 辅助工具 ──

    @staticmethod
    def check_prerequisites() -> dict[str, bool]:
        """检查前提条件是否满足。"""
        result = {"adb": False, "scrcpy": False, "device": False}
        try:
            r = subprocess.run(
                ["adb", "version"], capture_output=True, text=True, timeout=3
            )
            result["adb"] = r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            r = subprocess.run(
                ["scrcpy", "--version"], capture_output=True, text=True, timeout=3
            )
            result["scrcpy"] = r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if result["adb"]:
            try:
                r = subprocess.run(
                    ["adb", "devices"], capture_output=True, text=True, timeout=5
                )
                lines = [
                    l for l in r.stdout.strip().split("\n")[1:]
                    if l.strip() and "device" in l
                ]
                result["device"] = len(lines) > 0
            except Exception:
                pass

        return result

    @staticmethod
    def create_default_zones(game: str = "moba") -> dict[str, dict]:
        """生成常见游戏类型的默认触摸区域。

        Returns:
            {"action_name": {"x": float, "y": float, "r": float}, ...}
        """
        if game == "moba":
            # 王者荣耀/LoL 手游 典型布局（横屏）
            return {
                "move":     {"x": 0.13, "y": 0.72, "r": 0.10},  # 左侧摇杆
                "skill_1":  {"x": 0.82, "y": 0.72, "r": 0.05},  # 技能1
                "skill_2":  {"x": 0.88, "y": 0.58, "r": 0.05},  # 技能2
                "skill_3":  {"x": 0.94, "y": 0.42, "r": 0.05},  # 技能3（大招）
                "attack":   {"x": 0.92, "y": 0.72, "r": 0.06},  # 普攻
                "spell_1":  {"x": 0.72, "y": 0.88, "r": 0.04},  # 召唤师技能1
                "spell_2":  {"x": 0.78, "y": 0.88, "r": 0.04},  # 召唤师技能2
                "recall":   {"x": 0.60, "y": 0.88, "r": 0.04},  # 回城
                "minimap":  {"x": 0.10, "y": 0.25, "r": 0.08},  # 小地图
            }
        elif game == "fps":
            # FPS 手游（和平精英等）
            return {
                "move":     {"x": 0.13, "y": 0.72, "r": 0.10},
                "aim":      {"x": 0.70, "y": 0.50, "r": 0.20},  # 右侧视角区
                "fire":     {"x": 0.92, "y": 0.65, "r": 0.06},
                "scope":    {"x": 0.88, "y": 0.45, "r": 0.05},
                "reload":   {"x": 0.82, "y": 0.80, "r": 0.04},
                "crouch":   {"x": 0.75, "y": 0.85, "r": 0.04},
                "jump":     {"x": 0.90, "y": 0.85, "r": 0.04},
            }
        else:
            return {}
