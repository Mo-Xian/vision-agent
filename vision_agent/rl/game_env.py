"""游戏环境封装：截图 → 动作执行 → 奖励 → 下一状态。

将游戏抽象为标准 RL 环境接口：
  - reset():  重置环境，返回初始状态
  - step(action): 执行动作，返回 (next_state, reward, done, info)

通过 RemoteHub 获取远程设备画面，通过 RemoteHub 发送触控/按键指令。
支持 PC 客户端（RemoteCaptureClient）和 Android 客户端（App）。
"""

import logging
import math
import random
import time

import cv2
import numpy as np

from .reward import GameState, RewardConfig, create_reward_detector

logger = logging.getLogger(__name__)


class GameEnvironment:
    """游戏 RL 环境（基于 RemoteHub 远程控制）。

    通过 RemoteHub 获取远程设备画面并发送操控指令，
    提供标准的 reset/step 接口。

    支持 ONNX 模型检测对局开始（wzry_ai start.onnx），
    不可用时直接开始采集。

    Args:
        action_zones: 动作空间定义，格式:
            [{"name": "idle"}, {"name": "move", "x": 0.13, "y": 0.72, "r": 0.10}, ...]
            第 0 个必须是 idle（无操作）
        hub: RemoteHub 实例，用于获取画面和发送指令
        reward_config: 奖励配置
        start_model_path: 对局开始检测 ONNX 模型路径
        fps: 帧率控制（每秒最多执行多少步）
        on_log: 日志回调
    """

    def __init__(
        self,
        action_zones: list[dict],
        hub=None,
        reward_config: RewardConfig | None = None,
        start_model_path: str = "models/start.onnx",
        fps: int = 5,
        on_log=None,
    ):
        self._action_zones = action_zones
        self._hub = hub
        self._fps = fps
        self._frame_interval = 1.0 / fps
        self._on_log = on_log
        self._start_model_path = start_model_path

        self._reward_detector = create_reward_detector(reward_config)
        self._start_detector = None  # ONNX 对局开始检测
        self._encoder = None  # 延迟初始化

        # 设备信息（从 hub.client_meta 获取）
        self._screen_w = 0
        self._screen_h = 0

        self._step_count = 0
        self._episode_reward = 0.0

    @property
    def num_actions(self) -> int:
        return len(self._action_zones)

    @property
    def action_names(self) -> list[str]:
        return [z["name"] for z in self._action_zones]

    def setup(self):
        """初始化环境：加载编码器、检测模型、获取设备分辨率。"""
        from ..core.vision_encoder import VisionEncoder
        self._encoder = VisionEncoder()

        # 加载对局开始检测 ONNX 模型
        from .onnx_detector import OnnxDetector
        if self._start_model_path:
            det = OnnxDetector(self._start_model_path, classes=["started"])
            if det.is_available:
                self._start_detector = det
                self._log(f"[环境] 对局开始检测模型已加载: {self._start_model_path}")
            else:
                self._log(f"[环境] 对局开始检测模型不存在 ({self._start_model_path})，将直接开始采集")

        # 从 RemoteHub 获取设备分辨率
        if self._hub:
            self._screen_w, self._screen_h = self._hub.screen_size
            if not self._screen_w:
                # 等待客户端发送 meta 信息
                self._log("[环境] 等待远程设备分辨率信息...")
                for _ in range(20):
                    time.sleep(0.5)
                    self._screen_w, self._screen_h = self._hub.screen_size
                    if self._screen_w:
                        break

            if not self._screen_w:
                self._screen_w, self._screen_h = 1920, 1080
                self._log("[环境] 未获取到分辨率，使用默认 1920x1080")

        self._log(
            f"[环境] 就绪 | 动作数={self.num_actions} | "
            f"屏幕={self._screen_w}x{self._screen_h}"
        )

    def teardown(self):
        """清理环境（RemoteHub 由外部管理，这里不停止）。"""
        pass

    def reset(self) -> np.ndarray | None:
        """重置环境，返回初始状态（576 维嵌入向量）。"""
        self._reward_detector.reset()
        self._step_count = 0
        self._episode_reward = 0.0

        frame = self._capture_frame()
        if frame is None:
            return None

        return self._encoder.encode(frame)

    def step(self, action_idx: int) -> tuple[np.ndarray | None, float, bool, dict]:
        """执行一步：动作 → 截图 → 奖励。

        Args:
            action_idx: 动作索引

        Returns:
            (next_state, reward, done, info)
            - next_state: 576 维嵌入向量
            - reward: 奖励值
            - done: 是否终局
            - info: 调试信息
        """
        t0 = time.time()

        # 执行动作
        self._execute_action(action_idx)

        # 等待帧间隔
        elapsed = time.time() - t0
        if elapsed < self._frame_interval:
            time.sleep(self._frame_interval - elapsed)

        # 获取下一帧
        frame = self._capture_frame()
        if frame is None:
            return None, 0.0, True, {"error": "capture_failed"}

        # 检测奖励
        game_state = self._reward_detector.detect(frame, action_idx)

        # 编码
        next_state = self._encoder.encode(frame)

        self._step_count += 1
        self._episode_reward += game_state.raw_reward

        info = {
            "step": self._step_count,
            "action": self._action_zones[action_idx]["name"],
            "reward": round(game_state.raw_reward, 3),
            "my_hp": round(game_state.my_hp_ratio, 3),
            "enemy_hp": round(game_state.enemy_hp_ratio, 3),
            "is_dead": game_state.is_dead,
            "episode_reward": round(self._episode_reward, 3),
        }

        done = game_state.is_finished or game_state.is_dead

        return next_state, game_state.raw_reward, done, info

    def wait_for_game_start(self, timeout: float = 300) -> bool:
        """等待对局开始。

        优先使用 ONNX 模型（start.onnx）检测 "started" 类别。
        不可用时直接返回 True（跳过等待）。
        """
        if self._start_detector is None:
            self._log("[环境] 无对局检测模型，直接开始")
            return True

        self._log("[环境] 等待对局开始（检测 start.onnx）...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            frame = self._capture_frame()
            if frame is not None:
                label = self._start_detector.detect(frame)
                if label == "started":
                    self._log("[环境] 检测到对局已开始!")
                    return True

            time.sleep(0.5)

        self._log(f"[环境] 等待超时 ({timeout}s)")
        return False

    def check_game_finished(self, frame: np.ndarray) -> tuple[bool, str]:
        """检查对局是否结束。

        Returns:
            (is_finished, result): result 为 "win"/"lose"/""
        """
        return self._reward_detector._detect_finish(frame)

    # ── 画面获取 ──

    def _capture_frame(self) -> np.ndarray | None:
        """从 RemoteHub 获取远程设备画面。"""
        if not self._hub:
            return None
        return self._hub.get_frame()

    # ── 动作执行 ──

    def _execute_action(self, action_idx: int):
        """通过 RemoteHub 发送触控/按键指令。

        动作类型由 zone 配置的 "type" 字段决定：
          - "tap":   点击（默认）
          - "swipe": 随机方向滑动（用于摇杆/瞄准）
          - "hold":  长按
          - "idle":  无操作
        """
        if action_idx < 0 or action_idx >= len(self._action_zones):
            return
        if not self._hub:
            return

        zone = self._action_zones[action_idx]
        action_name = zone.get("name", "idle")

        if action_name == "idle":
            return

        nx = zone.get("x", 0.5)
        ny = zone.get("y", 0.5)
        radius = zone.get("r", 0)
        px = int(nx * self._screen_w)
        py = int(ny * self._screen_h)

        # 确定操作类型
        action_type = zone.get("type", "")
        if not action_type:
            if action_name in ("move", "aim"):
                action_type = "swipe"
            else:
                action_type = "tap"

        if action_type == "swipe":
            angle = random.uniform(0, 2 * math.pi)
            r_px = int(radius * self._screen_w * 0.6)
            end_x = px + int(r_px * math.cos(angle))
            end_y = py + int(r_px * math.sin(angle))
            self._hub.send_swipe(px, py, end_x, end_y, duration_ms=200)
        elif action_type == "hold":
            self._hub.send_swipe(px, py, px, py, duration_ms=500)
        else:
            self._hub.send_tap(px, py)

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass
