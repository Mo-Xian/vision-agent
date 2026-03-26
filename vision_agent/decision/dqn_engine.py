"""DQN 决策引擎：加载自对弈训练的模型，通过标准 Agent 接口执行。

支持两种执行模式：
  - PC 模式：输出键盘/鼠标动作（通过 action_key_map）
  - 手机模式：通过 ADB 发送触控操作（通过 touch_zones）

用法:
    engine = DQNEngine(
        model_dir="runs/selfplay/exp1/latest",
        touch_zones=[{"name": "idle"}, {"name": "attack", "x": 0.85, "y": 0.85, "r": 0.06}],
        device_serial="192.168.1.5:5555",
    )
    engine.on_start()
    actions = engine.decide(embedding=frame)
"""

import json
import logging
import math
import random
import re
import subprocess
from pathlib import Path

import numpy as np
import torch

from .base import Action, DecisionEngine

logger = logging.getLogger(__name__)


class DQNEngine(DecisionEngine):
    """DQN 决策引擎 — 自对弈模型的 Agent 部署接口。"""

    name = "dqn"

    def __init__(
        self,
        model_dir: str = "",
        touch_zones: list[dict] | None = None,
        action_key_map: dict | None = None,
        device_serial: str = "",
        confidence_threshold: float = 0.3,
        execute_actions: bool = True,
    ):
        self._model_dir = model_dir
        self._touch_zones = touch_zones or []
        self._action_key_map = action_key_map or {}
        self._device_serial = device_serial
        self._confidence_threshold = confidence_threshold
        self._execute_actions = execute_actions

        self._model = None
        self._encoder = None
        self._action_list: list[str] = []
        self._meta: dict = {}
        self._screen_w = 0
        self._screen_h = 0

    def on_start(self):
        from ..core.vision_encoder import VisionEncoder
        self._encoder = VisionEncoder()

        if not self._model_dir:
            logger.warning("DQNEngine: 未指定模型目录")
            return

        model_dir = Path(self._model_dir)
        meta_path = model_dir / "model.meta.json"
        model_path = model_dir / "model.pt"

        if not model_path.exists():
            logger.error(f"DQNEngine: 模型不存在: {model_path}")
            return

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)

        self._action_list = self._meta.get("action_list", [])
        num_actions = self._meta.get("num_actions", len(self._action_list))
        embed_dim = self._meta.get("embed_dim", 576)
        hidden_dims = self._meta.get("hidden_dims", [256, 128])

        from ..rl.dqn_agent import DQNNetwork
        self._model = DQNNetwork(embed_dim, num_actions, hidden_dims)
        self._model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self._model.eval()

        # 获取手机分辨率（用于触控执行）
        if self._touch_zones and self._device_serial:
            self._get_screen_size()

        logger.info(
            f"DQNEngine 加载完成: {num_actions} 动作, "
            f"train_steps={self._meta.get('train_steps', 0)}, "
            f"actions={self._action_list}"
        )

    def on_stop(self):
        self._model = None
        self._encoder = None

    def decide(self, embedding=None, **context) -> list[Action]:
        frame = embedding
        if frame is None or self._model is None or self._encoder is None:
            return [Action(name="idle", confidence=0.0, reason="DQN 引擎未就绪")]

        # 编码帧
        if isinstance(frame, np.ndarray) and frame.ndim == 3:
            frame = self._encoder.encode(frame)

        # Q 值推理
        with torch.no_grad():
            tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
            q_values = self._model(tensor).squeeze(0).numpy()

        # 选择最优动作
        action_idx = int(np.argmax(q_values))
        q_max = float(q_values[action_idx])

        # 用 softmax 作为置信度
        exp_q = np.exp(q_values - q_values.max())
        probs = exp_q / exp_q.sum()
        confidence = float(probs[action_idx])

        if action_idx < len(self._action_list):
            action_name = self._action_list[action_idx]
        else:
            action_name = "idle"

        if confidence < self._confidence_threshold:
            return [Action(
                name="idle", confidence=confidence,
                reason=f"置信度不足 ({confidence:.2f} < {self._confidence_threshold})",
            )]

        # 执行动作
        if self._execute_actions:
            self._execute(action_idx, action_name)

        params = {"q_value": round(q_max, 3), "action_idx": action_idx}

        # PC 模式：附加按键信息
        key_info = self._action_key_map.get(action_name, {})
        if key_info:
            params["key"] = key_info.get("key", "")

        return [Action(
            name=action_name,
            confidence=confidence,
            reason=f"DQN Q={q_max:.2f} (conf={confidence:.2f})",
            parameters=params,
        )]

    # ── 动作执行 ──

    def _execute(self, action_idx: int, action_name: str):
        """根据 touch_zones 配置执行动作。"""
        if action_name == "idle":
            return

        # 手机模式：ADB 触控
        if self._touch_zones and self._device_serial:
            self._execute_touch(action_idx)
            return

        # PC 模式：键盘按键
        key_info = self._action_key_map.get(action_name)
        if key_info:
            self._execute_key(key_info)

    def _execute_touch(self, action_idx: int):
        """通过 ADB 执行触控。"""
        if action_idx < 0 or action_idx >= len(self._touch_zones):
            return

        zone = self._touch_zones[action_idx]
        if zone.get("name") == "idle":
            return

        if not self._screen_w:
            self._get_screen_size()

        nx = zone.get("x", 0.5)
        ny = zone.get("y", 0.5)
        radius = zone.get("r", 0)
        px = int(nx * self._screen_w)
        py = int(ny * self._screen_h)

        action_type = zone.get("type", "")
        if not action_type:
            action_type = "swipe" if zone.get("name") in ("move", "aim") else "tap"

        if action_type == "swipe":
            angle = random.uniform(0, 2 * math.pi)
            r_px = int(radius * self._screen_w * 0.6)
            end_x = px + int(r_px * math.cos(angle))
            end_y = py + int(r_px * math.sin(angle))
            self._adb_cmd("shell", "input", "swipe",
                          str(px), str(py), str(end_x), str(end_y), "200")
        elif action_type == "hold":
            self._adb_cmd("shell", "input", "swipe",
                          str(px), str(py), str(px), str(py), "500")
        else:
            self._adb_cmd("shell", "input", "tap", str(px), str(py))

    def _execute_key(self, key_info: dict):
        """PC 端按键执行。"""
        try:
            import pynput.keyboard as kb
            controller = kb.Controller()
            key = key_info.get("key", "")
            if key:
                controller.press(key)
                controller.release(key)
        except Exception as e:
            logger.debug(f"按键执行失败: {e}")

    # ── ADB ──

    def _adb_cmd(self, *args):
        import shutil
        adb = shutil.which("adb") or "adb"
        cmd = [adb]
        if self._device_serial:
            cmd.extend(["-s", self._device_serial])
        cmd.extend(args)
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.debug(f"ADB 命令失败: {e}")

    def _get_screen_size(self):
        import shutil
        adb = shutil.which("adb") or "adb"
        cmd = [adb]
        if self._device_serial:
            cmd.extend(["-s", self._device_serial])
        cmd.extend(["shell", "wm", "size"])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            m = re.search(r"(\d+)x(\d+)", r.stdout)
            if m:
                self._screen_w = int(m.group(1))
                self._screen_h = int(m.group(2))
        except Exception:
            self._screen_w = 2400
            self._screen_h = 1080
