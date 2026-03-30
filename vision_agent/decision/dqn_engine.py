"""DQN 决策引擎：加载自对弈训练的模型，通过标准 Agent 接口执行。

PC 模式：输出键盘/鼠标动作（通过 action_key_map）

用法:
    engine = DQNEngine(
        model_dir="runs/selfplay/exp1/latest",
        action_key_map={"attack": {"key": "j"}, "skill1": {"key": "u"}},
    )
    engine.on_start()
    actions = engine.decide(embedding=frame)
"""

import json
import logging
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
        action_key_map: dict | None = None,
        confidence_threshold: float = 0.3,
        execute_actions: bool = True,
    ):
        self._model_dir = model_dir
        self._action_key_map = action_key_map or {}
        self._confidence_threshold = confidence_threshold
        self._execute_actions = execute_actions

        self._model = None
        self._encoder = None
        self._action_list: list[str] = []
        self._meta: dict = {}

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
            self._execute_key(action_name)

        params = {"q_value": round(q_max, 3), "action_idx": action_idx}

        key_info = self._action_key_map.get(action_name, {})
        if key_info:
            params["key"] = key_info.get("key", "")

        return [Action(
            name=action_name,
            confidence=confidence,
            reason=f"DQN Q={q_max:.2f} (conf={confidence:.2f})",
            parameters=params,
        )]

    def _execute_key(self, action_name: str):
        """PC 端按键执行。"""
        if action_name == "idle":
            return

        key_info = self._action_key_map.get(action_name)
        if not key_info:
            return

        try:
            import pynput.keyboard as kb
            controller = kb.Controller()
            key = key_info.get("key", "")
            if key:
                controller.press(key)
                controller.release(key)
        except Exception as e:
            logger.debug(f"按键执行失败: {e}")
