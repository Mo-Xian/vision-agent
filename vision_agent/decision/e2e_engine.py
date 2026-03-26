"""端到端决策引擎：截图 → MobileNetV3 编码 → MLP 推理 → 动作。

完整的视觉到决策推理管线，无需 YOLO 或手工特征。
CPU 推理延迟：~60-80ms/帧（编码 ~50ms + MLP <1ms）。
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch

from .base import Action, DecisionEngine

logger = logging.getLogger(__name__)


class E2EEngine(DecisionEngine):
    """端到端决策引擎。

    加载训练好的 E2E MLP 模型，配合 VisionEncoder 实现
    截图 → 特征 → 动作 的完整推理。

    用法:
        engine = E2EEngine(model_dir="runs/e2e/exp1")
        engine.on_start()
        action = engine.decide(detection_result, state, frame=frame)
    """

    name = "e2e"

    def __init__(
        self,
        model_dir: str = "",
        action_key_map: dict | None = None,
        confidence_threshold: float = 0.3,
    ):
        self._model_dir = model_dir
        self._action_key_map = action_key_map or {}
        self._confidence_threshold = confidence_threshold
        self._model = None
        self._encoder = None
        self._action_list: list[str] = []
        self._meta: dict = {}

    def on_start(self):
        """加载编码器和模型。"""
        # 加载视觉编码器
        from ..core.vision_encoder import VisionEncoder
        self._encoder = VisionEncoder()

        # 加载模型
        if not self._model_dir:
            logger.warning("E2EEngine: 未指定模型目录")
            return

        model_dir = Path(self._model_dir)
        meta_path = model_dir / "model.meta.json"
        model_path = model_dir / "model.pt"

        if not meta_path.exists() or not model_path.exists():
            logger.error(f"E2EEngine: 模型文件不存在: {model_dir}")
            return

        with open(meta_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)

        self._action_list = self._meta.get("action_list", [])
        num_actions = self._meta.get("num_actions", len(self._action_list))
        embed_dim = self._meta.get("embed_dim", 576)
        hidden_dims = self._meta.get("hidden_dims", [256, 128])

        from ..data.e2e_trainer import E2EMLP
        self._model = E2EMLP(
            input_dim=embed_dim,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
        )
        self._model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self._model.eval()

        logger.info(
            f"E2EEngine 加载完成: {num_actions} 动作, "
            f"val_acc={self._meta.get('best_val_acc', 0):.3f}"
        )

    def on_stop(self):
        self._model = None
        self._encoder = None

    def decide(self, detection_result, state, **kwargs) -> Action:
        """端到端决策：从画面帧直接推理动作。

        需要 kwargs['frame'] 传入原始 BGR 帧。
        """
        frame = kwargs.get("frame")
        if frame is None or self._model is None or self._encoder is None:
            return Action(tool_name="idle", confidence=0.0, reason="E2E 引擎未就绪")

        # 编码
        embedding = self._encoder.encode(frame)
        tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

        # 推理
        action_idx, confidence = self._model.predict_action(tensor)

        if action_idx < len(self._action_list):
            action_name = self._action_list[action_idx]
        else:
            action_name = "idle"

        # 置信度过低时 idle
        if confidence < self._confidence_threshold:
            return Action(
                tool_name="idle", confidence=confidence,
                reason=f"置信度不足 ({confidence:.2f} < {self._confidence_threshold})",
            )

        # 映射到按键
        key_info = self._action_key_map.get(action_name, {})

        return Action(
            tool_name=action_name,
            confidence=confidence,
            reason=f"E2E 视觉决策 (conf={confidence:.2f})",
            parameters={"key": key_info.get("key", "")} if key_info else {},
        )
