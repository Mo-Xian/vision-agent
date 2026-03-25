"""基于训练模型的决策引擎。

加载 DecisionTrainer 训练产出的模型和元数据，
在 Pipeline 中替代 RuleEngine 做实时决策。
"""

import json
import logging
from pathlib import Path

import numpy as np

from ..core.detector import DetectionResult
from ..core.state import SceneState
from .base import Action, DecisionEngine

logger = logging.getLogger(__name__)


class TrainedEngine(DecisionEngine):
    """用训练好的分类模型做决策。

    用法:
        engine = TrainedEngine("runs/decision/exp1")
        agent = ActionAgent(decision_engine=engine, tool_registry=tools)
    """

    # 语义动作名 → 无操作（不产生 Action）
    IDLE_ACTIONS = {"idle", "none", "wait", "no_action", "noop"}

    def __init__(
        self,
        model_dir: str,
        confidence_threshold: float = 0.3,
        action_config: dict[str, dict] | None = None,
        action_key_map: dict[str, dict] | None = None,
    ):
        """
        Args:
            model_dir: 训练产出目录（包含 model.pt/model.joblib + model.meta.json）
            confidence_threshold: 低于此置信度的预测不执行
            action_config: 动作键 → 工具参数映射（完整格式），如:
                {
                    "press:space": {"tool_name": "keyboard", "parameters": {"action": "press", "key": "space"}},
                }
            action_key_map: 语义动作名 → 实际按键/操作的简化映射，如:
                {
                    "attack": {"type": "key", "key": "a"},
                    "retreat": {"type": "key", "key": "s"},
                    "skill_1": {"type": "key", "key": "1"},
                    "idle": {"type": "none"},
                }
                type 可选: "key"(键盘按键), "click"(鼠标点击), "none"(无操作)
        """
        self._model_dir = Path(model_dir)
        self._confidence_threshold = confidence_threshold
        self._action_config = action_config or {}
        # action_key_map: 优先用参数传入，否则尝试从 model_dir/action_key_map.json 加载
        if action_key_map:
            self._action_key_map = action_key_map
        else:
            map_file = self._model_dir / "action_key_map.json"
            if map_file.exists():
                with open(map_file, "r", encoding="utf-8") as f:
                    self._action_key_map = json.load(f)
                logger.info(f"从 {map_file} 加载动作映射: {list(self._action_key_map.keys())}")
            else:
                self._action_key_map = {}

        # 加载元数据
        meta_path = self._model_dir / "model.meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)

        self._model_type = self._meta["model_type"]
        self._label_to_action = self._meta["label_to_action"]
        self._detection_classes = self._meta["detection_classes"]
        self._feature_mean = np.array(self._meta["feature_mean"], dtype=np.float32)
        self._feature_std = np.array(self._meta["feature_std"], dtype=np.float32)

        # 加载模型
        self._model = None
        self._load_model()

        logger.info(
            f"TrainedEngine 加载完成 | 模型: {self._model_type} | "
            f"动作数: {self._meta['num_classes']} | 阈值: {confidence_threshold}"
        )

    def _load_model(self):
        if self._model_type == "mlp":
            from ..data.train import ActionMLP
            self._model = ActionMLP(
                input_dim=self._meta["input_dim"],
                num_classes=self._meta["num_classes"],
                hidden_dims=self._meta.get("hidden_dims", [128, 64]),
                dropout=self._meta.get("dropout", 0.3),
            )
            model_path = self._model_dir / "model.pt"
            self._model.load(str(model_path))
        elif self._model_type == "rf":
            import joblib
            model_path = self._model_dir / "model.joblib"
            self._model = joblib.load(str(model_path))
        else:
            raise ValueError(f"未知模型类型: {self._model_type}")

    def decide(self, result: DetectionResult, state: SceneState) -> list[Action]:
        features = self._extract_features(result, state)
        safe_std = np.where(self._feature_std == 0, 1.0, self._feature_std)
        features_norm = (features - self._feature_mean) / safe_std

        if self._model_type == "mlp":
            proba = self._model.predict_proba(features_norm.reshape(1, -1))[0]
        else:
            proba = self._model.predict_proba(features_norm.reshape(1, -1))[0]

        best_idx = int(np.argmax(proba))
        best_conf = float(proba[best_idx])

        if best_conf < self._confidence_threshold:
            return []

        action_key = self._label_to_action.get(str(best_idx), "")
        if not action_key:
            return []

        action = self._action_key_to_action(action_key, best_conf, result)
        if action is None:
            return []

        return [action]

    def _extract_features(self, result: DetectionResult, state: SceneState) -> np.ndarray:
        """与训练时一致的特征提取逻辑。"""
        counts = state.object_counts
        count_features = [counts.get(c, 0) for c in self._detection_classes]

        if result.detections:
            det = result.detections[0]
            cx = (det.bbox_norm[0] + det.bbox_norm[2]) / 2
            cy = (det.bbox_norm[1] + det.bbox_norm[3]) / 2
            conf = det.confidence
        else:
            cx, cy, conf = 0.0, 0.0, 0.0

        feature_vec = count_features + [cx, cy, conf, len(result.detections)]
        return np.array(feature_vec, dtype=np.float32)

    def _action_key_to_action(self, action_key: str, confidence: float,
                              result: DetectionResult) -> Action | None:
        """将 action_key (如 "press:space" 或语义名 "attack") 转换为 Action 对象。"""
        # 优先使用完整 action_config
        if action_key in self._action_config:
            cfg = self._action_config[action_key]
            return Action(
                tool_name=cfg["tool_name"],
                parameters=cfg.get("parameters", {}),
                reason=f"trained:{action_key}",
                confidence=confidence,
            )

        # 解析 action_key 格式
        parts = action_key.split(":", 1)

        # ---- 处理 "press:xxx" 格式 ----
        if len(parts) == 2:
            action_type, detail = parts

            # 检查 detail 是否为语义动作名（如 "press:idle", "press:attack"）
            semantic = self._resolve_semantic(detail)
            if semantic is not None:
                return semantic(action_key, confidence, result)

            # 真实按键: "press:space", "press:a"
            if action_type in ("press", "release"):
                return Action(
                    tool_name="keyboard",
                    parameters={"action": action_type, "key": detail},
                    reason=f"trained:{action_key}",
                    confidence=confidence,
                )

            # 鼠标: "click:left", "click:right"
            if action_type == "click":
                params = {"action": "click"}
                if detail == "right":
                    params["action"] = "right_click"
                if result.detections:
                    det = result.detections[0]
                    cx = (det.bbox[0] + det.bbox[2]) / 2
                    cy = (det.bbox[1] + det.bbox[3]) / 2
                    params["x"] = int(cx)
                    params["y"] = int(cy)
                return Action(
                    tool_name="mouse",
                    parameters=params,
                    reason=f"trained:{action_key}",
                    confidence=confidence,
                )

        # ---- 处理纯语义名（如 "attack", "idle"）----
        else:
            semantic = self._resolve_semantic(action_key)
            if semantic is not None:
                return semantic(action_key, confidence, result)

        logger.warning(f"未处理的 action_key: {action_key}")
        return None

    def _resolve_semantic(self, name: str):
        """解析语义动作名，返回一个生成 Action 的 callable 或 None。

        返回值是 callable(action_key, confidence, result) -> Action | None
        如果 name 不是已知的语义名，返回 None。
        """
        name_lower = name.lower()

        # 1. idle 类动作 → 不执行
        if name_lower in self.IDLE_ACTIONS:
            return lambda ak, conf, res: None

        # 2. 用户定义的 action_key_map
        if name in self._action_key_map:
            mapping = self._action_key_map[name]
            return self._make_action_from_mapping(mapping)
        # 大小写不敏感再查一次
        for k, v in self._action_key_map.items():
            if k.lower() == name_lower:
                return self._make_action_from_mapping(v)

        # 3. 不在映射中且不是明显的真实按键 → 视为未知语义名，跳过
        # 判断是否为单字符或已知键名（真实按键）
        known_keys = {
            "space", "enter", "tab", "escape", "esc", "backspace", "delete",
            "up", "down", "left", "right", "shift", "ctrl", "alt",
            "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
        }
        if len(name) == 1 or name_lower in known_keys:
            return None  # 当作真实按键，交给外层处理

        # 多字符语义名但未映射 → 记日志、跳过
        logger.info(f"语义动作 '{name}' 未配置映射，跳过执行")
        return lambda ak, conf, res: None

    @staticmethod
    def _make_action_from_mapping(mapping: dict):
        """根据映射配置创建 Action 工厂函数。"""
        map_type = mapping.get("type", "key")

        if map_type == "none":
            return lambda ak, conf, res: None

        if map_type == "key":
            key = mapping["key"]
            action_type = mapping.get("action", "press")
            def factory(ak, conf, res):
                return Action(
                    tool_name="keyboard",
                    parameters={"action": action_type, "key": key},
                    reason=f"trained:{ak}→{key}",
                    confidence=conf,
                )
            return factory

        if map_type == "click":
            button = mapping.get("button", "left")
            def factory(ak, conf, res):
                params = {"action": "click" if button == "left" else "right_click"}
                if res.detections:
                    det = res.detections[0]
                    cx = (det.bbox[0] + det.bbox[2]) / 2
                    cy = (det.bbox[1] + det.bbox[3]) / 2
                    params["x"] = int(cx)
                    params["y"] = int(cy)
                return Action(
                    tool_name="mouse",
                    parameters=params,
                    reason=f"trained:{ak}→click:{button}",
                    confidence=conf,
                )
            return factory

        # 未知 type
        return lambda ak, conf, res: None
