import logging
import time
import json
import random
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from .base import DecisionEngine, Action
from ..core.detector import DetectionResult
from ..core.state import SceneState

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """一次状态转移记录。"""
    state_features: list[float]
    action_idx: int
    reward: float
    next_state_features: list[float]
    done: bool


class RewardDetector:
    """从检测结果变化中推断奖励信号。"""

    def __init__(self, reward_rules: dict | None = None):
        """
        Args:
            reward_rules: {
                "enemy_count_decrease": 2.0,  # 敌人数量减少
                "ally_count_increase": 1.0,   # 友方数量增加
                "idle_penalty": -0.1,         # 空闲惩罚
                "hp_decrease": -1.0,          # 血量降低（需要 ROI）
            }
        """
        self._rules = reward_rules or {
            "enemy_count_decrease": 2.0,
            "ally_count_increase": 1.0,
            "idle_penalty": -0.1,
        }
        self._prev_counts: dict[str, int] = {}

    def compute_reward(self, state: SceneState) -> float:
        reward = 0.0
        counts = state.object_counts

        # 比较前后帧的目标数量变化
        for cls_name, prev_count in self._prev_counts.items():
            curr_count = counts.get(cls_name, 0)
            diff = curr_count - prev_count

            # 敌方类别减少 = 好事
            if "enemy" in cls_name.lower() and diff < 0:
                reward += abs(diff) * self._rules.get("enemy_count_decrease", 2.0)
            # 友方类别增加 = 好事
            elif "ally" in cls_name.lower() and diff > 0:
                reward += diff * self._rules.get("ally_count_increase", 1.0)

        # 空闲惩罚
        if not state.current_result.detections:
            reward += self._rules.get("idle_penalty", -0.1)

        # 使用 ROI 特征（如果有）
        roi = state.custom_data.get("roi_features", {})
        if "hp_bar" in roi:
            hp = roi["hp_bar"].get("brightness", 0.5)
            if hp < 0.3:
                reward += self._rules.get("hp_decrease", -1.0)

        self._prev_counts = counts.copy()
        return reward

    def reset(self):
        self._prev_counts = {}


class RLEngine(DecisionEngine):
    """基于 DQN 的强化学习决策引擎。

    支持两种模式：
    1. 训练模式 (training=True): 探索+学习，定期保存模型
    2. 推理模式 (training=False): 纯利用，不更新权重
    """

    def __init__(
        self,
        actions: list[str],
        action_key_map: dict[str, dict] | None = None,
        state_dim: int = 0,
        hidden_dim: int = 128,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        memory_size: int = 10000,
        training: bool = True,
        model_path: str = "",
        save_dir: str = "runs/rl",
        save_interval: int = 500,
        reward_rules: dict | None = None,
    ):
        self._actions = actions
        self._action_key_map = action_key_map or {}
        self._n_actions = len(actions)
        self._state_dim = state_dim
        self._hidden_dim = hidden_dim
        self._lr = lr
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._batch_size = batch_size
        self._training = training
        self._save_dir = save_dir
        self._save_interval = save_interval

        self._memory: deque[Transition] = deque(maxlen=memory_size)
        self._reward_detector = RewardDetector(reward_rules)
        self._model = None  # 延迟初始化（需要知道 state_dim）
        self._optimizer = None
        self._loss_fn = None
        self._step_count = 0
        self._prev_state_features: list[float] | None = None
        self._prev_action_idx: int | None = None
        self._total_reward: float = 0
        self._episode_rewards: list[float] = []
        self._on_log = None

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def set_log_callback(self, callback):
        self._on_log = callback

    def decide(self, result: DetectionResult, state: SceneState) -> list[Action]:
        # 提取状态特征
        if hasattr(state, 'to_feature_vector'):
            features = state.to_feature_vector()
        else:
            features = self._simple_features(state)

        # 初始化模型（第一次知道维度时）
        if self._model is None:
            self._state_dim = len(features)
            self._init_model()

        # 学习：用上一步的结果计算奖励，存入经验
        if self._training and self._prev_state_features is not None:
            reward = self._reward_detector.compute_reward(state)
            self._total_reward += reward
            self._memory.append(Transition(
                state_features=self._prev_state_features,
                action_idx=self._prev_action_idx,
                reward=reward,
                next_state_features=features,
                done=False,
            ))

            # 经验回放
            if len(self._memory) >= self._batch_size:
                self._train_step()

            self._step_count += 1
            if self._step_count % self._save_interval == 0:
                self._save_model()
                self._log(
                    f"[RL] step={self._step_count} epsilon={self._epsilon:.3f} "
                    f"reward={self._total_reward:.1f} memory={len(self._memory)}"
                )
                self._episode_rewards.append(self._total_reward)
                self._total_reward = 0

        # 选择动作
        action_idx = self._select_action(features)

        self._prev_state_features = features
        self._prev_action_idx = action_idx

        action_name = self._actions[action_idx]

        # 转为 Action 输出
        key_map = self._action_key_map.get(action_name, {})
        if key_map:
            return [Action(
                tool_name="keyboard",
                parameters={"action": "press", "key": key_map.get("key", action_name)},
                reason=f"RL: {action_name} (eps={self._epsilon:.2f})",
                confidence=1.0 - self._epsilon,
            )]

        return [Action(
            tool_name="keyboard",
            parameters={"action": "press", "key": action_name},
            reason=f"RL: {action_name}",
        )]

    def _select_action(self, features: list[float]) -> int:
        """epsilon-greedy 选择动作。"""
        if self._training and random.random() < self._epsilon:
            return random.randrange(self._n_actions)

        import torch
        with torch.no_grad():
            state_t = torch.FloatTensor(features).unsqueeze(0)
            q_values = self._model(state_t)
            return int(q_values.argmax(dim=1).item())

    def _init_model(self):
        """初始化 DQN 网络。"""
        import torch
        import torch.nn as nn

        self._model = nn.Sequential(
            nn.Linear(self._state_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._n_actions),
        )
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        self._loss_fn = nn.MSELoss()
        self._log(f"[RL] DQN 初始化: state_dim={self._state_dim} actions={self._n_actions}")

    def _train_step(self):
        """从经验池采样训练一步。"""
        import torch

        batch = random.sample(list(self._memory), self._batch_size)

        states = torch.FloatTensor([t.state_features for t in batch])
        actions = torch.LongTensor([t.action_idx for t in batch]).unsqueeze(1)
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor([t.next_state_features for t in batch])
        dones = torch.FloatTensor([float(t.done) for t in batch])

        # 当前 Q 值
        q_values = self._model(states).gather(1, actions).squeeze()

        # 目标 Q 值
        with torch.no_grad():
            next_q = self._model(next_states).max(dim=1).values
            target = rewards + self._gamma * next_q * (1 - dones)

        loss = self._loss_fn(q_values, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # 衰减探索率
        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def _simple_features(self, state: SceneState) -> list[float]:
        """从基础 SceneState 提取简单特征。"""
        features = []
        for cls_name in sorted(state.object_counts.keys()):
            features.append(float(state.object_counts[cls_name]))

        # 最近目标的位置
        if state.current_result.detections:
            det = state.current_result.detections[0]
            cx = (det.bbox_norm[0] + det.bbox_norm[2]) / 2
            cy = (det.bbox_norm[1] + det.bbox_norm[3]) / 2
            features.extend([cx, cy])
        else:
            features.extend([0.5, 0.5])

        # 检测数量
        features.append(float(len(state.current_result.detections)))

        # 补齐到至少 state_dim（如果已初始化模型）
        if self._state_dim > 0:
            while len(features) < self._state_dim:
                features.append(0.0)
            features = features[:self._state_dim]

        return features

    def _save_model(self):
        import torch
        save_dir = Path(self._save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model": self._model.state_dict(),
            "epsilon": self._epsilon,
            "step_count": self._step_count,
            "state_dim": self._state_dim,
            "n_actions": self._n_actions,
            "actions": self._actions,
        }, str(save_dir / "rl_model.pt"))

        # 保存训练曲线
        with open(str(save_dir / "rewards.json"), "w") as f:
            json.dump(self._episode_rewards, f)

    def _load_model(self, path: str):
        import torch
        data = torch.load(path, weights_only=False)
        self._state_dim = data["state_dim"]
        self._n_actions = data["n_actions"]
        self._actions = data.get("actions", self._actions)
        self._epsilon = data.get("epsilon", self._epsilon)
        self._step_count = data.get("step_count", 0)
        self._init_model()
        self._model.load_state_dict(data["model"])
        self._log(f"[RL] 模型加载: {path} (step={self._step_count})")

    def on_start(self):
        self._reward_detector.reset()

    def on_stop(self):
        if self._training and self._model:
            self._save_model()
            self._log(f"[RL] 训练结束, 模型已保存 (steps={self._step_count})")

    def configure(self, **kwargs):
        if "training" in kwargs:
            self._training = kwargs["training"]
        if "epsilon" in kwargs:
            self._epsilon = kwargs["epsilon"]

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass
