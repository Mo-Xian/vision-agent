"""DQN 智能体：支持从行为克隆模型热启动。

架构与 E2EMLP 兼容（576→256→128→N），
可以直接加载 BC 预训练权重作为 Q 网络初始化。

核心特性：
  - Double DQN（policy + target 网络）
  - 经验回放
  - ε-greedy 探索（从 BC 热启动时 ε 可以更低）
  - 定期同步 target 网络
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .replay_buffer import ReplayBuffer, Transition

logger = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    """DQN Q 网络。

    架构与 E2EMLP 相同，输出的是各动作的 Q 值而非分类概率。
    这使得可以直接从 BC 模型加载预训练权重。
    """

    def __init__(self, input_dim: int = 576, num_actions: int = 10,
                 hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """DQN 智能体。

    用法:
        agent = DQNAgent(num_actions=9)
        agent.warm_start("runs/workshop/exp1/model")  # 可选：加载 BC 权重
        action = agent.select_action(state_embedding)
        agent.store(state, action, reward, next_state, done)
        loss = agent.train_step()
    """

    def __init__(
        self,
        num_actions: int = 10,
        input_dim: int = 576,
        hidden_dims: list[int] | None = None,
        lr: float = 0.0005,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.998,
        target_update_freq: int = 20,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self._num_actions = num_actions
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims or [256, 128]
        self._gamma = gamma
        self._epsilon = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._target_update_freq = target_update_freq
        self._batch_size = batch_size
        self._device = torch.device(device)

        # 网络
        self._policy_net = DQNNetwork(input_dim, num_actions, self._hidden_dims).to(self._device)
        self._target_net = DQNNetwork(input_dim, num_actions, self._hidden_dims).to(self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._optimizer = optim.Adam(self._policy_net.parameters(), lr=lr)
        self._criterion = nn.SmoothL1Loss()  # Huber loss

        # 经验回放
        self._buffer = ReplayBuffer(buffer_capacity)

        self._train_steps = 0
        self._total_steps = 0

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def train_steps(self) -> int:
        return self._train_steps

    def warm_start(self, model_dir: str) -> bool:
        """从 BC 模型加载预训练权重。

        BC 的 E2EMLP 和 DQNNetwork 架构兼容，
        但 E2EMLP 有 Dropout 层需要跳过。

        Args:
            model_dir: BC 模型目录（含 model.pt + model.meta.json）

        Returns:
            是否成功加载
        """
        model_dir = Path(model_dir)
        model_path = model_dir / "model.pt"
        meta_path = model_dir / "model.meta.json"

        if not model_path.exists():
            logger.warning(f"BC 模型不存在: {model_path}")
            return False

        # 读取元数据
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            bc_num_actions = meta.get("num_actions", self._num_actions)
            if bc_num_actions != self._num_actions:
                logger.warning(
                    f"BC 动作数 ({bc_num_actions}) != 当前动作数 ({self._num_actions})，"
                    f"跳过热启动"
                )
                return False

        # 加载 BC 权重，映射到 DQN 网络
        bc_state = torch.load(model_path, map_location=self._device)
        dqn_state = self._policy_net.state_dict()

        # E2EMLP 的 net 包含 [Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear]
        # DQNNetwork 的 net 包含 [Linear, ReLU, Linear, ReLU, Linear]
        # 需要映射对应的 Linear 层权重
        bc_linears = {k: v for k, v in bc_state.items()
                      if "weight" in k or "bias" in k}
        dqn_linears = {k: v for k, v in dqn_state.items()
                       if "weight" in k or "bias" in k}

        # 按序提取 Linear 层参数
        bc_params = []
        for k in sorted(bc_linears.keys()):
            bc_params.append((k, bc_linears[k]))

        dqn_params = []
        for k in sorted(dqn_linears.keys()):
            dqn_params.append((k, dqn_linears[k]))

        loaded = 0
        for (bc_k, bc_v), (dqn_k, dqn_v) in zip(bc_params, dqn_params):
            if bc_v.shape == dqn_v.shape:
                dqn_state[dqn_k] = bc_v
                loaded += 1

        if loaded > 0:
            self._policy_net.load_state_dict(dqn_state)
            self._target_net.load_state_dict(self._policy_net.state_dict())
            # BC 预训练后降低初始探索率
            self._epsilon = max(0.3, self._epsilon * 0.3)
            logger.info(f"DQN 热启动成功: 加载 {loaded} 个参数张量, ε={self._epsilon:.3f}")
            return True

        logger.warning("DQN 热启动失败: 无匹配参数")
        return False

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy 动作选择。

        Args:
            state: 576 维嵌入向量

        Returns:
            动作索引
        """
        self._total_steps += 1

        if np.random.random() < self._epsilon:
            return np.random.randint(self._num_actions)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)
            q_values = self._policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def store(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool):
        """存储经验到回放缓冲区。"""
        self._buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """执行一步 DQN 训练。

        Returns:
            loss 值，缓冲区不足时返回 None
        """
        if not self._buffer.is_ready:
            return None

        transitions = self._buffer.sample(self._batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self._device)
        actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self._device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).to(self._device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self._device)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(self._device)

        # 当前 Q(s, a)
        current_q = self._policy_net(states).gather(1, actions).squeeze(1)

        # 目标 Q = r + γ * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self._target_net(next_states).max(dim=1)[0]
            target_q = rewards + self._gamma * next_q * (1 - dones)

        loss = self._criterion(current_q, target_q)

        self._optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，稳定训练
        torch.nn.utils.clip_grad_norm_(self._policy_net.parameters(), max_norm=1.0)
        self._optimizer.step()

        self._train_steps += 1

        # 定期同步 target 网络
        if self._train_steps % self._target_update_freq == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())

        # 衰减 ε
        if self._epsilon > self._epsilon_end:
            self._epsilon *= self._epsilon_decay

        return loss.item()

    def save(self, save_dir: str, action_names: list[str] | None = None,
             extra_meta: dict | None = None):
        """保存模型和元数据。"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self._policy_net.state_dict(), save_dir / "model.pt")

        meta = {
            "model_type": "dqn",
            "input_dim": self._input_dim,
            "num_actions": self._num_actions,
            "hidden_dims": self._hidden_dims,
            "encoder": "mobilenet_v3_small",
            "embed_dim": self._input_dim,
            "action_list": action_names or [],
            "train_steps": self._train_steps,
            "epsilon": round(self._epsilon, 4),
            "buffer_size": len(self._buffer),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra_meta:
            meta.update(extra_meta)

        with open(save_dir / "model.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, model_dir: str) -> bool:
        """加载已保存的 DQN 模型。"""
        model_dir = Path(model_dir)
        model_path = model_dir / "model.pt"

        if not model_path.exists():
            return False

        self._policy_net.load_state_dict(
            torch.load(model_path, map_location=self._device)
        )
        self._target_net.load_state_dict(self._policy_net.state_dict())

        meta_path = model_dir / "model.meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._epsilon = meta.get("epsilon", self._epsilon)
            self._train_steps = meta.get("train_steps", 0)

        logger.info(f"DQN 模型加载: {model_dir}")
        return True
