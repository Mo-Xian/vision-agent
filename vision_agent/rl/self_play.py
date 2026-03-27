"""自对弈循环：采集 + 训练双线程并行。

核心流程：
  采集线程:  截图 → 选动作 → 执行 → 奖励 → 存经验（循环）
  训练线程:  从缓冲区采样 → DQN 训练 → 定期保存（循环）

支持：
  - 从 BC 模型热启动（大幅加速学习）
  - 自动检测对局开始/结束
  - 定期保存检查点
  - 训练统计和日志
"""

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from .dqn_agent import DQNAgent
from .game_env import GameEnvironment
from .reward import RewardConfig

logger = logging.getLogger(__name__)


class SelfPlayLoop:
    """自对弈学习循环。

    用法:
        loop = SelfPlayLoop(
            action_zones=[
                {"name": "idle"},
                {"name": "move", "x": 0.13, "y": 0.72, "r": 0.10},
                {"name": "attack", "x": 0.92, "y": 0.72, "r": 0.06},
                ...
            ],
            bc_model_dir="runs/workshop/exp1/model",  # 可选热启动
            output_dir="runs/selfplay/exp1",
        )
        loop.start()  # 开始自对弈
        # ... 等待或 Ctrl+C ...
        loop.stop()
    """

    def __init__(
        self,
        action_zones: list[dict],
        bc_model_dir: str = "",
        output_dir: str = "runs/selfplay/exp1",
        device_serial: str = "",
        reward_config: RewardConfig | None = None,
        start_model_path: str = "models/start.onnx",
        # DQN 超参
        lr: float = 0.0005,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.998,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        # 控制
        fps: int = 5,
        save_interval: int = 50,
        max_episodes: int = 0,
        max_steps_per_episode: int = 3000,
        on_log=None,
        on_stats=None,
    ):
        self._action_zones = action_zones
        self._bc_model_dir = bc_model_dir
        self._output_dir = Path(output_dir)
        self._save_interval = save_interval
        self._max_episodes = max_episodes
        self._max_steps_per_episode = max_steps_per_episode
        self._on_log = on_log
        self._on_stats = on_stats

        # 游戏环境
        self._env = GameEnvironment(
            action_zones=action_zones,
            device_serial=device_serial,
            reward_config=reward_config,
            start_model_path=start_model_path,
            fps=fps,
            on_log=on_log,
        )

        # DQN 智能体
        num_actions = len(action_zones)
        self._agent = DQNAgent(
            num_actions=num_actions,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
        )

        # 运行状态
        self._running = False
        self._stop_event = threading.Event()
        self._collector_thread = None
        self._trainer_thread = None

        # 统计
        self._episode_count = 0
        self._total_steps = 0
        self._episode_rewards: list[float] = []
        self._losses: list[float] = []
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        with self._lock:
            recent_rewards = self._episode_rewards[-10:] if self._episode_rewards else []
            recent_losses = self._losses[-100:] if self._losses else []
            episode_count = self._episode_count
            total_steps = self._total_steps
        return {
            "episodes": episode_count,
            "total_steps": total_steps,
            "epsilon": round(self._agent.epsilon, 4),
            "buffer_size": self._agent.buffer_size,
            "train_steps": self._agent.train_steps,
            "avg_reward_10ep": round(np.mean(recent_rewards), 3) if recent_rewards else 0,
            "avg_loss_100": round(np.mean(recent_losses), 6) if recent_losses else 0,
        }

    def start(self):
        """启动自对弈循环。"""
        if self._running:
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._log("[自对弈] 初始化环境...")
        self._env.setup()

        # 热启动
        if self._bc_model_dir:
            self._log(f"[自对弈] 从 BC 模型热启动: {self._bc_model_dir}")
            if self._agent.warm_start(self._bc_model_dir):
                self._log(f"[自对弈] 热启动成功, ε={self._agent.epsilon:.3f}")
            else:
                self._log("[自对弈] 热启动失败，将从零开始")

        # 检查是否有之前的检查点可恢复
        latest_ckpt = self._output_dir / "latest"
        if latest_ckpt.exists():
            if self._agent.load(str(latest_ckpt)):
                self._log(f"[自对弈] 恢复检查点: train_steps={self._agent.train_steps}")

        self._running = True
        self._stop_event.clear()

        # 启动采集线程
        self._collector_thread = threading.Thread(
            target=self._collect_loop, daemon=True, name="rl-collector",
        )
        self._collector_thread.start()

        # 启动训练线程
        self._trainer_thread = threading.Thread(
            target=self._train_loop, daemon=True, name="rl-trainer",
        )
        self._trainer_thread.start()

        self._log(
            f"[自对弈] 开始 | 动作={len(self._action_zones)} | "
            f"ε={self._agent.epsilon:.3f} | 输出={self._output_dir}"
        )

    def stop(self):
        """停止自对弈。"""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._collector_thread:
            self._collector_thread.join(timeout=10)
        if self._trainer_thread:
            self._trainer_thread.join(timeout=5)

        self._env.teardown()

        # 保存最终模型
        self._save_checkpoint("final")
        self._save_history()

        self._log(
            f"[自对弈] 结束 | 对局={self._episode_count} | "
            f"总步数={self._total_steps} | 训练步={self._agent.train_steps}"
        )

    # ── 采集线程 ──

    def _collect_loop(self):
        """数据采集循环：与游戏环境交互。"""
        self._log("[采集] 线程启动")

        while not self._stop_event.is_set():
            if self._max_episodes > 0 and self._episode_count >= self._max_episodes:
                self._log(f"[采集] 已达最大对局数 {self._max_episodes}")
                break

            # 等待对局开始（通过检测画面变化）
            state = self._wait_for_game_start()
            if state is None:
                break

            with self._lock:
                self._episode_count += 1
            episode_reward = 0.0
            episode_steps = 0

            self._log(f"[采集] 对局 {self._episode_count} 开始")

            while not self._stop_event.is_set():
                # 选择动作
                action = self._agent.select_action(state)

                # 执行动作
                next_state, reward, done, info = self._env.step(action)

                if next_state is None:
                    self._log("[采集] 截屏失败，跳过")
                    time.sleep(0.5)
                    continue

                # 存储经验
                self._agent.store(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                episode_steps += 1
                with self._lock:
                    self._total_steps += 1

                # 日志
                if episode_steps % 50 == 0:
                    self._log(
                        f"  步={episode_steps} | "
                        f"动作={info.get('action', '?')} | "
                        f"奖励={info.get('reward', 0):.3f} | "
                        f"累计={episode_reward:.1f} | "
                        f"ε={self._agent.epsilon:.3f}"
                    )

                if done or episode_steps >= self._max_steps_per_episode:
                    break

            # 对局结束
            with self._lock:
                self._episode_rewards.append(episode_reward)

            avg_10 = np.mean(self._episode_rewards[-10:]) if self._episode_rewards else 0
            self._log(
                f"[采集] 对局 {self._episode_count} 结束 | "
                f"步数={episode_steps} | 奖励={episode_reward:.1f} | "
                f"近10局均奖={avg_10:.1f}"
            )

            if self._on_stats:
                try:
                    self._on_stats(self.stats)
                except Exception:
                    pass

            # 对局间休息
            self._stop_event.wait(2)

        self._log("[采集] 线程结束")

    def _wait_for_game_start(self) -> np.ndarray | None:
        """等待对局开始，返回初始状态。

        使用 ONNX 模型（start.onnx）检测对局是否开始，
        不可用时直接开始采集。
        """
        self._log("[采集] 等待对局...")

        # 使用 ONNX 模型等待对局开始
        if not self._env.wait_for_game_start(timeout=300):
            if self._stop_event.is_set():
                return None
            self._log("[采集] 等待超时，尝试直接开始")

        # 对局开始，获取初始状态
        while not self._stop_event.is_set():
            state = self._env.reset()
            if state is not None:
                return state
            self._stop_event.wait(1)
        return None

    # ── 训练线程 ──

    def _train_loop(self):
        """训练循环：从缓冲区采样训练。"""
        self._log("[训练] 线程启动，等待经验积累...")

        while not self._stop_event.is_set():
            if not self._agent._buffer.is_ready:
                self._stop_event.wait(1)
                continue

            loss = self._agent.train_step()
            if loss is not None:
                with self._lock:
                    self._losses.append(loss)

                if self._agent.train_steps % 100 == 0:
                    avg_loss = np.mean(self._losses[-100:])
                    self._log(
                        f"[训练] step={self._agent.train_steps} | "
                        f"loss={loss:.6f} | avg_loss={avg_loss:.6f} | "
                        f"ε={self._agent.epsilon:.3f} | "
                        f"buffer={self._agent.buffer_size}"
                    )

                # 定期保存
                if self._agent.train_steps % self._save_interval == 0:
                    self._save_checkpoint("latest")

            # 控制训练速度，不要太快耗尽 CPU
            self._stop_event.wait(0.01)

        self._log("[训练] 线程结束")

    # ── 保存 ──

    def _save_checkpoint(self, tag: str = "latest"):
        """保存检查点。"""
        save_dir = str(self._output_dir / tag)
        self._agent.save(
            save_dir,
            action_names=self._env.action_names,
            extra_meta={
                "episodes": self._episode_count,
                "total_steps": self._total_steps,
                "avg_reward_10ep": round(
                    np.mean(self._episode_rewards[-10:]), 3
                ) if self._episode_rewards else 0,
            },
        )

    def _save_history(self):
        """保存训练历史。"""
        history = {
            "episode_rewards": [round(r, 3) for r in self._episode_rewards],
            "episodes": self._episode_count,
            "total_steps": self._total_steps,
            "train_steps": self._agent.train_steps,
            "final_epsilon": round(self._agent.epsilon, 4),
        }
        history_path = self._output_dir / "train_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass
