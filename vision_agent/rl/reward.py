"""奖励检测器：从游戏截图中提取奖励信号。

通用架构：
  - RewardDetector: 基类，定义检测接口
  - MobaRewardDetector: MOBA 游戏（王者荣耀/LoL 手游）
  - FpsRewardDetector: FPS 游戏（和平精英等）
  - PixelRewardDetector: 通用像素变化检测（任何游戏）

每种游戏类型实现自己的检测逻辑，通过 YAML 配置选择。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── 配置 ──

@dataclass
class RewardConfig:
    """通用奖励配置。"""
    # 游戏类型（决定使用哪个检测器）
    game_type: str = "moba"

    # 奖励值（通用）
    attack_reward: float = 2.0
    damage_penalty: float = -1.0
    death_penalty: float = -10.0
    win_reward: float = 100.0
    lose_penalty: float = -100.0
    idle_penalty: float = -0.1
    action_reward: float = 0.05

    # ONNX 模型路径（可选）
    death_model_path: str = "models/death.onnx"

    # 自定义检测区域（归一化坐标 0~1）
    # 不同游戏类型有不同的默认值，也可以在 YAML 中覆盖
    regions: dict = field(default_factory=dict)


@dataclass
class GameState:
    """当前游戏状态（所有游戏类型通用）。"""
    my_hp_ratio: float = 1.0
    enemy_hp_ratio: float = 1.0
    is_dead: bool = False
    is_finished: bool = False
    finish_result: str = ""      # "win" / "lose" / ""
    raw_reward: float = 0.0
    detail: str = ""             # 可读的状态描述


# ── 检测器基类 ──

class RewardDetector(ABC):
    """奖励检测器基类。

    子类实现具体游戏的检测逻辑，基类处理：
    - ONNX 模型加载（死亡检测）
    - 奖励计算公式
    - 状态追踪
    """

    def __init__(self, config: RewardConfig | None = None):
        self._config = config or RewardConfig()
        self._prev_my_hp = 1.0
        self._prev_enemy_hp = 1.0
        self._death_frames = 0

        # ONNX 死亡检测模型（可选，所有游戏类型通用）
        self._death_detector = None
        self._init_onnx_models()

    def _init_onnx_models(self):
        from .onnx_detector import OnnxDetector
        death_path = self._config.death_model_path
        if death_path:
            det = OnnxDetector(death_path, classes=["death"])
            if det.is_available:
                self._death_detector = det
                logger.info(f"[奖励] 死亡检测模型已加载: {death_path}")

    def reset(self):
        self._prev_my_hp = 1.0
        self._prev_enemy_hp = 1.0
        self._death_frames = 0

    def detect(self, frame: np.ndarray, action_idx: int) -> GameState:
        """分析一帧画面，返回游戏状态和奖励。"""
        state = GameState()

        state.my_hp_ratio = self.detect_my_hp(frame)
        state.enemy_hp_ratio = self.detect_enemy_hp(frame)
        state.is_dead = self.detect_death(frame)

        finished, result = self.detect_finish(frame)
        state.is_finished = finished
        state.finish_result = result

        state.raw_reward = self._calculate_reward(state, action_idx)

        self._prev_my_hp = state.my_hp_ratio
        self._prev_enemy_hp = state.enemy_hp_ratio

        return state

    def detect_death(self, frame: np.ndarray) -> bool:
        """检测死亡。优先 ONNX，子类可覆盖回退逻辑。"""
        if self._death_detector is not None:
            label = self._death_detector.detect(frame)
            if label == "death":
                self._death_frames += 1
            else:
                self._death_frames = 0
            return self._death_frames >= 2

        return self._detect_death_heuristic(frame)

    @abstractmethod
    def detect_my_hp(self, frame: np.ndarray) -> float:
        """检测己方血量比例 (0~1)。"""

    @abstractmethod
    def detect_enemy_hp(self, frame: np.ndarray) -> float:
        """检测敌方血量比例 (0~1)。"""

    @abstractmethod
    def detect_finish(self, frame: np.ndarray) -> tuple[bool, str]:
        """检测对局结束。返回 (is_finished, "win"/"lose"/"")。"""

    def _detect_death_heuristic(self, frame: np.ndarray) -> bool:
        """死亡启发式检测（灰屏）。子类可覆盖。"""
        h, w = frame.shape[:2]
        center = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        avg_sat = hsv[:, :, 1].mean()

        if avg_sat < 20:
            self._death_frames += 1
        else:
            self._death_frames = 0
        return self._death_frames >= 3

    def _calculate_reward(self, state: GameState, action_idx: int) -> float:
        """通用奖励计算（所有游戏类型共用）。"""
        cfg = self._config

        if state.is_finished:
            return cfg.win_reward if state.finish_result == "win" else cfg.lose_penalty

        if state.is_dead:
            return cfg.death_penalty

        reward = 0.0

        # 造成伤害
        enemy_delta = self._prev_enemy_hp - state.enemy_hp_ratio
        if enemy_delta > 0.01:
            reward += cfg.attack_reward * min(enemy_delta * 10, 5)

        # 受到伤害
        my_delta = self._prev_my_hp - state.my_hp_ratio
        if my_delta > 0.01:
            reward += cfg.damage_penalty * min(my_delta * 10, 5)

        # 动作奖励/惩罚
        reward += cfg.idle_penalty if action_idx == 0 else cfg.action_reward

        return reward


# ── MOBA 检测器（王者荣耀 / LoL 手游） ──

class MobaRewardDetector(RewardDetector):
    """MOBA 游戏奖励检测器。

    特点：
    - 己方血条：左上角，绿色
    - 敌方血条：顶部中间偏右，红色
    - 胜利：金色高亮画面
    - 失败：暗灰画面
    """

    # 默认区域（王者荣耀，来自 wzry_ai）
    DEFAULT_REGIONS = {
        "my_hp":    {"left": 0.03,  "top": 0.01,  "right": 0.18,  "bottom": 0.04},
        "enemy_hp": {"left": 0.57,  "top": 0.019, "right": 0.686, "bottom": 0.043},
    }

    def __init__(self, config: RewardConfig | None = None):
        super().__init__(config)
        regions = self._config.regions or self.DEFAULT_REGIONS
        self._my_hp_region = regions.get("my_hp", self.DEFAULT_REGIONS["my_hp"])
        self._enemy_hp_region = regions.get("enemy_hp", self.DEFAULT_REGIONS["enemy_hp"])

    def detect_my_hp(self, frame: np.ndarray) -> float:
        return self._detect_hp_bar(frame, self._my_hp_region, "green")

    def detect_enemy_hp(self, frame: np.ndarray) -> float:
        return self._detect_hp_bar(frame, self._enemy_hp_region, "red")

    def detect_finish(self, frame: np.ndarray) -> tuple[bool, str]:
        h, w = frame.shape[:2]
        center = frame[h // 4:h // 2, w // 3:2 * w // 3]
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)

        # 胜利：金色/黄色高亮
        gold_mask = cv2.inRange(hsv, np.array([15, 100, 180]), np.array([35, 255, 255]))
        gold_ratio = cv2.countNonZero(gold_mask) / max(center.shape[0] * center.shape[1], 1)
        if gold_ratio > 0.15:
            return True, "win"

        # 失败：暗灰
        avg_val = hsv[:, :, 2].mean()
        avg_sat = hsv[:, :, 1].mean()
        if avg_val < 60 and avg_sat < 30:
            return True, "lose"

        return False, ""

    @staticmethod
    def _detect_hp_bar(frame: np.ndarray, region: dict, color_mode: str) -> float:
        h, w = frame.shape[:2]
        x1 = int(region["left"] * w)
        y1 = int(region["top"] * h)
        x2 = int(region["right"] * w)
        y2 = int(region["bottom"] * h)

        if x2 <= x1 or y2 <= y1:
            return 1.0

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 1.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total = roi.shape[0] * roi.shape[1]

        if color_mode == "green":
            mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        else:
            mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)

        colored = cv2.countNonZero(mask)
        return min(colored / max(total * 0.3, 1), 1.0)


# ── FPS 检测器（和平精英 / CF 手游等） ──

class FpsRewardDetector(RewardDetector):
    """FPS 游戏奖励检测器。

    特点：
    - 己方血条：通常在屏幕底部
    - 击杀提示：屏幕中上方出现击杀 icon
    - 受伤：屏幕边缘红色闪烁
    - 胜利/失败：文字提示（需 OCR 或特征检测）
    """

    DEFAULT_REGIONS = {
        "my_hp":    {"left": 0.03,  "top": 0.90, "right": 0.20, "bottom": 0.96},
        "damage":   {"left": 0.0,   "top": 0.0,  "right": 1.0,  "bottom": 1.0},
    }

    def __init__(self, config: RewardConfig | None = None):
        super().__init__(config)
        regions = self._config.regions or self.DEFAULT_REGIONS
        self._my_hp_region = regions.get("my_hp", self.DEFAULT_REGIONS["my_hp"])

    def detect_my_hp(self, frame: np.ndarray) -> float:
        """检测底部血条（白色/绿色）。"""
        h, w = frame.shape[:2]
        r = self._my_hp_region
        roi = frame[int(r["top"]*h):int(r["bottom"]*h), int(r["left"]*w):int(r["right"]*w)]
        if roi.size == 0:
            return 1.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # FPS 游戏血条通常是白色或浅色
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
        green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        mask = cv2.bitwise_or(white_mask, green_mask)
        total = roi.shape[0] * roi.shape[1]
        return min(cv2.countNonZero(mask) / max(total * 0.3, 1), 1.0)

    def detect_enemy_hp(self, frame: np.ndarray) -> float:
        """FPS 一般没有持续显示的敌方血条，通过受伤指示检测。"""
        h, w = frame.shape[:2]
        # 检测屏幕边缘红色闪烁（受伤指示）
        edges = np.concatenate([
            frame[:, :w//10, :].reshape(-1, 3),        # 左边缘
            frame[:, -w//10:, :].reshape(-1, 3),       # 右边缘
            frame[:h//10, :, :].reshape(-1, 3),        # 上边缘
        ])
        hsv = cv2.cvtColor(edges.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
        red_ratio = (cv2.countNonZero(red1) + cv2.countNonZero(red2)) / max(edges.shape[0], 1)

        # 红色闪烁多 → 可能在造成/受到伤害
        # 返回 "反转" 值：红色越多 = 敌方血越低
        return max(1.0 - red_ratio * 5, 0.0)

    def detect_finish(self, frame: np.ndarray) -> tuple[bool, str]:
        """通过画面整体变化检测结束。"""
        h, w = frame.shape[:2]
        center = frame[h//3:2*h//3, w//4:3*w//4]
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)

        # 胜利：通常有金色/亮色提示
        bright_mask = cv2.inRange(hsv, np.array([15, 80, 200]), np.array([35, 255, 255]))
        bright_ratio = cv2.countNonZero(bright_mask) / max(center.shape[0] * center.shape[1], 1)
        if bright_ratio > 0.10:
            return True, "win"

        # 失败：暗沉画面
        avg_val = hsv[:, :, 2].mean()
        avg_sat = hsv[:, :, 1].mean()
        if avg_val < 50 and avg_sat < 25:
            return True, "lose"

        return False, ""


# ── 通用像素变化检测器（任何游戏） ──

class PixelRewardDetector(RewardDetector):
    """通用像素变化奖励检测器。

    不依赖游戏特定的 UI 布局，通过画面整体变化推测奖励：
    - 大面积变化 → 可能有事件发生
    - 画面变暗 → 可能死亡
    - 画面高亮 → 可能胜利

    适用于没有专门适配的游戏，或作为新游戏适配的起点。
    """

    def __init__(self, config: RewardConfig | None = None):
        super().__init__(config)
        self._prev_frame_gray = None

    def reset(self):
        super().reset()
        self._prev_frame_gray = None

    def detect_my_hp(self, frame: np.ndarray) -> float:
        """通过画面亮度变化估计。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = gray.mean() / 255.0

        if self._prev_frame_gray is not None:
            # 画面突然变暗可能表示受伤
            prev_brightness = self._prev_frame_gray.mean() / 255.0
            if prev_brightness - avg_brightness > 0.1:
                self._prev_frame_gray = gray
                return max(self._prev_my_hp - 0.1, 0.0)

        self._prev_frame_gray = gray
        return self._prev_my_hp

    def detect_enemy_hp(self, frame: np.ndarray) -> float:
        """通用场景下无法准确检测敌方血量。"""
        return self._prev_enemy_hp

    def detect_finish(self, frame: np.ndarray) -> tuple[bool, str]:
        """通过画面整体特征猜测结束。"""
        h, w = frame.shape[:2]
        center = frame[h//4:3*h//4, w//4:3*w//4]
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)

        avg_val = hsv[:, :, 2].mean()
        avg_sat = hsv[:, :, 1].mean()

        # 画面极亮+低饱和度 → 可能是结束画面
        if avg_val > 200 and avg_sat < 30:
            return True, "win"
        if avg_val < 40 and avg_sat < 20:
            return True, "lose"

        return False, ""


# ── 工厂函数 ──

def create_reward_detector(config: RewardConfig | None = None) -> RewardDetector:
    """根据配置创建对应的奖励检测器。

    Args:
        config: 奖励配置，game_type 决定检测器类型

    Returns:
        对应游戏类型的 RewardDetector 实例
    """
    config = config or RewardConfig()

    detectors = {
        "moba": MobaRewardDetector,
        "wzry": MobaRewardDetector,
        "fps": FpsRewardDetector,
        "generic": PixelRewardDetector,
    }

    cls = detectors.get(config.game_type, PixelRewardDetector)
    logger.info(f"[奖励] 使用检测器: {cls.__name__} (game_type={config.game_type})")
    return cls(config)
