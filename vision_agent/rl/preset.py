"""自对弈预设加载器。

内置预设：
  - wzry / moba:  王者荣耀 / LoL 手游
  - fps:          和平精英 / CF 手游
  - generic:      通用（任何游戏，最小配置）

扩展方式：
  1. 在 profiles/ 下创建 YAML 文件
  2. 在此文件中添加新的 PRESET 字典
  3. python main.py self-play --preset my_game.yaml
"""

import logging
from pathlib import Path

from .reward import RewardConfig

logger = logging.getLogger(__name__)


# ── 内置预设 ──

WZRY_PRESET = {
    "game_type": "moba",
    "touch_zones": {
        "move":          {"x": 0.164, "y": 0.798, "r": 0.10},
        "attack":        {"x": 0.85,  "y": 0.85,  "r": 0.06},
        "attack_minion": {"x": 0.776, "y": 0.91,  "r": 0.04},
        "attack_tower":  {"x": 0.88,  "y": 0.71,  "r": 0.04},
        "skill_1":       {"x": 0.71,  "y": 0.874, "r": 0.05},
        "skill_2":       {"x": 0.76,  "y": 0.69,  "r": 0.05},
        "skill_3":       {"x": 0.844, "y": 0.58,  "r": 0.05},
        "spell":         {"x": 0.64,  "y": 0.9,   "r": 0.04},
        "recall":        {"x": 0.518, "y": 0.9,   "r": 0.04},
        "heal":          {"x": 0.579, "y": 0.9,   "r": 0.04},
    },
    "reward_regions": {
        "my_hp":    {"left": 0.03,  "top": 0.01,  "right": 0.18,  "bottom": 0.04},
        "enemy_hp": {"left": 0.57,  "top": 0.019, "right": 0.686, "bottom": 0.043},
    },
    "rewards": {
        "attack_reward": 2.0,
        "damage_penalty": -1.0,
        "death_penalty": -10.0,
        "win_reward": 100.0,
        "lose_penalty": -100.0,
    },
    "dqn": {
        "lr": 0.0005,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.998,
        "buffer_capacity": 50000,
        "batch_size": 64,
    },
    "start_model_path": "models/start.onnx",
    "death_model_path": "models/death.onnx",
    "output_dir": "runs/selfplay/wzry",
}

FPS_PRESET = {
    "game_type": "fps",
    "touch_zones": {
        "move":    {"x": 0.13, "y": 0.72, "r": 0.10},
        "aim":     {"x": 0.70, "y": 0.50, "r": 0.20},
        "fire":    {"x": 0.92, "y": 0.65, "r": 0.06},
        "scope":   {"x": 0.88, "y": 0.45, "r": 0.05},
        "reload":  {"x": 0.82, "y": 0.80, "r": 0.04},
        "crouch":  {"x": 0.75, "y": 0.85, "r": 0.04},
        "jump":    {"x": 0.90, "y": 0.85, "r": 0.04},
    },
    "reward_regions": {
        "my_hp": {"left": 0.03, "top": 0.90, "right": 0.20, "bottom": 0.96},
    },
    "rewards": {
        "attack_reward": 3.0,
        "damage_penalty": -1.5,
        "death_penalty": -15.0,
        "win_reward": 100.0,
        "lose_penalty": -100.0,
    },
    "dqn": {
        "lr": 0.0005,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.997,
        "buffer_capacity": 50000,
        "batch_size": 64,
    },
    "start_model_path": "",
    "death_model_path": "",
    "output_dir": "runs/selfplay/fps",
}

GENERIC_PRESET = {
    "game_type": "generic",
    "touch_zones": {},
    "reward_regions": {},
    "rewards": {},
    "dqn": {},
    "start_model_path": "",
    "death_model_path": "",
    "output_dir": "runs/selfplay/generic",
}

# 预设注册表
PRESETS = {
    "wzry": WZRY_PRESET,
    "wzry_5v5": WZRY_PRESET,
    "moba": WZRY_PRESET,
    "fps": FPS_PRESET,
    "generic": GENERIC_PRESET,
}


def load_selfplay_preset(name: str = "wzry") -> dict:
    """加载自对弈预设。

    Args:
        name: 预设名称或 YAML 文件路径。支持:
            - "wzry" / "moba": 王者荣耀
            - "fps": FPS 游戏
            - "generic": 通用
            - 文件路径: 从 YAML 加载自定义配置
    """
    # 内置预设
    if name in PRESETS:
        return _build_preset(PRESETS[name])

    # YAML 文件
    path = Path(name)
    if path.exists() and path.suffix in (".yaml", ".yml"):
        return _load_yaml_preset(path)

    # profiles 目录搜索
    for candidate in [
        Path(f"profiles/{name}.yaml"),
        Path(f"profiles/{name}_selfplay.yaml"),
    ]:
        if candidate.exists():
            return _load_yaml_preset(candidate)

    logger.warning(f"未找到预设 '{name}'，使用通用配置")
    return _build_preset(GENERIC_PRESET)


def list_presets() -> list[dict]:
    """列出所有可用预设。"""
    seen = set()
    result = []
    for name, preset in PRESETS.items():
        game_type = preset.get("game_type", "unknown")
        if game_type not in seen:
            seen.add(game_type)
            result.append({
                "name": name,
                "game_type": game_type,
                "actions": len(preset.get("touch_zones", {})) + 1,
            })

    # 扫描 profiles 目录
    for yaml_file in Path("profiles").glob("*_selfplay.yaml"):
        result.append({
            "name": yaml_file.stem,
            "game_type": "custom",
            "actions": "?",
        })

    return result


def _build_preset(raw: dict) -> dict:
    """从原始字典构建标准化预设。"""
    game_type = raw.get("game_type", "generic")

    # action_zones
    action_zones = [{"name": "idle"}]
    for name, zone in raw.get("touch_zones", {}).items():
        action_zones.append({"name": name, **zone})

    # reward_config
    rewards = raw.get("rewards", {})
    reward_config = RewardConfig(
        game_type=game_type,
        attack_reward=rewards.get("attack_reward", 2.0),
        damage_penalty=rewards.get("damage_penalty", -1.0),
        death_penalty=rewards.get("death_penalty", -10.0),
        win_reward=rewards.get("win_reward", 100.0),
        lose_penalty=rewards.get("lose_penalty", -100.0),
        death_model_path=raw.get("death_model_path", ""),
        regions=raw.get("reward_regions", {}),
    )

    return {
        "game_type": game_type,
        "action_zones": action_zones,
        "reward_config": reward_config,
        "dqn_params": raw.get("dqn", {}),
        "bc_model_dir": raw.get("bc_model_dir", ""),
        "start_model_path": raw.get("start_model_path", ""),
        "output_dir": raw.get("output_dir", "runs/selfplay/exp1"),
    }


def _load_yaml_preset(path: Path) -> dict:
    """从 YAML 文件加载预设。"""
    try:
        import yaml
    except ImportError:
        logger.error("需要 pyyaml: pip install pyyaml")
        return _build_preset(GENERIC_PRESET)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    detection = data.get("detection", {})
    raw = {
        "game_type": data.get("game_type", "generic"),
        "touch_zones": data.get("touch_zones", {}),
        "reward_regions": data.get("reward_regions", {}),
        "rewards": data.get("rewards", {}),
        "dqn": data.get("dqn", {}),
        "bc_model_dir": data.get("bc_model_dir", ""),
        "start_model_path": detection.get("start_model", ""),
        "death_model_path": detection.get("death_model", ""),
        "output_dir": data.get("selfplay_output", "runs/selfplay/exp1"),
    }

    return _build_preset(raw)
