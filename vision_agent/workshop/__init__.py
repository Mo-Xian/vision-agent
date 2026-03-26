"""训练工坊模块：录制 → 行为克隆 → 模型训练。"""

from .learning_pipeline import LearningPipeline, LearningResult
from .model_registry import ModelRegistry
from .session import LearningSession
from .scene import Scene, SceneManager

__all__ = [
    "LearningPipeline", "LearningResult",
    "ModelRegistry",
    "LearningSession",
    "Scene", "SceneManager",
]
