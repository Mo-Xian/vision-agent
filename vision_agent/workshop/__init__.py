"""训练工坊模块：统一学习管线 — 教师教学 → 自主学习 → 自我实践。"""

from .learning_pipeline import LearningPipeline, LearningResult
from .unified_pipeline import UnifiedPipeline, UnifiedResult, LearningPhase
from .model_registry import ModelRegistry
from .session import LearningSession
from .scene import Scene, SceneManager

__all__ = [
    "LearningPipeline", "LearningResult",
    "UnifiedPipeline", "UnifiedResult", "LearningPhase",
    "ModelRegistry",
    "LearningSession",
    "Scene", "SceneManager",
]
