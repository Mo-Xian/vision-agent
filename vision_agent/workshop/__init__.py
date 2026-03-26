"""训练工坊模块：视频学习 → 结构化数据 → 模型训练 → 迭代强化。"""

from .video_analyzer import VideoAnalyzer, VideoInsight
from .learning_pipeline import LearningPipeline, LearningResult
from .model_registry import ModelRegistry
from .session import LearningSession
from .scene import Scene, SceneManager

__all__ = [
    "VideoAnalyzer", "VideoInsight",
    "LearningPipeline", "LearningResult",
    "ModelRegistry",
    "LearningSession",
    "Scene", "SceneManager",
]
