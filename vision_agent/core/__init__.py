"""核心模块。"""

from .detector import Detector, Detection, DetectionResult
from .visualizer import Visualizer
from .trainer import Trainer, TrainConfig
from .model_manager import ModelManager
from .state import SceneState, StateManager, SpatialInfo, EnhancedState, describe_position
from .scene_classifier import SceneClassifier
from .roi_extractor import ROIExtractor
