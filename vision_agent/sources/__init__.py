"""视频源模块。"""

from .base import BaseSource
from .camera import CameraSource
from .screen import ScreenSource
from .video_file import VideoFileSource
from .image import ImageSource


def create_source(config: dict) -> BaseSource:
    """根据配置创建视频源实例。"""
    source_type = config["type"]
    if source_type == "camera":
        cfg = config.get("camera", {})
        return CameraSource(**cfg)
    elif source_type == "screen":
        cfg = config.get("screen", {})
        return ScreenSource(**cfg)
    elif source_type == "video":
        cfg = config.get("video", {})
        return VideoFileSource(**cfg)
    elif source_type == "image":
        cfg = config.get("image", {})
        return ImageSource(**cfg)
    else:
        raise ValueError(f"未知的视频源类型: {source_type}")
