"""图片输入源：支持单张图片或目录批量输入。"""

import logging
from pathlib import Path
import cv2
import numpy as np
from .base import BaseSource

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class ImageSource(BaseSource):
    """图片输入源。

    Args:
        path: 图片文件路径或包含图片的目录路径
        loop: 批量模式下是否循环
    """

    def __init__(self, path: str, loop: bool = False):
        self._path = Path(path)
        self._loop = loop
        self._files: list[Path] = []
        self._index = 0
        self._started = False

    def start(self):
        if self._path.is_file():
            self._files = [self._path]
        elif self._path.is_dir():
            self._files = sorted(
                f for f in self._path.iterdir()
                if f.suffix.lower() in IMAGE_EXTS
            )
        else:
            raise FileNotFoundError(f"路径不存在: {self._path}")

        if not self._files:
            raise ValueError(f"未找到图片文件: {self._path}")

        self._index = 0
        self._started = True
        logger.info(f"图片源启动: {len(self._files)} 张图片, path={self._path}")

    def read(self) -> np.ndarray | None:
        if not self._started or self._index >= len(self._files):
            if self._loop and self._files:
                self._index = 0
            else:
                return None

        img_path = self._files[self._index]
        self._index += 1

        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"无法读取图片: {img_path}")
            return self.read()  # 跳过损坏的图片

        return frame

    def stop(self):
        self._started = False
        logger.info("图片源停止")
