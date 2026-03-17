"""视频源基类。"""

from abc import ABC, abstractmethod
import numpy as np


class BaseSource(ABC):
    """所有视频源的抽象基类。"""

    @abstractmethod
    def start(self):
        """启动视频源。"""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """读取一帧图像，返回 BGR numpy 数组，失败返回 None。"""

    @abstractmethod
    def stop(self):
        """停止视频源并释放资源。"""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
