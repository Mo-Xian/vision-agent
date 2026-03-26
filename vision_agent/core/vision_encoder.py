"""视觉编码器：将原始图像编码为语义特征向量。

使用预训练 MobileNetV3-Small 作为冻结特征提取器。
CPU 友好，单帧编码 ~50ms，输出 576 维特征向量。

用途：
  - 端到端学习：替代 YOLO 手工特征，保留完整视觉语义
  - 训练时：标注帧 → 编码 → 存储嵌入向量 → MLP 训练
  - 推理时：截图 → 编码 → MLP 推理 → 动作
"""

import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

logger = logging.getLogger(__name__)

# 预处理：ImageNet 标准化
_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VisionEncoder:
    """轻量视觉编码器 — MobileNetV3-Small 特征提取。

    输入：BGR 图像 (OpenCV 格式)
    输出：576 维 float32 特征向量

    特性：
      - 预训练权重冻结，不训练 backbone
      - CPU 单帧 ~50ms，批量更快
      - ImageNet 预训练已理解颜色/形状/纹理/空间关系
      - 游戏 UI（血条、技能栏、角色）属于可识别的视觉模式
    """

    EMBED_DIM = 576  # MobileNetV3-Small 特征维度

    def __init__(self, device: str = "cpu"):
        self._device = torch.device(device)
        self._model = self._build_model()
        self._model.to(self._device)
        self._model.eval()
        logger.info(f"VisionEncoder 初始化完成 (device={device}, dim={self.EMBED_DIM})")

    def _build_model(self) -> nn.Module:
        """构建冻结的 MobileNetV3-Small 特征提取器。"""
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = models.mobilenet_v3_small(weights=weights)
        # 移除分类头，只保留特征提取 + 全局池化
        # features → avgpool → flatten → 576 维
        model = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(),
        )
        # 冻结所有参数
        for p in model.parameters():
            p.requires_grad = False
        return model

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """编码单帧图像为特征向量。

        Args:
            frame: BGR 图像 (H, W, 3)，OpenCV 格式

        Returns:
            (576,) float32 特征向量
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = _TRANSFORM(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model(tensor)
        return features.squeeze(0).cpu().numpy()

    def encode_batch(self, frames: list[np.ndarray]) -> np.ndarray:
        """批量编码多帧图像。

        Args:
            frames: BGR 图像列表

        Returns:
            (N, 576) float32 特征矩阵
        """
        if not frames:
            return np.empty((0, self.EMBED_DIM), dtype=np.float32)

        tensors = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensors.append(_TRANSFORM(rgb))

        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            features = self._model(batch)
        return features.cpu().numpy()

    @property
    def embed_dim(self) -> int:
        return self.EMBED_DIM
