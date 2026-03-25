"""跨帧场景状态管理。"""

import math
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from .detector import Detection, DetectionResult

logger = logging.getLogger(__name__)


def describe_position(cx: float, cy: float) -> str:
    """将归一化坐标转换为方位描述。"""
    v = "上方" if cy < 0.33 else ("中部" if cy < 0.66 else "下方")
    h = "左侧" if cx < 0.33 else ("中间" if cx < 0.66 else "右侧")
    return f"{v}{h}"


@dataclass
class SpatialInfo:
    """目标的空间关系信息。"""
    nearest_distance: float = 1.0
    center_x: float = 0.5
    center_y: float = 0.5
    area_ratio: float = 0.0


@dataclass
class SceneState:
    """当前场景的完整状态快照（含可选的空间/ROI/场景信息）。"""
    current_result: DetectionResult
    history: list[DetectionResult]
    object_counts: dict[str, int]
    scene_summary: str
    custom_data: dict = field(default_factory=dict)
    timestamp: float = 0.0
    # 可选增强字段
    spatial_info: dict[str, SpatialInfo] = field(default_factory=dict)
    roi_features: dict[str, dict] = field(default_factory=dict)
    scene_name: str = ""

    def to_feature_vector(self) -> list[float]:
        """转为训练用的特征向量。"""
        features: list[float] = []
        for cls_name in sorted(self.object_counts.keys()):
            features.append(float(self.object_counts[cls_name]))
        for cls_name in sorted(self.spatial_info.keys()):
            info = self.spatial_info[cls_name]
            features.extend([info.nearest_distance, info.center_x,
                             info.center_y, info.area_ratio])
        for roi_name in sorted(self.roi_features.keys()):
            roi = self.roi_features[roi_name]
            if "brightness" in roi:
                features.append(roi["brightness"])
            if "color_ratio" in roi:
                for color in sorted(roi["color_ratio"].keys()):
                    features.append(roi["color_ratio"][color])
        return features


# 向后兼容别名
EnhancedState = SceneState


class StateManager:
    """维护跨帧的场景上下文，生成结构化状态供决策层使用。"""

    def __init__(self, history_size: int = 30):
        self._history: deque[DetectionResult] = deque(maxlen=history_size)
        self._custom_data: dict = {}

    def update(self, result: DetectionResult,
               roi_features: dict | None = None,
               scene_name: str = "") -> SceneState:
        """用新的检测结果更新状态，返回当前场景快照。"""
        self._history.append(result)

        counts = {}
        for det in result.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1

        summary = self._build_summary(result, counts)
        spatial = self._compute_spatial(result) if roi_features or scene_name else {}

        return SceneState(
            current_result=result,
            history=list(self._history),
            object_counts=counts,
            scene_summary=summary,
            custom_data=self._custom_data.copy(),
            timestamp=time.time(),
            spatial_info=spatial,
            roi_features=roi_features or {},
            scene_name=scene_name,
        )

    # 向后兼容：update_enhanced 合并到 update
    def update_enhanced(self, result: DetectionResult,
                        roi_features: dict | None = None,
                        scene_name: str = "unknown") -> SceneState:
        return self.update(result, roi_features=roi_features, scene_name=scene_name)

    def set_custom(self, key: str, value) -> None:
        self._custom_data[key] = value

    def _build_summary(self, result: DetectionResult, counts: dict[str, int]) -> str:
        if not result.detections:
            return "当前画面未检测到任何目标。"

        parts = []
        for name, count in sorted(counts.items(), key=lambda x: -x[1]):
            parts.append(f"{name}×{count}" if count > 1 else name)

        detail_lines = []
        for det in result.detections[:10]:
            cx = (det.bbox_norm[0] + det.bbox_norm[2]) / 2
            cy = (det.bbox_norm[1] + det.bbox_norm[3]) / 2
            pos = describe_position(cx, cy)
            detail_lines.append(f"  - {det.class_name} (conf={det.confidence:.2f}) 位于{pos}")

        summary = f"检测到 {len(result.detections)} 个目标: {', '.join(parts)}\n"
        summary += "\n".join(detail_lines)
        if len(result.detections) > 10:
            summary += f"\n  ... 等共 {len(result.detections)} 个目标"

        return summary

    def _compute_spatial(self, result: DetectionResult) -> dict[str, SpatialInfo]:
        if not result.detections:
            return {}

        groups: dict[str, list[Detection]] = {}
        for det in result.detections:
            groups.setdefault(det.class_name, []).append(det)

        fw = result.frame_width or 1
        fh = result.frame_height or 1
        frame_area = fw * fh

        spatial: dict[str, SpatialInfo] = {}
        for cls_name, dets in groups.items():
            cx_sum, cy_sum, area_sum = 0.0, 0.0, 0.0
            centers = []
            for d in dets:
                x1, y1, x2, y2 = d.bbox_norm
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centers.append((cx, cy))
                cx_sum += cx
                cy_sum += cy
                bw = (x2 - x1) * fw
                bh = (y2 - y1) * fh
                area_sum += bw * bh

            n = len(dets)
            avg_cx = cx_sum / n
            avg_cy = cy_sum / n
            area_ratio = area_sum / frame_area

            nearest = 1.0
            if n >= 2:
                min_dist = float("inf")
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = centers[i][0] - centers[j][0]
                        dy = centers[i][1] - centers[j][1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist < min_dist:
                            min_dist = dist
                nearest = min_dist

            spatial[cls_name] = SpatialInfo(
                nearest_distance=round(nearest, 4),
                center_x=round(avg_cx, 4),
                center_y=round(avg_cy, 4),
                area_ratio=round(area_ratio, 6),
            )

        return spatial
