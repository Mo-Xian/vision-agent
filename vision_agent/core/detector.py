"""YOLO 目标检测器封装。"""

import time
import numpy as np
from ultralytics import YOLO


class Detection:
    """单个检测结果。"""
    __slots__ = ("class_id", "class_name", "confidence", "bbox", "bbox_norm")

    def __init__(self, class_id: int, class_name: str, confidence: float,
                 bbox: tuple, bbox_norm: tuple):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox              # (x1, y1, x2, y2) 像素坐标
        self.bbox_norm = bbox_norm    # (x1, y1, x2, y2) 归一化坐标 [0,1]

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 1) for v in self.bbox],
            "bbox_norm": [round(v, 4) for v in self.bbox_norm],
        }


class DetectionResult:
    """一帧的检测结果集合。"""

    def __init__(self, detections: list[Detection], frame_id: int,
                 timestamp: float, inference_ms: float,
                 frame_width: int, frame_height: int):
        self.detections = detections
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.inference_ms = inference_ms
        self.frame_width = frame_width
        self.frame_height = frame_height

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "inference_ms": round(self.inference_ms, 1),
            "frame_size": [self.frame_width, self.frame_height],
            "count": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
        }


class Detector:
    """YOLO 检测器。"""

    def __init__(self, model: str = "yolov8n.pt", confidence: float = 0.5,
                 iou: float = 0.45, classes: list[int] | None = None,
                 device: str | None = None, imgsz: int = 640):
        self.confidence = confidence
        self.iou = iou
        self.classes = classes
        self.device = device
        self.imgsz = imgsz
        self._model_path = model
        self._model = YOLO(model)
        self._frame_id = 0

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """对一帧图像执行目标检测。"""
        h, w = frame.shape[:2]
        self._frame_id += 1

        t0 = time.perf_counter()
        results = self._model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = self._model.names[cls_id]
                bbox_norm = (x1 / w, y1 / h, x2 / w, y2 / h)
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    bbox_norm=bbox_norm,
                ))

        return DetectionResult(
            detections=detections,
            frame_id=self._frame_id,
            timestamp=time.time(),
            inference_ms=inference_ms,
            frame_width=w,
            frame_height=h,
        )
