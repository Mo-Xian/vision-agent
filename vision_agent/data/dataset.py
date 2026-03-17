"""训练数据集加载与预处理。

从 DataRecorder 生成的 JSONL 文件中加载数据，提取特征和标签，
供后续训练决策模型使用。
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ActionDataset:
    """从 JSONL 录制文件加载训练数据。

    用法:
        dataset = ActionDataset("data/recordings")
        dataset.load()  # 加载所有 .jsonl 文件
        print(dataset.summary())

        # 获取训练用的 (features, labels)
        features, labels = dataset.to_feature_label(
            action_map={"press:space": 0, "press:a": 1, "click:left": 2}
        )
    """

    def __init__(self, data_dir: str):
        self._data_dir = Path(data_dir)
        self._samples: list[dict] = []

    def load(self, files: list[str] | None = None) -> int:
        """加载 JSONL 文件。

        Args:
            files: 指定文件列表，None 则加载目录下所有 .jsonl

        Returns:
            加载的样本总数
        """
        self._samples.clear()
        if files:
            paths = [Path(f) for f in files]
        else:
            paths = sorted(self._data_dir.glob("*.jsonl"))

        for path in paths:
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        self._samples.append(sample)
                        count += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过无效行 ({path.name}): {e}")
            logger.info(f"加载 {path.name}: {count} 条样本")

        logger.info(f"总计加载 {len(self._samples)} 条样本")
        return len(self._samples)

    def summary(self) -> dict:
        """统计数据集概况。"""
        if not self._samples:
            return {"total": 0}

        action_counts: dict[str, int] = {}
        class_counts: dict[str, int] = {}

        for s in self._samples:
            # 统计动作分布
            ha = s.get("human_action", {})
            action_key = f"{ha.get('action', '?')}:{ha.get('key', ha.get('button', '?'))}"
            action_counts[action_key] = action_counts.get(action_key, 0) + 1

            # 统计出现的目标类别
            for cls_name, cnt in s.get("object_counts", {}).items():
                class_counts[cls_name] = class_counts.get(cls_name, 0) + cnt

        return {
            "total": len(self._samples),
            "action_distribution": dict(sorted(action_counts.items(), key=lambda x: -x[1])),
            "detection_classes": dict(sorted(class_counts.items(), key=lambda x: -x[1])),
        }

    def to_feature_label(
        self,
        action_map: dict[str, int] | None = None,
        min_detections: int = 0,
    ) -> tuple[list[list[float]], list[int]]:
        """将样本转换为特征向量和标签。

        特征向量（按顺序）:
            - 各类别目标计数 (按 class 名称排序)
            - 最近目标的归一化坐标 (cx, cy)
            - 最近目标的置信度
            - 目标总数

        Args:
            action_map: 动作到标签的映射。如 {"press:space": 0, "click:left": 1}。
                        None 则自动生成。
            min_detections: 最少检测目标数，低于此值的样本被过滤。

        Returns:
            (features, labels) 元组
        """
        # 收集所有类别名
        all_classes = sorted({
            cls for s in self._samples
            for cls in s.get("object_counts", {}).keys()
        })
        class_to_idx = {c: i for i, c in enumerate(all_classes)}

        # 自动生成 action_map
        if action_map is None:
            action_keys = set()
            for s in self._samples:
                ha = s.get("human_action", {})
                key = f"{ha.get('action', '?')}:{ha.get('key', ha.get('button', '?'))}"
                action_keys.add(key)
            action_map = {k: i for i, k in enumerate(sorted(action_keys))}
            logger.info(f"自动生成 action_map ({len(action_map)} 类): {action_map}")

        features = []
        labels = []

        for s in self._samples:
            detections = s.get("detections", [])
            if len(detections) < min_detections:
                continue

            ha = s.get("human_action", {})
            action_key = f"{ha.get('action', '?')}:{ha.get('key', ha.get('button', '?'))}"
            if action_key not in action_map:
                continue

            # 各类别计数特征
            counts = s.get("object_counts", {})
            count_features = [counts.get(c, 0) for c in all_classes]

            # 最近目标特征 (取第一个检测框)
            if detections:
                det = detections[0]
                bbox = det.get("bbox_norm", [0, 0, 0, 0])
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                conf = det.get("confidence", 0)
            else:
                cx, cy, conf = 0.0, 0.0, 0.0

            feature_vec = count_features + [cx, cy, conf, len(detections)]
            features.append(feature_vec)
            labels.append(action_map[action_key])

        logger.info(f"生成特征: {len(features)} 条, 维度: {len(features[0]) if features else 0}")
        return features, labels

    @property
    def samples(self) -> list[dict]:
        return self._samples

    @property
    def action_map(self) -> dict[str, int]:
        """自动扫描生成 action_map。"""
        keys = set()
        for s in self._samples:
            ha = s.get("human_action", {})
            key = f"{ha.get('action', '?')}:{ha.get('key', ha.get('button', '?'))}"
            keys.add(key)
        return {k: i for i, k in enumerate(sorted(keys))}
