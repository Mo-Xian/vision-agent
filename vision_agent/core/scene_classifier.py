"""根据检测结果自动判断当前场景类型。"""

import logging
from collections import Counter, deque
from .detector import DetectionResult

logger = logging.getLogger(__name__)


class SceneClassifier:
    """根据检测结果中的类别名与注册的场景关键词做交集打分，
    结合历史平滑避免频繁切换。"""

    def __init__(self, stability_threshold: int = 10):
        self._profiles: dict[str, set[str]] = {}
        self._history: deque[str] = deque(maxlen=30)
        self._current_scene: str = "unknown"
        self._stability_threshold = stability_threshold

    def register_profile(self, name: str, keywords: list[str]):
        """注册一个场景的关键词集合。"""
        self._profiles[name] = set(keywords)
        logger.debug("注册场景 '%s'，关键词: %s", name, keywords)

    def classify(self, result: DetectionResult) -> str:
        """根据当前帧检测结果分类场景。
        连续 stability_threshold 帧指向同一场景才切换。"""
        detected_names = {det.class_name for det in result.detections}

        # 计算各场景得分（交集大小 / 关键词总数）
        best_scene = "unknown"
        best_score = 0.0
        for name, keywords in self._profiles.items():
            if not keywords:
                continue
            overlap = len(detected_names & keywords)
            score = overlap / len(keywords)
            if score > best_score:
                best_score = score
                best_scene = name

        # 没有任何交集则归为 unknown
        if best_score == 0.0:
            best_scene = "unknown"

        self._history.append(best_scene)

        # 历史平滑：最近 N 帧中最频繁的场景
        recent = list(self._history)[-self._stability_threshold:]
        counter = Counter(recent)
        most_common, count = counter.most_common(1)[0]

        if count >= self._stability_threshold and most_common != self._current_scene:
            old = self._current_scene
            self._current_scene = most_common
            logger.info("场景切换: %s -> %s", old, self._current_scene)

        return self._current_scene

    @property
    def current_scene(self) -> str:
        return self._current_scene

    def reset(self):
        self._history.clear()
        self._current_scene = "unknown"
