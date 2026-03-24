from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .base import SceneProfile


def load_profile(path: str) -> "SceneProfile":
    from .base import SceneProfile

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    roi = data.get("roi_regions", {})
    roi_tuples = {k: tuple(v) for k, v in roi.items()} if roi else {}

    return SceneProfile(
        name=data["name"],
        display_name=data.get("display_name", data["name"]),
        yolo_model=data.get("yolo_model", "yolov8n.pt"),
        actions=data.get("actions", []),
        action_key_map=data.get("action_key_map", {}),
        action_descriptions=data.get("action_descriptions", {}),
        decision_model_dir=data.get("decision_model_dir", ""),
        decision_engine=data.get("decision_engine", "rule"),
        roi_regions=roi_tuples,
        scene_keywords=data.get("scene_keywords", []),
        auto_train=data.get("auto_train", {}),
        extra=data.get("extra", {}),
    )


def save_profile(profile: "SceneProfile", path: str) -> None:
    from .base import SceneProfile

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            profile.to_dict(),
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
