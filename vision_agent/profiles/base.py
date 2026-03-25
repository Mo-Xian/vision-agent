from dataclasses import dataclass, field
from pathlib import Path

from .loader import load_profile


@dataclass
class SceneProfile:
    name: str
    display_name: str
    yolo_model: str
    actions: list[str] = field(default_factory=list)
    action_key_map: dict[str, dict] = field(default_factory=dict)
    action_descriptions: dict[str, str] = field(default_factory=dict)
    decision_model_dir: str = ""
    decision_engine: str = "rule"
    roi_regions: dict[str, tuple] = field(default_factory=dict)
    scene_keywords: list[str] = field(default_factory=list)
    auto_train: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "yolo_model": self.yolo_model,
            "actions": self.actions,
            "action_key_map": self.action_key_map,
            "action_descriptions": self.action_descriptions,
            "decision_model_dir": self.decision_model_dir,
            "decision_engine": self.decision_engine,
            "roi_regions": {k: list(v) for k, v in self.roi_regions.items()},
            "scene_keywords": self.scene_keywords,
            "auto_train": self.auto_train,
        }


class ProfileManager:
    def __init__(self, profile_dir: str = "profiles"):
        self._dir = Path(profile_dir)
        self._profiles: dict[str, SceneProfile] = {}

    def load_all(self) -> dict[str, SceneProfile]:
        self._profiles.clear()
        if not self._dir.exists():
            return self._profiles
        for f in sorted(self._dir.glob("*.yaml")):
            try:
                p = load_profile(str(f))
                self._profiles[p.name] = p
            except Exception as e:
                print(f"[ProfileManager] 加载失败 {f}: {e}")
        return self._profiles

    def get(self, name: str) -> SceneProfile | None:
        if not self._profiles:
            self.load_all()
        return self._profiles.get(name)

    def list_profiles(self) -> list[str]:
        if not self._profiles:
            self.load_all()
        return list(self._profiles.keys())
