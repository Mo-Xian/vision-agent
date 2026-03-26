"""场景管理：创建、持久化、切换训练场景。

每个场景是一个独立的工作空间，包含：
  - 视频源列表
  - LLM 分析见解
  - 动作空间配置
  - 训练历史（多次 session）
  - 最佳模型引用

目录结构:
  runs/workshop/scenes/<name>/
    ├── scene.json          # 场景元数据
    ├── sessions/           # 训练会话
    │   ├── 20260326_100000/
    │   │   ├── annotations/
    │   │   ├── model/
    │   │   └── session.json
    │   └── ...
    └── profile.yaml        # 导出的 Profile
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_SCENES_ROOT = "runs/workshop/scenes"


@dataclass
class Scene:
    """训练场景。"""
    name: str = ""
    display_name: str = ""
    description: str = ""
    scene_dir: str = ""
    video_sources: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    action_descriptions: dict[str, str] = field(default_factory=dict)
    knowledge: str = ""  # 场景先验知识（规则/教程/提示）
    scene_type: str = ""
    analysis_summary: str = ""
    yolo_model: str = ""
    best_model_dir: str = ""
    best_val_acc: float = 0.0
    profile_path: str = ""
    session_count: int = 0
    total_annotated: int = 0
    created_at: str = ""
    updated_at: str = ""
    status: str = "idle"  # idle / analyzing / annotating / training / ready

    def save(self):
        """持久化场景到 scene.json。"""
        if not self.scene_dir:
            return
        self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        path = Path(self.scene_dir) / "scene.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "scene_dir": self.scene_dir,
            "video_sources": self.video_sources,
            "actions": self.actions,
            "action_descriptions": self.action_descriptions,
            "knowledge": self.knowledge,
            "scene_type": self.scene_type,
            "analysis_summary": self.analysis_summary,
            "yolo_model": self.yolo_model,
            "best_model_dir": self.best_model_dir,
            "best_val_acc": self.best_val_acc,
            "profile_path": self.profile_path,
            "session_count": self.session_count,
            "total_annotated": self.total_annotated,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
        }

    @classmethod
    def load(cls, scene_dir: str) -> "Scene":
        path = Path(scene_dir) / "scene.json"
        if not path.exists():
            raise FileNotFoundError(f"场景文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def add_videos(self, paths: list[str]):
        """添加视频源（去重）。"""
        existing = set(self.video_sources)
        for p in paths:
            if p not in existing:
                self.video_sources.append(p)
                existing.add(p)
        self.save()

    def remove_videos(self, paths: list[str]):
        """移除视频源。"""
        to_remove = set(paths)
        self.video_sources = [v for v in self.video_sources if v not in to_remove]
        self.save()

    def new_session_dir(self) -> str:
        """创建新的训练会话目录。"""
        sessions_dir = Path(self.scene_dir) / "sessions"
        session_dir = sessions_dir / time.strftime("%Y%m%d_%H%M%S")
        session_dir.mkdir(parents=True, exist_ok=True)
        self.session_count += 1
        return str(session_dir)

    def list_sessions(self) -> list[dict]:
        """列出所有训练会话。"""
        sessions_dir = Path(self.scene_dir) / "sessions"
        if not sessions_dir.exists():
            return []
        results = []
        for d in sorted(sessions_dir.iterdir(), reverse=True):
            session_file = d / "session.json"
            if session_file.exists():
                try:
                    with open(session_file, "r", encoding="utf-8") as f:
                        results.append(json.load(f))
                except Exception:
                    pass
        return results

    def update_best_model(self, model_dir: str, val_acc: float, profile_path: str = ""):
        """更新最佳模型。"""
        if val_acc > self.best_val_acc or not self.best_model_dir:
            self.best_model_dir = model_dir
            self.best_val_acc = val_acc
            if profile_path:
                self.profile_path = profile_path
            self.status = "ready"
            self.save()

    def update_from_insight(self, insight_dict: dict):
        """从 LLM 分析见解更新场景信息。"""
        if insight_dict.get("scene_type"):
            self.scene_type = insight_dict["scene_type"]
        if insight_dict.get("suggested_actions"):
            self.actions = insight_dict["suggested_actions"]
        if insight_dict.get("action_descriptions"):
            self.action_descriptions = insight_dict["action_descriptions"]
        if insight_dict.get("analysis_summary"):
            self.analysis_summary = insight_dict["analysis_summary"]
        self.save()


class SceneManager:
    """场景管理器：创建、列出、加载、删除场景。"""

    def __init__(self, root_dir: str = _SCENES_ROOT):
        self._root = Path(root_dir)

    def create(self, name: str, display_name: str = "", description: str = "") -> Scene:
        """创建新场景。"""
        # 清理名称（用于目录名）
        safe_name = name.strip().replace(" ", "_").lower()
        for ch in "（）()【】[]{}，。、/\\:：\"'":
            safe_name = safe_name.replace(ch, "_")
        safe_name = safe_name[:50] or "unnamed"

        scene_dir = self._root / safe_name
        if scene_dir.exists():
            # 已存在则直接加载
            return Scene.load(str(scene_dir))

        scene = Scene(
            name=safe_name,
            display_name=display_name or name,
            description=description,
            scene_dir=str(scene_dir),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        scene.save()
        logger.info(f"创建场景: {safe_name} -> {scene_dir}")
        return scene

    def list_scenes(self) -> list[Scene]:
        """列出所有场景。"""
        scenes = []
        if not self._root.exists():
            return scenes
        for d in sorted(self._root.iterdir()):
            scene_file = d / "scene.json"
            if scene_file.exists():
                try:
                    scenes.append(Scene.load(str(d)))
                except Exception as e:
                    logger.warning(f"加载场景失败 {d}: {e}")
        return scenes

    def load(self, name: str) -> Scene | None:
        """按名称加载场景。"""
        scene_dir = self._root / name
        if not (scene_dir / "scene.json").exists():
            return None
        return Scene.load(str(scene_dir))

    def delete(self, name: str) -> bool:
        """删除场景（移到回收站）。"""
        import shutil
        scene_dir = self._root / name
        if not scene_dir.exists():
            return False
        # 移到 _trash 而非直接删除
        trash_dir = self._root / "_trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        dest = trash_dir / f"{name}_{int(time.time())}"
        shutil.move(str(scene_dir), str(dest))
        logger.info(f"场景已移至回收站: {name}")
        return True

    def get_or_create(self, name: str, display_name: str = "") -> Scene:
        """获取已有场景或创建新场景。"""
        scene = self.load(name)
        if scene:
            return scene
        return self.create(name, display_name)
