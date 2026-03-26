"""学习会话：持久化单次学习过程的完整状态和产出。"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LearningSession:
    """学习会话记录。

    持久化到 runs/workshop/<timestamp>/session.json，
    记录一次完整学习的输入、过程和产出。
    """
    session_dir: str = ""
    video_paths: list[str] = field(default_factory=list)
    description: str = ""
    scene_type: str = ""
    actions: list[str] = field(default_factory=list)
    annotated_count: int = 0
    model_dir: str = ""
    rl_dir: str = ""
    profile_path: str = ""
    metrics: dict = field(default_factory=dict)
    created_at: str = ""
    status: str = "pending"  # pending / running / completed / failed

    def save(self):
        """保存会话到 session.json。"""
        if not self.session_dir:
            return
        path = Path(self.session_dir) / "session.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def to_dict(self) -> dict:
        return {
            "session_dir": self.session_dir,
            "video_paths": self.video_paths,
            "description": self.description,
            "scene_type": self.scene_type,
            "actions": self.actions,
            "annotated_count": self.annotated_count,
            "model_dir": self.model_dir,
            "rl_dir": self.rl_dir,
            "profile_path": self.profile_path,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "status": self.status,
        }

    @classmethod
    def load(cls, session_dir: str) -> "LearningSession":
        """从目录加载会话。"""
        path = Path(session_dir) / "session.json"
        if not path.exists():
            raise FileNotFoundError(f"会话文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @staticmethod
    def list_sessions(base_dir: str = "runs/workshop") -> list["LearningSession"]:
        """列出所有历史学习会话。"""
        sessions = []
        root = Path(base_dir)
        if not root.exists():
            return sessions
        for session_file in sorted(root.rglob("session.json"), reverse=True):
            try:
                session = LearningSession.load(str(session_file.parent))
                sessions.append(session)
            except Exception as e:
                logger.warning(f"加载会话失败 {session_file}: {e}")
        return sessions
