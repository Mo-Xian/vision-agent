import logging
import threading
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class AutoPilot:
    """自动驾驶编排器：场景识别 → Profile 路由 → 自动训练 → 热加载决策。

    嵌入到检测流程中，作为 Agent 运行。监控检测结果，
    当识别到新场景时自动触发训练流程。
    """

    def __init__(
        self,
        profile_manager,
        scene_classifier,
        frame_buffer_size: int = 900,
        auto_train_enabled: bool = True,
        on_log=None,
        on_scene_changed=None,
        on_engine_ready=None,
    ):
        self._profile_mgr = profile_manager
        self._classifier = scene_classifier
        self._frame_buffer: deque = deque(maxlen=frame_buffer_size)
        self._auto_train_enabled = auto_train_enabled
        self._on_log = on_log
        self._on_scene_changed = on_scene_changed
        self._on_engine_ready = on_engine_ready

        self._current_scene = "unknown"
        self._lock = threading.Lock()
        self._active_engines: dict[str, object] = {}
        self._training_in_progress: set[str] = set()
        self._running = False

    def start(self):
        """初始化：注册所有 profile 的 scene_keywords 到 classifier。"""
        self._running = True
        for name, profile in self._profile_mgr.load_all().items():
            self._classifier.register_profile(name, profile.scene_keywords)
            if profile.decision_model_dir and Path(profile.decision_model_dir).exists():
                self._try_load_engine(profile)
        self._log(f"AutoPilot 启动, {len(self._profile_mgr.list_profiles())} 个 Profile 已注册")

    def stop(self):
        self._running = False

    def on_frame(self, frame: np.ndarray, result) -> str:
        """每帧调用：缓存帧 + 场景分类 + 触发训练。

        Args:
            frame: 原始帧
            result: DetectionResult

        Returns:
            当前场景名
        """
        self._frame_buffer.append(frame.copy())

        scene = self._classifier.classify(result)

        if scene != self._current_scene:
            old = self._current_scene
            self._current_scene = scene
            self._log(f"场景切换: {old} → {scene}")
            if self._on_scene_changed:
                self._on_scene_changed(old, scene)

            with self._lock:
                should_train = (
                    scene != "unknown"
                    and scene not in self._active_engines
                    and scene not in self._training_in_progress
                    and self._auto_train_enabled
                )
                if should_train:
                    self._training_in_progress.add(scene)
            if should_train:
                profile = self._profile_mgr.get(scene)
                if profile and profile.auto_train.get("enabled", False):
                    self._trigger_auto_train(profile)
                else:
                    with self._lock:
                        self._training_in_progress.discard(scene)

        return self._current_scene

    def get_engine(self, scene_name: str = ""):
        """获取当前场景（或指定场景）的决策引擎。"""
        name = scene_name or self._current_scene
        with self._lock:
            return self._active_engines.get(name)

    @property
    def current_scene(self) -> str:
        return self._current_scene

    @property
    def available_engines(self) -> list[str]:
        with self._lock:
            return list(self._active_engines.keys())

    @property
    def buffer_size(self) -> int:
        return len(self._frame_buffer)

    def _trigger_auto_train(self, profile):
        """在后台线程触发自动训练。"""
        self._log(f"[自动训练] 触发: {profile.name} (缓冲区 {len(self._frame_buffer)} 帧)")

        thread = threading.Thread(
            target=self._auto_train_thread,
            args=(profile,),
            daemon=True,
        )
        thread.start()

    def _auto_train_thread(self, profile):
        """后台自动训练线程。"""
        try:
            from .auto_trainer import AutoTrainer

            trainer = AutoTrainer(output_base_dir="runs/auto")
            if self._on_log:
                trainer.set_log_callback(self._on_log)

            frames = list(self._frame_buffer)
            if len(frames) < 30:
                self._log(f"[自动训练] {profile.name} 帧数不足 ({len(frames)}), 跳过")
                return

            auto_cfg = profile.auto_train
            result = trainer.train_from_buffer(
                frames=frames,
                profile_name=profile.name,
                yolo_model=profile.yolo_model,
                actions=profile.actions,
                action_descriptions=profile.action_descriptions,
                llm_provider_name=auto_cfg.get("llm_provider", "claude"),
                llm_api_key=auto_cfg.get("llm_api_key", ""),
                llm_model=auto_cfg.get("llm_model", "claude-sonnet-4-20250514"),
                model_type=auto_cfg.get("model_type", "mlp"),
                epochs=auto_cfg.get("epochs", 100),
            )

            if result.get("model_dir") and Path(result["model_dir"]).exists():
                profile.decision_model_dir = result["model_dir"]
                profile.decision_engine = "trained"
                self._try_load_engine(profile)
                # 持久化到 YAML，重启后不丢失
                self._save_profile(profile)
                self._log(f"[自动训练] {profile.name} 完成, 模型已热加载并保存")
            else:
                self._log(f"[自动训练] {profile.name} 训练失败或数据不足")

        except Exception as e:
            self._log(f"[自动训练] {profile.name} 异常: {e}")
        finally:
            with self._lock:
                self._training_in_progress.discard(profile.name)

    def _try_load_engine(self, profile):
        """尝试加载 profile 对应的决策引擎。"""
        try:
            if profile.decision_engine == "trained" and profile.decision_model_dir:
                from ..decision.trained_engine import TrainedEngine

                engine = TrainedEngine(
                    model_dir=profile.decision_model_dir,
                    action_key_map=profile.action_key_map,
                )
                with self._lock:
                    self._active_engines[profile.name] = engine
                if self._on_engine_ready:
                    self._on_engine_ready(profile.name, engine)
                self._log(f"引擎加载: {profile.name} (trained)")
        except Exception as e:
            self._log(f"引擎加载失败: {profile.name} → {e}")

    def _save_profile(self, profile):
        """将 profile 变更写回 YAML 文件。"""
        try:
            from ..profiles.loader import save_profile
            profile_dir = self._profile_mgr._dir
            save_path = str(profile_dir / f"{profile.name}.yaml")
            save_profile(profile, save_path)
            self._log(f"Profile 已保存: {save_path}")
        except Exception as e:
            self._log(f"Profile 保存失败: {e}")

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass
