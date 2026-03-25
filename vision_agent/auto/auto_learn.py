"""LLM 驱动的全自动学习管线。

完整闭环: 用户兴趣 → 搜索下载 → YOLO 检测 → LLM 标注 → MLP 训练 → RL 强化 → 决策模型
"""

import json
import logging
import time
from pathlib import Path

from ..decision.base import LoggingMixin

logger = logging.getLogger(__name__)


class AutoLearn(LoggingMixin):
    """全自动学习系统：从用户兴趣描述到可用的决策模型。

    6 个阶段:
        1. 规划 (Plan)     - LLM 分析兴趣，生成搜索策略和动作集
        2. 获取 (Fetch)    - 搜索下载相关视频/图片
        3. 标注 (Annotate) - YOLO 检测 + LLM 标注动作
        4. 训练 (Train)    - MLP/RF 监督学习训练决策模型
        5. 强化 (RL)       - DQN 在标注数据上自我强化
        6. 产出 (Export)   - 生成完整的场景 Profile + 决策模型
    """

    def __init__(
        self,
        llm_provider_name: str = "claude",
        llm_api_key: str = "",
        llm_model: str = "claude-sonnet-4-20250514",
        llm_base_url: str = "",
        yolo_model: str = "yolov8n.pt",
        output_dir: str = "runs/auto_learn",
        on_log=None,
        on_progress=None,
    ):
        self._llm_provider_name = llm_provider_name
        self._llm_api_key = llm_api_key
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._yolo_model = yolo_model
        self._output_dir = Path(output_dir)
        self._on_log = on_log
        self._on_progress = on_progress
        self._stop = False

        self._provider = None
        self._result = {}

    def stop(self):
        self._stop = True

    def _progress(self, phase: str, pct: float):
        if self._on_progress:
            try:
                self._on_progress(phase, pct)
            except Exception:
                pass

    def _create_provider(self):
        if self._provider is None:
            from ..decision.llm_provider import create_provider
            self._provider = create_provider(
                self._llm_provider_name,
                self._llm_api_key or "ollama",
                self._llm_model,
                self._llm_base_url,
            )
        return self._provider

    def run(self, interest: str, resource_type: str = "video",
            max_resources: int = 3, sample_count: int = 300,
            rl_steps: int = 2000, model_type: str = "mlp",
            epochs: int = 100, skip_fetch: bool = False,
            video_paths: list[str] | None = None) -> dict:
        """执行完整的自动学习管线。

        Args:
            interest: 用户兴趣描述，如 "王者荣耀5v5团战"
            resource_type: "video" 或 "image"
            max_resources: 最大下载资源数
            sample_count: 标注采样帧数
            rl_steps: RL 强化学习步数
            model_type: 监督学习模型类型 "mlp" 或 "rf"
            epochs: 训练轮数
            skip_fetch: 跳过下载，直接用 video_paths
            video_paths: 指定本地视频路径列表（skip_fetch=True 时使用）

        Returns:
            {"profile_path": str, "model_dir": str, "rl_dir": str, ...}
        """
        self._stop = False
        run_dir = self._output_dir / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "run_dir": str(run_dir),
            "interest": interest,
            "phases": {},
        }
        self._result = result

        provider = self._create_provider()

        try:
            # ── Phase 1: 规划 ──
            self._emit_log("=" * 50)
            self._emit_log("Phase 1/6: LLM 规划搜索策略")
            self._progress("plan", 0.0)

            from ..data.resource_fetcher import ResourceFetcher
            fetcher = ResourceFetcher(
                llm_provider=provider,
                output_dir=str(run_dir / "fetched"),
                on_log=self._on_log,
            )
            plan = fetcher.plan_search(interest, resource_type)
            result["phases"]["plan"] = plan
            self._emit_log(f"  动作集: {plan['suggested_actions']}")
            self._emit_log(f"  关键词: {plan['keywords']}")

            actions = plan.get("suggested_actions", ["attack", "defend", "idle"])
            action_descriptions = plan.get("action_descriptions", {})
            if "idle" not in actions:
                actions.append("idle")

            if self._stop:
                return result

            # ── Phase 2: 获取资源 ──
            local_files = video_paths or []
            if not skip_fetch:
                self._emit_log("=" * 50)
                self._emit_log("Phase 2/6: 搜索下载资源")
                self._progress("fetch", 0.1)

                fetch_result = fetcher.fetch(
                    interest, resource_type, max_results=max_resources,
                    progress_callback=lambda p, pct: self._progress("fetch", 0.1 + pct * 0.15),
                )
                local_files = fetch_result.get("local_files", [])
                result["phases"]["fetch"] = {
                    "total_found": len(fetch_result.get("resources", [])),
                    "downloaded": len(local_files),
                    "files": local_files,
                }

                if not local_files:
                    self._emit_log("[警告] 未下载到任何资源")
                    return result
            else:
                self._emit_log("Phase 2/6: 跳过下载（使用指定视频）")
                result["phases"]["fetch"] = {"files": local_files, "skipped": True}

            if self._stop or not local_files:
                return result

            # ── Phase 3: LLM 标注 ──
            self._emit_log("=" * 50)
            self._emit_log(f"Phase 3/6: LLM 自动标注 ({len(local_files)} 个文件)")
            self._progress("annotate", 0.25)

            annotate_dir = run_dir / "annotations"
            annotate_dir.mkdir(exist_ok=True)
            total_annotated = 0

            from ..core.detector import Detector
            from ..data.auto_annotator import AutoAnnotator

            detector = Detector(model=self._yolo_model)

            # 只处理视频文件
            video_files = [f for f in local_files if f.endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm'))]
            if not video_files:
                self._emit_log("[警告] 没有可处理的视频文件")
                return result

            for i, vpath in enumerate(video_files):
                if self._stop:
                    break

                self._emit_log(f"  标注 [{i+1}/{len(video_files)}]: {Path(vpath).name}")
                save_path = str(annotate_dir / f"annotated_{i}.jsonl")

                annotator = AutoAnnotator(
                    video_path=vpath,
                    detector=detector,
                    provider=provider,
                    actions=actions,
                    action_descriptions=action_descriptions,
                    sample_interval=30,
                    max_frames=sample_count // len(video_files),
                    use_tool_calling=True,
                    progress_callback=lambda cur, total, ann: self._progress(
                        "annotate", 0.25 + 0.25 * ((i + cur / max(total, 1)) / len(video_files))
                    ),
                )
                if self._on_log:
                    annotator.set_log_callback(self._on_log)

                try:
                    stats = annotator.run(save_path=save_path)
                    total_annotated += stats["annotated"]
                    self._emit_log(f"  标注完成: {stats['annotated']} 条")
                except Exception as e:
                    self._emit_log(f"  标注失败: {e}")

            result["phases"]["annotate"] = {"total_annotated": total_annotated}

            if total_annotated < 10:
                self._emit_log("[终止] 标注数据太少 (<10)，无法训练")
                return result

            if self._stop:
                return result

            # ── Phase 4: 监督训练 ──
            self._emit_log("=" * 50)
            self._emit_log(f"Phase 4/6: 监督学习训练 ({model_type})")
            self._progress("train", 0.5)

            model_dir = str(run_dir / "model")

            from ..data.train import DecisionTrainer

            trainer = DecisionTrainer(
                data_dir=str(annotate_dir),
                output_dir=model_dir,
                model_type=model_type,
                epochs=epochs,
                progress_callback=lambda ep, total, loss, tacc, vacc: self._progress(
                    "train", 0.5 + 0.2 * ep / max(total, 1)
                ),
            )

            try:
                metrics = trainer.run()
                result["phases"]["train"] = metrics
                result["model_dir"] = model_dir
                self._emit_log(f"  训练完成: val_acc={metrics.get('best_val_acc', 0):.3f}")
            except Exception as e:
                self._emit_log(f"  训练失败: {e}")
                return result

            if self._stop:
                return result

            # ── Phase 5: RL 强化学习 ──
            self._emit_log("=" * 50)
            self._emit_log(f"Phase 5/6: DQN 强化学习 ({rl_steps} 步)")
            self._progress("rl", 0.7)

            rl_dir = str(run_dir / "rl_model")
            rl_result = self._run_rl(
                video_files, detector, actions, action_descriptions,
                rl_steps, rl_dir,
            )
            result["phases"]["rl"] = rl_result
            result["rl_dir"] = rl_dir
            self._emit_log(f"  RL 训练完成: {rl_result.get('total_steps', 0)} 步")

            if self._stop:
                return result

            # ── Phase 6: 导出 Profile ──
            self._emit_log("=" * 50)
            self._emit_log("Phase 6/6: 导出场景 Profile")
            self._progress("export", 0.95)

            profile_path = self._export_profile(
                run_dir, interest, plan, actions, action_descriptions, model_dir,
            )
            result["profile_path"] = profile_path
            self._emit_log(f"  Profile 已保存: {profile_path}")

            self._progress("done", 1.0)
            self._emit_log("=" * 50)
            self._emit_log("全部完成！")
            self._emit_log(f"  模型目录: {model_dir}")
            self._emit_log(f"  RL 模型: {rl_dir}")
            self._emit_log(f"  Profile: {profile_path}")

        except Exception as e:
            self._emit_log(f"[错误] 自动学习异常: {e}")
            logger.exception("AutoLearn error")

        # 保存完整记录
        record_path = run_dir / "learn_record.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        return result

    def _run_rl(self, video_files: list[str], detector, actions: list[str],
                action_descriptions: dict, rl_steps: int, save_dir: str) -> dict:
        """在标注数据上运行 RL 强化学习。"""
        import cv2
        from ..decision.rl_engine import RLEngine
        from ..core.state import StateManager

        # 构建 action_key_map（简单映射）
        action_key_map = {}
        for a in actions:
            if a != "idle":
                action_key_map[a] = {"key": a[0]}  # 取首字母

        rl_engine = RLEngine(
            actions=actions,
            action_key_map=action_key_map,
            training=True,
            save_dir=save_dir,
            save_interval=500,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.995,
        )
        rl_engine.set_log_callback(self._on_log)
        rl_engine.on_start()

        state_mgr = StateManager()
        step = 0
        total_reward = 0

        for vpath in video_files:
            if self._stop or step >= rl_steps:
                break

            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                continue

            self._emit_log(f"  RL 训练: {Path(vpath).name}")

            while step < rl_steps and not self._stop:
                ret, frame = cap.read()
                if not ret:
                    break

                # 每 5 帧处理一次
                if step % 5 != 0:
                    step += 1
                    continue

                result = detector.detect(frame)
                state = state_mgr.update(result)

                rl_engine.decide(result, state)
                step += 1

                if step % 200 == 0:
                    self._progress("rl", 0.7 + 0.25 * step / rl_steps)

            cap.release()

        rl_engine.on_stop()

        return {
            "total_steps": step,
            "epsilon_final": rl_engine._epsilon,
            "memory_size": len(rl_engine._memory),
        }

    def _export_profile(self, run_dir: Path, interest: str, plan: dict,
                        actions: list[str], action_descriptions: dict,
                        model_dir: str) -> str:
        """导出为场景 Profile YAML。"""
        import yaml

        # 生成 profile 名称（简化兴趣描述）
        name = interest.replace(" ", "_")[:30].lower()
        for ch in "（）()【】[]{}，。、/\\:：":
            name = name.replace(ch, "_")

        # 构建 action_key_map
        action_key_map = {}
        key_pool = list("qwertyuiop")
        for i, a in enumerate(actions):
            if a == "idle":
                continue
            key = key_pool[i % len(key_pool)]
            action_key_map[a] = {"type": "keyboard", "action": "press", "key": key}

        profile_data = {
            "name": name,
            "display_name": interest,
            "yolo_model": self._yolo_model,
            "actions": actions,
            "action_key_map": action_key_map,
            "action_descriptions": action_descriptions,
            "decision_model_dir": model_dir,
            "decision_engine": "trained",
            "roi_regions": {},
            "scene_keywords": plan.get("keywords", []),
            "auto_train": {
                "enabled": True,
                "source": "auto_learn",
                "interest": interest,
            },
        }

        profile_path = str(run_dir / f"{name}.yaml")
        with open(profile_path, "w", encoding="utf-8") as f:
            yaml.dump(profile_data, f, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)

        # 也复制到 profiles 目录
        profiles_dir = Path("profiles")
        if profiles_dir.exists():
            import shutil
            shutil.copy2(profile_path, str(profiles_dir / f"{name}.yaml"))

        return profile_path
