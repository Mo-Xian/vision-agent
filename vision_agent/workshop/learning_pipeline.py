"""学习管线：录制人类操作 → 行为克隆训练 → 伪标签扩展。

流程：
  1. 动作发现 — LLM 分析按键+画面，识别语义动作（可选）
  2. 训练     — MobileNetV3 编码 + MLP 行为克隆
  3. RL 微调  — 策略梯度强化（可选）
  + 教练诊断 — LLM 分析模型弱点（可选）
  + 伪标签扩展 — 用已有模型标注新视频，高置信度样本扩充数据集
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..decision.base import LoggingMixin

logger = logging.getLogger(__name__)


@dataclass
class LearningResult:
    """学习管线的完整产出。"""
    run_dir: str = ""
    data_source: str = "recording"
    annotated_count: int = 0
    model_dir: str = ""
    rl_dir: str = ""
    profile_path: str = ""
    metrics: dict = field(default_factory=dict)
    phases: dict = field(default_factory=dict)
    coach_advice: dict = field(default_factory=dict)
    success: bool = False

    def to_dict(self) -> dict:
        return {
            "run_dir": self.run_dir,
            "data_source": self.data_source,
            "annotated_count": self.annotated_count,
            "model_dir": self.model_dir,
            "rl_dir": self.rl_dir,
            "profile_path": self.profile_path,
            "metrics": self.metrics,
            "coach_advice": self.coach_advice,
            "success": self.success,
        }


class LearningPipeline(LoggingMixin):
    """行为克隆学习管线：录制数据 → 动作发现 → 训练 → RL微调。"""

    def __init__(
        self,
        llm_provider_name: str = "minimax",
        llm_api_key: str = "",
        llm_model: str = "MiniMax-M2.7",
        llm_base_url: str = "",
        output_dir: str = "runs/workshop",
        on_log=None,
        on_progress=None,
        on_train_step=None,
        provider=None,
    ):
        self._llm_provider_name = llm_provider_name
        self._llm_api_key = llm_api_key
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._output_dir = Path(output_dir)
        self._on_log = on_log
        self._on_progress = on_progress
        self._on_train_step = on_train_step
        self._stop = False
        self._provider = provider  # 可复用外部已创建的 provider

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

    # ================================================================
    #  行为克隆训练（录制数据）
    # ================================================================

    def learn_from_recordings(
        self,
        recording_dirs: list[str],
        description: str = "",
        epochs: int = 100,
        rl_steps: int = 0,
        knowledge: str = "",
    ) -> LearningResult:
        """从录制数据学习（行为克隆模式）。

        3 个阶段:
            1. 动作发现 — LLM 分析按键+画面，识别语义动作
            2. 训练     — MobileNetV3 编码 + MLP 行为克隆
            3. RL 微调  — 策略梯度强化（可选）
        """
        self._stop = False
        run_dir = self._output_dir / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        result = LearningResult(run_dir=str(run_dir), data_source="recording")

        # 验证录制目录
        valid_dirs = []
        for d in recording_dirs:
            dp = Path(d)
            if (dp / "recording.mp4").exists() and (dp / "actions.jsonl").exists():
                valid_dirs.append(d)
            else:
                self._emit_log(f"[跳过] 录制目录不完整: {d}")

        if not valid_dirs:
            self._emit_log("[错误] 没有有效的录制数据")
            return result

        try:
            # LLM 用于动作发现和训练诊断（可选）
            try:
                provider = self._create_provider()
            except Exception:
                provider = None

            # ── Phase 1: 动作发现 ──
            self._emit_log("=" * 50)
            self._emit_log(f"Phase 1/3: 动作发现 ({len(valid_dirs)} 个录制)")
            self._progress("discover", 0.0)

            actions, action_map, action_descriptions = self._discover_actions(
                valid_dirs, provider, knowledge
            )
            result.phases["discover"] = {
                "actions": actions,
                "action_map": action_map,
            }

            if self._stop:
                return result

            # ── Phase 2: 训练 ──
            model_dir = str(run_dir / "model")
            self._emit_log("=" * 50)
            self._emit_log("Phase 2/3: 行为克隆训练")
            self._progress("train", 0.2)

            metrics = self._train_from_recordings(
                valid_dirs, actions, action_map, model_dir, epochs,
            )
            result.metrics = metrics
            result.model_dir = model_dir
            result.phases["train"] = metrics
            result.annotated_count = metrics.get("total_samples", 0)

            if metrics.get("error"):
                self._emit_log(f"  训练失败: {metrics['error']}")
                return result

            self._emit_log(f"  训练完成: val_acc={metrics.get('best_val_acc', 0):.3f}")

            # ── Phase 3: RL（由 SelfPlayLoop 在真实设备上执行） ──
            if rl_steps > 0:
                self._emit_log("=" * 50)
                self._emit_log("Phase 3/3: RL 需要真实设备交互，将在学习完成后由 Agent 部署执行")
                result.phases["rl"] = {"status": "deferred", "rl_steps": rl_steps}
            else:
                self._emit_log("Phase 3/3: 跳过 RL")

            # ── 教练诊断（LLM 可用时） ──
            if provider and not self._stop:
                self._emit_log("─" * 30)
                self._emit_log("[教练] 训练诊断...")
                from ..decision.llm_coach import LLMCoach
                coach = LLMCoach(provider, knowledge, on_log=self._on_log)
                advice = coach.diagnose_training(
                    metrics,
                    metrics.get("action_dist"),
                )
                result.coach_advice = {
                    "weaknesses": advice.weaknesses,
                    "suggestions": advice.suggestions,
                    "focus_areas": advice.focus_areas,
                    "overall_assessment": advice.overall_assessment,
                }
                if advice.suggestions:
                    self._emit_log(f"[教练建议] {'; '.join(advice.suggestions[:3])}")

            # ── 导出 Profile ──
            profile_path = self._export_profile(
                run_dir, description, actions, action_descriptions,
                model_dir, action_map,
            )
            result.profile_path = profile_path
            result.success = True

            self._progress("done", 1.0)
            self._emit_log("=" * 50)
            self._emit_log("学习完成!")
            self._emit_log(f"  模型: {model_dir}")
            self._emit_log(f"  Profile: {profile_path}")

        except Exception as e:
            self._emit_log(f"[错误] 学习管线异常: {e}")
            logger.exception("LearningPipeline error")

        self._save_session(run_dir, result)
        return result

    def _discover_actions(
        self, recording_dirs, provider, knowledge
    ) -> tuple[list[str], dict, dict]:
        """从录制数据发现动作集。"""
        if provider:
            from ..decision.llm_coach import LLMCoach
            coach = LLMCoach(provider, knowledge, on_log=self._on_log)
            result = coach.discover_actions(recording_dirs[0])
            actions = result.get("actions", ["idle"])
            action_map = result.get("action_map", {})
            action_descriptions = result.get("action_descriptions", {})
        else:
            # 无 LLM：直接从录制的按键提取动作
            key_set = set()
            for d in recording_dirs:
                actions_path = Path(d) / "actions.jsonl"
                if actions_path.exists():
                    with open(actions_path, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                s = json.loads(line.strip())
                                key = s.get("human_action", {}).get("key", "")
                                if key:
                                    key_set.add(key)
                            except Exception:
                                pass
            actions = sorted(key_set)
            if "idle" not in actions:
                actions.append("idle")
            action_map = {a: a for a in actions if a != "idle"}
            action_descriptions = {}

        self._emit_log(f"  动作集: {actions}")
        return actions, action_map, action_descriptions

    def _train_from_recordings(
        self, recording_dirs, actions, action_map, model_dir, epochs,
    ) -> dict:
        """从录制数据训练 E2E 模型。"""
        import cv2
        from ..core.vision_encoder import VisionEncoder
        from ..data.e2e_dataset import E2EDataset
        from ..data.e2e_trainer import E2ETrainer

        encoder = VisionEncoder()
        dataset = E2EDataset()
        dataset.set_actions(actions)

        total_samples = 0

        for di, rec_dir in enumerate(recording_dirs):
            if self._stop:
                break

            rec_path = Path(rec_dir)
            video_path = str(rec_path / "recording.mp4")
            actions_path = str(rec_path / "actions.jsonl")

            # 加载动作标注
            frame_actions = {}
            with open(actions_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        s = json.loads(line.strip())
                        fid = s.get("frame_id", 0)
                        key = s.get("human_action", {}).get("key", "")
                        if key:
                            # 映射到动作名 (action_map 是 key→action 方向)
                            mapped = action_map.get(key, key)
                            if mapped in actions:
                                frame_actions[fid] = mapped
                    except Exception:
                        pass

            if not frame_actions:
                continue

            self._emit_log(
                f"[训练] 录制 [{di+1}/{len(recording_dirs)}]: "
                f"{len(frame_actions)} 帧标注"
            )

            # 读取视频帧并编码
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            frame_idx = 0
            batch_frames = []
            batch_labels = []

            while not self._stop:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                label = frame_actions.get(frame_idx)
                if label is None:
                    continue

                batch_frames.append(frame)
                batch_labels.append(label)

                if len(batch_frames) >= 32:
                    embeddings = encoder.encode_batch(batch_frames)
                    for emb, lbl in zip(embeddings, batch_labels):
                        dataset.add_sample(emb, lbl)
                        # 嵌入空间数据增强：50% 概率添加高斯噪声扰动副本，增加样本多样性
                        if np.random.random() < 0.5:
                            augmented = emb + np.random.normal(0, 0.02, emb.shape)
                            dataset.add_sample(augmented, lbl)
                    total_samples += len(batch_frames)
                    batch_frames.clear()
                    batch_labels.clear()

                    self._progress(
                        "train",
                        0.3 + 0.35 * min(total_samples / max(len(frame_actions) * len(recording_dirs), 1), 1.0)
                    )

            if batch_frames:
                embeddings = encoder.encode_batch(batch_frames)
                for emb, lbl in zip(embeddings, batch_labels):
                    dataset.add_sample(emb, lbl)
                    # 嵌入空间数据增强：50% 概率添加高斯噪声扰动副本，增加样本多样性
                    if np.random.random() < 0.5:
                        augmented = emb + np.random.normal(0, 0.02, emb.shape)
                        dataset.add_sample(augmented, lbl)
                total_samples += len(batch_frames)

            cap.release()

        self._emit_log(f"[训练] 编码完成: {len(dataset)} 样本, {dataset.num_actions} 动作")

        if len(dataset) < 10:
            return {"error": "insufficient_data", "count": len(dataset)}

        dataset_path = str(Path(model_dir) / "e2e_dataset.npz")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        dataset.save(dataset_path)

        self._progress("train", 0.65)
        def _train_cb(ep, total, loss, tacc, vacc):
            self._progress("train", 0.65 + 0.2 * ep / max(total, 1))
            if self._on_train_step:
                try:
                    self._on_train_step(loss, tacc, vacc)
                except Exception:
                    pass

        trainer = E2ETrainer(
            dataset=dataset,
            output_dir=model_dir,
            epochs=epochs,
            progress_callback=_train_cb,
            on_log=self._on_log,
        )

        metrics = trainer.train()
        metrics["total_samples"] = total_samples
        return metrics

    def _export_profile(self, run_dir, description, actions,
                        action_descriptions, model_dir,
                        action_map=None) -> str:
        """导出场景 Profile YAML。"""
        import yaml

        name = description.replace(" ", "_")[:30].lower()
        for ch in "（）()【】[]{}，。、/\\:：":
            name = name.replace(ch, "_")

        # 使用真实按键映射
        # Invert action_map (key→action) to (action→key) for profile
        action_key_map = {}
        if action_map:
            for key, action_name in action_map.items():
                if action_name != "idle":
                    action_key_map[action_name] = {
                        "type": "keyboard", "action": "press", "key": key,
                    }
        else:
            key_pool = list("qwertyuiop")
            for i, a in enumerate(actions):
                if a == "idle":
                    continue
                key = key_pool[i % len(key_pool)]
                action_key_map[a] = {"type": "keyboard", "action": "press", "key": key}

        profile_data = {
            "name": name,
            "display_name": description,
            "actions": actions,
            "action_key_map": action_key_map,
            "action_descriptions": action_descriptions or {},
            "decision_model_dir": model_dir,
        }

        profile_path = str(run_dir / f"{name}.yaml")
        with open(profile_path, "w", encoding="utf-8") as f:
            yaml.dump(profile_data, f, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)

        profiles_dir = Path("profiles")
        if profiles_dir.exists():
            import shutil
            shutil.copy2(profile_path, str(profiles_dir / f"{name}.yaml"))

        return profile_path

    # ================================================================
    #  伪标签扩展：用已训练模型标注新视频
    # ================================================================

    def expand_from_videos(
        self,
        model_dir: str,
        video_paths: list[str],
        confidence_threshold: float = 0.85,
        max_idle_ratio: float = 0.3,
        epochs: int = 100,
        mix_ratio: float = 0.3,
    ) -> LearningResult:
        """用已有模型对新视频做伪标签，扩充数据后重新训练。

        Args:
            model_dir: 已训练模型目录（含 model.pt + model.meta.json）
            video_paths: 新视频文件路径
            confidence_threshold: 伪标签置信度阈值（低于此值丢弃）
            max_idle_ratio: idle 动作最大保留比例（防止 idle 主导）
            epochs: 重新训练轮数
            mix_ratio: 原始数据混入比例（0.3 = 至少 30% 原始数据）

        Returns:
            新一轮训练的 LearningResult
        """
        import cv2
        import torch

        self._stop = False
        run_dir = self._output_dir / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        result = LearningResult(run_dir=str(run_dir), data_source="pseudo_label")

        model_path = Path(model_dir) / "model.pt"
        meta_path = Path(model_dir) / "model.meta.json"
        dataset_path = Path(model_dir) / "e2e_dataset.npz"

        if not model_path.exists() or not meta_path.exists():
            self._emit_log("[错误] 模型文件不存在")
            return result

        # 加载模型和元数据（兼容 BC 和 DQN 两种模型）
        from ..core.vision_encoder import VisionEncoder
        from ..data.e2e_dataset import E2EDataset

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        actions = meta.get("action_list", [])
        action_map_inv = {v: k for k, v in meta.get("action_map", {}).items()}

        model_type = meta.get("model_type", "e2e_mlp")
        embed_dim = meta.get("embed_dim", 576)
        num_actions = meta.get("num_actions", len(actions))
        hidden_dims = meta.get("hidden_dims", [256, 128])

        if model_type == "dqn":
            from ..rl.dqn_agent import DQNNetwork
            model = DQNNetwork(embed_dim, num_actions, hidden_dims)
        else:
            from ..data.e2e_trainer import E2EMLP
            model = E2EMLP(embed_dim, num_actions, hidden_dims)

        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        encoder = VisionEncoder()

        self._emit_log("=" * 50)
        self._emit_log(f"伪标签扩展: {len(video_paths)} 个视频")
        self._emit_log(f"  模型: {model_dir}")
        self._emit_log(f"  动作集: {actions}")
        self._emit_log(f"  置信度阈值: {confidence_threshold}")
        self._progress("expand", 0.0)

        # ── Phase 1: 伪标签生成 ──
        pseudo_samples = []  # (embedding, action_name, confidence)
        total_frames = 0
        accepted = 0
        rejected = 0
        action_counts = defaultdict(int)

        for vi, vpath in enumerate(video_paths):
            if self._stop:
                break

            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                self._emit_log(f"  [跳过] 无法打开: {vpath}")
                continue

            video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._emit_log(
                f"  [{vi+1}/{len(video_paths)}] {Path(vpath).name} ({video_total} 帧)"
            )

            batch_frames = []
            frame_idx = 0

            while not self._stop:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                batch_frames.append(frame)

                if len(batch_frames) >= 32:
                    self._process_pseudo_batch(
                        encoder, model, batch_frames, actions,
                        confidence_threshold, pseudo_samples,
                        action_counts,
                    )
                    total_frames += len(batch_frames)
                    accepted = len(pseudo_samples)
                    rejected = total_frames - accepted
                    batch_frames.clear()

                    if frame_idx % 100 == 0:
                        pct = (vi + frame_idx / max(video_total, 1)) / len(video_paths)
                        self._progress("expand", pct * 0.5)

            # 尾巴
            if batch_frames:
                self._process_pseudo_batch(
                    encoder, model, batch_frames, actions,
                    confidence_threshold, pseudo_samples,
                    action_counts,
                )
                total_frames += len(batch_frames)

            cap.release()

        accepted = len(pseudo_samples)
        rejected = total_frames - accepted

        self._emit_log(f"  伪标签完成: 总帧={total_frames}, "
                       f"接受={accepted} ({accepted/max(total_frames,1)*100:.1f}%), "
                       f"拒绝={rejected}")
        self._emit_log(f"  动作分布: {dict(action_counts)}")

        if accepted < 10:
            self._emit_log("[错误] 高置信度样本太少，无法扩展")
            return result

        # ── idle 比例控制 ──
        idle_idx = None
        for i, a in enumerate(actions):
            if a == "idle":
                idle_idx = i
                break

        if idle_idx is not None and action_counts.get("idle", 0) > 0:
            non_idle = sum(v for k, v in action_counts.items() if k != "idle")
            max_idle = int(non_idle * max_idle_ratio / (1 - max_idle_ratio))
            current_idle = action_counts.get("idle", 0)
            if current_idle > max_idle:
                # 随机采样 idle
                import random
                idle_samples = [s for s in pseudo_samples if s[1] == "idle"]
                other_samples = [s for s in pseudo_samples if s[1] != "idle"]
                random.shuffle(idle_samples)
                pseudo_samples = other_samples + idle_samples[:max_idle]
                self._emit_log(f"  idle 限制: {current_idle} → {min(current_idle, max_idle)}")

        # ── Phase 2: 合并数据集 ──
        self._emit_log("=" * 50)
        self._emit_log("合并数据集...")
        self._progress("expand", 0.55)

        new_dataset = E2EDataset()
        new_dataset.set_actions(actions)

        # 加载原始数据集
        orig_count = 0
        if dataset_path.exists():
            orig_dataset = E2EDataset.load(str(dataset_path))
            orig_count = len(orig_dataset)

            # 确保原始数据至少占 mix_ratio
            min_orig = int(len(pseudo_samples) * mix_ratio / (1 - mix_ratio))
            orig_to_add = max(orig_count, min_orig)

            for emb, lbl in zip(orig_dataset.embeddings[:orig_to_add],
                                orig_dataset.labels[:orig_to_add]):
                action_name = actions[lbl] if lbl < len(actions) else "idle"
                new_dataset.add_sample(emb, action_name)

            self._emit_log(f"  原始数据: {orig_count} 样本")
        else:
            self._emit_log("  [注意] 未找到原始数据集，仅使用伪标签")

        # 添加伪标签数据
        for emb, action_name, conf in pseudo_samples:
            new_dataset.add_sample(emb, action_name)

        self._emit_log(
            f"  合并后: {len(new_dataset)} 样本 "
            f"(原始 {orig_count} + 伪标签 {len(pseudo_samples)})"
        )

        # ── Phase 3: 重新训练 ──
        new_model_dir = str(run_dir / "model")
        self._emit_log("=" * 50)
        self._emit_log(f"重新训练 ({epochs} 轮)")
        self._progress("train", 0.6)

        from ..data.e2e_trainer import E2ETrainer

        Path(new_model_dir).mkdir(parents=True, exist_ok=True)
        new_dataset.save(str(Path(new_model_dir) / "e2e_dataset.npz"))

        trainer = E2ETrainer(
            dataset=new_dataset,
            output_dir=new_model_dir,
            epochs=epochs,
            progress_callback=lambda ep, total, loss, tacc, vacc: self._progress(
                "train", 0.6 + 0.35 * ep / max(total, 1)
            ),
            on_log=self._on_log,
        )

        metrics = trainer.train()
        metrics["total_samples"] = len(new_dataset)
        metrics["pseudo_samples"] = len(pseudo_samples)
        metrics["orig_samples"] = orig_count
        metrics["confidence_threshold"] = confidence_threshold

        result.metrics = metrics
        result.model_dir = new_model_dir
        result.annotated_count = len(new_dataset)

        if metrics.get("error"):
            self._emit_log(f"  训练失败: {metrics['error']}")
            return result

        old_acc = meta.get("best_val_acc", 0)
        new_acc = metrics.get("best_val_acc", 0)
        self._emit_log(f"  训练完成: val_acc={new_acc:.3f} (之前: {old_acc:.3f})")

        if new_acc > old_acc:
            self._emit_log(f"  精度提升: +{(new_acc-old_acc)*100:.1f}%")
        else:
            self._emit_log(f"  [注意] 精度未提升，可能需要更高置信度阈值或更多数据")

        result.success = True
        self._progress("done", 1.0)
        self._emit_log("=" * 50)
        self._emit_log("伪标签扩展完成!")
        self._emit_log(f"  新模型: {new_model_dir}")

        self._save_session(run_dir, result)
        return result

    @staticmethod
    def _process_pseudo_batch(
        encoder, model, frames, actions,
        threshold, out_samples, action_counts,
    ):
        """处理一批帧的伪标签。"""
        import torch

        embeddings = encoder.encode_batch(frames)
        tensors = torch.tensor(
            np.array(embeddings, dtype=np.float32)
        )

        with torch.no_grad():
            logits = model(tensors)
            probs = torch.softmax(logits, dim=-1)
            confidences, indices = probs.max(dim=-1)

        for emb, conf, idx in zip(embeddings, confidences, indices):
            conf_val = conf.item()
            action_idx = idx.item()
            if conf_val >= threshold and action_idx < len(actions):
                action_name = actions[action_idx]
                out_samples.append((emb, action_name, conf_val))
                action_counts[action_name] += 1

    def _save_session(self, run_dir, result):
        """保存会话记录。"""
        record_path = run_dir / "session.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2, default=str)
