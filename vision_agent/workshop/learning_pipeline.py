"""学习管线：编排 分析→标注→训练→强化 全流程。

核心理念："像人看视频学习"——
  1. 看视频（LLM 直接看截图，分析场景和内容）
  2. LLM 标注帧数据（积累决策经验）
  3. 视觉编码 + 标签传播 + MLP 训练
  4. 策略梯度 RL（可选）
  5. 产出可用模型和 Profile
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..decision.base import LoggingMixin
from .video_analyzer import VideoAnalyzer, VideoInsight

logger = logging.getLogger(__name__)


@dataclass
class LearningResult:
    """学习管线的完整产出。"""
    run_dir: str = ""
    insight: VideoInsight | None = None
    annotated_count: int = 0
    model_dir: str = ""
    rl_dir: str = ""
    profile_path: str = ""
    metrics: dict = field(default_factory=dict)
    phases: dict = field(default_factory=dict)
    success: bool = False

    def to_dict(self) -> dict:
        return {
            "run_dir": self.run_dir,
            "insight": self.insight.to_dict() if self.insight else None,
            "annotated_count": self.annotated_count,
            "model_dir": self.model_dir,
            "rl_dir": self.rl_dir,
            "profile_path": self.profile_path,
            "metrics": self.metrics,
            "success": self.success,
        }


class LearningPipeline(LoggingMixin):
    """端到端学习管线：视频 → 分析 → 标注 → 训练 → 强化 → 模型。

    4 个核心阶段:
        1. Analyze  — LLM 视觉分析视频内容
        2. Annotate — LLM 逐帧标注决策数据
        3. Train    — 视觉编码 + 标签传播 + MLP 训练
        4. Reinforce — 策略梯度 RL（可选）

    用法:
        pipeline = LearningPipeline(
            llm_provider_name="minimax",
            llm_api_key=api_key,
            llm_model="MiniMax-M2.7",
        )
        result = pipeline.learn_from_videos(
            video_paths=["video1.mp4"],
            description="王者荣耀5v5",
        )
    """

    def __init__(
        self,
        llm_provider_name: str = "minimax",
        llm_api_key: str = "",
        llm_model: str = "MiniMax-M2.7",
        llm_base_url: str = "",
        output_dir: str = "runs/workshop",
        on_log=None,
        on_progress=None,
    ):
        self._llm_provider_name = llm_provider_name
        self._llm_api_key = llm_api_key
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._output_dir = Path(output_dir)
        self._on_log = on_log
        self._on_progress = on_progress
        self._stop = False
        self._provider = None

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

    def learn_from_videos(
        self,
        video_paths: list[str],
        description: str = "",
        sample_count: int = 300,
        epochs: int = 100,
        rl_steps: int = 2000,
        send_image: bool = True,
        analyze_samples: int = 20,
        batch_size: int = 5,
        knowledge: str = "",
    ) -> LearningResult:
        """从视频列表学习，完整执行 4 阶段管线。"""
        self._stop = False
        run_dir = self._output_dir / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        result = LearningResult(run_dir=str(run_dir))
        provider = self._create_provider()

        # 过滤有效视频
        video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.ts', '.m4v'}
        video_files = [f for f in video_paths if Path(f).suffix.lower() in video_exts]
        if not video_files:
            self._emit_log("[错误] 没有可处理的视频文件")
            return result

        try:
            # ── Phase 1: 分析视频 ──
            self._emit_log("=" * 50)
            self._emit_log(f"Phase 1/4: 分析视频 ({len(video_files)} 个)")
            self._emit_log("  模式: LLM 视觉分析")
            self._progress("analyze", 0.0)

            analyzer = VideoAnalyzer(
                provider=provider,
                detector=None,
                sample_count=analyze_samples,
                knowledge=knowledge,
                on_log=self._on_log,
            )
            insights = analyzer.analyze_batch(video_files)
            insight = analyzer.merge_insights(insights) if insights else VideoInsight()
            result.insight = insight
            result.phases["analyze"] = insight.to_dict()

            actions = insight.suggested_actions or ["attack", "defend", "idle"]
            action_descriptions = insight.action_descriptions or {}
            if "idle" not in actions:
                actions.append("idle")

            self._emit_log(f"  场景: {insight.scene_type}")
            self._emit_log(f"  动作集: {actions}")
            self._emit_log(f"  分析: {insight.analysis_summary[:200]}")

            if self._stop:
                return result

            # ── Phase 2: LLM 标注 ──
            self._emit_log("=" * 50)
            self._emit_log(f"Phase 2/4: LLM 标注 ({len(video_files)} 个视频)")
            self._progress("annotate", 0.2)

            annotate_dir = run_dir / "annotations"
            annotate_dir.mkdir(exist_ok=True)
            total_annotated = 0

            from ..data.auto_annotator import AutoAnnotator
            from ..core.keyframe import KeyFrameSampler

            # 关键帧检测
            video_keyframes = {}
            self._emit_log("  [关键帧] 检测高价值帧...")
            for vpath in video_files:
                target_per_video = max(sample_count // len(video_files), 30)
                sampler = KeyFrameSampler(
                    target_count=target_per_video,
                    on_log=self._on_log,
                )
                try:
                    indices = sampler.sample_indices(vpath)
                    video_keyframes[vpath] = indices
                    self._emit_log(f"  [关键帧] {Path(vpath).name}: {len(indices)} 帧")
                except Exception as ex:
                    self._emit_log(f"  [关键帧] {Path(vpath).name} 失败: {ex}")
                    video_keyframes[vpath] = None

            for i, vpath in enumerate(video_files):
                if self._stop:
                    break

                self._emit_log(f"  标注 [{i+1}/{len(video_files)}]: {Path(vpath).name}")
                save_path = str(annotate_dir / f"annotated_{i}.jsonl")

                target_frames = max(sample_count // len(video_files), 20)
                kf_indices = video_keyframes.get(vpath)

                annotator = AutoAnnotator(
                    video_path=vpath,
                    provider=provider,
                    actions=actions,
                    detector=None,
                    action_descriptions=action_descriptions,
                    sample_interval=10,
                    max_frames=target_frames,
                    send_image=send_image,
                    use_tool_calling=True,
                    batch_size=batch_size,
                    knowledge=knowledge,
                    keyframe_indices=kf_indices,
                    progress_callback=lambda cur, total, ann: self._progress(
                        "annotate", 0.2 + 0.3 * ((i + cur / max(total, 1)) / len(video_files))
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

            result.annotated_count = total_annotated
            result.phases["annotate"] = {"total_annotated": total_annotated}

            if total_annotated < 10:
                self._emit_log("[终止] 标注数据太少 (<10)，无法训练")
                return result

            if self._stop:
                return result

            model_dir = str(run_dir / "model")

            # ── Phase 3: 端到端训练 ──
            self._emit_log("=" * 50)
            self._emit_log("Phase 3/4: 端到端视觉编码 + MLP 训练")
            self._progress("encode", 0.5)

            e2e_result = self._run_e2e_train(
                video_files, annotate_dir, actions, model_dir,
                sample_interval=10,
                max_frames=max(sample_count // max(len(video_files), 1), 20),
                epochs=epochs,
            )
            metrics = e2e_result
            result.metrics = metrics
            result.model_dir = model_dir
            result.phases["train"] = metrics

            if metrics.get("error"):
                self._emit_log(f"  端到端训练失败: {metrics['error']}")
                return result
            self._emit_log(f"  训练完成: val_acc={metrics.get('best_val_acc', 0):.3f}")

            # Phase 4: RL 强化
            if rl_steps > 0 and not self._stop:
                self._emit_log("=" * 50)
                self._emit_log(f"Phase 4/4: 端到端 RL 强化 ({rl_steps} 步)")
                self._progress("rl", 0.75)
                rl_result = self._run_e2e_rl(
                    video_files, actions, model_dir, rl_steps,
                )
                result.rl_dir = model_dir
                result.phases["rl"] = rl_result
                self._emit_log(f"  RL 完成: {rl_result.get('total_steps', 0)} 步")
            else:
                self._emit_log("Phase 4/4: 跳过 RL")

            # ── 导出 Profile ──
            profile_path = self._export_profile(
                run_dir, description or insight.scene_type,
                actions, action_descriptions, model_dir, insight,
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

        # 保存会话记录
        record_path = run_dir / "session.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2, default=str)

        return result

    def _run_e2e_train(
        self, video_files, annotate_dir, actions, model_dir,
        sample_interval=10, max_frames=300, epochs=100,
        encode_interval=3, propagate_radius=5,
    ) -> dict:
        """端到端训练：视频密集抽帧 → 视觉编码 → 标签传播 → 训练 MLP。"""
        import cv2
        from ..core.vision_encoder import VisionEncoder
        from ..data.e2e_dataset import E2EDataset
        from ..data.e2e_trainer import E2ETrainer

        # ── 加载 LLM 标注 ──
        annotated_actions = {}
        for jsonl_file in sorted(Path(annotate_dir).glob("*.jsonl")):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        fid = sample.get("frame_id", 0)
                        action = sample.get("human_action", {}).get("key", "")
                        if action:
                            annotated_actions[fid] = action
                    except Exception:
                        pass

        if not annotated_actions:
            self._emit_log("[E2E] 无标注数据可用")
            return {"error": "no_annotations"}

        self._emit_log(f"[E2E] LLM 标注: {len(annotated_actions)} 帧")

        # ── 标签传播 ──
        sorted_fids = sorted(annotated_actions.keys())

        def get_label_for_frame(fid: int) -> str | None:
            import bisect
            pos = bisect.bisect_left(sorted_fids, fid)
            best_dist = float('inf')
            best_label = None
            for idx in (pos - 1, pos):
                if 0 <= idx < len(sorted_fids):
                    dist = abs(fid - sorted_fids[idx])
                    if dist < best_dist:
                        best_dist = dist
                        best_label = annotated_actions[sorted_fids[idx]]
            if best_dist <= propagate_radius * encode_interval:
                return best_label
            return None

        # ── 视觉编码 ──
        self._emit_log("[E2E] 加载视觉编码器 (MobileNetV3-Small)...")
        encoder = VisionEncoder()

        dataset = E2EDataset()
        dataset.set_actions(actions)

        total_encoded = 0
        total_propagated = 0

        for vi, vpath in enumerate(video_files):
            if self._stop:
                break

            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._emit_log(
                f"[E2E] 编码视频 [{vi+1}/{len(video_files)}]: "
                f"{Path(vpath).name} ({total_frames} 帧, 每 {encode_interval} 帧取 1 帧)"
            )

            frame_idx = 0
            batch_frames = []
            batch_fids = []

            while not self._stop:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % encode_interval != 0:
                    continue

                label = annotated_actions.get(frame_idx) or get_label_for_frame(frame_idx)
                if label is None:
                    continue

                if frame_idx in annotated_actions:
                    total_encoded += 1
                else:
                    total_propagated += 1

                batch_frames.append(frame)
                batch_fids.append((frame_idx, label))

                if len(batch_frames) >= 16:
                    embeddings = encoder.encode_batch(batch_frames)
                    for emb, (fid, lbl) in zip(embeddings, batch_fids):
                        dataset.add_sample(emb, lbl)
                    batch_frames.clear()
                    batch_fids.clear()

                    done = total_encoded + total_propagated
                    self._progress("encode", 0.5 + 0.15 * min(done / max(len(annotated_actions) * 5, 1), 1.0))

            if batch_frames:
                embeddings = encoder.encode_batch(batch_frames)
                for emb, (fid, lbl) in zip(embeddings, batch_fids):
                    dataset.add_sample(emb, lbl)

            cap.release()

        self._emit_log(
            f"[E2E] 编码完成: {len(dataset)} 样本 "
            f"(精确标注 {total_encoded} + 传播 {total_propagated}), "
            f"{dataset.num_actions} 动作"
        )

        if len(dataset) < 10:
            return {"error": "insufficient_data", "count": len(dataset)}

        dataset_path = str(Path(model_dir) / "e2e_dataset.npz")
        dataset.save(dataset_path)

        # ── 训练 ──
        self._progress("train", 0.65)
        trainer = E2ETrainer(
            dataset=dataset,
            output_dir=model_dir,
            epochs=epochs,
            progress_callback=lambda ep, total, loss, tacc, vacc: self._progress(
                "train", 0.65 + 0.2 * ep / max(total, 1)
            ),
            on_log=self._on_log,
        )

        return trainer.train()

    def _run_e2e_rl(self, video_files, actions, model_dir, rl_steps) -> dict:
        """端到端 RL：在视觉嵌入空间做策略梯度强化。"""
        import cv2
        import torch
        import torch.nn.functional as F
        from ..core.vision_encoder import VisionEncoder
        from ..data.e2e_trainer import E2EMLP

        encoder = VisionEncoder()
        meta_path = Path(model_dir) / "model.meta.json"
        model_path = Path(model_dir) / "model.pt"
        if not model_path.exists():
            return {"error": "no_model"}

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model = E2EMLP(
            input_dim=meta.get("embed_dim", 576),
            num_actions=meta.get("num_actions", len(actions)),
            hidden_dims=meta.get("hidden_dims", [256, 128]),
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        step = 0
        total_reward = 0.0
        action_counts = {a: 0 for a in actions}
        log_probs_buffer = []
        rewards_buffer = []
        prev_action = -1

        for vpath in video_files:
            if self._stop or step >= rl_steps:
                break

            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                continue

            frame_counter = 0
            while step < rl_steps and not self._stop:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_counter += 1
                if frame_counter % 5 != 0:
                    continue

                embedding = encoder.encode(frame)
                tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                logits = model(tensor)
                probs = F.softmax(logits, dim=-1)

                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)

                reward = 0.0
                a_idx = action_idx.item()
                if a_idx != prev_action:
                    reward += 0.1
                if a_idx < len(actions) and actions[a_idx] != "idle":
                    reward += 0.05
                if a_idx < len(actions):
                    action_counts[actions[a_idx]] = action_counts.get(actions[a_idx], 0) + 1

                log_probs_buffer.append(log_prob)
                rewards_buffer.append(reward)
                total_reward += reward
                prev_action = a_idx
                step += 1

                if len(log_probs_buffer) >= 32:
                    self._policy_update(optimizer, log_probs_buffer, rewards_buffer)
                    log_probs_buffer.clear()
                    rewards_buffer.clear()

                if step % 200 == 0:
                    self._progress("rl", 0.85 + 0.1 * step / rl_steps)
                    self._emit_log(f"  [RL] step={step}, avg_reward={total_reward/step:.3f}")

            cap.release()

        if log_probs_buffer:
            self._policy_update(optimizer, log_probs_buffer, rewards_buffer)

        torch.save(model.state_dict(), model_path)
        self._emit_log(f"[RL] 模型已更新: {model_path}")

        return {
            "total_steps": step,
            "avg_reward": round(total_reward / max(step, 1), 4),
            "action_dist": {k: v for k, v in action_counts.items() if v > 0},
        }

    @staticmethod
    def _policy_update(optimizer, log_probs, rewards):
        """REINFORCE 策略梯度更新。"""
        import torch
        returns = []
        R = 0
        gamma = 0.99
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        if returns.std() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for lp, G in zip(log_probs, returns):
            loss -= lp * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _export_profile(self, run_dir, description, actions,
                        action_descriptions, model_dir, insight) -> str:
        """导出场景 Profile YAML。"""
        import yaml

        name = description.replace(" ", "_")[:30].lower()
        for ch in "（）()【】[]{}，。、/\\:：":
            name = name.replace(ch, "_")

        action_key_map = {}
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
            "action_descriptions": action_descriptions,
            "decision_model_dir": model_dir,
            "scene_keywords": insight.scene_keywords if insight else [],
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
