"""学习管线：编排 分析→标注→训练→强化 全流程。

核心理念："像人看视频学习"——
  1. 看视频（LLM 直接看截图，分析场景和内容）
  2. LLM 标注帧数据（积累决策经验）
  3. 训练决策模型（总结规律）
  4. 强化学习（优化策略，可选）
  5. 产出可用模型和 Profile

YOLO 检测为可选辅助，LLM 视觉分析是主要手段。
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
    """学习管线：视频 → 分析 → 标注 → 训练 → 强化 → 模型。

    4 个核心阶段:
        1. Analyze  — LLM 视觉分析视频内容（截图为主，YOLO 可选辅助）
        2. Annotate — LLM 逐帧标注决策数据
        3. Train    — MLP/RF 监督训练
        4. Reinforce — DQN 强化学习（可选）

    用法:
        pipeline = LearningPipeline(
            llm_provider_name="claude",
            llm_api_key=api_key,
            llm_model="claude-sonnet-4-20250514",
        )
        result = pipeline.learn_from_videos(
            video_paths=["video1.mp4", "video2.mp4"],
            description="王者荣耀5v5",
        )
    """

    def __init__(
        self,
        llm_provider_name: str = "claude",
        llm_api_key: str = "",
        llm_model: str = "claude-sonnet-4-20250514",
        llm_base_url: str = "",
        yolo_model: str = "",
        output_dir: str = "runs/workshop",
        on_log=None,
        on_progress=None,
    ):
        self._llm_provider_name = llm_provider_name
        self._llm_api_key = llm_api_key
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._yolo_model = yolo_model  # 空字符串 = 不使用 YOLO
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

    def _create_detector(self):
        """创建 YOLO 检测器（如果配置了 yolo_model）。"""
        if not self._yolo_model:
            return None
        from ..core.detector import Detector
        return Detector(model=self._yolo_model)

    def learn_from_videos(
        self,
        video_paths: list[str],
        description: str = "",
        sample_count: int = 300,
        model_type: str = "mlp",
        epochs: int = 100,
        rl_steps: int = 2000,
        send_image: bool = True,
        analyze_samples: int = 20,
        batch_size: int = 5,
        knowledge: str = "",
        e2e: bool = False,
    ) -> LearningResult:
        """从视频列表学习，完整执行 4 阶段管线。"""
        self._stop = False
        run_dir = self._output_dir / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        result = LearningResult(run_dir=str(run_dir))
        provider = self._create_provider()
        detector = self._create_detector()  # 可能为 None

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
            if detector:
                self._emit_log(f"  YOLO 辅助: {self._yolo_model}")
            else:
                self._emit_log("  模式: LLM 纯视觉分析（无 YOLO）")
            self._progress("analyze", 0.0)

            analyzer = VideoAnalyzer(
                provider=provider,
                detector=detector,
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

            # 关键帧检测（e2e 模式自动启用）
            video_keyframes = {}  # vpath → list[int] or None
            if e2e:
                from ..core.keyframe import KeyFrameSampler
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
                    detector=detector,  # 可选
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

            if e2e:
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

                # Phase 4: 端到端 RL（基于视觉嵌入）
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
            else:
                # ── Phase 3: 传统特征 + MLP 训练 ──
                self._emit_log("=" * 50)
                self._emit_log(f"Phase 3/4: 监督学习训练 ({model_type})")
                self._progress("train", 0.5)

                from ..data.train import DecisionTrainer

                trainer = DecisionTrainer(
                    data_dir=str(annotate_dir),
                    output_dir=model_dir,
                    model_type=model_type,
                    epochs=epochs,
                    progress_callback=lambda ep, total, loss, tacc, vacc: self._progress(
                        "train", 0.5 + 0.25 * ep / max(total, 1)
                    ),
                )

                try:
                    metrics = trainer.run()
                    result.metrics = metrics
                    result.model_dir = model_dir
                    result.phases["train"] = metrics
                    self._emit_log(f"  训练完成: val_acc={metrics.get('best_val_acc', 0):.3f}")
                except Exception as e:
                    self._emit_log(f"  训练失败: {e}")
                    return result

                if self._stop:
                    return result

                # ── Phase 4: RL 强化学习（可选，需要 YOLO）──
                if rl_steps > 0 and detector is not None:
                    self._emit_log("=" * 50)
                    self._emit_log(f"Phase 4/4: DQN 强化学习 ({rl_steps} 步)")
                    self._progress("rl", 0.75)

                    rl_dir = str(run_dir / "rl_model")
                    rl_result = self._run_rl(
                        video_files, detector, actions, rl_steps, rl_dir,
                    )
                    result.rl_dir = rl_dir
                    result.phases["rl"] = rl_result
                    self._emit_log(f"  RL 完成: {rl_result.get('total_steps', 0)} 步")
                elif rl_steps > 0:
                    self._emit_log("Phase 4/4: 跳过 RL（需要 YOLO 检测器）")
                else:
                    self._emit_log("Phase 4/4: 跳过 RL 强化学习")

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

    def incremental_learn(
        self,
        model_dir: str,
        new_videos: list[str],
        **kwargs,
    ) -> LearningResult:
        """增量学习：在已有模型基础上，用新视频继续训练。"""
        meta_path = Path(model_dir) / "model.meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            existing_actions = list(meta.get("action_map", {}).keys())
            if existing_actions:
                kwargs.setdefault("description", f"增量学习 ({len(new_videos)} 新视频)")

        return self.learn_from_videos(video_paths=new_videos, **kwargs)

    def _run_e2e_train(
        self, video_files, annotate_dir, actions, model_dir,
        sample_interval=10, max_frames=300, epochs=100,
        encode_interval=3, propagate_radius=5,
    ) -> dict:
        """端到端训练：视频密集抽帧 → 视觉编码 → 标签传播 → 训练 MLP。

        核心策略：LLM 标注少量关键帧，标签传播到大量相邻帧。
          - LLM 标注: 每 sample_interval 帧标注 1 帧（如每 10 帧）→ ~300 条
          - 视觉编码: 每 encode_interval 帧编码 1 帧（如每 3 帧）→ ~3000+ 条
          - 标签传播: 每个标注覆盖前后 propagate_radius 帧 → 10x 数据放大

        Args:
            encode_interval: 编码采样间隔（越小数据越多，但编码越慢）
            propagate_radius: 标签传播半径（标注帧前后各多少帧继承同一标签）
        """
        import cv2
        from ..core.vision_encoder import VisionEncoder
        from ..data.e2e_dataset import E2EDataset
        from ..data.e2e_trainer import E2ETrainer

        # ── 加载 LLM 标注 ──
        annotated_actions = {}  # frame_id → action
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

        # ── 标签传播：从稀疏标注生成密集标签 ──
        # 对每帧找最近的已标注帧，在 propagate_radius 内继承其标签
        sorted_fids = sorted(annotated_actions.keys())

        def get_label_for_frame(fid: int) -> str | None:
            """查找帧 fid 最近的标注，如果在传播半径内则返回标签。"""
            # 二分查找最近的标注帧
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
            # 传播半径内才使用（换算成实际帧距）
            if best_dist <= propagate_radius * encode_interval:
                return best_label
            return None

        # ── 视觉编码：密集抽帧 ──
        self._emit_log(f"[E2E] 加载视觉编码器 (MobileNetV3-Small)...")
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

                # 查找标签（精确标注 或 传播标签）
                label = annotated_actions.get(frame_idx) or get_label_for_frame(frame_idx)
                if label is None:
                    continue

                if frame_idx in annotated_actions:
                    total_encoded += 1
                else:
                    total_propagated += 1

                batch_frames.append(frame)
                batch_fids.append((frame_idx, label))

                # 批量编码
                if len(batch_frames) >= 16:
                    embeddings = encoder.encode_batch(batch_frames)
                    for emb, (fid, lbl) in zip(embeddings, batch_fids):
                        dataset.add_sample(emb, lbl)
                    batch_frames.clear()
                    batch_fids.clear()

                    done = total_encoded + total_propagated
                    self._progress("encode", 0.5 + 0.15 * min(done / max(len(annotated_actions) * 5, 1), 1.0))

            # 处理剩余
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

        # ── 保存数据集 ──
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

        # 加载编码器和模型
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

        # 简单策略梯度 RL：用动作多样性和时序一致性作为奖励
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

                # 编码 + 前向
                embedding = encoder.encode(frame)
                tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                logits = model(tensor)
                probs = F.softmax(logits, dim=-1)

                # 采样动作
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)

                # 简单奖励设计：
                # +0.1 动作变化（鼓励多样性）
                # +0.05 不全是 idle
                # -0.1 连续相同动作超过 5 次
                reward = 0.0
                a_idx = action_idx.item()
                if a_idx != prev_action:
                    reward += 0.1
                if a_idx < len(actions) and actions[a_idx] != "idle":
                    reward += 0.05
                # 惩罚过于集中的动作
                if a_idx < len(actions):
                    action_counts[actions[a_idx]] = action_counts.get(actions[a_idx], 0) + 1

                log_probs_buffer.append(log_prob)
                rewards_buffer.append(reward)
                total_reward += reward
                prev_action = a_idx
                step += 1

                # 每 32 步更新一次
                if len(log_probs_buffer) >= 32:
                    self._policy_update(optimizer, log_probs_buffer, rewards_buffer)
                    log_probs_buffer.clear()
                    rewards_buffer.clear()

                if step % 200 == 0:
                    self._progress("rl", 0.85 + 0.1 * step / rl_steps)
                    self._emit_log(f"  [RL] step={step}, avg_reward={total_reward/step:.3f}")

            cap.release()

        # 最后一批
        if log_probs_buffer:
            self._policy_update(optimizer, log_probs_buffer, rewards_buffer)

        # 保存 RL 微调后的模型
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
        # 计算折扣回报
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

    def _run_rl(self, video_files, detector, actions, rl_steps, save_dir) -> dict:
        """在视频上运行 RL 强化学习（需要 detector）。"""
        import cv2
        from ..decision.rl_engine import RLEngine
        from ..core.state import StateManager

        action_key_map = {}
        for a in actions:
            if a != "idle":
                action_key_map[a] = {"key": a[0]}

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

                result = detector.detect(frame)
                state = state_mgr.update(result)
                rl_engine.decide(result, state)
                step += 1

                if step % 200 == 0:
                    self._progress("rl", 0.75 + 0.2 * step / rl_steps)

            cap.release()

        rl_engine.on_stop()
        return {
            "total_steps": step,
            "epsilon_final": rl_engine._epsilon,
            "memory_size": len(rl_engine._memory),
        }

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
            "yolo_model": self._yolo_model or "none",
            "actions": actions,
            "action_key_map": action_key_map,
            "action_descriptions": action_descriptions,
            "decision_model_dir": model_dir,
            "decision_engine": "trained",
            "roi_regions": {},
            "scene_keywords": insight.scene_keywords if insight else [],
            "auto_train": {
                "enabled": True,
                "source": "workshop",
            },
        }

        profile_path = str(run_dir / f"{name}.yaml")
        with open(profile_path, "w", encoding="utf-8") as f:
            yaml.dump(profile_data, f, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)

        # 复制到 profiles 目录
        profiles_dir = Path("profiles")
        if profiles_dir.exists():
            import shutil
            shutil.copy2(profile_path, str(profiles_dir / f"{name}.yaml"))

        return profile_path
