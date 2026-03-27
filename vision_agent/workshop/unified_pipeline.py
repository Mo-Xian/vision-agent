"""统一学习管线 — 模拟人类学习过程。

流程：
  Phase 1: 教师教学 — 人类录制操作 → 行为克隆
  Phase 2: 自主学习 — 搜索在线视频 → 伪标签/LLM辅助标注 → 扩充训练
  Phase 3: 自我实践 — 自对弈 RL 强化

像人一样学习：
  - 有老师教 → 先模仿（BC）
  - 老师教完 → 自己找资料学（在线视频 + 伪标签）
  - 遇到不懂的 → 问 LLM 教练
  - LLM 也不确定 → 标记待人工补充
  - 学够了 → 实战练习（自对弈 RL）
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from ..decision.base import LoggingMixin

logger = logging.getLogger(__name__)


class LearningPhase(str, Enum):
    """学习阶段。"""
    IDLE = "idle"
    DETECTING_INPUT = "detecting_input"       # 检测输入类型
    TEACHER_LEARNING = "teacher_learning"     # 教师教学 (BC)
    SEARCHING_VIDEOS = "searching_videos"     # 搜索在线视频
    SELF_STUDY = "self_study"                 # 自主学习 (伪标签+LLM)
    ASKING_COACH = "asking_coach"             # 询问 LLM 教练
    WAITING_HUMAN = "waiting_human"           # 等待人工补充
    PRACTICE = "practice"                     # 自我实践 (RL)
    COMPLETE = "complete"


@dataclass
class UnifiedResult:
    """统一学习产出。"""
    run_dir: str = ""
    phase_history: list[dict] = field(default_factory=list)
    model_dir: str = ""
    metrics: dict = field(default_factory=dict)
    skill_gaps: list[str] = field(default_factory=list)
    human_review_items: list[dict] = field(default_factory=list)
    coach_advice: dict = field(default_factory=dict)
    success: bool = False

    def to_dict(self) -> dict:
        return {
            "run_dir": self.run_dir,
            "phase_history": self.phase_history,
            "model_dir": self.model_dir,
            "metrics": self.metrics,
            "skill_gaps": self.skill_gaps,
            "human_review_items": self.human_review_items,
            "coach_advice": self.coach_advice,
            "success": self.success,
        }


class UnifiedPipeline(LoggingMixin):
    """统一学习管线 — 自动编排 BC → 自主学习 → RL 全流程。

    使用方式:
        pipeline = UnifiedPipeline(llm_provider_name="minimax", ...)

        # 一键学习：自动检测输入类型并编排流程
        result = pipeline.run(
            recording_dirs=["recordings/session1"],  # 人类录制（可选）
            extra_videos=["video1.mp4"],             # 额外视频（可选）
            description="王者荣耀5v5",
            device_serial="xxx",                     # 手机设备（可选，用于RL）
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
        on_phase_change=None,
        on_need_human=None,
    ):
        self._llm_provider_name = llm_provider_name
        self._llm_api_key = llm_api_key
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._output_dir = Path(output_dir)
        self._on_log = on_log
        self._on_progress = on_progress
        self._on_phase_change = on_phase_change
        self._on_need_human = on_need_human
        self._stop = False
        self._provider = None
        self._phase = LearningPhase.IDLE

    def stop(self):
        self._stop = True

    @property
    def phase(self) -> LearningPhase:
        return self._phase

    def _set_phase(self, phase: LearningPhase):
        self._phase = phase
        if self._on_phase_change:
            try:
                self._on_phase_change(phase.value)
            except Exception:
                pass

    def _progress(self, pct: float, detail: str = ""):
        if self._on_progress:
            try:
                self._on_progress(self._phase.value, pct, detail)
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
    #  主入口
    # ================================================================

    def run(
        self,
        recording_dirs: list[str] | None = None,
        extra_videos: list[str] | None = None,
        description: str = "",
        knowledge: str = "",
        epochs: int = 100,
        device_serial: str = "",
        selfplay_preset: str = "",
        selfplay_episodes: int = 0,
        confidence_threshold: float = 0.85,
        search_online: bool = True,
    ) -> UnifiedResult:
        """一键启动统一学习流程。

        自动检测输入类型并编排：教师教学 → 自主学习 → 自我实践。
        """
        self._stop = False
        run_dir = self._output_dir / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        result = UnifiedResult(run_dir=str(run_dir))
        recording_dirs = recording_dirs or []
        extra_videos = extra_videos or []

        try:
            provider = self._create_provider()
        except Exception as e:
            provider = None
            self._emit_log(f"[警告] LLM 连接失败: {e}（将跳过 LLM 辅助功能）")

        model_dir = ""
        actions = []
        action_map = {}

        # ────────────────────────────────────
        #  Phase 1: 检测输入 + 教师教学
        # ────────────────────────────────────
        self._set_phase(LearningPhase.DETECTING_INPUT)

        human_recordings = []
        plain_videos = []

        for d in recording_dirs:
            if self._is_human_recording(d):
                human_recordings.append(d)
                self._emit_log(f"[检测] {Path(d).name} → 人类操作录制")
            else:
                # 有视频但无动作数据 → 当作额外视频
                video_path = Path(d) / "recording.mp4"
                if video_path.exists():
                    plain_videos.append(str(video_path))
                    self._emit_log(f"[检测] {Path(d).name} → 纯视频（无操作数据）")

        plain_videos.extend(extra_videos)

        if self._stop:
            return result

        # ── 教师教学（有人类录制时） ──
        if human_recordings:
            self._set_phase(LearningPhase.TEACHER_LEARNING)
            self._emit_log("=" * 50)
            self._emit_log("Phase 1: 教师教学（行为克隆）")
            self._emit_log(f"  {len(human_recordings)} 个人类录制")

            from .learning_pipeline import LearningPipeline
            bc_pipeline = LearningPipeline(
                llm_provider_name=self._llm_provider_name,
                llm_api_key=self._llm_api_key,
                llm_model=self._llm_model,
                llm_base_url=self._llm_base_url,
                output_dir=str(run_dir / "bc"),
                on_log=self._on_log,
                on_progress=lambda phase, pct: self._progress(pct * 0.3, f"BC: {phase}"),
            )

            bc_result = bc_pipeline.learn_from_recordings(
                recording_dirs=human_recordings,
                description=description,
                epochs=epochs,
                knowledge=knowledge,
                rl_steps=selfplay_episodes if device_serial else 0,
            )

            result.phase_history.append({
                "phase": "teacher_learning",
                "success": bc_result.success,
                "model_dir": bc_result.model_dir,
                "metrics": bc_result.metrics,
            })

            if bc_result.success:
                model_dir = bc_result.model_dir
                # 读取动作集
                meta_path = Path(model_dir) / "model.meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    actions = meta.get("action_list", [])
                    action_map = meta.get("action_map", {})
                self._emit_log(f"  BC 完成: val_acc={bc_result.metrics.get('best_val_acc', 0):.3f}")
            else:
                self._emit_log("[警告] BC 训练失败，尝试继续后续阶段")

        if self._stop:
            result.model_dir = model_dir
            return result

        # ────────────────────────────────────
        #  Phase 2: 自主学习
        # ────────────────────────────────────

        # 2a. 分析技能缺口 — 让 LLM 教练评估还缺什么
        if provider and model_dir and not self._stop:
            self._set_phase(LearningPhase.ASKING_COACH)
            self._emit_log("=" * 50)
            self._emit_log("Phase 2a: 技能差距分析")

            skill_gaps = self._analyze_skill_gaps(
                provider, actions, description, knowledge,
                result.phase_history[0].get("metrics", {}) if result.phase_history else {},
            )
            result.skill_gaps = skill_gaps

            if skill_gaps:
                self._emit_log(f"  教练发现 {len(skill_gaps)} 个技能缺口:")
                for gap in skill_gaps:
                    self._emit_log(f"    - {gap}")

        if self._stop:
            result.model_dir = model_dir
            return result

        # 2b. 搜索在线学习资源（如果启用且有 LLM）
        search_videos = []
        if search_online and provider and result.skill_gaps and not self._stop:
            self._set_phase(LearningPhase.SEARCHING_VIDEOS)
            self._emit_log("=" * 50)
            self._emit_log("Phase 2b: 搜索在线学习资源")

            suggestions = self._suggest_search_queries(
                provider, description, result.skill_gaps,
            )
            if suggestions:
                self._emit_log("  建议搜索以下关键词来获取训练视频:")
                for q in suggestions:
                    self._emit_log(f"    🔍 {q}")

                # 尝试 MiniMax MCP Search
                search_results = self._search_online(provider, suggestions)
                if search_results:
                    self._emit_log(f"  找到 {len(search_results)} 个相关资源:")
                    for sr in search_results[:5]:
                        self._emit_log(f"    • {sr.get('title', '')} — {sr.get('url', '')}")

                # 搜索结果目前需要人工下载，标记待处理
                if search_results:
                    result.human_review_items.append({
                        "type": "download_videos",
                        "reason": "在线搜索到相关学习资源，请下载后导入",
                        "search_results": search_results[:10],
                    })

        if self._stop:
            result.model_dir = model_dir
            return result

        # 2c. 伪标签扩展 — 用现有模型标注额外视频
        if plain_videos and model_dir and not self._stop:
            self._set_phase(LearningPhase.SELF_STUDY)
            self._emit_log("=" * 50)
            self._emit_log(f"Phase 2c: 自主学习（{len(plain_videos)} 个视频）")

            expand_result = self._self_study(
                model_dir, plain_videos, provider, actions,
                knowledge, confidence_threshold, epochs, run_dir,
            )

            result.phase_history.append({
                "phase": "self_study",
                "success": expand_result.get("success", False),
                "model_dir": expand_result.get("model_dir", ""),
                "metrics": expand_result.get("metrics", {}),
                "uncertain_frames": expand_result.get("uncertain_count", 0),
            })

            if expand_result.get("success") and expand_result.get("model_dir"):
                model_dir = expand_result["model_dir"]
                self._emit_log(f"  自主学习完成: 新模型 → {model_dir}")

            # 有无法处理的帧 → 标记待人工
            if expand_result.get("human_review"):
                result.human_review_items.extend(expand_result["human_review"])
                self._emit_log(
                    f"  [待人工] {len(expand_result['human_review'])} 个问题需要人工确认"
                )

        if self._stop:
            result.model_dir = model_dir
            return result

        # ────────────────────────────────────
        #  Phase 3: 自我实践 (RL)
        # ────────────────────────────────────
        if device_serial and selfplay_preset and not self._stop:
            self._set_phase(LearningPhase.PRACTICE)
            self._emit_log("=" * 50)
            self._emit_log("Phase 3: 自我实践（自对弈 RL）")
            self._emit_log(f"  设备: {device_serial}")
            self._emit_log(f"  预设: {selfplay_preset}")
            self._emit_log(f"  热启动模型: {model_dir or '无（从零开始）'}")

            result.phase_history.append({
                "phase": "practice",
                "status": "ready",
                "device": device_serial,
                "preset": selfplay_preset,
                "bc_model": model_dir,
            })

            # 自对弈需要实时设备交互，通过信号通知 GUI 启动
            # 这里只记录配置，实际执行由 main_window 的 SelfPlayLoop 处理
            self._emit_log("  RL 自对弈已就绪，请在 Agent 部署 Tab 启动")
        elif not device_serial:
            self._emit_log("─" * 30)
            self._emit_log("[提示] 未连接设备，跳过自对弈实践。连接手机后可在训练工坊启动 RL。")

        # ────────────────────────────────────
        #  完成
        # ────────────────────────────────────
        self._set_phase(LearningPhase.COMPLETE)
        result.model_dir = model_dir
        result.success = bool(model_dir)

        # 最终教练总结
        if provider and model_dir and not self._stop:
            self._emit_log("─" * 30)
            self._emit_log("[教练] 最终总结...")
            summary = self._coach_summary(provider, result)
            result.coach_advice = summary
            if summary.get("next_steps"):
                self._emit_log("  下一步建议:")
                for step in summary["next_steps"]:
                    self._emit_log(f"    → {step}")

        self._progress(1.0, "完成")
        self._emit_log("=" * 50)
        self._emit_log("统一学习流程完成!")
        if model_dir:
            self._emit_log(f"  最终模型: {model_dir}")
        if result.human_review_items:
            self._emit_log(f"  待人工处理: {len(result.human_review_items)} 项")

        # 保存会话
        session_path = run_dir / "unified_session.json"
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2, default=str)

        return result

    # ================================================================
    #  输入检测
    # ================================================================

    @staticmethod
    def _is_human_recording(rec_dir: str) -> bool:
        """检测录制目录是否包含人类操作数据。

        判断依据：
        - 有 actions.jsonl 且包含有效的 human_action 事件
        - 按键时间间隔有自然波动（非机器人）
        """
        rec_path = Path(rec_dir)
        video = rec_path / "recording.mp4"
        actions = rec_path / "actions.jsonl"

        if not video.exists():
            return False
        if not actions.exists():
            return False

        # 检查动作文件内容
        timestamps = []
        action_count = 0
        with open(actions, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    ha = sample.get("human_action", {})
                    if ha.get("key"):
                        action_count += 1
                        ts = sample.get("timestamp", 0)
                        if ts:
                            timestamps.append(ts)
                except Exception:
                    pass

        if action_count < 5:
            return False

        # 检查时间间隔的自然性（标准差 > 0.05s 说明是人类操作）
        if len(timestamps) > 5:
            intervals = [timestamps[i+1] - timestamps[i]
                         for i in range(len(timestamps)-1) if timestamps[i+1] > timestamps[i]]
            if intervals:
                std = np.std(intervals)
                return std > 0.05  # 人类操作有自然波动

        return action_count > 5

    # ================================================================
    #  技能差距分析
    # ================================================================

    def _analyze_skill_gaps(
        self, provider, actions, description, knowledge, metrics,
    ) -> list[str]:
        """让 LLM 分析当前模型还缺什么技能。"""
        prompt = (
            f"我训练了一个游戏 AI（{description or '未知游戏'}），"
            f"目前学会了以下动作：{', '.join(actions)}\n\n"
        )
        if metrics:
            acc = metrics.get("best_val_acc", 0)
            samples = metrics.get("total_samples", 0)
            dist = metrics.get("action_dist", {})
            prompt += f"训练样本: {samples}，准确率: {acc:.1%}\n"
            if dist:
                prompt += f"动作分布: {json.dumps(dist, ensure_ascii=False)}\n"
        if knowledge:
            prompt += f"\n游戏背景知识:\n{knowledge}\n"
        prompt += (
            "\n请分析:\n"
            "1. 当前 AI 可能缺少哪些关键技能/策略？\n"
            "2. 哪些动作的训练数据可能不足？\n"
            "3. 应该重点学习什么？\n\n"
            '返回 JSON: {"skill_gaps": ["缺失技能1", ...], '
            '"weak_actions": ["数据不足的动作", ...], '
            '"priority": "最应该优先学习的内容"}'
        )

        try:
            resp = provider.chat(
                messages=[{"role": "user", "content": prompt}],
                system="你是游戏 AI 教练。分析 AI 的技能缺口，只返回 JSON。",
                max_tokens=1024,
            )
            data = self._parse_json(resp.text)
            gaps = data.get("skill_gaps", [])
            weak = data.get("weak_actions", [])
            return gaps + weak
        except Exception as e:
            self._emit_log(f"  [教练] 技能分析失败: {e}")
            return []

    # ================================================================
    #  在线搜索
    # ================================================================

    def _suggest_search_queries(
        self, provider, description, skill_gaps,
    ) -> list[str]:
        """让 LLM 建议搜索关键词。"""
        prompt = (
            f"我在训练 {description or '游戏'} AI，需要更多训练视频。\n"
            f"当前技能缺口: {', '.join(skill_gaps)}\n\n"
            "请建议 3-5 个搜索关键词（中文），用于在 B 站/YouTube 找到相关的游戏教学视频。\n"
            '返回 JSON: {"queries": ["关键词1", "关键词2", ...]}'
        )

        try:
            resp = provider.chat(
                messages=[{"role": "user", "content": prompt}],
                system="建议精准的视频搜索关键词。只返回 JSON。",
                max_tokens=512,
            )
            data = self._parse_json(resp.text)
            return data.get("queries", [])
        except Exception:
            return []

    def _search_online(self, provider, queries: list[str]) -> list[dict]:
        """通过 MiniMax MCP Search 搜索在线视频。"""
        results = []
        try:
            from ..decision.minimax_mcp import MiniMaxMCPTools
            api_key = self._llm_api_key
            if not api_key:
                return []
            mcp = MiniMaxMCPTools(api_key=api_key)
            for q in queries[:3]:
                try:
                    search_result = mcp.web_search(f"{q} 教学视频 游戏攻略")
                    if isinstance(search_result, list):
                        results.extend(search_result[:3])
                    elif isinstance(search_result, str):
                        results.append({"title": q, "content": search_result[:200]})
                except Exception:
                    pass
        except ImportError:
            pass

        return results

    # ================================================================
    #  自主学习（智能伪标签）
    # ================================================================

    def _self_study(
        self, model_dir, video_paths, provider, actions,
        knowledge, confidence_threshold, epochs, run_dir,
    ) -> dict:
        """智能自主学习 — 结合伪标签 + LLM 辅助。

        对于每帧：
        - 模型置信度高 → 直接用模型标签
        - 模型不确定 → 询问 LLM 教练
        - LLM 也不确定 → 标记待人工
        """
        import cv2
        import torch
        from ..core.vision_encoder import VisionEncoder
        from ..data.e2e_dataset import E2EDataset

        model_path = Path(model_dir) / "model.pt"
        meta_path = Path(model_dir) / "model.meta.json"
        dataset_path = Path(model_dir) / "e2e_dataset.npz"

        if not model_path.exists() or not meta_path.exists():
            return {"success": False, "error": "model_not_found"}

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_type = meta.get("model_type", "e2e_mlp")
        embed_dim = meta.get("embed_dim", 576)
        num_actions = meta.get("num_actions", len(actions))
        hidden_dims = meta.get("hidden_dims", [256, 128])

        if not actions:
            actions = meta.get("action_list", [])

        if model_type == "dqn":
            from ..rl.dqn_agent import DQNNetwork
            model = DQNNetwork(embed_dim, num_actions, hidden_dims)
        else:
            from ..data.e2e_trainer import E2EMLP
            model = E2EMLP(embed_dim, num_actions, hidden_dims)

        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        encoder = VisionEncoder()

        # LLM 教练（用于辅助标注不确定帧）
        coach = None
        if provider:
            from ..decision.llm_coach import LLMCoach
            coach = LLMCoach(provider, knowledge, on_log=self._on_log)

        # 收集样本
        confident_samples = []    # 模型自信的
        coach_samples = []        # LLM 教练标注的
        uncertain_frames = []     # 都不确定的
        total_frames = 0

        low_confidence_threshold = 0.5  # 低于此值才请教 LLM
        coach_batch_frames = []
        coach_batch_embeddings = []

        for vi, vpath in enumerate(video_paths):
            if self._stop:
                break

            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                continue

            video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._emit_log(f"  [{vi+1}/{len(video_paths)}] {Path(vpath).name}")

            frame_idx = 0
            batch_frames = []

            while not self._stop:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % 3 != 0:  # 每3帧取1帧
                    continue

                batch_frames.append(frame)

                if len(batch_frames) >= 16:
                    self._classify_batch(
                        encoder, model, batch_frames, actions,
                        confidence_threshold, low_confidence_threshold,
                        confident_samples, coach_batch_frames,
                        coach_batch_embeddings, uncertain_frames,
                    )
                    total_frames += len(batch_frames)
                    batch_frames.clear()

                    pct = (vi + frame_idx / max(video_total, 1)) / len(video_paths)
                    self._progress(0.3 + pct * 0.3, f"分析帧 {frame_idx}/{video_total}")

            if batch_frames:
                self._classify_batch(
                    encoder, model, batch_frames, actions,
                    confidence_threshold, low_confidence_threshold,
                    confident_samples, coach_batch_frames,
                    coach_batch_embeddings, uncertain_frames,
                )
                total_frames += len(batch_frames)

            cap.release()

        self._emit_log(
            f"  帧分类: 自信={len(confident_samples)}, "
            f"待教练={len(coach_batch_frames)}, "
            f"不确定={len(uncertain_frames)}"
        )

        # ── 询问 LLM 教练处理中等置信度帧 ──
        if coach and coach_batch_frames and not self._stop:
            self._set_phase(LearningPhase.ASKING_COACH)
            self._emit_log(f"  [教练] 分析 {len(coach_batch_frames)} 个不确定帧...")

            # 分批让 LLM 分析
            batch_size = 8
            for i in range(0, len(coach_batch_frames), batch_size):
                if self._stop:
                    break
                batch = coach_batch_frames[i:i+batch_size]
                batch_emb = coach_batch_embeddings[i:i+batch_size]
                coach_labels = self._ask_coach_label(
                    coach, batch, actions,
                )

                for emb, label in zip(batch_emb, coach_labels):
                    if label and label in actions:
                        coach_samples.append((emb, label, 0.7))
                    else:
                        uncertain_frames.append(emb)

                self._emit_log(
                    f"    教练标注: {i+len(batch)}/{len(coach_batch_frames)}, "
                    f"有效={len(coach_samples)}"
                )

        # 人工审核标记
        human_review = []
        if uncertain_frames:
            human_review.append({
                "type": "uncertain_frames",
                "count": len(uncertain_frames),
                "reason": "模型和 LLM 教练都无法确定这些帧的正确动作",
                "suggestion": "请提供更多相关场景的录制数据",
            })

        # ── 合并数据集训练 ──
        all_samples = confident_samples + coach_samples
        if len(all_samples) < 10:
            self._emit_log("[警告] 有效样本太少，跳过重新训练")
            return {
                "success": False,
                "uncertain_count": len(uncertain_frames),
                "human_review": human_review,
            }

        self._set_phase(LearningPhase.SELF_STUDY)
        new_dataset = E2EDataset()
        new_dataset.set_actions(actions)

        # 混入原始数据
        if dataset_path.exists():
            orig = E2EDataset.load(str(dataset_path))
            for emb, lbl in zip(orig.embeddings, orig.labels):
                action_name = actions[lbl] if lbl < len(actions) else "idle"
                new_dataset.add_sample(emb, action_name)
            self._emit_log(f"  原始数据: {len(orig)} 样本")

        # 添加新样本
        for emb, action_name, conf in all_samples:
            new_dataset.add_sample(emb, action_name)

        self._emit_log(
            f"  合并数据: {len(new_dataset)} 样本 "
            f"(模型标注 {len(confident_samples)} + 教练标注 {len(coach_samples)})"
        )

        # 训练
        new_model_dir = str(run_dir / "model")
        Path(new_model_dir).mkdir(parents=True, exist_ok=True)
        new_dataset.save(str(Path(new_model_dir) / "e2e_dataset.npz"))

        from ..data.e2e_trainer import E2ETrainer
        trainer = E2ETrainer(
            dataset=new_dataset,
            output_dir=new_model_dir,
            epochs=epochs,
            progress_callback=lambda ep, total, loss, tacc, vacc: self._progress(
                0.7 + 0.25 * ep / max(total, 1), f"训练 {ep}/{total}"
            ),
            on_log=self._on_log,
        )

        metrics = trainer.train()
        return {
            "success": not metrics.get("error"),
            "model_dir": new_model_dir,
            "metrics": metrics,
            "confident_count": len(confident_samples),
            "coach_count": len(coach_samples),
            "uncertain_count": len(uncertain_frames),
            "human_review": human_review,
        }

    def _classify_batch(
        self, encoder, model, frames, actions,
        high_threshold, low_threshold,
        confident_out, coach_frames_out, coach_emb_out, uncertain_out,
    ):
        """将一批帧分为三类：自信/待教练/不确定。"""
        import torch

        embeddings = encoder.encode_batch(frames)
        tensors = torch.tensor(np.array(embeddings, dtype=np.float32))

        with torch.no_grad():
            logits = model(tensors)
            probs = torch.softmax(logits, dim=-1)
            confidences, indices = probs.max(dim=-1)

        for frame, emb, conf, idx in zip(frames, embeddings, confidences, indices):
            conf_val = conf.item()
            action_idx = idx.item()

            if conf_val >= high_threshold and action_idx < len(actions):
                # 模型自信 → 直接标注
                confident_out.append((emb, actions[action_idx], conf_val))
            elif conf_val >= low_threshold:
                # 中等置信度 → 请教 LLM 教练
                coach_frames_out.append(frame)
                coach_emb_out.append(emb)
            else:
                # 完全不确定 → 待人工
                uncertain_out.append(emb)

    def _ask_coach_label(
        self, coach, frames, actions,
    ) -> list[str | None]:
        """让 LLM 教练为一批帧标注动作。"""
        import base64
        import cv2

        content = []
        content.append({"type": "text", "text": (
            f"请分析以下 {len(frames)} 个游戏截图，判断每帧应执行什么动作。\n"
            f"可选动作: {', '.join(actions)}\n\n"
            '返回 JSON 数组: [{"frame": 1, "action": "动作名", "confidence": "high/medium/low"}, ...]'
        )})

        for i, frame in enumerate(frames):
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            b64 = base64.b64encode(buf).decode()
            content.append({"type": "text", "text": f"--- 帧 {i+1} ---"})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }})

        try:
            # 使用 pipeline 持有的 provider，避免访问 coach 私有属性
            resp = self._provider.chat(
                messages=[{"role": "user", "content": content}],
                system="你是游戏 AI 教练，为游戏截图标注最合适的动作。只返回 JSON 数组。",
                max_tokens=1024,
            )
            data = self._parse_json_array(resp.text)
            labels = []
            for i in range(len(frames)):
                if i < len(data):
                    action = data[i].get("action", "")
                    conf = data[i].get("confidence", "low")
                    # 只接受教练有信心的标注
                    if action in actions and conf in ("high", "medium"):
                        labels.append(action)
                    else:
                        labels.append(None)
                else:
                    labels.append(None)
            return labels
        except Exception as e:
            self._emit_log(f"    [教练] 标注失败: {e}")
            return [None] * len(frames)

    # ================================================================
    #  教练总结
    # ================================================================

    def _coach_summary(self, provider, result: UnifiedResult) -> dict:
        """让教练总结学习成果并建议下一步。"""
        prompt = (
            "以下是 AI 的学习过程总结:\n\n"
            f"经历阶段: {[p['phase'] for p in result.phase_history]}\n"
            f"最终模型: {result.model_dir or '未产出'}\n"
        )

        for ph in result.phase_history:
            metrics = ph.get("metrics", {})
            if metrics:
                prompt += (
                    f"\n{ph['phase']} 结果:\n"
                    f"  准确率: {metrics.get('best_val_acc', 'N/A')}\n"
                    f"  样本数: {metrics.get('total_samples', 'N/A')}\n"
                )

        if result.skill_gaps:
            prompt += f"\n技能缺口: {', '.join(result.skill_gaps)}\n"
        if result.human_review_items:
            prompt += f"\n待人工处理: {len(result.human_review_items)} 项\n"

        prompt += (
            "\n请总结学习成果，并建议下一步:\n"
            '返回 JSON: {"summary": "总体评价", '
            '"next_steps": ["建议1", ...], '
            '"ready_for_practice": true/false}'
        )

        try:
            resp = provider.chat(
                messages=[{"role": "user", "content": prompt}],
                system="你是游戏 AI 教练。总结学习成果，给出下一步建议。只返回 JSON。",
                max_tokens=1024,
            )
            return self._parse_json(resp.text)
        except Exception:
            return {"summary": "学习流程完成", "next_steps": [], "ready_for_practice": True}

    # ================================================================
    #  工具方法
    # ================================================================

    @staticmethod
    def _parse_json(text: str) -> dict:
        """从 LLM 回复中提取 JSON。"""
        import re
        # 尝试直接解析
        text = text.strip()
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        # 尝试提取 JSON 块
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # 尝试找第一个 { ... }
        m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {}

    @staticmethod
    def _parse_json_array(text: str) -> list[dict]:
        """从 LLM 回复中提取 JSON 数组。"""
        import re
        text = text.strip()
        if text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return []
