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
        # 从 phase_history 提取蒸馏结果
        distill_result = {}
        for ph in self.phase_history:
            if ph.get("phase") == "practice" and ph.get("distilled_samples"):
                distill_result = {
                    "distilled_samples": ph["distilled_samples"],
                    "rl_episodes": ph.get("rl_episodes", 0),
                    "new_val_acc": ph.get("metrics", {}).get("best_val_acc", 0),
                }
        return {
            "run_dir": self.run_dir,
            "phase_history": self.phase_history,
            "model_dir": self.model_dir,
            "metrics": self.metrics,
            "skill_gaps": self.skill_gaps,
            "human_review_items": self.human_review_items,
            "coach_advice": self.coach_advice,
            "success": self.success,
            "distill_result": distill_result,
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
        on_train_step=None,
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
        self._on_train_step = on_train_step
        self._stop = False
        self._provider = None
        self._encoder = None
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

    def _make_train_callback(self, progress_base: float, progress_range: float):
        """创建训练回调，同时更新进度条和图表。"""
        def callback(epoch, total, loss, train_acc, val_acc):
            self._progress(
                progress_base + progress_range * epoch / max(total, 1),
                f"训练 {epoch}/{total}",
            )
            if self._on_train_step:
                try:
                    self._on_train_step(loss, train_acc, val_acc)
                except Exception:
                    pass
        return callback

    def _get_encoder(self):
        if self._encoder is None:
            from ..core.vision_encoder import VisionEncoder
            self._encoder = VisionEncoder()
        return self._encoder

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
        max_improve_rounds: int = 3,
        video_sources: list[str] | None = None,
        max_videos_per_round: int = 5,
        max_video_duration: int = 600,
    ) -> UnifiedResult:
        """一键启动统一学习流程。

        自动检测输入类型并编排：教师教学 → 自主学习 → 自我实践。

        Args:
            max_improve_rounds: 自主改进最大轮数（搜索→下载→标注→训练）。
                设为 0 则跳过自主搜索，只处理用户提供的视频。
            video_sources: 用户配置的视频源 URL 列表。有则优先从这些 URL
                下载，用完后再通过 LLM 搜索补充。
            max_videos_per_round: 每轮最多下载视频数。
            max_video_duration: 单个视频最大时长（秒）。
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
                on_train_step=self._on_train_step,
                provider=provider,  # 复用已创建的 LLM provider
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
        #  Phase 2: 自主改进循环
        # ────────────────────────────────────
        #
        #  Round 0: 处理用户提供的额外视频（plain_videos）
        #  Round 1+: 自动搜索→下载→伪标签→训练→评估→继续/停止
        #
        #  停止条件：
        #    - 精度不再提升（< +0.5%）
        #    - 达到最大轮数 max_improve_rounds
        #    - 无技能缺口
        #    - 搜不到/下不到新视频 → 标记待人工
        #    - 用户手动停止

        prev_acc = 0.0
        if result.phase_history:
            prev_acc = result.phase_history[0].get("metrics", {}).get("best_val_acc", 0)

        # ── Round 0: 处理用户提供的额外视频 ──
        if plain_videos and model_dir and not self._stop:
            self._set_phase(LearningPhase.SELF_STUDY)
            self._emit_log("=" * 50)
            self._emit_log(f"Phase 2 Round 0: 自主学习（{len(plain_videos)} 个用户视频）")

            expand_result = self._self_study(
                model_dir, plain_videos, provider, actions,
                knowledge, confidence_threshold, epochs, run_dir,
            )

            result.phase_history.append({
                "phase": "self_study_round0",
                "success": expand_result.get("success", False),
                "model_dir": expand_result.get("model_dir", ""),
                "metrics": expand_result.get("metrics", {}),
                "uncertain_frames": expand_result.get("uncertain_count", 0),
            })

            if expand_result.get("success") and expand_result.get("model_dir"):
                model_dir = expand_result["model_dir"]
                new_acc = expand_result["metrics"].get("best_val_acc", 0)
                self._emit_log(
                    f"  Round 0 完成: val_acc {prev_acc:.3f} → {new_acc:.3f}"
                )
                prev_acc = new_acc

            if expand_result.get("human_review"):
                result.human_review_items.extend(expand_result["human_review"])

        if self._stop:
            result.model_dir = model_dir
            return result

        # ── Round 1+: 自主搜索→下载→标注→训练 ──
        #
        # 视频获取优先级：
        #   1. 用户配置的 video_sources URL → 直接下载（不需要搜索）
        #   2. video_sources 用完后 → LLM 分析技能缺口 → 搜索 → 下载
        #   3. 搜索也无结果 → 标记待人工

        video_sources = list(video_sources or [])
        has_user_sources = bool(video_sources)
        can_auto_improve = (model_dir and max_improve_rounds > 0
                            and (has_user_sources or (search_online and provider)))

        if can_auto_improve:
            from .video_downloader import download_videos, is_ytdlp_available

            for round_num in range(1, max_improve_rounds + 1):
                if self._stop:
                    break

                self._emit_log("=" * 50)
                self._emit_log(f"Phase 2 Round {round_num}/{max_improve_rounds}: 自主改进")

                download_dir = str(run_dir / f"downloads_round{round_num}")
                downloaded = []

                # ─ 策略 A: 用户配置的视频源（优先） ─
                if video_sources:
                    batch_urls = video_sources[:max_videos_per_round]
                    video_sources = video_sources[max_videos_per_round:]
                    self._emit_log(f"  从配置视频源下载 ({len(batch_urls)} 个 URL)...")

                    if is_ytdlp_available():
                        downloaded = download_videos(
                            batch_urls, download_dir,
                            max_count=max_videos_per_round,
                            max_duration=max_video_duration,
                            on_log=self._on_log,
                        )
                    else:
                        self._emit_log("  [提示] yt-dlp 未安装，请运行: pip install yt-dlp")
                        result.human_review_items.append({
                            "type": "download_videos",
                            "reason": f"Round {round_num}: 请安装 yt-dlp 或手动下载以下视频",
                            "urls": batch_urls,
                        })
                        break

                # ─ 策略 B: LLM 搜索在线视频 ─
                elif search_online and provider:
                    # 技能差距分析
                    self._set_phase(LearningPhase.ASKING_COACH)
                    latest_metrics = (
                        result.phase_history[-1].get("metrics", {})
                        if result.phase_history else {}
                    )
                    skill_gaps = self._analyze_skill_gaps(
                        provider, actions, description, knowledge, latest_metrics,
                    )
                    result.skill_gaps = skill_gaps

                    if not skill_gaps:
                        self._emit_log("  教练认为无明显技能缺口，停止自主改进")
                        break

                    self._emit_log(f"  技能缺口: {', '.join(skill_gaps[:5])}")

                    # 搜索在线视频
                    self._set_phase(LearningPhase.SEARCHING_VIDEOS)
                    suggestions = self._suggest_search_queries(
                        provider, description, skill_gaps,
                    )
                    if not suggestions:
                        self._emit_log("  无搜索建议，停止自主改进")
                        break

                    search_results = self._search_online(provider, suggestions)
                    if not search_results:
                        self._emit_log("  搜索无结果，停止自主改进")
                        break

                    self._emit_log(f"  找到 {len(search_results)} 个相关资源")

                    # 下载
                    urls = [r.get("url", "") for r in search_results]
                    if is_ytdlp_available():
                        downloaded = download_videos(
                            urls, download_dir,
                            max_count=max_videos_per_round,
                            max_duration=max_video_duration,
                            on_log=self._on_log,
                        )
                    else:
                        downloaded = []

                    if not downloaded:
                        self._emit_log("  无法自动下载视频，标记待人工处理")
                        result.human_review_items.append({
                            "type": "download_videos",
                            "reason": f"Round {round_num}: 搜索到视频但无法自动下载，请手动下载后导入",
                            "search_results": search_results[:10],
                        })
                        break
                else:
                    break  # 无视频源也无搜索能力

                if not downloaded:
                    break

                # ─ 伪标签 + 重新训练 ─
                self._set_phase(LearningPhase.SELF_STUDY)
                self._emit_log(f"  自主学习: {len(downloaded)} 个下载视频")

                round_dir = run_dir / f"round{round_num}"
                expand_result = self._self_study(
                    model_dir, downloaded, provider, actions,
                    knowledge, confidence_threshold, epochs, round_dir,
                )

                result.phase_history.append({
                    "phase": f"self_study_round{round_num}",
                    "success": expand_result.get("success", False),
                    "model_dir": expand_result.get("model_dir", ""),
                    "metrics": expand_result.get("metrics", {}),
                    "videos_downloaded": len(downloaded),
                    "uncertain_frames": expand_result.get("uncertain_count", 0),
                })

                if expand_result.get("human_review"):
                    result.human_review_items.extend(expand_result["human_review"])

                if not expand_result.get("success"):
                    self._emit_log(f"  Round {round_num} 训练失败，停止自主改进")
                    break

                # ─ 评估改进幅度 ─
                new_acc = expand_result["metrics"].get("best_val_acc", 0)
                improvement = new_acc - prev_acc
                self._emit_log(
                    f"  Round {round_num} 结果: val_acc {prev_acc:.3f} → {new_acc:.3f} "
                    f"({'+' if improvement >= 0 else ''}{improvement*100:.1f}%)"
                )

                if new_acc > prev_acc:
                    model_dir = expand_result["model_dir"]
                    prev_acc = new_acc

                if improvement < 0.005:
                    self._emit_log("  精度未显著提升，停止自主改进")
                    break

            self._emit_log(f"  自主改进结束，最终 val_acc={prev_acc:.3f}")

        if self._stop:
            result.model_dir = model_dir
            return result

        # ────────────────────────────────────
        #  Phase 3: 自我实践 (RL) + 经验蒸馏回 BC
        # ────────────────────────────────────
        if device_serial and selfplay_preset and selfplay_episodes > 0 and not self._stop:
            self._set_phase(LearningPhase.PRACTICE)
            self._emit_log("=" * 50)
            self._emit_log("Phase 3: 自我实践（自对弈 RL → 经验蒸馏 → BC 重训练）")
            self._emit_log(f"  设备: {device_serial}")
            self._emit_log(f"  预设: {selfplay_preset}")
            self._emit_log(f"  热启动模型: {model_dir or '无（从零开始）'}")
            self._emit_log(f"  对局数: {selfplay_episodes}")

            distill_result = self._rl_and_distill(
                model_dir=model_dir,
                actions=actions,
                device_serial=device_serial,
                selfplay_preset=selfplay_preset,
                selfplay_episodes=selfplay_episodes,
                epochs=epochs,
                run_dir=run_dir,
            )

            result.phase_history.append({
                "phase": "practice",
                "success": distill_result.get("success", False),
                "model_dir": distill_result.get("model_dir", ""),
                "metrics": distill_result.get("metrics", {}),
                "rl_episodes": distill_result.get("rl_episodes", 0),
                "distilled_samples": distill_result.get("distilled_samples", 0),
            })

            if distill_result.get("success") and distill_result.get("model_dir"):
                model_dir = distill_result["model_dir"]
                self._emit_log(f"  RL 蒸馏完成，最终模型: {model_dir}")

        elif device_serial and selfplay_preset:
            # 有设备但未指定对局数 → 记录配置供手动启动
            result.coach_advice["rl_ready"] = {
                "device": device_serial,
                "preset": selfplay_preset,
                "bc_model": model_dir,
                "episodes": selfplay_episodes,
            }
            self._emit_log("─" * 30)
            self._emit_log("[提示] 未指定自对弈对局数，可在 Agent 面板手动启动 RL。")
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
            rl_ready = result.coach_advice.get("rl_ready")
            result.coach_advice = summary
            if rl_ready:
                result.coach_advice["rl_ready"] = rl_ready
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
    #  RL 自对弈 + 经验蒸馏回 BC
    # ================================================================

    def _rl_and_distill(
        self,
        model_dir: str,
        actions: list[str],
        device_serial: str,
        selfplay_preset: str,
        selfplay_episodes: int,
        epochs: int,
        run_dir,
    ) -> dict:
        """执行 RL 自对弈，然后将高奖励经验蒸馏回 BC 模型。

        流程：
          1. 加载预设 → 启动 SelfPlayLoop（BC 热启动）
          2. 自对弈 N 局，收集高奖励 episode 的 (embedding, action)
          3. 停止 RL → 将经验注入 BC 数据集 → 重新训练 E2EMLP
          4. 产出: 融合了 RL 经验的 E2EMLP 模型

        Returns:
            {"success": bool, "model_dir": str, "metrics": dict, ...}
        """
        from ..rl.preset import load_selfplay_preset
        from ..rl.self_play import SelfPlayLoop

        # 加载预设
        try:
            preset = load_selfplay_preset(selfplay_preset)
        except Exception as e:
            self._emit_log(f"  [错误] 加载预设失败: {e}")
            return {"success": False, "error": str(e)}

        action_zones = preset.get("action_zones", [])
        if not action_zones:
            self._emit_log("  [错误] 预设中无动作区域定义")
            return {"success": False, "error": "no_action_zones"}

        rl_output = str(run_dir / "rl")

        # 启动自对弈
        loop = SelfPlayLoop(
            action_zones=action_zones,
            bc_model_dir=model_dir,
            output_dir=rl_output,
            device_serial=device_serial,
            reward_config=preset.get("reward_config"),
            start_model_path=preset.get("start_model_path", "models/start.onnx"),
            max_episodes=selfplay_episodes,
            on_log=self._on_log,
            on_stats=lambda s: self._progress(
                0.85 + 0.1 * min(s.get("episodes", 0) / max(selfplay_episodes, 1), 1),
                f"RL: {s.get('episodes', 0)}/{selfplay_episodes} 局",
            ),
        )

        self._emit_log(f"  [RL] 开始自对弈: {selfplay_episodes} 局...")
        loop.start()

        # 等待完成或用户停止
        while loop.is_running and not self._stop:
            import time as _t
            _t.sleep(2)

        if self._stop:
            loop.stop()
            return {"success": False, "error": "stopped"}

        # 等待线程自然结束
        loop.stop()

        # 获取统计
        stats = loop.stats
        self._emit_log(
            f"  [RL] 自对弈完成: {stats['episodes']} 局, "
            f"均奖={stats['avg_reward_10ep']:.1f}, "
            f"高奖励对局={loop.good_episode_count}"
        )

        # ── 蒸馏：高奖励经验 → BC 数据集 → 重新训练 ──
        experience = loop.get_good_experience()
        if experience is None or len(experience[0]) < 10:
            self._emit_log("  [蒸馏] 高奖励经验不足，跳过 BC 重训练")
            return {
                "success": False,
                "rl_episodes": stats["episodes"],
                "distilled_samples": 0,
            }

        rl_embeddings, rl_actions = experience
        self._emit_log(
            f"  [蒸馏] {len(rl_actions)} 条高奖励经验 → 注入 BC 数据集"
        )

        # 加载原始 BC 数据集
        from ..data.e2e_dataset import E2EDataset

        dataset_path = Path(model_dir) / "e2e_dataset.npz"
        if dataset_path.exists():
            dataset = E2EDataset.load(str(dataset_path))
        else:
            dataset = E2EDataset()
            dataset.set_actions(actions)

        orig_count = len(dataset)

        # 动作名映射（RL 用的是 zone index，需要对应到 action name）
        zone_names = [z["name"] for z in action_zones]

        injected = 0
        for emb, act_idx in zip(rl_embeddings, rl_actions):
            if act_idx < len(zone_names):
                action_name = zone_names[act_idx]
                if action_name in dataset.action_map:
                    dataset.add_sample(emb, action_name)
                    injected += 1

        self._emit_log(
            f"  [蒸馏] 数据集: {orig_count} → {len(dataset)} "
            f"(+{injected} RL 经验)"
        )

        if injected < 5:
            self._emit_log("  [蒸馏] 有效注入样本太少，跳过重训练")
            return {
                "success": False,
                "rl_episodes": stats["episodes"],
                "distilled_samples": injected,
            }

        # 重新训练 E2EMLP
        distill_model_dir = str(run_dir / "distilled_model")
        Path(distill_model_dir).mkdir(parents=True, exist_ok=True)
        dataset.save(str(Path(distill_model_dir) / "e2e_dataset.npz"))

        from ..data.e2e_trainer import E2ETrainer

        self._emit_log(f"  [蒸馏] 重新训练 E2EMLP ({epochs} 轮)...")
        trainer = E2ETrainer(
            dataset=dataset,
            output_dir=distill_model_dir,
            epochs=epochs,
            progress_callback=self._make_train_callback(0.92, 0.06),
            on_log=self._on_log,
        )

        metrics = trainer.train()
        metrics["rl_episodes"] = stats["episodes"]
        metrics["distilled_samples"] = injected
        metrics["orig_samples"] = orig_count

        if metrics.get("error"):
            self._emit_log(f"  [蒸馏] 训练失败: {metrics['error']}")
            return {"success": False, "metrics": metrics}

        self._emit_log(
            f"  [蒸馏] 完成: val_acc={metrics.get('best_val_acc', 0):.3f} "
            f"(含 {injected} 条 RL 经验)"
        )

        return {
            "success": True,
            "model_dir": distill_model_dir,
            "metrics": metrics,
            "rl_episodes": stats["episodes"],
            "distilled_samples": injected,
        }

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

        encoder = self._get_encoder()

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

                if len(batch_frames) >= 32:
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

            # 去重：跳过与已包含帧余弦相似度 >0.95 的重复帧，减少 LLM API 调用
            coach_batch_frames, coach_batch_embeddings = self._deduplicate_frames(
                coach_batch_frames, coach_batch_embeddings
            )

            self._emit_log(f"  [教练] 分析 {len(coach_batch_frames)} 个不确定帧（已去重）...")

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
            progress_callback=self._make_train_callback(0.7, 0.25),
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

    @staticmethod
    def _deduplicate_frames(
        frames: list,
        embeddings: list,
        similarity_threshold: float = 0.95,
    ) -> tuple[list, list]:
        """去除与已包含帧高度相似的重复帧，减少 LLM API 调用。

        对 embeddings 计算余弦相似度，跳过与已选帧相似度 > similarity_threshold 的帧。

        Returns:
            (deduplicated_frames, deduplicated_embeddings)
        """
        if not frames:
            return frames, embeddings

        selected_frames = []
        selected_embeddings = []
        selected_norms = []  # 预计算各已选向量的 L2 范数，避免重复计算

        for frame, emb in zip(frames, embeddings):
            emb_arr = np.array(emb, dtype=np.float32)
            emb_norm = np.linalg.norm(emb_arr)

            # 与所有已选帧比较余弦相似度
            is_duplicate = False
            if selected_embeddings:
                for sel_arr, sel_norm in zip(selected_embeddings, selected_norms):
                    denom = emb_norm * sel_norm
                    if denom < 1e-8:
                        continue
                    sim = float(np.dot(emb_arr, sel_arr) / denom)
                    if sim > similarity_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                selected_frames.append(frame)
                selected_embeddings.append(emb_arr)
                selected_norms.append(emb_norm if emb_norm > 1e-8 else 1.0)

        return selected_frames, selected_embeddings

    def _classify_batch(
        self, encoder, model, frames, actions,
        high_threshold, low_threshold,
        confident_out, coach_frames_out, coach_emb_out, uncertain_out,
    ):
        """将一批帧分为三类：自信/待教练/不确定。"""
        import torch

        embeddings = encoder.encode_batch(frames)
        tensors = torch.from_numpy(embeddings)

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
