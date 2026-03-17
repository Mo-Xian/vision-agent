"""LLM 自动标注器：用大模型离线标注视频，生成决策训练数据。

流程:
  加载视频 → 抽帧 → YOLO检测 → 构建场景描述 → LLM决策 → 保存JSONL

产出与 DataRecorder 格式一致，可直接用于 DecisionTrainer 训练。
"""

import json
import logging
import time
from pathlib import Path

from ..core.detector import Detector, DetectionResult
from ..core.state import StateManager
from ..decision.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# 默认系统提示词
DEFAULT_SYSTEM_PROMPT = """\
你是一个游戏AI决策标注员。根据画面中YOLO检测到的目标，判断当前应该执行什么动作。

可用动作（你必须从中选一个）:
{action_list}

规则:
1. 分析检测到的目标类别、数量、位置
2. 从可用动作中选择最合适的一个
3. 如果不确定，选 idle
4. 只返回 JSON，不要返回任何其他文字

{{"action": "从可用动作中选一个", "reason": "简短理由"}}"""


class AutoAnnotator:
    """用 LLM 对视频帧自动标注决策动作。

    用法:
        annotator = AutoAnnotator(
            video_path="test_wzry.mp4",
            detector=Detector(model="runs/.../best.pt"),
            provider=create_provider("claude", api_key, model),
            actions=["attack", "retreat", "skill_1", "idle"],
        )
        annotator.run(save_path="data/recordings/auto_annotated.jsonl")
    """

    def __init__(
        self,
        video_path: str,
        detector: Detector,
        provider: LLMProvider,
        actions: list[str],
        action_descriptions: dict[str, str] | None = None,
        system_prompt: str | None = None,
        sample_interval: int = 10,
        max_frames: int = 0,
        max_tokens: int = 256,
        progress_callback=None,
    ):
        """
        Args:
            video_path: 视频文件路径
            detector: YOLO 检测器
            provider: LLM 供应商实例
            actions: 可用动作名称列表
            action_descriptions: 动作描述 {"attack": "攻击最近的敌人", ...}
            system_prompt: 自定义系统提示词，None 使用默认
            sample_interval: 每 N 帧采样一次（跳帧）
            max_frames: 最大处理帧数，0=不限制
            max_tokens: LLM 最大输出 token
            progress_callback: (current_frame, total_frames, annotated_count) 回调
        """
        self.video_path = video_path
        self.detector = detector
        self.provider = provider
        self.actions = actions
        self.action_descriptions = action_descriptions or {}
        self.sample_interval = sample_interval
        self.max_frames = max_frames
        # thinking 模型需要更多 token（推理过程 + 实际输出）
        provider_model = getattr(provider, '_model', '') or ''
        self._is_thinking_model = (
            'thinking' in provider_model.lower() or 'reasoner' in provider_model.lower()
        )
        if self._is_thinking_model:
            self.max_tokens = max(max_tokens, 4096)
        else:
            self.max_tokens = max_tokens
        self.progress_callback = progress_callback

        # 构建系统提示词
        action_lines = []
        for a in actions:
            desc = self.action_descriptions.get(a, "")
            action_lines.append(f"  - {a}" + (f": {desc}" if desc else ""))
        action_list = "\n".join(action_lines)

        if system_prompt:
            self._system_prompt = system_prompt
        else:
            self._system_prompt = DEFAULT_SYSTEM_PROMPT.format(action_list=action_list)

        self._state_manager = StateManager()
        self._stop_flag = False
        self._log_callback = None

    def set_log_callback(self, callback):
        """设置日志回调，用于 GUI 显示。"""
        self._log_callback = callback

    def _log(self, msg: str):
        logger.info(msg)
        if self._log_callback:
            try:
                self._log_callback(msg)
            except Exception:
                pass

    def stop(self):
        """外部请求停止。"""
        self._stop_flag = True

    def run(self, save_path: str) -> dict:
        """执行自动标注，返回统计信息。

        Args:
            save_path: JSONL 输出路径

        Returns:
            {"total_frames": N, "sampled": N, "annotated": N, "action_dist": {...}}
        """
        import cv2

        self._stop_flag = False
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.max_frames > 0:
            total_frames = min(total_frames, self.max_frames * self.sample_interval)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        out_file = open(save_path, "w", encoding="utf-8")

        stats = {"total_frames": 0, "sampled": 0, "annotated": 0, "skipped": 0,
                 "errors": 0, "action_dist": {}}
        frame_idx = 0

        logger.info(
            f"自动标注开始 | 视频: {self.video_path} | "
            f"总帧数: {total_frames} | 采样间隔: {self.sample_interval}"
        )

        try:
            while not self._stop_flag:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                stats["total_frames"] = frame_idx

                if self.max_frames > 0 and stats["sampled"] >= self.max_frames:
                    break

                # 抽帧
                if frame_idx % self.sample_interval != 0:
                    continue

                stats["sampled"] += 1

                # YOLO 检测
                result = self.detector.detect(frame)
                state = self._state_manager.update(result)

                # 跳过无检测结果的帧
                if not result.detections:
                    stats["skipped"] += 1
                    if self.progress_callback:
                        self.progress_callback(frame_idx, total_frames, stats["annotated"])
                    continue

                # LLM 标注
                action_key = self._query_llm(state.scene_summary)
                if action_key is None:
                    stats["errors"] += 1
                    if self.progress_callback:
                        self.progress_callback(frame_idx, total_frames, stats["annotated"])
                    continue

                # 构建与 DataRecorder 兼容的样本
                counts = {}
                for det in result.detections:
                    counts[det.class_name] = counts.get(det.class_name, 0) + 1

                sample = {
                    "frame_id": result.frame_id,
                    "timestamp": round(result.timestamp, 3),
                    "frame_size": [result.frame_width, result.frame_height],
                    "inference_ms": round(result.inference_ms, 1),
                    "detections": [d.to_dict() for d in result.detections],
                    "object_counts": counts,
                    "human_action": {
                        "type": "llm_annotated",
                        "action": "press",
                        "key": action_key,
                    },
                    "action_timestamp": round(time.time(), 3),
                }

                out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["annotated"] += 1
                stats["action_dist"][action_key] = stats["action_dist"].get(action_key, 0) + 1

                if stats["annotated"] % 10 == 0:
                    out_file.flush()

                if self.progress_callback:
                    self.progress_callback(frame_idx, total_frames, stats["annotated"])

        finally:
            out_file.flush()
            out_file.close()
            cap.release()

        logger.info(
            f"自动标注完成 | 采样: {stats['sampled']} | "
            f"标注: {stats['annotated']} | 跳过: {stats['skipped']} | "
            f"错误: {stats['errors']}"
        )
        logger.info(f"动作分布: {stats['action_dist']}")
        return stats

    def _query_llm(self, scene_summary: str) -> str | None:
        """向 LLM 查询当前场景的动作。"""
        messages = [{"role": "user", "content": f"当前画面:\n{scene_summary}"}]

        try:
            response = self.provider.chat(
                messages=messages,
                system=self._system_prompt,
                tools=None,
                max_tokens=self.max_tokens,
            )

            # 收集所有可能包含 JSON 的文本
            candidates = []

            # 1. 标准 text（经 provider 处理后的，可能来自 content 或 reasoning_content）
            if response.text:
                candidates.append(response.text)

            # 2. 对 thinking 模型，还要检查 raw 中的 content（最终回答）和 reasoning
            if self._is_thinking_model and response.raw:
                raw = response.raw
                if hasattr(raw, 'choices') and raw.choices:
                    msg = raw.choices[0].message
                    # content 是最终回答（可能有 JSON）
                    if msg.content and msg.content not in candidates:
                        candidates.insert(0, msg.content)  # 优先
                    # reasoning_content / model_extra
                    reasoning = getattr(msg, 'reasoning_content', None)
                    if not reasoning:
                        extra = getattr(msg, 'model_extra', None) or {}
                        reasoning = extra.get('reasoning_content') or extra.get('reasoning', '')
                    if reasoning and reasoning not in candidates:
                        candidates.append(reasoning)

            # 3. tool_calls 兜底
            if not candidates and response.tool_calls:
                for tc in response.tool_calls:
                    inp = tc.get("input", {})
                    if "action" in inp:
                        candidates.append(json.dumps(inp, ensure_ascii=False))
                        break

            if not candidates:
                self._log("[LLM] 返回完全空响应")
                return None

            # 逐个候选文本尝试解析
            for text in candidates:
                action = self._parse_action(text)
                if action is not None:
                    return action

            # 全部解析失败
            preview = candidates[0][:200] if candidates else "empty"
            self._log(f"[LLM] 解析失败, 候选数={len(candidates)}, 首段: {preview}")
            return None

        except Exception as e:
            self._log(f"[LLM 错误] {type(e).__name__}: {e}")
            return None

    def _parse_action(self, text: str) -> str | None:
        """从 LLM 响应中解析动作名称。"""
        action = ""

        # 尝试解析 JSON
        json_str = text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in json_str:
            json_str = json_str.split("```", 1)[1].split("```", 1)[0]

        try:
            data = json.loads(json_str.strip())
            action = data.get("action", "")
        except json.JSONDecodeError:
            # fallback: 在文本中查找 JSON 对象
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                    action = data.get("action", "")
                except json.JSONDecodeError:
                    pass

        # 最后手段：在文本中匹配动作关键词
        if not action:
            text_lower = text.lower()
            for a in self.actions:
                if a.lower() in text_lower:
                    action = a
                    break

        if not action:
            return None

        # 验证动作合法性（精确匹配）
        if action in self.actions:
            return action

        # 模糊匹配（忽略大小写）
        action_lower = action.lower()
        for a in self.actions:
            if a.lower() == action_lower:
                return a

        # 未知动作也保留，记录警告
        self._log(f"[LLM] 未知动作 '{action}' 不在预设列表中，仍然保留")
        return action
