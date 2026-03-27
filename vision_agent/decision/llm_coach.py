"""LLM 教练：动作发现、训练诊断、奖励评估。

LLM 不再逐帧标注，而是作为"教练"角色：
  1. 动作发现 — 从录制数据中自动识别有意义的动作集
  2. 训练诊断 — 分析模型弱点，给出改进建议
  3. 奖励评估 — 评估动作序列的合理性，为 RL 提供奖励信号
"""

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class CoachAdvice:
    """教练建议。"""
    weaknesses: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    focus_areas: list[str] = field(default_factory=list)
    overall_assessment: str = ""


class LLMCoach:
    """LLM 教练 — 指导训练而非逐帧标注。

    用法：
        coach = LLMCoach(provider=llm_provider)

        # 从录制数据发现动作集
        actions = coach.discover_actions(recording_dir)

        # 训练后诊断
        advice = coach.diagnose_training(metrics, action_dist)

        # RL 奖励评估
        rewards = coach.evaluate_sequence(frames, actions)
    """

    def __init__(
        self,
        provider: LLMProvider,
        knowledge: str = "",
        on_log=None,
    ):
        self._provider = provider
        self._knowledge = knowledge
        self._on_log = on_log

    def discover_actions(
        self,
        recording_dir: str,
        max_sample_frames: int = 20,
    ) -> dict:
        """从录制数据中自动识别有意义的动作集。

        分析录制的视频帧 + 键盘事件，让 LLM 判断哪些是有意义的游戏动作，
        并为每个按键分配语义名称。

        Returns:
            {"actions": ["attack", "skill_1", ...],
             "action_map": {"q": "attack", "w": "skill_1", ...},  # key→action mapping
             "action_descriptions": {"attack": "普通攻击", ...},
             "scene_type": "MOBA游戏"}
        """
        rec_dir = Path(recording_dir)
        video_path = rec_dir / "recording.mp4"
        actions_path = rec_dir / "actions.jsonl"

        if not video_path.exists() or not actions_path.exists():
            self._log("[教练] 录制文件不完整")
            return {"actions": [], "action_map": {}, "action_descriptions": {}}

        # 统计按键频率
        key_freq: dict[str, int] = {}
        with open(actions_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    key = sample.get("human_action", {}).get("key", "")
                    if key and key != "idle":
                        key_freq[key] = key_freq.get(key, 0) + 1
                except Exception:
                    pass

        if not key_freq:
            return {"actions": ["idle"], "action_map": {}, "action_descriptions": {}}

        # 抽取视频帧
        frames_b64 = self._sample_frames(str(video_path), max_sample_frames)

        # 构建 LLM 请求
        content = []
        content.append({"type": "text", "text": (
            "我录制了一段游戏操作，请分析以下信息：\n\n"
            f"录制的按键及频率：\n"
            + "\n".join(f"  - '{k}': 按了 {v} 次" for k, v in
                        sorted(key_freq.items(), key=lambda x: -x[1]))
            + "\n\n以下是游戏画面的一些截图（按时间顺序）：\n"
        )})

        for i, img_b64 in enumerate(frames_b64):
            content.append({"type": "text", "text": f"--- 截图 {i+1} ---"})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }})

        knowledge_hint = f"\n\n场景知识：{self._knowledge}" if self._knowledge else ""

        system = (
            "你是一个游戏AI教练。分析录制的游戏画面和按键操作，识别有意义的动作。\n"
            "请返回 JSON 格式：\n"
            '{"scene_type": "游戏类型",'
            ' "actions": ["action1", "action2", ...],'
            ' "action_map": {"按键": "动作名", ...},'
            ' "action_descriptions": {"动作名": "描述", ...},'
            ' "analysis": "简短分析"}'
            "\n\n规则：\n"
            "1. 动作名用英文小写（如 attack, skill_1, move_up）\n"
            "2. 忽略明显无关的按键（如截图键）\n"
            "3. 必须包含 idle（无操作）\n"
            "4. 只返回 JSON"
            + knowledge_hint
        )

        try:
            response = self._provider.chat(
                messages=[{"role": "user", "content": content}],
                system=system,
                max_tokens=2048,
            )
            result = self._parse_json(response.text)
            if "actions" in result:
                if "idle" not in result["actions"]:
                    result["actions"].append("idle")
                self._log(f"[教练] 发现 {len(result['actions'])} 个动作: {result['actions']}")
                return result
        except Exception as e:
            self._log(f"[教练] 动作发现失败: {e}")

        # 回退：直接用按键名作为动作
        actions = list(key_freq.keys()) + ["idle"]
        return {
            "actions": actions,
            "action_map": {k: k for k in key_freq},
            "action_descriptions": {},
            "scene_type": "unknown",
        }

    def diagnose_training(
        self,
        metrics: dict,
        action_dist: dict | None = None,
        video_path: str = "",
    ) -> CoachAdvice:
        """分析训练结果，给出改进建议。

        Args:
            metrics: 训练指标 {"best_val_acc": 0.75, "train_loss": [...], ...}
            action_dist: 动作分布 {"attack": 300, "idle": 1000, ...}
            video_path: 可选，用于截帧辅助分析
        """
        prompt = "请分析以下模型训练结果，给出改进建议：\n\n"
        prompt += f"训练指标：\n{json.dumps(metrics, indent=2, ensure_ascii=False)}\n\n"

        if action_dist:
            prompt += f"动作分布：\n{json.dumps(action_dist, indent=2, ensure_ascii=False)}\n\n"

        prompt += (
            "请返回 JSON:\n"
            '{"weaknesses": ["弱点1", ...], '
            '"suggestions": ["建议1", ...], '
            '"focus_areas": ["需要更多数据的场景/动作", ...], '
            '"overall_assessment": "总体评价"}'
        )

        try:
            response = self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                system="你是游戏AI教练，擅长分析训练数据和模型表现。只返回 JSON。",
                max_tokens=2048,
            )
            data = self._parse_json(response.text)
            advice = CoachAdvice(
                weaknesses=data.get("weaknesses", []),
                suggestions=data.get("suggestions", []),
                focus_areas=data.get("focus_areas", []),
                overall_assessment=data.get("overall_assessment", ""),
            )
            self._log(f"[教练] 诊断完成: {advice.overall_assessment[:100]}")
            return advice
        except Exception as e:
            self._log(f"[教练] 诊断失败: {e}")
            return CoachAdvice(overall_assessment=f"诊断失败: {e}")

    def evaluate_sequence(
        self,
        frames: list[np.ndarray],
        actions: list[str],
        action_names: list[str],
    ) -> list[float]:
        """评估一组 帧-动作 对的合理性，返回奖励分数。

        用于 RL 微调阶段。LLM 给出 -1.0 到 1.0 的奖励信号。
        批量评估，一次 LLM 调用处理多帧。

        Returns:
            rewards: 每帧的奖励分数列表
        """
        if not frames:
            return []

        # 限制批量大小
        batch_size = min(len(frames), 10)
        frames = frames[:batch_size]
        actions = actions[:batch_size]

        content = []
        content.append({"type": "text", "text": (
            f"评估以下 {len(frames)} 个游戏画面-动作对的合理性。\n"
            f"可用动作: {', '.join(action_names)}\n"
            "为每个评分 -1.0（很差）到 1.0（很好），0 为中性。\n\n"
            '返回: [{"frame": 1, "score": 0.5, "reason": "..."}, ...]'
        )})

        for i, (frame, action) in enumerate(zip(frames, actions)):
            content.append({"type": "text", "text": f"--- 帧 {i+1}: 动作={action} ---"})
            img_b64 = self._encode_frame(frame)
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }})

        try:
            response = self._provider.chat(
                messages=[{"role": "user", "content": content}],
                system="你是游戏AI教练，评估动作选择的合理性。只返回 JSON 数组。",
                max_tokens=1024,
            )
            scores = self._parse_json_array(response.text)
            rewards = []
            for i in range(len(frames)):
                if i < len(scores):
                    r = float(scores[i].get("score", 0.0))
                    rewards.append(max(-1.0, min(1.0, r)))
                else:
                    rewards.append(0.0)
            return rewards
        except Exception as e:
            self._log(f"[教练] 奖励评估失败: {e}")
            return [0.0] * len(frames)

    # ── 工具方法 ──

    def _sample_frames(self, video_path: str, count: int) -> list[str]:
        """从视频中均匀抽帧，返回 base64 列表。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // count)
        frames_b64 = []

        for i in range(0, total, step):
            if len(frames_b64) >= count:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames_b64.append(self._encode_frame(frame))

        cap.release()
        return frames_b64

    @staticmethod
    def _encode_frame(frame: np.ndarray, max_short_side: int = 512) -> str:
        """编码帧为 base64 JPEG。"""
        h, w = frame.shape[:2]
        short_side = min(h, w)
        if short_side > max_short_side:
            scale = max_short_side / short_side
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf.tobytes()).decode('ascii')

    def _parse_json(self, text: str) -> dict:
        """从 LLM 响应提取 JSON 对象。"""
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        return {}

    def _parse_json_array(self, text: str) -> list[dict]:
        """从 LLM 响应提取 JSON 数组。"""
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        return []

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass
