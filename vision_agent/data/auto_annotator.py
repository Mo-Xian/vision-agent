"""LLM 自动标注器：用大模型离线标注视频，生成决策训练数据。

流程:
  加载视频 → 抽帧 → YOLO检测 → 构建场景描述 → LLM决策 → 保存JSONL

产出与 DataRecorder 格式一致，可直接用于 DecisionTrainer 训练。
"""

import base64
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

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

# 带视觉的系统提示词
VISION_SYSTEM_PROMPT = """\
你是一个游戏AI决策标注员。你将看到游戏画面截图和YOLO检测结果，判断当前应该执行什么动作。

可用动作（你必须从中选一个）:
{action_list}

规则:
1. 结合画面截图和检测结果进行综合判断
2. 从可用动作中选择最合适的一个
3. 如果不确定，选 idle
4. 只返回 JSON，不要返回任何其他文字

{{"action": "从可用动作中选一个", "reason": "简短理由"}}"""

# 重试配置
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # 秒，指数退避基数
_MIN_REQUEST_INTERVAL = 0.5  # 秒，两次请求最小间隔


def _build_action_tool(actions: list[str], descriptions: dict[str, str] | None = None) -> dict:
    """构建 Tool Calling 的工具定义（Claude tools 格式）。

    这样 LLM 就不会自由发挥格式，而是被强制填入结构化参数。
    """
    desc_map = descriptions or {}
    enum_desc = "\n".join(
        f"  - {a}: {desc_map[a]}" if a in desc_map else f"  - {a}"
        for a in actions
    )
    return {
        "name": "decide_action",
        "description": f"根据当前画面的检测结果，选择要执行的动作。\n可选动作:\n{enum_desc}",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": actions,
                    "description": "选择的动作名称",
                },
                "reason": {
                    "type": "string",
                    "description": "简短理由（一句话）",
                },
            },
            "required": ["action"],
        },
    }


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
        send_image: bool = False,
        is_thinking_model: bool = False,
        use_tool_calling: bool = True,
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
            send_image: 是否将帧图像一并发送给 LLM（多模态）
            is_thinking_model: 是否为 thinking/reasoner 模型
            use_tool_calling: 是否使用 Tool Calling 规范化输出（推荐开启）
        """
        self.video_path = video_path
        self.detector = detector
        self.provider = provider
        self.actions = actions
        self.action_descriptions = action_descriptions or {}
        self.sample_interval = sample_interval
        self.max_frames = max_frames
        self.progress_callback = progress_callback
        self.send_image = send_image

        # thinking 模型判断：显式参数 > 模型名推断
        provider_model = getattr(provider, '_model', '') or ''
        self._is_thinking_model = is_thinking_model or (
            'thinking' in provider_model.lower() or 'reasoner' in provider_model.lower()
        )
        if self._is_thinking_model:
            self.max_tokens = max(max_tokens, 4096)
        else:
            self.max_tokens = max_tokens

        # 构建系统提示词
        action_lines = []
        for a in actions:
            desc = self.action_descriptions.get(a, "")
            action_lines.append(f"  - {a}" + (f": {desc}" if desc else ""))
        action_list = "\n".join(action_lines)

        if system_prompt:
            self._system_prompt = system_prompt
        elif send_image:
            self._system_prompt = VISION_SYSTEM_PROMPT.format(action_list=action_list)
        else:
            self._system_prompt = DEFAULT_SYSTEM_PROMPT.format(action_list=action_list)

        # Tool Calling 配置
        self.use_tool_calling = use_tool_calling
        if use_tool_calling:
            self._tool_def = _build_action_tool(actions, self.action_descriptions)
        else:
            self._tool_def = None

        self._state_manager = StateManager()
        self._stop_flag = False
        self._log_callback = None
        self._last_request_time = 0.0

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
        self._stop_flag = False
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.max_frames > 0:
            total_frames = min(total_frames, self.max_frames * self.sample_interval)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        stats = {"total_frames": 0, "sampled": 0, "annotated": 0, "skipped": 0,
                 "errors": 0, "action_dist": {}}
        frame_idx = 0

        logger.info(
            f"自动标注开始 | 视频: {self.video_path} | "
            f"总帧数: {total_frames} | 采样间隔: {self.sample_interval}"
        )

        try:
            with open(save_path, "w", encoding="utf-8") as out_file:
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
                    action_key, action_reason = self._query_llm(state.scene_summary, frame)
                    if action_key is None:
                        stats["errors"] += 1
                        if self.progress_callback:
                            self.progress_callback(frame_idx, total_frames, stats["annotated"])
                        continue

                    # 构建与 DataRecorder 兼容的样本
                    counts = {}
                    for det in result.detections:
                        counts[det.class_name] = counts.get(det.class_name, 0) + 1

                    video_time = frame_idx / fps

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
                            "reason": action_reason,
                        },
                        "action_timestamp": round(video_time, 3),
                    }

                    out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    stats["annotated"] += 1
                    stats["action_dist"][action_key] = stats["action_dist"].get(action_key, 0) + 1

                    if stats["annotated"] % 10 == 0:
                        out_file.flush()

                    if self.progress_callback:
                        self.progress_callback(frame_idx, total_frames, stats["annotated"])
        finally:
            cap.release()

        logger.info(
            f"自动标注完成 | 采样: {stats['sampled']} | "
            f"标注: {stats['annotated']} | 跳过: {stats['skipped']} | "
            f"错误: {stats['errors']}"
        )
        logger.info(f"动作分布: {stats['action_dist']}")
        return stats

    def _encode_frame(self, frame: np.ndarray) -> str:
        """将帧编码为 base64 JPEG 字符串。"""
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf.tobytes()).decode('ascii')

    def _rate_limit(self):
        """简单速率限制：确保两次请求之间有最小间隔。"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _query_llm(self, scene_summary: str, frame: np.ndarray) -> tuple[str | None, str]:
        """向 LLM 查询当前场景的动作，带速率限制和重试。

        Returns:
            (action_key, reason) — action_key 为 None 表示失败
        """
        # 构建消息内容
        if self.send_image:
            img_b64 = self._encode_frame(frame)
            content = [
                {"type": "text", "text": f"当前画面检测结果:\n{scene_summary}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]
        else:
            content = f"当前画面:\n{scene_summary}"

        messages = [{"role": "user", "content": content}]
        tools = [self._tool_def] if self.use_tool_calling else None

        for attempt in range(_MAX_RETRIES):
            self._rate_limit()

            try:
                response = self.provider.chat(
                    messages=messages,
                    system=self._system_prompt,
                    tools=tools,
                    max_tokens=self.max_tokens,
                )

                # ── 优先从 tool_calls 提取（结构化输出，最可靠） ──
                if response.tool_calls:
                    for tc in response.tool_calls:
                        if tc.get("name") == "decide_action":
                            tc_input = tc.get("input", {})
                            action = tc_input.get("action", "")
                            reason = tc_input.get("reason", "")
                            if action and action in self.actions:
                                return action, reason
                            # tool call 返回了非法动作
                            if action:
                                self._log(f"[LLM] tool_call 返回未知动作 '{action}'，已丢弃")
                                return None, ""

                # ── Fallback：从文本响应中解析（兼容不支持 tool calling 的模型） ──
                candidates = []

                # 1. 标准 text
                if response.text:
                    candidates.append(response.text)

                # 2. thinking 模型：检查 raw 中的 content 和 reasoning
                if self._is_thinking_model and response.raw:
                    raw = response.raw
                    if hasattr(raw, 'choices') and raw.choices:
                        msg = raw.choices[0].message
                        if msg.content and msg.content not in candidates:
                            candidates.insert(0, msg.content)
                        reasoning = getattr(msg, 'reasoning_content', None)
                        if not reasoning:
                            extra = getattr(msg, 'model_extra', None) or {}
                            reasoning = extra.get('reasoning_content') or extra.get('reasoning', '')
                        if reasoning and reasoning not in candidates:
                            candidates.append(reasoning)

                if not candidates:
                    self._log("[LLM] 返回完全空响应")
                    return None, ""

                # 逐个候选文本尝试解析
                for text in candidates:
                    action, reason = self._parse_action(text)
                    if action is not None:
                        return action, reason

                # 全部解析失败
                preview = candidates[0][:200] if candidates else "empty"
                self._log(f"[LLM] 解析失败, 候选数={len(candidates)}, 首段: {preview}")
                return None, ""

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = '429' in err_str or 'rate' in err_str or 'too many' in err_str
                is_server_error = any(code in err_str for code in ('500', '502', '503', '504'))

                if (is_rate_limit or is_server_error) and attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    self._log(f"[LLM] {type(e).__name__}, {delay:.0f}s 后重试 ({attempt+1}/{_MAX_RETRIES})")
                    time.sleep(delay)
                    continue

                self._log(f"[LLM 错误] {type(e).__name__}: {e}")
                return None, ""

        return None, ""

    def _parse_action(self, text: str) -> tuple[str | None, str]:
        """从 LLM 响应中解析动作名称和理由。

        Returns:
            (action, reason) — action 为 None 表示解析失败或动作不合法
        """
        action = ""
        reason = ""

        # 尝试解析 JSON
        json_str = text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in json_str:
            json_str = json_str.split("```", 1)[1].split("```", 1)[0]

        try:
            data = json.loads(json_str.strip())
            action = data.get("action", "")
            reason = data.get("reason", "")
        except json.JSONDecodeError:
            # fallback: 在文本中查找 JSON 对象
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                    action = data.get("action", "")
                    reason = data.get("reason", "")
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
            return None, ""

        # 验证动作合法性（精确匹配）
        if action in self.actions:
            return action, reason

        # 模糊匹配（忽略大小写）
        action_lower = action.lower()
        for a in self.actions:
            if a.lower() == action_lower:
                return a, reason

        # 未知动作 → 丢弃，避免污染训练数据
        self._log(f"[LLM] 未知动作 '{action}' 不在预设列表 {self.actions} 中，已丢弃")
        return None, ""
