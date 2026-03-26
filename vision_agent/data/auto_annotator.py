"""LLM 自动标注器：用大模型离线标注视频，生成决策训练数据。

两种标注策略:
  1. 批量标注（默认）: 一次发 N 帧截图 → LLM 批量返回 N 个决策
     - 效率高：1 次 API 调用标注多帧
     - 上下文连贯：LLM 能看到前后帧的变化趋势
  2. 逐帧标注: 每帧单独调用一次 LLM（兼容旧模式）

YOLO 检测为可选辅助，send_image=True 时 LLM 直接看截图。
"""

import base64
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from ..decision.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# ── 批量标注提示词 ──

BATCH_VISION_PROMPT = """\
你是一个AI决策标注员。你将看到一组连续的游戏画面截图（按时间顺序排列），
请为每一帧判断当前应该执行什么动作。

可用动作（必须从中选）:
{action_list}

规则:
1. 根据每帧画面的实际状态选择动作
2. 注意前后帧的变化趋势（如敌人靠近→应攻击，血量下降→应撤退）
3. 如果不确定，选 idle
4. 严格按以下 JSON 数组格式返回，每帧一个对象，顺序与截图顺序一致:

```json
[
  {{"frame": 1, "action": "动作名", "reason": "简短理由"}},
  {{"frame": 2, "action": "动作名", "reason": "简短理由"}}
]
```

只返回 JSON 数组，不要其他内容。"""

BATCH_VISION_WITH_YOLO_PROMPT = """\
你是一个AI决策标注员。你将看到一组连续的画面截图和可选的检测结果，
请为每一帧判断当前应该执行什么动作。

可用动作（必须从中选）:
{action_list}

规则:
1. 优先根据画面截图判断，检测结果仅作参考
2. 注意前后帧的变化趋势
3. 如果不确定，选 idle
4. 严格按以下 JSON 数组格式返回:

```json
[
  {{"frame": 1, "action": "动作名", "reason": "简短理由"}},
  {{"frame": 2, "action": "动作名", "reason": "简短理由"}}
]
```

只返回 JSON 数组，不要其他内容。"""

# ── 批量标注 Tool Calling ──

def _build_batch_action_tool(actions: list[str], batch_size: int,
                             descriptions: dict[str, str] | None = None) -> dict:
    """构建批量标注的 Tool Calling 工具定义。"""
    desc_map = descriptions or {}
    enum_desc = "\n".join(
        f"  - {a}: {desc_map[a]}" if a in desc_map else f"  - {a}"
        for a in actions
    )
    return {
        "name": "annotate_frames",
        "description": f"为一批连续帧标注决策动作。\n可选动作:\n{enum_desc}",
        "input_schema": {
            "type": "object",
            "properties": {
                "annotations": {
                    "type": "array",
                    "description": f"每帧的标注结果，数量应与输入帧数一致（{batch_size}帧）",
                    "items": {
                        "type": "object",
                        "properties": {
                            "frame": {
                                "type": "integer",
                                "description": "帧序号（从1开始）",
                            },
                            "action": {
                                "type": "string",
                                "enum": actions,
                                "description": "选择的动作",
                            },
                            "reason": {
                                "type": "string",
                                "description": "简短理由",
                            },
                        },
                        "required": ["frame", "action"],
                    },
                },
            },
            "required": ["annotations"],
        },
    }


# ── 单帧标注提示词（兼容旧模式） ──

SINGLE_VISION_PROMPT = """\
你是一个AI决策标注员。你将看到画面截图，判断当前应该执行什么动作。

可用动作（你必须从中选一个）:
{action_list}

规则:
1. 根据画面截图判断场景和状态
2. 从可用动作中选择最合适的一个
3. 如果不确定，选 idle
4. 只返回 JSON，不要返回任何其他文字

{{"action": "从可用动作中选一个", "reason": "简短理由"}}"""

TEXT_SYSTEM_PROMPT = """\
你是一个AI决策标注员。根据画面中检测到的目标，判断当前应该执行什么动作。

可用动作（你必须从中选一个）:
{action_list}

规则:
1. 分析检测到的目标类别、数量、位置
2. 从可用动作中选择最合适的一个
3. 如果不确定，选 idle
4. 只返回 JSON，不要返回任何其他文字

{{"action": "从可用动作中选一个", "reason": "简短理由"}}"""


def _build_single_action_tool(actions: list[str], descriptions: dict[str, str] | None = None) -> dict:
    """构建单帧 Tool Calling 工具定义。"""
    desc_map = descriptions or {}
    enum_desc = "\n".join(
        f"  - {a}: {desc_map[a]}" if a in desc_map else f"  - {a}"
        for a in actions
    )
    return {
        "name": "decide_action",
        "description": f"根据当前画面选择要执行的动作。\n可选动作:\n{enum_desc}",
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


# 重试配置
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0
_MIN_REQUEST_INTERVAL = 0.5


class AutoAnnotator:
    """用 LLM 对视频帧自动标注决策动作。

    推荐使用批量模式（batch_size > 1）：一次 LLM 调用标注多帧，
    效率提升 N 倍，且 LLM 能看到连续帧的变化趋势做出更准确的判断。

    用法:
        # 批量标注（推荐，一次标注 5 帧）
        annotator = AutoAnnotator(
            video_path="video.mp4",
            provider=provider,
            actions=["attack", "retreat", "skill_1", "idle"],
            batch_size=5,
        )
        annotator.run(save_path="data/auto_annotated.jsonl")
    """

    def __init__(
        self,
        video_path: str,
        provider: LLMProvider,
        actions: list[str],
        detector=None,
        action_descriptions: dict[str, str] | None = None,
        system_prompt: str | None = None,
        sample_interval: int = 10,
        max_frames: int = 0,
        max_tokens: int = 2048,
        progress_callback=None,
        send_image: bool = True,
        is_thinking_model: bool = False,
        use_tool_calling: bool = True,
        batch_size: int = 5,
        knowledge: str = "",
        keyframe_indices: list[int] | None = None,
    ):
        """
        Args:
            batch_size: 每次 LLM 调用标注的帧数（>1 为批量模式，1 为逐帧模式）
            knowledge: 场景先验知识（规则/教程），注入系统提示词
            keyframe_indices: 关键帧列表（如提供，则只标注这些帧，忽略 sample_interval）
        """
        self.video_path = video_path
        self.detector = detector
        self.provider = provider
        self.actions = actions
        self.action_descriptions = action_descriptions or {}
        self.sample_interval = sample_interval
        self.max_frames = max_frames
        self._keyframe_set = set(keyframe_indices) if keyframe_indices else None
        self.progress_callback = progress_callback
        self.send_image = send_image
        self.batch_size = max(1, batch_size)
        self._knowledge = knowledge

        # thinking 模型判断
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
        elif self.batch_size > 1 and send_image:
            if detector:
                self._system_prompt = BATCH_VISION_WITH_YOLO_PROMPT.format(action_list=action_list)
            else:
                self._system_prompt = BATCH_VISION_PROMPT.format(action_list=action_list)
        elif send_image:
            self._system_prompt = SINGLE_VISION_PROMPT.format(action_list=action_list)
        else:
            self._system_prompt = TEXT_SYSTEM_PROMPT.format(action_list=action_list)

        # 注入场景先验知识
        if self._knowledge:
            self._system_prompt += (
                "\n\n## 场景知识（请作为决策参考）:\n"
                + self._knowledge
            )

        # Tool Calling 配置
        self.use_tool_calling = use_tool_calling

        self._stop_flag = False
        self._log_callback = None
        self._last_request_time = 0.0

    def set_log_callback(self, callback):
        self._log_callback = callback

    def _log(self, msg: str):
        logger.info(msg)
        if self._log_callback:
            try:
                self._log_callback(msg)
            except Exception:
                pass

    def stop(self):
        self._stop_flag = True

    def run(self, save_path: str) -> dict:
        """执行自动标注，返回统计信息。"""
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

        use_keyframes = self._keyframe_set is not None
        mode = f"batch={self.batch_size}" if self.batch_size > 1 else "single"
        sample_mode = f"关键帧 {len(self._keyframe_set)} 个" if use_keyframes else f"间隔 {self.sample_interval}"
        self._log(
            f"[标注] 开始 | {Path(self.video_path).name} | "
            f"模式: {mode} | 采样: {sample_mode} | send_image: {self.send_image}"
        )

        # 收集待标注帧
        pending_frames = []  # [(frame_idx, frame, result_or_none, scene_summary)]

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

                    # 关键帧模式 vs 均匀采样模式
                    if use_keyframes:
                        if frame_idx not in self._keyframe_set:
                            continue
                    else:
                        if frame_idx % self.sample_interval != 0:
                            continue

                    stats["sampled"] += 1

                    # 可选 YOLO 检测
                    result = None
                    scene_summary = ""
                    if self.detector is not None:
                        result = self.detector.detect(frame)
                        if hasattr(result, 'detections') and result.detections:
                            names = [d.class_name for d in result.detections[:5]]
                            scene_summary = f"检测到: {', '.join(names)}"

                    # 无截图模式下，没有检测结果则跳过
                    if not self.send_image and (result is None or not result.detections):
                        stats["skipped"] += 1
                        if self.progress_callback:
                            self.progress_callback(frame_idx, total_frames, stats["annotated"])
                        continue

                    pending_frames.append((frame_idx, frame.copy(), result, scene_summary))

                    # 积累到 batch_size 则批量标注
                    if len(pending_frames) >= self.batch_size:
                        annotated = self._annotate_batch(pending_frames, fps, out_file, stats)
                        stats["annotated"] += annotated
                        pending_frames.clear()

                        if self.progress_callback:
                            self.progress_callback(frame_idx, total_frames, stats["annotated"])

                # 处理剩余帧
                if pending_frames and not self._stop_flag:
                    annotated = self._annotate_batch(pending_frames, fps, out_file, stats)
                    stats["annotated"] += annotated
                    pending_frames.clear()

        finally:
            cap.release()

        self._log(
            f"[标注] 完成 | 采样: {stats['sampled']} | "
            f"标注: {stats['annotated']} | 跳过: {stats['skipped']} | "
            f"错误: {stats['errors']}"
        )
        self._log(f"[标注] 动作分布: {stats['action_dist']}")
        return stats

    def _annotate_batch(self, frames_data: list, fps: float, out_file, stats: dict) -> int:
        """批量标注一组帧，返回成功标注数。"""
        if self.batch_size > 1 and self.send_image and len(frames_data) > 1:
            return self._annotate_batch_multi(frames_data, fps, out_file, stats)
        else:
            # 逐帧标注
            count = 0
            for frame_idx, frame, result, scene_summary in frames_data:
                action_key, reason = self._query_llm_single(scene_summary, frame)
                if action_key is None:
                    stats["errors"] += 1
                    continue
                self._write_sample(out_file, frame_idx, frame, result, action_key, reason, fps, stats)
                count += 1
            return count

    def _annotate_batch_multi(self, frames_data: list, fps: float, out_file, stats: dict) -> int:
        """一次 LLM 调用标注多帧。"""
        # 构建多帧消息
        content = []
        content.append({"type": "text", "text": f"以下是连续 {len(frames_data)} 帧游戏画面，请为每帧标注动作:"})

        for i, (fidx, frame, result, scene_summary) in enumerate(frames_data):
            ts = fidx / fps
            label = f"--- 第 {i+1} 帧 (@{ts:.1f}s) ---"
            if scene_summary:
                label += f"\n检测参考: {scene_summary}"
            content.append({"type": "text", "text": label})

            img_b64 = self._encode_frame(frame)
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }})

        messages = [{"role": "user", "content": content}]

        # Tool Calling
        tool_def = None
        tools = None
        if self.use_tool_calling:
            tool_def = _build_batch_action_tool(
                self.actions, len(frames_data), self.action_descriptions
            )
            tools = [tool_def]

        # 调用 LLM
        annotations = self._call_llm_batch(messages, tools)

        if not annotations:
            stats["errors"] += len(frames_data)
            return 0

        # 写入标注结果
        count = 0
        for i, (fidx, frame, result, _) in enumerate(frames_data):
            if i < len(annotations):
                ann = annotations[i]
                action_key = ann.get("action", "")
                reason = ann.get("reason", "")
                if action_key and action_key in self.actions:
                    self._write_sample(out_file, fidx, frame, result, action_key, reason, fps, stats)
                    count += 1
                else:
                    stats["errors"] += 1
            else:
                stats["errors"] += 1

        return count

    def _call_llm_batch(self, messages: list, tools: list | None) -> list[dict]:
        """调用 LLM 获取批量标注结果。"""
        for attempt in range(_MAX_RETRIES):
            self._rate_limit()
            try:
                response = self.provider.chat(
                    messages=messages,
                    system=self._system_prompt,
                    tools=tools,
                    max_tokens=self.max_tokens,
                )

                # 从 tool_calls 提取
                if response.tool_calls:
                    for tc in response.tool_calls:
                        if tc.get("name") == "annotate_frames":
                            annotations = tc.get("input", {}).get("annotations", [])
                            if annotations:
                                return annotations

                # 从文本解析 JSON 数组
                if response.text:
                    parsed = self._parse_batch_response(response.text)
                    if parsed:
                        return parsed

                self._log(f"[LLM] 批量标注解析失败")
                return []

            except Exception as e:
                err_str = str(e).lower()
                is_retryable = '429' in err_str or 'rate' in err_str or 'too many' in err_str
                is_retryable = is_retryable or any(c in err_str for c in ('500', '502', '503', '504'))

                if is_retryable and attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    self._log(f"[LLM] {type(e).__name__}, {delay:.0f}s 后重试")
                    time.sleep(delay)
                    continue

                self._log(f"[LLM 错误] {type(e).__name__}: {e}")
                return []

        return []

    def _parse_batch_response(self, text: str) -> list[dict]:
        """从 LLM 文本响应中解析批量标注 JSON 数组。"""
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        text = text.strip()

        # 尝试解析 JSON 数组
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # 尝试找到 [ ... ] 部分
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        # 最后尝试：按行解析多个 JSON 对象
        results = []
        for line in text.split("\n"):
            line = line.strip().rstrip(",")
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    if "action" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    pass
        return results

    def _write_sample(self, out_file, frame_idx: int, frame: np.ndarray,
                      result, action_key: str, reason: str, fps: float, stats: dict):
        """写入一条标注样本到 JSONL。"""
        counts = {}
        detections_data = []
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        inference_ms = 0.0

        if result is not None:
            for det in result.detections:
                counts[det.class_name] = counts.get(det.class_name, 0) + 1
            detections_data = [d.to_dict() for d in result.detections]
            frame_width = result.frame_width
            frame_height = result.frame_height
            inference_ms = result.inference_ms

        video_time = frame_idx / fps

        sample = {
            "frame_id": frame_idx,
            "timestamp": round(video_time, 3),
            "frame_size": [frame_width, frame_height],
            "inference_ms": round(inference_ms, 1),
            "detections": detections_data,
            "object_counts": counts,
            "human_action": {
                "type": "llm_annotated",
                "action": "press",
                "key": action_key,
                "reason": reason,
            },
            "action_timestamp": round(video_time, 3),
        }

        out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
        stats["action_dist"][action_key] = stats["action_dist"].get(action_key, 0) + 1

        if (stats.get("annotated", 0) + 1) % 10 == 0:
            out_file.flush()

    # ── 单帧标注（兼容旧模式） ──

    def _query_llm_single(self, scene_summary: str, frame: np.ndarray) -> tuple[str | None, str]:
        """单帧 LLM 标注。"""
        if self.send_image:
            img_b64 = self._encode_frame(frame)
            parts = []
            if scene_summary:
                parts.append({"type": "text", "text": f"检测参考:\n{scene_summary}"})
            parts.append({"type": "text", "text": "当前画面截图:"})
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
            content = parts
        else:
            content = f"当前画面:\n{scene_summary}"

        messages = [{"role": "user", "content": content}]

        tool_def = _build_single_action_tool(self.actions, self.action_descriptions) if self.use_tool_calling else None
        tools = [tool_def] if tool_def else None

        for attempt in range(_MAX_RETRIES):
            self._rate_limit()
            try:
                response = self.provider.chat(
                    messages=messages,
                    system=self._system_prompt,
                    tools=tools,
                    max_tokens=self.max_tokens,
                )

                if response.tool_calls:
                    for tc in response.tool_calls:
                        if tc.get("name") == "decide_action":
                            tc_input = tc.get("input", {})
                            action = tc_input.get("action", "")
                            reason = tc_input.get("reason", "")
                            if action and action in self.actions:
                                return action, reason
                            if action:
                                return None, ""

                if response.text:
                    action, reason = self._parse_single_action(response.text)
                    if action is not None:
                        return action, reason

                return None, ""

            except Exception as e:
                err_str = str(e).lower()
                is_retryable = '429' in err_str or 'rate' in err_str or any(c in err_str for c in ('500', '502', '503'))
                if is_retryable and attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                    continue
                self._log(f"[LLM 错误] {type(e).__name__}: {e}")
                return None, ""

        return None, ""

    def _parse_single_action(self, text: str) -> tuple[str | None, str]:
        """从单帧 LLM 响应中解析动作。"""
        import re
        action = ""
        reason = ""

        json_str = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if "```json" in json_str:
            json_str = json_str.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in json_str:
            json_str = json_str.split("```", 1)[1].split("```", 1)[0]

        try:
            data = json.loads(json_str.strip())
            action = data.get("action", "")
            reason = data.get("reason", "")
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                    action = data.get("action", "")
                    reason = data.get("reason", "")
                except json.JSONDecodeError:
                    pass

        if not action:
            text_lower = text.lower()
            for a in self.actions:
                if a.lower() in text_lower:
                    action = a
                    break

        if not action:
            return None, ""

        if action in self.actions:
            return action, reason

        action_lower = action.lower()
        for a in self.actions:
            if a.lower() == action_lower:
                return a, reason

        return None, ""

    def _encode_frame(self, frame: np.ndarray, max_short_side: int = 512) -> str:
        """编码帧为 base64 JPEG，自动缩放以控制 token 消耗。"""
        h, w = frame.shape[:2]
        short_side = min(h, w)
        if short_side > max_short_side:
            scale = max_short_side / short_side
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf.tobytes()).decode('ascii')

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()
