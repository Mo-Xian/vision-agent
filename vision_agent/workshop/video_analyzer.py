"""视频分析器：抽帧 + LLM 视觉分析，产出结构化见解。

核心能力：给定视频源，自动分析视频内容、识别场景类型、
建议动作空间、生成学习策略。类似"人先看一遍视频再规划怎么学"。

YOLO 检测为可选辅助信息，LLM 直接看截图是主要分析方式。
"""

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..decision.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# 分析用系统提示词（纯视觉，不依赖 YOLO）
_ANALYZE_PROMPT = """\
你是一个视觉场景分析专家。你将看到从同一个视频中均匀抽取的多帧截图（覆盖视频的不同时间段）。
请综合所有截图分析视频内容，并为 AI Agent 规划学习策略。

分析要求:
- 根据截图判断场景类型（游戏/监控/体育/其他）
- 如果是游戏，识别具体类型（MOBA、FPS、格斗、赛车、RTS 等）和具体游戏名
- 识别画面中的关键元素（UI、角色、地图、技能栏、血条、小地图等）
- 注意不同截图可能展示了不同阶段（如对线期、团战、打野、回城等），应全面覆盖
- 根据场景设计完整的动作空间，覆盖所有阶段可能的操作，使用英文命名
- 动作空间应细致实用，不要过于笼统

请严格按以下 JSON 格式回复（只返回 JSON，不要其他内容）：
```json
{
  "scene_type": "场景类型（如：MOBA游戏-王者荣耀、FPS射击-CS2 等，尽量具体）",
  "scene_description": "对视频内容的描述，包括观察到的不同阶段",
  "detected_objects": ["画面中可见的主要元素，如：英雄角色、小兵、防御塔、技能按钮、小地图、血条"],
  "suggested_actions": ["完整的动作空间，覆盖所有观察到的游戏阶段"],
  "action_descriptions": {"动作名": "动作描述和适用场景"},
  "scene_keywords": ["用于场景识别的关键词"],
  "analysis_summary": "综合分析：各截图展示的不同阶段、场景特点、关键元素、建议的学习策略",
  "difficulty": "easy/medium/hard",
  "recommended_model": "推荐的 YOLO 模型（yolov8n.pt/yolov8s.pt/yolov8m.pt），如果不需要物体检测可填 none"
}
```"""

# 带 YOLO 辅助的分析提示词
_ANALYZE_PROMPT_WITH_YOLO = """\
你是一个视觉场景分析专家。你将看到从同一个视频中均匀抽取的多帧截图（覆盖不同时间段），以及可选的 YOLO 检测结果。
请综合所有截图分析视频内容，并为 AI Agent 规划学习策略。

注意: YOLO 使用的是通用预训练模型，检测类别可能不准确（如把游戏 UI 识别为 clock），请以截图为主要判断依据。

分析要求:
- 综合所有截图判断场景类型，注意不同截图可能展示不同阶段
- 设计覆盖所有阶段的完整动作空间

请严格按以下 JSON 格式回复（只返回 JSON，不要其他内容）：
```json
{
  "scene_type": "场景类型（如：MOBA游戏-王者荣耀、FPS射击-CS2 等，尽量具体）",
  "scene_description": "对视频内容的描述，包括观察到的不同阶段",
  "detected_objects": ["画面中实际可见的主要元素（不限于YOLO检测结果）"],
  "suggested_actions": ["完整的动作空间，覆盖所有观察到的游戏阶段"],
  "action_descriptions": {"动作名": "动作描述和适用场景"},
  "scene_keywords": ["用于场景识别的关键词"],
  "analysis_summary": "综合分析：各截图展示的不同阶段、场景特点、关键元素、建议的学习策略",
  "difficulty": "easy/medium/hard",
  "recommended_model": "推荐的 YOLO 模型（yolov8n.pt/yolov8s.pt/yolov8m.pt），如果不需要可填 none"
}
```"""


@dataclass
class VideoInsight:
    """视频分析产出的结构化见解。"""
    source_path: str = ""
    scene_type: str = ""
    scene_description: str = ""
    detected_objects: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)
    action_descriptions: dict[str, str] = field(default_factory=dict)
    scene_keywords: list[str] = field(default_factory=list)
    analysis_summary: str = ""
    difficulty: str = "medium"
    recommended_model: str = "yolov8n.pt"
    frame_samples: list[dict] = field(default_factory=list)
    total_frames: int = 0
    fps: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "scene_type": self.scene_type,
            "scene_description": self.scene_description,
            "detected_objects": self.detected_objects,
            "suggested_actions": self.suggested_actions,
            "action_descriptions": self.action_descriptions,
            "scene_keywords": self.scene_keywords,
            "analysis_summary": self.analysis_summary,
            "difficulty": self.difficulty,
            "recommended_model": self.recommended_model,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration_seconds": round(self.duration_seconds, 1),
        }


class VideoAnalyzer:
    """视频分析器：抽帧 → LLM 视觉分析 → VideoInsight。

    默认直接把截图发给 LLM 分析，不依赖 YOLO。
    可选传入 detector 提供辅助检测信息。

    用法:
        # 纯视觉分析（推荐）
        analyzer = VideoAnalyzer(provider=provider)
        insight = analyzer.analyze("path/to/video.mp4")

        # 带 YOLO 辅助
        analyzer = VideoAnalyzer(provider=provider, detector=detector)
        insight = analyzer.analyze("path/to/video.mp4")
    """

    def __init__(
        self,
        provider: LLMProvider,
        detector=None,
        sample_count: int = 20,
        max_image_count: int = 10,
        max_tokens: int = 2048,
        knowledge: str = "",
        on_log=None,
    ):
        self._provider = provider
        self._detector = detector  # 可选
        self._sample_count = sample_count
        self._max_image_count = max_image_count
        self._max_tokens = max_tokens
        self._knowledge = knowledge
        self._on_log = on_log
        self._stop = False

    def stop(self):
        self._stop = True

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass

    def analyze(self, video_path: str, progress_callback=None) -> VideoInsight:
        """分析单个视频文件，返回结构化见解。"""
        self._stop = False
        insight = VideoInsight(source_path=video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        insight.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        insight.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        insight.duration_seconds = insight.total_frames / insight.fps if insight.fps > 0 else 0

        self._log(f"[分析] 视频: {Path(video_path).name} ({insight.duration_seconds:.0f}s, {insight.total_frames} 帧)")

        # 均匀抽帧
        sample_interval = max(1, insight.total_frames // self._sample_count)
        samples = []
        all_classes = set()
        frame_idx = 0

        while not self._stop:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % sample_interval != 0:
                continue
            if len(samples) >= self._sample_count:
                break

            sample_info = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / insight.fps,
            }

            # 可选 YOLO 检测
            if self._detector is not None:
                result = self._detector.detect(frame)
                sample_info["detection_count"] = len(result.detections)
                sample_info["classes"] = {}
                for det in result.detections:
                    all_classes.add(det.class_name)
                    sample_info["classes"][det.class_name] = sample_info["classes"].get(det.class_name, 0) + 1

            # 编码帧图像供 LLM 视觉分析（缩放以控制 token）
            h, w = frame.shape[:2]
            short_side = min(h, w)
            enc_frame = frame
            if short_side > 512:
                scale = 512 / short_side
                enc_frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode('.jpg', enc_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            sample_info["image_b64"] = base64.b64encode(buf.tobytes()).decode('ascii')

            samples.append(sample_info)

            if progress_callback:
                progress_callback(len(samples), self._sample_count)

        cap.release()
        insight.frame_samples = [{k: v for k, v in s.items() if k != "image_b64"} for s in samples]
        insight.detected_objects = sorted(all_classes)

        if not samples:
            self._log("[分析] 未能采样到有效帧")
            return insight

        # 构建 LLM 分析请求
        llm_insight = self._query_llm(samples, all_classes)
        if llm_insight:
            insight.scene_type = llm_insight.get("scene_type", "")
            insight.scene_description = llm_insight.get("scene_description", "")
            insight.suggested_actions = llm_insight.get("suggested_actions", [])
            insight.action_descriptions = llm_insight.get("action_descriptions", {})
            insight.scene_keywords = llm_insight.get("scene_keywords", [])
            insight.analysis_summary = llm_insight.get("analysis_summary", "")
            insight.difficulty = llm_insight.get("difficulty", "medium")
            insight.recommended_model = llm_insight.get("recommended_model", "yolov8n.pt")
            # LLM 识别到的元素覆盖 YOLO 检测结果
            if llm_insight.get("detected_objects"):
                insight.detected_objects = llm_insight["detected_objects"]

        # 确保有 idle 动作
        if insight.suggested_actions and "idle" not in insight.suggested_actions:
            insight.suggested_actions.append("idle")

        self._log(f"[分析] 完成: 场景={insight.scene_type}, 动作={insight.suggested_actions}")
        return insight

    def analyze_batch(self, video_paths: list[str], progress_callback=None) -> list[VideoInsight]:
        """批量分析多个视频文件。"""
        insights = []
        for i, path in enumerate(video_paths):
            if self._stop:
                break
            self._log(f"[分析] 批量 [{i+1}/{len(video_paths)}]: {Path(path).name}")
            try:
                insight = self.analyze(path)
                insights.append(insight)
            except Exception as e:
                self._log(f"[分析] 跳过 {Path(path).name}: {e}")
            if progress_callback:
                progress_callback(i + 1, len(video_paths))
        return insights

    def merge_insights(self, insights: list[VideoInsight]) -> VideoInsight:
        """合并多个视频的分析结果为统一见解。"""
        if not insights:
            return VideoInsight()
        if len(insights) == 1:
            return insights[0]

        merged = VideoInsight()
        merged.scene_type = insights[0].scene_type

        all_objects = set()
        all_actions = []
        all_descriptions = {}
        all_keywords = set()

        for ins in insights:
            all_objects.update(ins.detected_objects)
            for a in ins.suggested_actions:
                if a not in all_actions:
                    all_actions.append(a)
            all_descriptions.update(ins.action_descriptions)
            all_keywords.update(ins.scene_keywords)

        merged.detected_objects = sorted(all_objects)
        merged.suggested_actions = all_actions
        merged.action_descriptions = all_descriptions
        merged.scene_keywords = sorted(all_keywords)
        merged.analysis_summary = "\n".join(
            f"[{Path(ins.source_path).name}] {ins.analysis_summary}" for ins in insights
        )
        return merged

    def _query_llm(self, samples: list[dict], all_classes: set) -> dict | None:
        """用 LLM 分析采样帧截图。"""
        has_yolo = any("classes" in s for s in samples)

        # 均匀选取截图
        image_samples = [s for s in samples if "image_b64" in s]
        if not image_samples:
            self._log("[分析] 无有效截图可供分析")
            return None

        count = min(self._max_image_count, len(image_samples))
        step = max(1, len(image_samples) // count)
        selected = [image_samples[i * step] for i in range(count)
                    if i * step < len(image_samples)]

        # 构建消息内容：截图为主
        content = []

        if has_yolo:
            # 有 YOLO 辅助信息时附上统计
            stats_lines = []
            for s in samples:
                ts = s.get("timestamp", 0)
                classes = s.get("classes", {})
                if classes:
                    cls_str = ", ".join(f"{k}x{v}" for k, v in sorted(classes.items(), key=lambda x: -x[1]))
                    stats_lines.append(f"  @{ts:.1f}s: [{cls_str}]")
            if stats_lines:
                content.append({"type": "text", "text": (
                    f"YOLO 通用模型检测参考（可能不准确）:\n"
                    f"检测到的类别: {sorted(all_classes)}\n"
                    + "\n".join(stats_lines)
                )})

        content.append({"type": "text", "text": f"以下是视频中均匀抽取的 {len(selected)} 帧截图，请分析:"})

        for i, s in enumerate(selected):
            content.append({"type": "text", "text": f"--- 截图 {i+1}/{len(selected)} (@{s.get('timestamp', 0):.1f}s) ---"})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{s['image_b64']}"
            }})

        messages = [{"role": "user", "content": content}]
        system_prompt = _ANALYZE_PROMPT_WITH_YOLO if has_yolo else _ANALYZE_PROMPT

        # 注入场景先验知识
        if self._knowledge:
            system_prompt += (
                "\n\n## 用户提供的场景知识（请作为重要参考）:\n"
                + self._knowledge
            )

        try:
            response = self._provider.chat(
                messages=messages,
                system=system_prompt,
                max_tokens=self._max_tokens,
            )
            if response.text:
                return self._parse_json(response.text)
        except Exception as e:
            self._log(f"[分析] LLM 调用失败: {e}")
        return None

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """从 LLM 响应中解析 JSON。"""
        import re
        # 去除 thinking 标签（MiniMax-M2.7 等模型会输出 <think>...</think>）
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        return None
