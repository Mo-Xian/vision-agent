"""测试 AutoAnnotator 核心逻辑：YOLO 检测 → 场景描述 → LLM 决策 → 动作解析。

不依赖真实的 YOLO 模型和 LLM API，全部用 mock 替代。
"""

import json
import sys
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_agent.core.detector import Detection, DetectionResult
from vision_agent.core.state import StateManager
from vision_agent.decision.llm_provider import LLMResponse, LLMProvider


# ── 辅助函数 ──

def make_detections(specs: list[tuple]) -> list[Detection]:
    """快速构造 Detection 列表。
    specs: [(class_name, confidence, cx_norm, cy_norm), ...]
    """
    dets = []
    for i, (name, conf, cx, cy) in enumerate(specs):
        w_half, h_half = 0.05, 0.05
        bbox_norm = (cx - w_half, cy - h_half, cx + w_half, cy + h_half)
        bbox = tuple(v * 1920 if j % 2 == 0 else v * 1080 for j, v in enumerate(bbox_norm))
        dets.append(Detection(
            class_id=i, class_name=name, confidence=conf,
            bbox=bbox, bbox_norm=bbox_norm,
        ))
    return dets


def make_result(detections: list[Detection], frame_id: int = 1) -> DetectionResult:
    return DetectionResult(
        detections=detections, frame_id=frame_id,
        timestamp=0.033 * frame_id, inference_ms=12.5,
        frame_width=1920, frame_height=1080,
    )


class MockProvider(LLMProvider):
    """可配置响应的 mock LLM Provider。

    responses 支持三种类型：
    - str: 模拟纯文本响应（text=str, tool_calls=[]）
    - Exception: 模拟 API 异常
    - LLMResponse: 直接返回（用于模拟 tool_calls 场景）
    """

    def __init__(self, responses: list):
        self._responses = list(responses)
        self._call_count = 0
        self._model = "mock-model"
        self.last_messages = None
        self.last_system = None
        self.last_tools = None

    @property
    def provider_name(self):
        return "mock"

    def chat(self, messages, system="", tools=None, max_tokens=1024):
        self.last_messages = messages
        self.last_system = system
        self.last_tools = tools
        self._call_count += 1

        if not self._responses:
            return LLMResponse(text="", tool_calls=[], raw=None)

        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, LLMResponse):
            return resp
        return LLMResponse(text=resp, tool_calls=[], raw=None)

    def test_connection(self):
        return True


ACTIONS = ["attack", "retreat", "skill_1", "idle"]


def make_annotator(provider, actions=None, send_image=False):
    """构造一个 AutoAnnotator，mock 掉 detector（不会真正用到）。"""
    from vision_agent.data.auto_annotator import AutoAnnotator

    mock_detector = MagicMock()
    annotator = AutoAnnotator(
        video_path="fake.mp4",
        detector=mock_detector,
        provider=provider,
        actions=actions or ACTIONS,
        sample_interval=1,
        send_image=send_image,
    )
    # 禁用速率限制以加快测试
    annotator._last_request_time = 0
    return annotator


# ── 测试用例 ──

class TestParseAction(unittest.TestCase):
    """测试 _parse_action：从 LLM 文本响应中提取动作。"""

    def setUp(self):
        self.provider = MockProvider([])
        self.annotator = make_annotator(self.provider)

    def test_standard_json(self):
        """标准 JSON 响应。"""
        text = '{"action": "attack", "reason": "敌人在近处"}'
        self.assertEqual(self.annotator._parse_action(text), "attack")

    def test_json_in_code_block(self):
        """```json 代码块包裹的响应。"""
        text = '```json\n{"action": "retreat", "reason": "血量低"}\n```'
        self.assertEqual(self.annotator._parse_action(text), "retreat")

    def test_json_in_generic_code_block(self):
        """``` 代码块包裹（无 json 标记）。"""
        text = '```\n{"action": "skill_1"}\n```'
        self.assertEqual(self.annotator._parse_action(text), "skill_1")

    def test_json_embedded_in_text(self):
        """JSON 嵌在多余文字中间。"""
        text = '根据分析，我认为应该执行 {"action": "idle", "reason": "无敌人"} 这个动作。'
        self.assertEqual(self.annotator._parse_action(text), "idle")

    def test_case_insensitive_match(self):
        """大小写不敏感匹配。"""
        text = '{"action": "Attack", "reason": "test"}'
        self.assertEqual(self.annotator._parse_action(text), "attack")

    def test_keyword_fallback(self):
        """无法解析 JSON 时，从文本中匹配关键词。"""
        text = "当前应该执行 retreat 动作来保命"
        self.assertEqual(self.annotator._parse_action(text), "retreat")

    def test_unknown_action_rejected(self):
        """未知动作应被丢弃。"""
        text = '{"action": "dance", "reason": "for fun"}'
        self.assertIsNone(self.annotator._parse_action(text))

    def test_empty_text(self):
        """空文本返回 None。"""
        self.assertIsNone(self.annotator._parse_action(""))

    def test_no_action_field(self):
        """JSON 中无 action 字段。"""
        text = '{"reasoning": "不知道该做什么"}'
        self.assertIsNone(self.annotator._parse_action(text))


class TestQueryLLM(unittest.TestCase):
    """测试 _query_llm：构建消息 → 调用 provider → 解析响应。"""

    def _make_frame(self):
        """构造一个假的 numpy 图像帧。"""
        import numpy as np
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_text_only_mode(self):
        """纯文本模式：content 应为字符串。"""
        provider = MockProvider(['{"action": "attack", "reason": "敌人近"}'])
        annotator = make_annotator(provider, send_image=False)

        result = annotator._query_llm("检测到 enemy×2", self._make_frame())

        self.assertEqual(result, "attack")
        # 验证发给 LLM 的消息是纯字符串
        msg = provider.last_messages[0]
        self.assertIsInstance(msg["content"], str)
        self.assertIn("enemy×2", msg["content"])

    def test_image_mode(self):
        """多模态模式：content 应包含 text 和 image_url。"""
        provider = MockProvider(['{"action": "retreat", "reason": "血少"}'])
        annotator = make_annotator(provider, send_image=True)

        result = annotator._query_llm("检测到 hero×1", self._make_frame())

        self.assertEqual(result, "retreat")
        # 验证消息结构
        msg = provider.last_messages[0]
        self.assertIsInstance(msg["content"], list)
        types = [item["type"] for item in msg["content"]]
        self.assertIn("text", types)
        self.assertIn("image_url", types)
        # 验证图片是 base64
        img_item = [item for item in msg["content"] if item["type"] == "image_url"][0]
        self.assertTrue(img_item["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_system_prompt_contains_actions(self):
        """系统提示词应包含所有可用动作。"""
        provider = MockProvider(['{"action": "idle"}'])
        annotator = make_annotator(provider)

        annotator._query_llm("test", self._make_frame())

        for action in ACTIONS:
            self.assertIn(action, provider.last_system)

    def test_empty_response(self):
        """LLM 返回空响应时应返回 None。"""
        provider = MockProvider([""])
        annotator = make_annotator(provider)

        result = annotator._query_llm("test", self._make_frame())
        self.assertIsNone(result)

    def test_unparseable_response(self):
        """LLM 返回无法解析的内容时应返回 None。"""
        provider = MockProvider(["我不知道该怎么办，没有任何线索。"])
        annotator = make_annotator(provider)

        result = annotator._query_llm("test", self._make_frame())
        self.assertIsNone(result)

    def test_retry_on_rate_limit(self):
        """遇到 429 时应重试，最终成功。"""
        provider = MockProvider([
            Exception("Error 429: rate limit exceeded"),
            '{"action": "attack"}',
        ])
        annotator = make_annotator(provider)

        with patch("vision_agent.data.auto_annotator.time.sleep"):
            result = annotator._query_llm("test", self._make_frame())

        self.assertEqual(result, "attack")
        self.assertEqual(provider._call_count, 2)

    def test_retry_on_server_error(self):
        """遇到 500 时应重试。"""
        provider = MockProvider([
            Exception("Server error 502"),
            '{"action": "idle"}',
        ])
        annotator = make_annotator(provider)

        with patch("vision_agent.data.auto_annotator.time.sleep"):
            result = annotator._query_llm("test", self._make_frame())

        self.assertEqual(result, "idle")

    def test_no_retry_on_auth_error(self):
        """非限流/服务器错误（如认证失败）不重试。"""
        provider = MockProvider([
            Exception("401 Unauthorized"),
        ])
        annotator = make_annotator(provider)

        with patch("vision_agent.data.auto_annotator.time.sleep"):
            result = annotator._query_llm("test", self._make_frame())

        self.assertIsNone(result)
        self.assertEqual(provider._call_count, 1)

    def test_all_retries_exhausted(self):
        """重试全部耗尽后返回 None。"""
        provider = MockProvider([
            Exception("429 too many requests"),
            Exception("429 too many requests"),
            Exception("429 too many requests"),
        ])
        annotator = make_annotator(provider)

        with patch("vision_agent.data.auto_annotator.time.sleep"):
            result = annotator._query_llm("test", self._make_frame())

        self.assertIsNone(result)
        self.assertEqual(provider._call_count, 3)


class TestToolCalling(unittest.TestCase):
    """测试 Tool Calling 模式：LLM 通过结构化工具调用返回结果。"""

    def _make_frame(self):
        import numpy as np
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_tool_call_success(self):
        """LLM 正确调用 decide_action 工具时，直接提取 action。"""
        tool_response = LLMResponse(
            text="",
            tool_calls=[{
                "id": "tc_1",
                "name": "decide_action",
                "input": {"action": "attack", "reason": "enemy nearby"},
            }],
            raw=None,
        )
        provider = MockProvider([tool_response])
        annotator = make_annotator(provider)

        result = annotator._query_llm("检测到 enemy×1", self._make_frame())

        self.assertEqual(result, "attack")
        # 验证 tools 参数被传给了 provider
        self.assertIsNotNone(provider.last_tools)
        self.assertEqual(len(provider.last_tools), 1)
        self.assertEqual(provider.last_tools[0]["name"], "decide_action")

    def test_tool_call_with_enum(self):
        """tool 定义中的 enum 应包含所有预设动作。"""
        from vision_agent.data.auto_annotator import _build_action_tool

        tool = _build_action_tool(ACTIONS)
        enum_values = tool["input_schema"]["properties"]["action"]["enum"]

        self.assertEqual(enum_values, ACTIONS)
        self.assertIn("action", tool["input_schema"]["required"])

    def test_tool_call_unknown_action_rejected(self):
        """tool_call 返回不在列表中的动作时应丢弃。"""
        tool_response = LLMResponse(
            text="",
            tool_calls=[{
                "id": "tc_1",
                "name": "decide_action",
                "input": {"action": "fly", "reason": "want to fly"},
            }],
            raw=None,
        )
        provider = MockProvider([tool_response])
        annotator = make_annotator(provider)

        result = annotator._query_llm("test", self._make_frame())
        self.assertIsNone(result)

    def test_fallback_to_text_when_no_tool_call(self):
        """模型未调用工具时，应 fallback 到文本解析。"""
        # 有些模型可能忽略 tools 直接返回文本
        provider = MockProvider(['{"action": "idle", "reason": "no threats"}'])
        annotator = make_annotator(provider)

        result = annotator._query_llm("test", self._make_frame())
        self.assertEqual(result, "idle")

    def test_tool_calling_disabled(self):
        """use_tool_calling=False 时，不传 tools 参数。"""
        provider = MockProvider(['{"action": "retreat"}'])
        from vision_agent.data.auto_annotator import AutoAnnotator

        mock_detector = MagicMock()
        annotator = AutoAnnotator(
            video_path="fake.mp4",
            detector=mock_detector,
            provider=provider,
            actions=ACTIONS,
            sample_interval=1,
            use_tool_calling=False,
        )
        annotator._last_request_time = 0

        result = annotator._query_llm("test", self._make_frame())

        self.assertEqual(result, "retreat")
        self.assertIsNone(provider.last_tools)

    def test_tool_with_descriptions(self):
        """带描述的 tool 定义应在 description 中包含动作描述。"""
        from vision_agent.data.auto_annotator import _build_action_tool

        desc = {"attack": "攻击最近的敌人", "idle": "原地待命"}
        tool = _build_action_tool(["attack", "idle"], desc)

        self.assertIn("攻击最近的敌人", tool["description"])
        self.assertIn("原地待命", tool["description"])

    def test_tool_call_prioritized_over_text(self):
        """同时有 tool_call 和 text 时，优先使用 tool_call。"""
        tool_response = LLMResponse(
            text='{"action": "retreat"}',  # 文本里说 retreat
            tool_calls=[{
                "id": "tc_1",
                "name": "decide_action",
                "input": {"action": "attack"},  # tool_call 里说 attack
            }],
            raw=None,
        )
        provider = MockProvider([tool_response])
        annotator = make_annotator(provider)

        result = annotator._query_llm("test", self._make_frame())
        # 应优先返回 tool_call 的结果
        self.assertEqual(result, "attack")


class TestSceneSummary(unittest.TestCase):
    """测试 StateManager 生成的 scene_summary 是否包含检测信息。"""

    def test_summary_with_detections(self):
        dets = make_detections([
            ("enemy_hero", 0.92, 0.5, 0.5),
            ("minion", 0.85, 0.2, 0.8),
            ("minion", 0.78, 0.3, 0.8),
        ])
        result = make_result(dets)
        state = StateManager().update(result)

        self.assertIn("3", state.scene_summary)        # 3 个目标
        self.assertIn("enemy_hero", state.scene_summary)
        self.assertIn("minion", state.scene_summary)
        self.assertIn("minion×2", state.scene_summary)  # 数量

    def test_summary_empty(self):
        result = make_result([])
        state = StateManager().update(result)
        self.assertIn("未检测到", state.scene_summary)


class TestEndToEnd(unittest.TestCase):
    """端到端测试：模拟视频 → YOLO 检测 → LLM 标注 → JSONL 输出。"""

    def test_full_pipeline(self):
        """模拟 3 帧视频（sample_interval=1），验证输出 JSONL。"""
        import numpy as np
        from vision_agent.data.auto_annotator import AutoAnnotator

        # 准备 mock 数据
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]

        # 第 1 帧：有检测，LLM 返回 attack
        det1 = make_detections([("enemy", 0.9, 0.5, 0.5)])
        res1 = make_result(det1, frame_id=1)

        # 第 2 帧：无检测（应跳过）
        res2 = make_result([], frame_id=2)

        # 第 3 帧：有检测，LLM 返回 idle
        det3 = make_detections([("tower", 0.8, 0.7, 0.3)])
        res3 = make_result(det3, frame_id=3)

        # Mock detector
        mock_detector = MagicMock()
        mock_detector.detect = MagicMock(side_effect=[res1, res2, res3])

        # Mock provider
        provider = MockProvider([
            '{"action": "attack", "reason": "enemy nearby"}',
            '{"action": "idle", "reason": "safe zone"}',
        ])

        # Mock video capture
        frame_iter = iter(frames)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 3.0  # 3 帧 / 30fps
        mock_cap.__enter__ = MagicMock(return_value=mock_cap)
        mock_cap.__exit__ = MagicMock(return_value=False)

        def mock_read():
            try:
                return True, next(frame_iter)
            except StopIteration:
                return False, None

        mock_cap.read = mock_read

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_output.jsonl")

            annotator = AutoAnnotator(
                video_path="fake.mp4",
                detector=mock_detector,
                provider=provider,
                actions=ACTIONS,
                sample_interval=1,
            )
            annotator._last_request_time = 0

            with patch("vision_agent.data.auto_annotator.cv2.VideoCapture", return_value=mock_cap):
                with patch("vision_agent.data.auto_annotator.time.sleep"):
                    stats = annotator.run(save_path=save_path)

            # 验证统计
            self.assertEqual(stats["total_frames"], 3)
            self.assertEqual(stats["sampled"], 3)
            self.assertEqual(stats["annotated"], 2)  # 帧1 和 帧3
            self.assertEqual(stats["skipped"], 1)    # 帧2 无检测
            self.assertEqual(stats["errors"], 0)
            self.assertEqual(stats["action_dist"], {"attack": 1, "idle": 1})

            # 验证 JSONL 内容
            with open(save_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 2)

            sample1 = json.loads(lines[0])
            self.assertEqual(sample1["human_action"]["key"], "attack")
            self.assertEqual(sample1["human_action"]["type"], "llm_annotated")
            self.assertIn("enemy", sample1["object_counts"])
            # action_timestamp 应该是视频时间而非系统时间
            self.assertLess(sample1["action_timestamp"], 10.0)

            sample2 = json.loads(lines[1])
            self.assertEqual(sample2["human_action"]["key"], "idle")
            self.assertIn("tower", sample2["object_counts"])

    def test_unknown_action_not_saved(self):
        """LLM 返回未知动作时不应写入 JSONL。"""
        import numpy as np
        from vision_agent.data.auto_annotator import AutoAnnotator

        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        det = make_detections([("enemy", 0.9, 0.5, 0.5)])
        res = make_result(det, frame_id=1)

        mock_detector = MagicMock()
        mock_detector.detect = MagicMock(return_value=res)

        # LLM 返回不在列表中的动作
        provider = MockProvider(['{"action": "dance", "reason": "party time"}'])

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 1.0
        frame_iter = iter(frames)

        def mock_read():
            try:
                return True, next(frame_iter)
            except StopIteration:
                return False, None

        mock_cap.read = mock_read

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_output.jsonl")

            annotator = AutoAnnotator(
                video_path="fake.mp4",
                detector=mock_detector,
                provider=provider,
                actions=ACTIONS,
                sample_interval=1,
            )
            annotator._last_request_time = 0

            with patch("vision_agent.data.auto_annotator.cv2.VideoCapture", return_value=mock_cap):
                with patch("vision_agent.data.auto_annotator.time.sleep"):
                    stats = annotator.run(save_path=save_path)

            self.assertEqual(stats["annotated"], 0)
            self.assertEqual(stats["errors"], 1)

            # JSONL 应为空
            with open(save_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read().strip(), "")


class TestClaudeImageConversion(unittest.TestCase):
    """测试 ClaudeProvider._convert_messages 对图片格式的转换。"""

    def test_image_url_to_claude_format(self):
        from vision_agent.decision.llm_provider import ClaudeProvider

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "描述图片"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/abc123"}},
                ],
            }
        ]

        converted = ClaudeProvider._convert_messages(messages)

        self.assertEqual(len(converted), 1)
        content = converted[0]["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]["type"], "text")

        img_block = content[1]
        self.assertEqual(img_block["type"], "image")
        self.assertEqual(img_block["source"]["type"], "base64")
        self.assertEqual(img_block["source"]["media_type"], "image/jpeg")
        self.assertEqual(img_block["source"]["data"], "/9j/abc123")

    def test_text_only_unchanged(self):
        from vision_agent.decision.llm_provider import ClaudeProvider

        messages = [{"role": "user", "content": "hello"}]
        converted = ClaudeProvider._convert_messages(messages)
        self.assertEqual(converted, messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
