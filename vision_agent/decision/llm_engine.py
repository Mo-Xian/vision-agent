"""基于 LLM 的智能决策引擎，支持多供应商。"""

import json
import logging
import time
from ..core.detector import DetectionResult
from ..core.state import SceneState
from .base import Action, DecisionEngine, LoggingMixin
from .llm_provider import LLMProvider, create_provider

logger = logging.getLogger(__name__)


class LLMEngine(LoggingMixin, DecisionEngine):
    """LLM 决策引擎，通过 Provider 抽象层支持 Claude / OpenAI / 兼容 API。"""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        system_prompt: str = "",
        tools_schema: list[dict] | None = None,
        decision_interval: float = 1.0,
        max_tokens: int = 1024,
        # 兼容旧参数，直接创建 ClaudeProvider
        api_key: str | None = None,
        model: str | None = None,
        provider_name: str = "claude",
        base_url: str = "",
    ):
        if provider:
            self._provider = provider
        elif api_key:
            self._provider = create_provider(
                provider_name=provider_name,
                api_key=api_key,
                model=model or "claude-sonnet-4-20250514",
                base_url=base_url,
            )
        else:
            self._provider = None

        self._system_prompt = system_prompt
        self._tools_schema = tools_schema or []
        self._decision_interval = decision_interval
        self._max_tokens = max_tokens
        self._last_call_time = 0.0
        self._conversation: list[dict] = []
        self._max_history = 10
        # 是否使用文本模式（不支持 function calling 的供应商）
        self._text_mode: bool | None = None  # None=自动检测

    @property
    def provider(self) -> LLMProvider | None:
        return self._provider

    def set_provider(self, provider: LLMProvider) -> None:
        """运行时切换 Provider。"""
        self._provider = provider
        self._conversation.clear()
        self._text_mode = None  # 重新检测
        logger.info(f"LLM Provider 切换为: {provider.provider_name}")

    def on_start(self) -> None:
        if not self._provider:
            logger.warning("LLMEngine 未配置 Provider")
            return
        logger.info(f"LLMEngine 已启动: provider={self._provider.provider_name}")

    def on_stop(self) -> None:
        self._conversation.clear()

    def set_tools_schema(self, tools_schema: list[dict]) -> None:
        self._tools_schema = tools_schema

    def configure(self, **kwargs) -> None:
        if "system_prompt" in kwargs:
            self._system_prompt = kwargs["system_prompt"]
        if "decision_interval" in kwargs:
            self._decision_interval = kwargs["decision_interval"]

    def decide(self, result: DetectionResult, state: SceneState) -> list[Action]:
        now = time.time()
        if now - self._last_call_time < self._decision_interval:
            return []
        self._last_call_time = now

        if not self._provider:
            return []

        if not self._tools_schema:
            logger.warning("LLMEngine 没有可用工具")
            return []

        return self._call_llm(state)

    def _should_use_text_mode(self) -> bool:
        """判断是否应该使用文本模式。"""
        if self._text_mode is not None:
            return self._text_mode
        if not self._provider:
            return True
        # 已知不支持 function calling 的供应商
        provider_name = self._provider.provider_name
        # OpenAIProvider 的 provider_name 统一返回 "openai"，
        # 需要通过 base_url 来判断实际供应商
        base_url = getattr(self._provider, '_base_url', '') or ''
        if any(name in base_url for name in ['minimax', 'ollama', 'localhost:11434']):
            return True
        return False

    def _build_text_mode_prompt(self) -> str:
        """构建文本模式的系统提示词，把工具定义嵌入提示词中。"""
        tools_desc = []
        for t in self._tools_schema:
            name = t.get("name", "")
            desc = t.get("description", "")
            params = t.get("input_schema", {}).get("properties", {})
            param_list = ", ".join(f'{k}: {v.get("type", "string")}' for k, v in params.items())
            tools_desc.append(f"  - {name}({param_list}): {desc}")

        tools_text = "\n".join(tools_desc)

        return f"""{self._system_prompt}

你可以使用以下工具来执行动作:
{tools_text}

请根据画面检测结果分析当前局势，如果需要执行动作，请严格按以下 JSON 格式回复:
```json
{{
  "thinking": "你的分析思考过程",
  "actions": [
    {{"tool": "工具名称", "params": {{"参数名": "参数值"}}}}
  ]
}}
```

如果当前不需要执行任何动作，返回:
```json
{{
  "thinking": "你的分析",
  "actions": []
}}
```

只返回 JSON，不要返回其他内容。"""

    def _parse_text_response(self, text: str) -> list[Action]:
        """从文本响应中解析 JSON 格式的动作。"""
        # 尝试提取 JSON
        json_str = text.strip()
        # 去除 markdown 代码块
        if "```json" in json_str:
            json_str = json_str.split("```json", 1)[1]
            json_str = json_str.split("```", 1)[0]
        elif "```" in json_str:
            json_str = json_str.split("```", 1)[1]
            json_str = json_str.split("```", 1)[0]

        try:
            data = json.loads(json_str.strip())
        except json.JSONDecodeError:
            # 尝试找到 JSON 对象
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    self._emit_log(f"[LLM] 无法解析 JSON 响应")
                    return []
            else:
                return []

        thinking = data.get("thinking", "")
        if thinking:
            self._emit_log(f"[LLM 思考] {thinking[:500]}")

        actions = []
        for item in data.get("actions", []):
            tool_name = item.get("tool", "")
            params = item.get("params", {})
            if tool_name:
                actions.append(Action(
                    tool_name=tool_name,
                    parameters=params,
                    reason=f"llm:{self._provider.provider_name}",
                    confidence=0.85,
                ))
        return actions

    def _call_llm(self, state: SceneState) -> list[Action]:
        """调用 LLM，支持工具调用模式和文本模式。"""
        user_msg = f"当前画面检测结果:\n{state.scene_summary}"
        if state.custom_data:
            user_msg += f"\n\n附加上下文:\n{json.dumps(state.custom_data, ensure_ascii=False)}"

        use_text_mode = self._should_use_text_mode()

        if use_text_mode:
            return self._call_llm_text_mode(user_msg)
        else:
            return self._call_llm_tool_mode(user_msg)

    def _call_llm_tool_mode(self, user_msg: str) -> list[Action]:
        """工具调用模式（Claude / OpenAI 等支持 function calling 的模型）。"""
        self._conversation.append({"role": "user", "content": user_msg})
        if len(self._conversation) > self._max_history * 2:
            self._conversation = self._conversation[-self._max_history * 2:]

        try:
            self._emit_log(f"[LLM] 工具模式调用 {self._provider.provider_name} ...")
            response = self._provider.chat(
                messages=self._conversation,
                system=self._system_prompt,
                tools=self._tools_schema,
                max_tokens=self._max_tokens,
            )
            self._emit_log(f"[LLM] 响应 | text={bool(response.text)} | tool_calls={len(response.tool_calls)}")

            # 如果返回为空，自动切换到文本模式
            if not response.text and not response.tool_calls:
                self._emit_log("[LLM] 工具模式返回为空，自动切换为文本模式")
                self._text_mode = True
                self._conversation.pop()  # 移除刚加的 user 消息
                return self._call_llm_text_mode(user_msg)

            actions = []
            for tc in response.tool_calls:
                self._emit_log(f"[LLM] 工具调用: {tc['name']}({json.dumps(tc.get('input', {}), ensure_ascii=False)[:200]})")
                actions.append(Action(
                    tool_name=tc["name"],
                    parameters=tc.get("input", {}),
                    reason=f"llm:{self._provider.provider_name}",
                    confidence=0.9,
                ))

            if response.text:
                self._emit_log(f"[LLM 思考] {response.text[:500]}")

            # 记录助手回复
            assistant_content = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use", "id": tc.get("id", ""),
                    "name": tc["name"], "input": tc.get("input", {}),
                })
            self._conversation.append({
                "role": "assistant",
                "content": assistant_content if assistant_content else response.text or "",
            })

            if actions:
                self._emit_log(f"[LLM 决策] 共 {len(actions)} 个动作: {[a.tool_name for a in actions]}")
            else:
                self._emit_log("[LLM] 本轮无决策动作")

            return actions

        except Exception as e:
            import traceback
            self._emit_log(f"[LLM 错误] {type(e).__name__}: {e}")
            self._emit_log(f"[LLM] 尝试切换为文本模式")
            self._text_mode = True
            # 移除刚加的消息，避免污染对话
            if self._conversation and self._conversation[-1].get("role") == "user":
                self._conversation.pop()
            return self._call_llm_text_mode(user_msg)

    def _call_llm_text_mode(self, user_msg: str) -> list[Action]:
        """文本模式（不支持 function calling 的模型，用 JSON 输出代替）。"""
        system_prompt = self._build_text_mode_prompt()

        # 文本模式用独立的简短对话，避免格式混乱
        messages = [{"role": "user", "content": user_msg}]

        try:
            self._emit_log(f"[LLM] 文本模式调用 {self._provider.provider_name} ...")
            response = self._provider.chat(
                messages=messages,
                system=system_prompt,
                tools=None,  # 不传工具
                max_tokens=self._max_tokens,
            )

            if response.text:
                self._emit_log(f"[LLM] 收到响应 ({len(response.text)} 字符)")
                actions = self._parse_text_response(response.text)
                if actions:
                    self._emit_log(f"[LLM 决策] 共 {len(actions)} 个动作: {[a.tool_name for a in actions]}")
                else:
                    self._emit_log("[LLM] 本轮无决策动作")
                return actions
            else:
                self._emit_log("[LLM] 文本模式也返回为空")
                return []

        except Exception as e:
            self._emit_log(f"[LLM 错误] {type(e).__name__}: {e}")
            return []
