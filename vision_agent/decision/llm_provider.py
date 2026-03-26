"""LLM 供应商抽象层，支持 Claude / OpenAI / OpenAI 兼容 API。"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .minimax_mcp import MiniMaxMCPTools, preprocess_messages_with_vlm

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 统一响应格式。"""
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{"name": ..., "input": ...}]
    raw: object = None


class LLMProvider(ABC):
    """LLM 供应商基类。"""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """供应商名称。"""

    @abstractmethod
    def chat(self, messages: list[dict], system: str = "",
             tools: list[dict] | None = None, max_tokens: int = 1024) -> LLMResponse:
        """发送对话请求。

        Args:
            messages: [{"role": "user"|"assistant", "content": ...}]
            system: 系统提示词
            tools: 工具定义列表 (Claude tools 格式)
            max_tokens: 最大输出 token
        """

    @abstractmethod
    def test_connection(self) -> bool:
        """测试连接是否可用。"""


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API。"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514",
                 base_url: str | None = None):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

    @property
    def provider_name(self):
        return "claude"

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = anthropic.Anthropic(**kwargs)

    @staticmethod
    def _convert_messages(messages: list[dict]) -> list[dict]:
        """将 OpenAI 格式的 image_url 转为 Claude 的 image source 格式。"""
        converted = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            # data:image/jpeg;base64,xxxxx
                            header, b64_data = url.split(",", 1)
                            media_type = header.split(":")[1].split(";")[0]
                            new_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64_data,
                                },
                            })
                        else:
                            new_content.append({"type": "text", "text": f"[image: {url}]"})
                    else:
                        new_content.append(item)
                converted.append({"role": msg["role"], "content": new_content})
            else:
                converted.append(msg)
        return converted

    def chat(self, messages, system="", tools=None, max_tokens=1024):
        self._ensure_client()
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": self._convert_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = self._client.messages.create(**kwargs)

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            raw=response,
        )

    def test_connection(self):
        try:
            self._ensure_client()
            self._client.messages.create(
                model=self._model, max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception as e:
            logger.error(f"Claude 连接测试失败: {e}")
            return False


class OpenAIProvider(LLMProvider):
    """OpenAI API 及兼容接口（如 DeepSeek, 通义千问, 本地 Ollama 等）。"""

    def __init__(self, api_key: str, model: str = "gpt-4o",
                 base_url: str | None = None,
                 mcp_config: dict | None = None):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None
        # MCP 工具配置（当前仅 MiniMax 使用）
        self._mcp_tools: MiniMaxMCPTools | None = None
        if mcp_config and mcp_config.get("enabled"):
            self._mcp_tools = MiniMaxMCPTools(
                api_key=mcp_config.get("api_key", api_key),
                api_host=mcp_config.get("api_host", "https://api.minimaxi.com"),
            )
            logger.info(f"MCP 工具已启用: VLM + 搜索 (host={mcp_config.get('api_host', 'https://api.minimaxi.com')})")

    @property
    def provider_name(self):
        return "openai"

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)

    def _convert_tools(self, claude_tools: list[dict]) -> list[dict]:
        """将 Claude tools 格式转为 OpenAI function calling 格式。"""
        result = []
        for t in claude_tools:
            result.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            })
        return result

    def chat(self, messages, system="", tools=None, max_tokens=1024):
        self._ensure_client()

        # MCP VLM 预处理：将图片通过 VLM 转为文本描述
        if self._mcp_tools is not None:
            messages = self._preprocess_images_via_vlm(messages)

        # OpenAI 的 system 放在 messages 里
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})

        for msg in messages:
            oai_messages.append({"role": msg["role"], "content": self._flatten_content(msg["content"])})

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": oai_messages,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        # 对 thinking 模型，尝试关闭 thinking 让它直接返回 content
        model_lower = self._model.lower()
        if "thinking" in model_lower or "reasoner" in model_lower:
            try:
                kwargs["extra_body"] = {"enable_thinking": False}
                response = self._client.chat.completions.create(**kwargs)
            except Exception:
                # 框架不支持 enable_thinking，去掉后重试
                kwargs.pop("extra_body", None)
                response = self._client.chat.completions.create(**kwargs)
        else:
            response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        # 标准 content
        text = msg.content or ""

        # thinking 模型（qwen3-thinking, deepseek-reasoner 等）的处理:
        # 实际回答在 content，推理过程在 reasoning_content / model_extra["reasoning_content"]
        # 但如果 content 为空（token 不够、或模型把所有内容放进了 reasoning），
        # 需要 fallback 到 reasoning_content 中提取有效内容
        if not text:
            # 方式1: 标准 reasoning_content 属性
            reasoning = getattr(msg, "reasoning_content", None)
            # 方式2: model_extra 字典（openai SDK 将未知字段放在这里）
            if not reasoning:
                extra = getattr(msg, "model_extra", None) or {}
                reasoning = extra.get("reasoning_content") or extra.get("reasoning", "")
            if reasoning:
                text = reasoning

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": args,
                })

        return LLMResponse(text=text, tool_calls=tool_calls, raw=response)

    def test_connection(self):
        try:
            self._ensure_client()
            self._client.chat.completions.create(
                model=self._model, max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI 连接测试失败: {e}")
            return False

    @staticmethod
    def _flatten_content(content):
        """将 content 转为 OpenAI 兼容格式。

        如果包含图片等多模态内容，保留列表格式；
        如果只有文本，返回字符串。
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # 检查是否包含图片等非文本内容
            has_media = any(
                isinstance(item, dict) and item.get("type") in ("image_url", "image")
                for item in content
            )
            if has_media:
                # 保留原始列表格式，OpenAI API 原生支持
                return content
            # 纯文本列表，合并为字符串
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content)

    def _preprocess_images_via_vlm(self, messages: list[dict]) -> list[dict]:
        """通过 MCP VLM 将图片预处理为文本描述（用于不支持原生视觉的模型）。"""
        has_images = False
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        has_images = True
                        break
            if has_images:
                break

        if not has_images:
            return messages

        logger.info(f"[MCP VLM] 检测到图片，通过 VLM 转为文本描述...")
        return preprocess_messages_with_vlm(messages, self._mcp_tools)


# 预设配置（供 GUI 填充下拉框和自动填入 base_url）
PROVIDER_PRESETS = {
    "claude": {
        "class": ClaudeProvider,
        "models": ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001", "claude-opus-4-20250514"],
        "base_url": "",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "class": OpenAIProvider,
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-mini"],
        "base_url": "",
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "class": OpenAIProvider,
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "class": OpenAIProvider,
        "models": ["qwen-plus", "qwen-turbo", "qwen-max"],
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "QWEN_API_KEY",
    },
    "minimax": {
        "class": OpenAIProvider,
        "models": ["MiniMax-M2.7", "MiniMax-Text-01", "abab6.5s-chat", "abab5.5-chat"],
        "base_url": "https://api.minimax.chat/v1",
        "api_key_env": "MINIMAX_API_KEY",
        "mcp": {
            "enabled": True,
            "api_host": "https://api.minimaxi.com",
            "features": ["vlm", "web_search"],
        },
    },
    "ollama": {
        "class": OpenAIProvider,
        "models": ["llama3", "qwen2", "mistral"],
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "",
    },
    "custom": {
        "class": OpenAIProvider,
        "models": [],
        "base_url": "",
        "api_key_env": "",
    },
}


def create_provider(provider_name: str, api_key: str, model: str,
                    base_url: str = "") -> LLMProvider:
    """工厂方法：创建 LLM Provider 实例。"""
    preset = PROVIDER_PRESETS.get(provider_name)
    if preset is None:
        raise ValueError(f"未知供应商: {provider_name}，可用: {list(PROVIDER_PRESETS.keys())}")

    cls = preset["class"]
    kwargs = {"api_key": api_key, "model": model}
    if base_url:
        kwargs["base_url"] = base_url
    else:
        preset_url = preset.get("base_url", "")
        if preset_url:
            kwargs["base_url"] = preset_url

    # MCP 工具配置（当前仅 MiniMax 使用）
    mcp_preset = preset.get("mcp")
    if mcp_preset and mcp_preset.get("enabled") and cls == OpenAIProvider:
        kwargs["mcp_config"] = {
            "enabled": True,
            "api_key": api_key,
            "api_host": mcp_preset.get("api_host", "https://api.minimaxi.com"),
        }

    return cls(**kwargs)
