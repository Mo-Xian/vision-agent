"""工具系统基类，参考 Claude tool-use 模式设计。"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """工具执行结果。"""
    success: bool
    output: dict = field(default_factory=dict)
    error: str = ""


class BaseTool(ABC):
    """工具基类。每个工具需定义名称、描述、参数 Schema 和执行逻辑。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具唯一名称。"""

    @property
    @abstractmethod
    def description(self) -> str:
        """工具功能描述（供 LLM 理解）。"""

    @property
    @abstractmethod
    def parameters_schema(self) -> dict:
        """JSON Schema 格式的参数定义。"""

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具动作，返回结果。"""

    def to_claude_tool(self) -> dict:
        """转换为 Claude API 的 tool 定义格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema,
        }


class ToolRegistry:
    """工具注册表，管理所有可用工具。"""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册一个工具。"""
        if tool.name in self._tools:
            logger.warning(f"工具 '{tool.name}' 已存在，将被覆盖")
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def get(self, name: str) -> BaseTool:
        """按名称获取工具。"""
        if name not in self._tools:
            raise KeyError(f"工具 '{name}' 未注册。可用工具: {list(self._tools.keys())}")
        return self._tools[name]

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """按名称执行工具。"""
        tool = self.get(tool_name)
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"工具 '{tool_name}' 执行失败: {e}")
            return ToolResult(success=False, error=str(e))

    def list_tools(self) -> list[dict]:
        """返回所有工具的摘要信息。"""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    def to_claude_tools(self) -> list[dict]:
        """转换为 Claude API 的 tools 参数格式。"""
        return [t.to_claude_tool() for t in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)
