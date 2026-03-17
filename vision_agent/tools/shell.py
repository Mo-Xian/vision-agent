"""Shell 命令执行工具。"""

import logging
import subprocess
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ShellTool(BaseTool):
    """执行 Shell 命令并返回输出。"""

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return "执行 Shell 命令并返回标准输出和错误输出"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的命令"},
                "cwd": {"type": "string", "description": "工作目录，默认当前目录"},
                "timeout": {"type": "number", "description": "超时时间(秒)，默认 30"},
            },
            "required": ["command"],
        }

    def __init__(self, allowed_commands: list[str] | None = None):
        """
        Args:
            allowed_commands: 允许执行的命令前缀白名单，None 表示不限制
        """
        self._allowed = allowed_commands

    def execute(self, command: str, cwd: str = None, timeout: float = 30, **kwargs) -> ToolResult:
        if self._allowed is not None:
            cmd_prefix = command.split()[0] if command.strip() else ""
            if cmd_prefix not in self._allowed:
                return ToolResult(
                    success=False,
                    error=f"命令 '{cmd_prefix}' 不在白名单中。允许: {self._allowed}",
                )

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            return ToolResult(
                success=proc.returncode == 0,
                output={
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[:5000],   # 截断避免过长
                    "stderr": proc.stderr[:2000],
                },
                error=proc.stderr[:500] if proc.returncode != 0 else "",
            )

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error=f"命令超时 ({timeout}s): {command}")
        except Exception as e:
            logger.error(f"Shell 执行失败: {e}")
            return ToolResult(success=False, error=str(e))
