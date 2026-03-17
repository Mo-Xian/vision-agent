"""鼠标模拟工具。"""

import logging
import time
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

try:
    import pynput.mouse as ms
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class MouseTool(BaseTool):
    """模拟鼠标操作：移动、点击、拖拽、滚轮。"""

    @property
    def name(self) -> str:
        return "mouse"

    @property
    def description(self) -> str:
        return "模拟鼠标操作，支持移动、点击、双击、拖拽、滚轮"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["move", "click", "double_click", "right_click", "drag", "scroll"],
                    "description": "鼠标动作类型",
                },
                "x": {"type": "integer", "description": "目标 X 坐标 (像素)"},
                "y": {"type": "integer", "description": "目标 Y 坐标 (像素)"},
                "end_x": {"type": "integer", "description": "拖拽终点 X，仅 drag 时使用"},
                "end_y": {"type": "integer", "description": "拖拽终点 Y，仅 drag 时使用"},
                "scroll_amount": {"type": "integer", "description": "滚轮量，正数向上负数向下，仅 scroll 时使用"},
                "duration": {"type": "number", "description": "拖拽持续时间(秒)，默认 0.5"},
            },
            "required": ["action"],
        }

    def __init__(self):
        if not _AVAILABLE:
            raise ImportError("需要安装 pynput: pip install pynput")
        self._controller = ms.Controller()

    def execute(self, action: str, x: int = None, y: int = None,
                end_x: int = None, end_y: int = None,
                scroll_amount: int = 0, duration: float = 0.5, **kwargs) -> ToolResult:
        try:
            if action == "move":
                self._controller.position = (x, y)
                return ToolResult(success=True, output={"action": "move", "x": x, "y": y})

            elif action == "click":
                if x is not None and y is not None:
                    self._controller.position = (x, y)
                self._controller.click(ms.Button.left)
                return ToolResult(success=True, output={"action": "click", "x": x, "y": y})

            elif action == "double_click":
                if x is not None and y is not None:
                    self._controller.position = (x, y)
                self._controller.click(ms.Button.left, 2)
                return ToolResult(success=True, output={"action": "double_click", "x": x, "y": y})

            elif action == "right_click":
                if x is not None and y is not None:
                    self._controller.position = (x, y)
                self._controller.click(ms.Button.right)
                return ToolResult(success=True, output={"action": "right_click", "x": x, "y": y})

            elif action == "drag":
                if None in (x, y, end_x, end_y):
                    return ToolResult(success=False, error="drag 需要 x, y, end_x, end_y")
                self._controller.position = (x, y)
                time.sleep(0.05)
                self._controller.press(ms.Button.left)
                steps = max(int(duration * 60), 5)
                dx = (end_x - x) / steps
                dy = (end_y - y) / steps
                for i in range(steps):
                    self._controller.position = (int(x + dx * (i + 1)), int(y + dy * (i + 1)))
                    time.sleep(duration / steps)
                self._controller.release(ms.Button.left)
                return ToolResult(success=True, output={"action": "drag", "from": [x, y], "to": [end_x, end_y]})

            elif action == "scroll":
                if x is not None and y is not None:
                    self._controller.position = (x, y)
                self._controller.scroll(0, scroll_amount)
                return ToolResult(success=True, output={"action": "scroll", "amount": scroll_amount})

            else:
                return ToolResult(success=False, error=f"未知 action: {action}")

        except Exception as e:
            logger.error(f"鼠标操作失败: {e}")
            return ToolResult(success=False, error=str(e))
