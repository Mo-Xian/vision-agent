"""键盘模拟工具。"""

import logging
import time
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

try:
    import pynput.keyboard as kb
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class KeyboardTool(BaseTool):
    """模拟键盘按键操作。"""

    # pynput 特殊键映射
    _SPECIAL_KEYS = {
        "space": "space", "enter": "enter", "tab": "tab",
        "esc": "esc", "escape": "esc",
        "backspace": "backspace", "delete": "delete",
        "up": "up", "down": "down", "left": "left", "right": "right",
        "shift": "shift", "ctrl": "ctrl", "ctrl_l": "ctrl_l", "ctrl_r": "ctrl_r",
        "alt": "alt", "alt_l": "alt_l", "alt_r": "alt_r",
        "cmd": "cmd", "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4",
        "f5": "f5", "f6": "f6", "f7": "f7", "f8": "f8",
        "f9": "f9", "f10": "f10", "f11": "f11", "f12": "f12",
    }

    @property
    def name(self) -> str:
        return "keyboard"

    @property
    def description(self) -> str:
        return "模拟键盘按键操作，支持单键按下、组合键、按住一段时间"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["press", "hold", "hotkey", "type_text"],
                    "description": "press=按一下, hold=按住一段时间, hotkey=组合键, type_text=输入文本",
                },
                "key": {
                    "type": "string",
                    "description": "按键名称，如 space, a, enter, f1",
                },
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "组合键列表，如 ['ctrl', 'c']，仅 hotkey 时使用",
                },
                "text": {
                    "type": "string",
                    "description": "要输入的文本，仅 type_text 时使用",
                },
                "duration": {
                    "type": "number",
                    "description": "按住时长(秒)，仅 hold 时使用，默认 0.1",
                },
            },
            "required": ["action"],
        }

    def __init__(self):
        if not _AVAILABLE:
            raise ImportError("需要安装 pynput: pip install pynput")
        self._controller = kb.Controller()

    def execute(self, action: str, key: str = None, keys: list[str] = None,
                text: str = None, duration: float = 0.1, **kwargs) -> ToolResult:
        try:
            if action == "press":
                k = self._parse_key(key)
                self._controller.press(k)
                self._controller.release(k)
                return ToolResult(success=True, output={"action": "press", "key": key})

            elif action == "hold":
                k = self._parse_key(key)
                self._controller.press(k)
                time.sleep(duration)
                self._controller.release(k)
                return ToolResult(success=True, output={"action": "hold", "key": key, "duration": duration})

            elif action == "hotkey":
                if not keys:
                    return ToolResult(success=False, error="hotkey 需要 keys 参数")
                parsed = [self._parse_key(k) for k in keys]
                for k in parsed:
                    self._controller.press(k)
                for k in reversed(parsed):
                    self._controller.release(k)
                return ToolResult(success=True, output={"action": "hotkey", "keys": keys})

            elif action == "type_text":
                if not text:
                    return ToolResult(success=False, error="type_text 需要 text 参数")
                self._controller.type(text)
                return ToolResult(success=True, output={"action": "type_text", "length": len(text)})

            else:
                return ToolResult(success=False, error=f"未知 action: {action}")

        except Exception as e:
            logger.error(f"键盘操作失败: {e}")
            return ToolResult(success=False, error=str(e))

    def _parse_key(self, key_str: str):
        """将字符串解析为 pynput 按键。"""
        if not key_str:
            raise ValueError("key 不能为空")
        low = key_str.lower()
        if low in self._SPECIAL_KEYS:
            return getattr(kb.Key, self._SPECIAL_KEYS[low])
        if len(key_str) == 1:
            return key_str
        raise ValueError(f"无法识别的按键: {key_str}")
