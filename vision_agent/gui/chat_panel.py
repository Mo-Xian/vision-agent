"""LLM 对话面板：通过自然语言直接控制键盘鼠标等工具。"""

import base64
import json
import logging
import threading
import traceback

import cv2
import numpy as np
from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QCheckBox, QLabel,
)

from ..decision.llm_provider import LLMProvider
from ..tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# 系统提示词
_SYSTEM_PROMPT = """\
你是一个桌面操作助手。用户会用自然语言告诉你要做什么，你需要调用工具来完成操作。

重要规则:
1. 仔细理解用户意图，选择合适的工具和参数
2. 如果用户的指令不够明确，先回复文字确认，再执行
3. 可以一次调用多个工具来完成复合操作
4. 执行完后简短汇报结果
5. 如果有截图附带，结合截图内容理解用户需求
6. 如果用户只是在聊天（不需要操作键盘鼠标），直接用文字回复即可，不要调用工具"""

# 文本模式提示词（不支持 function calling 时）
_TEXT_MODE_SUFFIX = """

你可以使用以下工具:
{tools_desc}

当需要执行操作时，严格按以下 JSON 格式回复:
```json
{{
  "thinking": "你的分析",
  "actions": [
    {{"tool": "工具名", "params": {{"参数名": "参数值"}}}}
  ]
}}
```

如果只需要回复文字（不需要执行操作），直接用自然语言回复即可。"""


class ChatPanel(QWidget):
    """LLM 对话面板，支持自然语言控制键鼠。"""

    log_signal = Signal(str)
    # 跨线程信号：后台线程 → GUI 线程
    _sig_append = Signal(str)
    _sig_status = Signal(str)
    _sig_done = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._provider: LLMProvider | None = None
        self._registry: ToolRegistry | None = None
        self._conversation: list[dict] = []
        self._max_history = 20
        self._working = False
        self._current_frame: np.ndarray | None = None
        self._dryrun = False
        self._text_mode: bool | None = None
        self._try_auto_init: "callable | None" = None
        self._log_callback = None

        self._init_ui()

        # 连接跨线程信号（保证 GUI 线程安全）
        self._sig_append.connect(self._do_append)
        self._sig_status.connect(self._do_set_status)
        self._sig_done.connect(self._on_llm_done)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # 消息显示区
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Microsoft YaHei", 10))
        self.chat_display.setStyleSheet(
            "QTextEdit { border: 1px solid #253352; border-radius: 6px; "
            "padding: 6px; background-color: #0f1525; color: #e8ecf4; }"
        )
        layout.addWidget(self.chat_display, 1)

        # 状态行
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #8892a8; font-size: 11px; padding: 0 4px;")
        layout.addWidget(self.status_label)

        # 选项行
        opts_row = QHBoxLayout()
        opts_row.setSpacing(6)

        self.attach_screen_check = QCheckBox("附带截图")
        self.attach_screen_check.setChecked(False)
        self.attach_screen_check.setToolTip("发送消息时附带当前画面截图")
        opts_row.addWidget(self.attach_screen_check)

        self.dryrun_check = QCheckBox("仅模拟")
        self.dryrun_check.setChecked(False)
        self.dryrun_check.setToolTip("工具调用不会真正执行，仅显示将要执行的操作")
        self.dryrun_check.toggled.connect(self._on_dryrun_changed)
        opts_row.addWidget(self.dryrun_check)

        opts_row.addStretch()

        self.clear_btn = QPushButton("清空")
        self.clear_btn.setMaximumWidth(50)
        self.clear_btn.clicked.connect(self.clear_conversation)
        opts_row.addWidget(self.clear_btn)

        layout.addLayout(opts_row)

        # 输入行
        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self.input_edit = QLineEdit()
        self.input_edit.setFont(QFont("Microsoft YaHei", 11))
        self.input_edit.setPlaceholderText("输入指令或聊天内容...")
        self.input_edit.returnPressed.connect(self._on_send)
        self.input_edit.setStyleSheet(
            "QLineEdit { border: 1px solid #253352; border-radius: 6px; "
            "padding: 8px; background-color: #1a2338; color: #e8ecf4; }"
            "QLineEdit:focus { border-color: #4a7dff; }"
        )
        input_row.addWidget(self.input_edit)

        self.send_btn = QPushButton("发送")
        self.send_btn.setMinimumWidth(60)
        self.send_btn.setStyleSheet(
            "QPushButton { background-color: #4a7dff; color: white; "
            "font-weight: bold; padding: 8px 16px; border-radius: 6px; border: none; }"
            "QPushButton:hover { background-color: #3a6de8; }"
            "QPushButton:disabled { background-color: #253352; color: #5a6478; }"
        )
        self.send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self.send_btn)

        layout.addLayout(input_row)

    # ── 外部接口 ──

    def set_provider(self, provider: LLMProvider | None):
        self._provider = provider
        self._text_mode = None
        if provider:
            self._emit_log(f"[对话] Provider: {provider.provider_name}")

    def set_tool_registry(self, registry: ToolRegistry | None):
        self._registry = registry
        if registry:
            self._emit_log(f"[对话] 工具: {len(registry)} 个")

    def set_log_callback(self, callback):
        self._log_callback = callback

    def update_frame(self, frame: np.ndarray):
        self._current_frame = frame

    def _on_dryrun_changed(self, checked: bool):
        self._dryrun = checked

    def _emit_log(self, msg: str):
        logger.info(msg)
        if self._log_callback:
            try:
                self._log_callback(msg)
            except Exception:
                pass

    # ── 对话逻辑 ──

    def _on_send(self):
        text = self.input_edit.text().strip()
        if not text or self._working:
            return

        if not self._provider:
            if self._try_auto_init:
                if not self._try_auto_init():
                    self._append_system("LLM 未配置。请先在 LLM 页面设置。")
                    return
            else:
                self._append_system("LLM 未配置。请先在 LLM 页面设置。")
                return

        if not self._provider:
            self._append_system("LLM 初始化失败。")
            return

        self.input_edit.clear()
        self._append_user(text)
        self._sig_status.emit("思考中...")

        user_content = self._build_user_content(text)
        self._conversation.append({"role": "user", "content": user_content})

        if len(self._conversation) > self._max_history * 2:
            self._conversation = self._conversation[-self._max_history * 2:]

        self._working = True
        self.send_btn.setEnabled(False)
        self.input_edit.setEnabled(False)

        self._emit_log(f"[对话] 发送: {text[:80]}")
        threading.Thread(target=self._call_llm, daemon=True).start()

    def _build_user_content(self, text: str):
        if not self.attach_screen_check.isChecked():
            return text

        frame = self._current_frame
        if frame is None:
            frame = self._capture_screen()

        if frame is not None:
            h, w = frame.shape[:2]
            if w > 800:
                scale = 800 / w
                frame = cv2.resize(frame, (800, int(h * scale)))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf).decode("utf-8")
            return [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": text},
            ]
        return text

    @staticmethod
    def _capture_screen() -> "np.ndarray | None":
        try:
            import mss
            with mss.mss() as sct:
                return np.array(sct.grab(sct.monitors[1]))[:, :, :3]
        except ImportError:
            pass
        try:
            from PIL import ImageGrab
            return np.array(ImageGrab.grab())[:, :, ::-1]
        except ImportError:
            return None

    def _has_tools(self) -> bool:
        return self._registry is not None and len(self._registry) > 0

    def _should_use_text_mode(self) -> bool:
        if self._text_mode is not None:
            return self._text_mode
        if not self._provider:
            return True
        base_url = getattr(self._provider, '_base_url', '') or ''
        return any(s in base_url for s in ['minimax', 'ollama', 'localhost:11434'])

    def _call_llm(self):
        """后台线程：调用 LLM。"""
        try:
            has_tools = self._has_tools()
            if not has_tools:
                self._emit_log("[对话] 纯聊天（无工具）")
                actions, reply = self._call_chat_mode()
            elif self._should_use_text_mode():
                self._emit_log("[对话] 文本模式")
                actions, reply = self._call_text_mode()
            else:
                self._emit_log("[对话] 工具模式")
                actions, reply = self._call_tool_mode()

            self._emit_log(f"[对话] 结果: {len(actions)} 个动作, {len(reply or '')} 字")

            if actions:
                results = self._execute_actions(actions)
                for action, result in results:
                    if self._dryrun:
                        self._append_tool_call(action, "[模拟] 未执行")
                    elif result.success:
                        self._append_tool_call(action, f"成功: {result.output}")
                    else:
                        self._append_tool_call(action, f"失败: {result.error}")
            elif reply:
                self._append_assistant(reply)
            else:
                self._append_assistant("(无响应)")

        except Exception as e:
            logger.error(f"LLM error: {traceback.format_exc()}")
            self._emit_log(f"[对话] 异常: {e}")
            self._append_system(f"调用失败: {e}")
        finally:
            self._sig_status.emit("")
            self._sig_done.emit()

    def _call_chat_mode(self) -> tuple[list[dict], str]:
        """纯聊天，不传工具。"""
        response = self._provider.chat(
            messages=self._conversation,
            system="你是一个智能助手，用简洁的中文回答用户问题。",
            tools=None,
            max_tokens=2048,
        )
        reply = response.text or ""
        if reply:
            self._conversation.append({"role": "assistant", "content": reply})
        return [], reply

    def _call_tool_mode(self) -> tuple[list[dict], str]:
        """工具调用模式（function calling）。"""
        try:
            tools = self._registry.to_claude_tools() if self._registry else []
            response = self._provider.chat(
                messages=self._conversation,
                system=_SYSTEM_PROMPT,
                tools=tools,
                max_tokens=2048,
            )

            if not response.text and not response.tool_calls:
                self._text_mode = True
                self._conversation.pop()
                return self._call_text_mode()

            actions = [{"tool": tc["name"], "params": tc.get("input", {})}
                       for tc in response.tool_calls]

            # 记录到对话历史
            content = []
            if response.text:
                content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                content.append({"type": "tool_use", "id": tc.get("id", ""),
                                "name": tc["name"], "input": tc.get("input", {})})
            self._conversation.append({
                "role": "assistant",
                "content": content if content else response.text or "",
            })

            return actions, response.text if not actions else ""

        except Exception as e:
            self._emit_log(f"[对话] 工具模式异常: {e}，切文本模式")
            self._text_mode = True
            return self._call_text_mode()

    def _call_text_mode(self) -> tuple[list[dict], str]:
        """文本模式。"""
        tools_desc = []
        if self._registry:
            for t in self._registry.to_claude_tools():
                name = t.get("name", "")
                desc = t.get("description", "")
                params = t.get("input_schema", {}).get("properties", {})
                param_list = ", ".join(f'{k}: {v.get("type", "string")}' for k, v in params.items())
                tools_desc.append(f"  - {name}({param_list}): {desc}")

        system = _SYSTEM_PROMPT + _TEXT_MODE_SUFFIX.format(tools_desc="\n".join(tools_desc))
        messages = self._conversation[-2:] if len(self._conversation) > 2 else self._conversation

        response = self._provider.chat(
            messages=messages, system=system, tools=None, max_tokens=2048)

        if not response.text:
            return [], ""

        actions, thinking = self._parse_text_response(response.text)
        self._conversation.append({"role": "assistant", "content": response.text})

        if actions:
            return actions, ""
        return [], response.text

    def _parse_text_response(self, text: str) -> tuple[list[dict], str]:
        json_str = text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in json_str:
            parts = json_str.split("```")
            if len(parts) >= 3:
                json_str = parts[1]

        try:
            data = json.loads(json_str.strip())
        except json.JSONDecodeError:
            s, e = text.find("{"), text.rfind("}") + 1
            if s >= 0 and e > s:
                try:
                    data = json.loads(text[s:e])
                except json.JSONDecodeError:
                    return [], ""
            else:
                return [], ""

        if not isinstance(data, dict) or "actions" not in data:
            return [], ""

        thinking = data.get("thinking", "")
        actions = [{"tool": i.get("tool", ""), "params": i.get("params", {})}
                   for i in data.get("actions", []) if i.get("tool")]
        return actions, thinking

    def _execute_actions(self, actions: list[dict]) -> list[tuple[dict, ToolResult]]:
        results = []
        for action in actions:
            if self._dryrun:
                results.append((action, ToolResult(success=True, output={"dryrun": True})))
            elif not self._registry:
                results.append((action, ToolResult(success=False, error="工具未初始化")))
            else:
                try:
                    result = self._registry.execute(action["tool"], **action["params"])
                    results.append((action, result))
                except Exception as e:
                    results.append((action, ToolResult(success=False, error=str(e))))
        return results

    # ── 消息显示（全部通过 Signal，线程安全） ──

    def _append_user(self, text: str):
        self._sig_append.emit(
            f'<p style="margin:6px 0;color:#4a7dff;"><b>你:</b> {_esc(text)}</p>')

    def _append_assistant(self, text: str):
        clean = _strip_think(text)
        if not clean:
            return
        self._sig_append.emit(
            f'<p style="margin:6px 0;color:#2ecc71;"><b>AI:</b> '
            f'<span style="color:#e8ecf4;">{_esc(clean)}</span></p>')

    def _append_tool_call(self, action: dict, result_text: str):
        tool = action.get("tool", "?")
        params = action.get("params", {})
        ps = ", ".join(f'{k}={v}' for k, v in params.items())
        self._sig_append.emit(
            f'<p style="margin:3px 0;color:#f39c12;font-size:12px;">'
            f'[工具] {_esc(tool)}({_esc(ps)}) → '
            f'<span style="color:#8892a8;">{_esc(result_text)}</span></p>')

    def _append_system(self, text: str):
        self._sig_append.emit(
            f'<p style="margin:4px 0;color:#e74c3c;font-size:12px;">{_esc(text)}</p>')

    @Slot(str)
    def _do_append(self, html: str):
        self.chat_display.append(html)
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    @Slot(str)
    def _do_set_status(self, text: str):
        self.status_label.setText(text)

    @Slot()
    def _on_llm_done(self):
        self._working = False
        self.send_btn.setEnabled(True)
        self.input_edit.setEnabled(True)
        self.input_edit.setFocus()

    def clear_conversation(self):
        self._conversation.clear()
        self.chat_display.clear()
        self._text_mode = None
        self.status_label.setText("")


import re

_THINK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)


def _strip_think(text: str) -> str:
    """去除推理模型的 <think>...</think> 标签内容。"""
    return _THINK_RE.sub('', text).strip()


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace("\n", "<br/>"))
