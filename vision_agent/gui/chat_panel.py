"""LLM 对话面板：通过自然语言直接控制键盘鼠标等工具。"""

import base64
import json
import logging
import threading
import traceback

import cv2
import numpy as np
from PySide6.QtCore import Signal, Slot, QTimer
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

    log_signal = Signal(str)  # 日志输出

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
        self._try_auto_init: "callable | None" = None  # 由父窗口设置
        self._log_callback = None  # 外部日志回调

        self._init_ui()

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
            "padding: 6px; background-color: #0f1525; }"
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
        self.attach_screen_check.setToolTip("发送消息时附带当前画面截图，让 LLM 能看到屏幕")
        opts_row.addWidget(self.attach_screen_check)

        self.dryrun_check = QCheckBox("仅模拟")
        self.dryrun_check.setChecked(False)
        self.dryrun_check.setToolTip("勾选后工具调用不会真正执行，仅显示将要执行的操作")
        self.dryrun_check.toggled.connect(self._on_dryrun_changed)
        opts_row.addWidget(self.dryrun_check)

        opts_row.addStretch()

        self.clear_btn = QPushButton("清空对话")
        self.clear_btn.setMaximumWidth(70)
        self.clear_btn.clicked.connect(self.clear_conversation)
        opts_row.addWidget(self.clear_btn)

        layout.addLayout(opts_row)

        # 输入行
        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self.input_edit = QLineEdit()
        self.input_edit.setFont(QFont("Microsoft YaHei", 11))
        self.input_edit.setPlaceholderText("输入指令，如：点击屏幕中间、按下 Ctrl+C ...")
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
        """设置 LLM Provider。"""
        self._provider = provider
        self._text_mode = None
        if provider:
            self._emit_log(f"[对话] Provider 已设置: {provider.provider_name}")

    def set_tool_registry(self, registry: ToolRegistry | None):
        """设置工具注册表。"""
        self._registry = registry
        if registry:
            self._emit_log(f"[对话] 工具已注册: {len(registry)} 个")

    def set_log_callback(self, callback):
        """设置外部日志回调。"""
        self._log_callback = callback

    def update_frame(self, frame: np.ndarray):
        """更新当前画面帧（供截图附带用）。"""
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

    def _set_status(self, text: str):
        QTimer.singleShot(0, lambda: self.status_label.setText(text))

    # ── 对话逻辑 ──

    def _on_send(self):
        text = self.input_edit.text().strip()
        if not text or self._working:
            return

        # 检查 Provider
        if not self._provider:
            self._emit_log("[对话] Provider 为空，尝试自动初始化")
            if self._try_auto_init:
                ok = self._try_auto_init()
                self._emit_log(f"[对话] 自动初始化结果: {ok}")
                if not ok:
                    self._append_system("LLM 未配置。请先在 LLM 页面设置供应商和 API Key。")
                    return
            else:
                self._append_system("LLM 未配置。请先在 LLM 页面设置供应商和 API Key。")
                return

        # 再次检查（auto_init 可能设置了 provider）
        if not self._provider:
            self._append_system("LLM Provider 初始化失败，请检查 LLM 配置。")
            return

        if not self._registry or len(self._registry) == 0:
            self._append_system("工具未注册。请确保 pynput 已安装。")
            return

        self.input_edit.clear()
        self._append_user(text)
        self._set_status("思考中...")

        # 构建消息
        user_content = self._build_user_content(text)
        self._conversation.append({"role": "user", "content": user_content})

        # 截断历史
        if len(self._conversation) > self._max_history * 2:
            self._conversation = self._conversation[-self._max_history * 2:]

        # 异步调用 LLM
        self._working = True
        self.send_btn.setEnabled(False)
        self.input_edit.setEnabled(False)

        self._emit_log(f"[对话] 发送消息: {text[:50]}...")
        thread = threading.Thread(target=self._call_llm, daemon=True)
        thread.start()

    def _build_user_content(self, text: str):
        """构建用户消息内容（可能包含截图）。"""
        if not self.attach_screen_check.isChecked():
            return text

        frame = self._current_frame
        # 如果没有从检测管线获取帧，尝试直接截屏
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
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
                {"type": "text", "text": text},
            ]
        return text

    @staticmethod
    def _capture_screen() -> "np.ndarray | None":
        """截取当前屏幕（不依赖检测管线）。"""
        try:
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # 主显示器
                img = sct.grab(monitor)
                frame = np.array(img)[:, :, :3]  # BGRA → BGR
                return frame
        except ImportError:
            pass
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            frame = np.array(img)[:, :, ::-1]  # RGB → BGR
            return frame
        except ImportError:
            return None

    def _should_use_text_mode(self) -> bool:
        if self._text_mode is not None:
            return self._text_mode
        if not self._provider:
            return True
        base_url = getattr(self._provider, '_base_url', '') or ''
        if any(name in base_url for name in ['minimax', 'ollama', 'localhost:11434']):
            return True
        return False

    def _call_llm(self):
        """在后台线程调用 LLM 并处理响应。"""
        try:
            use_text_mode = self._should_use_text_mode()
            self._emit_log(f"[对话] 调用模式: {'文本' if use_text_mode else '工具'}")

            if use_text_mode:
                actions, reply_text = self._call_text_mode()
            else:
                actions, reply_text = self._call_tool_mode()

            self._emit_log(f"[对话] 返回: actions={len(actions)}, text={len(reply_text or '')}字")

            # 处理工具调用
            if actions:
                results = self._execute_actions(actions)
                # 显示执行结果
                for action, result in results:
                    if self._dryrun:
                        self._append_tool_call(action, "[模拟] 未实际执行")
                    elif result.success:
                        self._append_tool_call(action, f"成功: {result.output}")
                    else:
                        self._append_tool_call(action, f"失败: {result.error}")
            elif reply_text:
                self._append_assistant(reply_text)
            else:
                self._append_assistant("(无响应)")

        except Exception as e:
            err_detail = traceback.format_exc()
            logger.error(f"LLM 调用失败: {e}\n{err_detail}")
            self._emit_log(f"[对话] 异常: {e}")
            self._append_system(f"LLM 调用失败: {e}")
        finally:
            self._set_status("")
            # 必须在 GUI 线程恢复控件状态
            QTimer.singleShot(0, self._on_llm_done)

    def _call_tool_mode(self) -> tuple[list[dict], str]:
        """工具调用模式。返回 (actions, reply_text)。"""
        try:
            tools = self._registry.to_claude_tools() if self._registry else []
            self._emit_log(f"[对话] tool_mode: {len(tools)} 个工具, "
                           f"{len(self._conversation)} 条消息")

            response = self._provider.chat(
                messages=self._conversation,
                system=_SYSTEM_PROMPT,
                tools=tools,
                max_tokens=2048,
            )

            self._emit_log(f"[对话] 响应: text={len(response.text or '')}字, "
                           f"tool_calls={len(response.tool_calls)}")

            # 如果工具调用和文本都为空，尝试切换到文本模式
            if not response.text and not response.tool_calls:
                self._emit_log("[对话] 响应为空，切换文本模式")
                self._text_mode = True
                self._conversation.pop()  # 移除刚加的消息
                return self._call_text_mode()

            actions = []
            for tc in response.tool_calls:
                actions.append({
                    "tool": tc["name"],
                    "params": tc.get("input", {}),
                })

            # 记录助手回复到对话
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

            if actions and response.text:
                self._append_assistant(response.text)

            return actions, response.text if not actions else ""

        except Exception as e:
            # function calling 失败，切换到文本模式
            self._emit_log(f"[对话] 工具模式异常: {e}，切换文本模式")
            self._text_mode = True
            return self._call_text_mode()

    def _call_text_mode(self) -> tuple[list[dict], str]:
        """文本模式（不支持 function calling 的模型）。"""
        tools_desc = []
        if self._registry:
            for t in self._registry.to_claude_tools():
                name = t.get("name", "")
                desc = t.get("description", "")
                params = t.get("input_schema", {}).get("properties", {})
                param_list = ", ".join(f'{k}: {v.get("type", "string")}' for k, v in params.items())
                tools_desc.append(f"  - {name}({param_list}): {desc}")

        system = _SYSTEM_PROMPT + _TEXT_MODE_SUFFIX.format(tools_desc="\n".join(tools_desc))

        # 文本模式用简短对话
        messages = self._conversation[-2:] if len(self._conversation) > 2 else self._conversation

        self._emit_log(f"[对话] text_mode: {len(messages)} 条消息")

        response = self._provider.chat(
            messages=messages,
            system=system,
            tools=None,
            max_tokens=2048,
        )

        self._emit_log(f"[对话] text_mode 响应: {len(response.text or '')}字")

        if not response.text:
            return [], ""

        # 尝试解析 JSON 格式的工具调用
        actions, thinking = self._parse_text_response(response.text)

        # 记录到对话
        self._conversation.append({"role": "assistant", "content": response.text})

        if actions:
            if thinking:
                self._append_assistant(thinking)
            return actions, ""
        else:
            return [], response.text

    def _parse_text_response(self, text: str) -> tuple[list[dict], str]:
        """从文本响应解析工具调用 JSON。"""
        json_str = text.strip()

        # 去除 markdown 代码块
        if "```json" in json_str:
            json_str = json_str.split("```json", 1)[1]
            json_str = json_str.split("```", 1)[0]
        elif "```" in json_str:
            parts = json_str.split("```")
            if len(parts) >= 3:
                json_str = parts[1]

        try:
            data = json.loads(json_str.strip())
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    return [], ""
            else:
                return [], ""

        if not isinstance(data, dict) or "actions" not in data:
            return [], ""

        thinking = data.get("thinking", "")
        actions = []
        for item in data.get("actions", []):
            tool_name = item.get("tool", "")
            params = item.get("params", {})
            if tool_name:
                actions.append({"tool": tool_name, "params": params})

        return actions, thinking

    def _execute_actions(self, actions: list[dict]) -> list[tuple[dict, ToolResult]]:
        """执行工具调用列表。"""
        results = []
        for action in actions:
            tool_name = action["tool"]
            params = action["params"]

            if self._dryrun:
                results.append((action, ToolResult(success=True, output={"dryrun": True})))
                continue

            if not self._registry:
                results.append((action, ToolResult(success=False, error="工具注册表未初始化")))
                continue

            try:
                result = self._registry.execute(tool_name, **params)
                results.append((action, result))
            except Exception as e:
                results.append((action, ToolResult(success=False, error=str(e))))

        return results

    # ── 消息显示 ──

    def _append_user(self, text: str):
        QTimer.singleShot(0, lambda: self._do_append(
            f'<div style="margin: 6px 0; text-align: right;">'
            f'<span style="background-color: #1a3a6e; color: #e8ecf4; '
            f'padding: 6px 12px; border-radius: 10px; display: inline-block; '
            f'max-width: 80%; text-align: left;">{_escape(text)}</span>'
            f'<span style="color: #4a7dff; font-size: 12px;"> 你</span></div>'
        ))

    def _append_assistant(self, text: str):
        QTimer.singleShot(0, lambda: self._do_append(
            f'<div style="margin: 6px 0;">'
            f'<span style="color: #2ecc71; font-size: 12px;">AI </span>'
            f'<span style="background-color: #1a2338; color: #e8ecf4; '
            f'padding: 6px 12px; border-radius: 10px; display: inline-block; '
            f'max-width: 80%;">{_escape(text)}</span></div>'
        ))

    def _append_tool_call(self, action: dict, result_text: str):
        tool = action.get("tool", "?")
        params = action.get("params", {})
        params_str = ", ".join(f'{k}={v}' for k, v in params.items())
        QTimer.singleShot(0, lambda: self._do_append(
            f'<div style="margin: 4px 0; padding: 4px 8px; '
            f'background-color: #151d30; border-left: 3px solid #f39c12; '
            f'border-radius: 4px; font-family: Consolas, monospace; font-size: 12px;">'
            f'<span style="color: #f39c12;">[工具]</span> '
            f'<span style="color: #9b59b6;">{_escape(tool)}</span>'
            f'(<span style="color: #8892a8;">{_escape(params_str)}</span>)'
            f'<br/><span style="color: #5a6478;">{_escape(result_text)}</span></div>'
        ))

    def _append_system(self, text: str):
        QTimer.singleShot(0, lambda: self._do_append(
            f'<div style="margin: 4px 0; text-align: center; '
            f'color: #e74c3c; font-size: 12px;">{_escape(text)}</div>'
        ))

    def _do_append(self, html: str):
        self.chat_display.append(html)
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot()
    def _on_llm_done(self):
        self._working = False
        self.send_btn.setEnabled(True)
        self.input_edit.setEnabled(True)
        self.input_edit.setFocus()

    def clear_conversation(self):
        """清空对话历史。"""
        self._conversation.clear()
        self.chat_display.clear()
        self._text_mode = None
        self.status_label.setText("")


def _escape(text: str) -> str:
    """HTML 转义。"""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>"))
