"""HTTP API 调用工具。"""

import json
import logging
import urllib.request
import urllib.error
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ApiCallTool(BaseTool):
    """发起 HTTP 请求调用外部 API。"""

    @property
    def name(self) -> str:
        return "api_call"

    @property
    def description(self) -> str:
        return "发起 HTTP 请求（GET/POST/PUT/DELETE），调用外部 API 并返回响应"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "请求 URL"},
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE"],
                    "description": "HTTP 方法，默认 GET",
                },
                "headers": {
                    "type": "object",
                    "description": "请求头 (key-value)",
                },
                "body": {
                    "type": "object",
                    "description": "请求体 (JSON)，仅 POST/PUT 时使用",
                },
                "timeout": {
                    "type": "number",
                    "description": "超时时间(秒)，默认 10",
                },
            },
            "required": ["url"],
        }

    def execute(self, url: str, method: str = "GET", headers: dict = None,
                body: dict = None, timeout: float = 10, **kwargs) -> ToolResult:
        try:
            data = None
            req_headers = headers or {}
            if body and method in ("POST", "PUT"):
                data = json.dumps(body).encode("utf-8")
                req_headers.setdefault("Content-Type", "application/json")

            req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_body = resp.read().decode("utf-8")
                try:
                    resp_json = json.loads(resp_body)
                except json.JSONDecodeError:
                    resp_json = resp_body

                return ToolResult(success=True, output={
                    "status": resp.status,
                    "body": resp_json,
                })

        except urllib.error.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP {e.code}: {e.reason}")
        except Exception as e:
            logger.error(f"API 调用失败: {e}")
            return ToolResult(success=False, error=str(e))
