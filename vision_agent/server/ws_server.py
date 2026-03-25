"""WebSocket 服务器，向连接的客户端推送检测结果。"""

import asyncio
import json
import logging
import threading
import websockets

from ..core.detector import DetectionResult

logger = logging.getLogger(__name__)


class WebSocketServer:
    """异步 WebSocket 服务器，在独立线程中运行。"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        """在后台线程中启动 WebSocket 服务器。"""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"WebSocket 服务器启动: ws://{self.host}:{self.port}")

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()  # run forever

    async def _handler(self, websocket):
        self._clients.add(websocket)
        remote = websocket.remote_address
        logger.info(f"客户端连接: {remote}")
        try:
            async for message in websocket:
                # 可扩展：接收客户端指令
                pass
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info(f"客户端断开: {remote}")

    def broadcast(self, result: DetectionResult):
        """向所有连接的客户端广播检测结果。"""
        if not self._clients or self._loop is None:
            return
        data = json.dumps(result.to_dict(), ensure_ascii=False)
        asyncio.run_coroutine_threadsafe(self._broadcast(data), self._loop)

    async def _broadcast(self, data: str):
        if not self._clients:
            return
        dead = set()
        for ws in self._clients:
            try:
                await ws.send(data)
            except websockets.ConnectionClosed:
                dead.add(ws)
        self._clients -= dead

    def broadcast_decision(self, actions):
        """向所有客户端推送决策结果。"""
        if not self._clients or self._loop is None:
            return
        decision_data = {
            "type": "decision",
            "actions": [
                {
                    "tool": a.tool_name,
                    "parameters": a.parameters,
                    "reason": a.reason,
                    "priority": a.priority,
                }
                for a in actions
            ],
        }
        data = json.dumps(decision_data, ensure_ascii=False)
        asyncio.run_coroutine_threadsafe(self._broadcast(data), self._loop)

    def stop(self):
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

    @property
    def client_count(self) -> int:
        return len(self._clients)
