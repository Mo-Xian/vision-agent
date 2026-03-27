"""远程中转服务 — 运行在 Vision Agent 主机上。

接收远程客户端推送的画面和键鼠事件，提供给录制和 Agent 模块使用。
同时转发 Agent 的控制指令到远程客户端执行。

架构:
    远程 PC (客户端)  ──WebSocket──>  Vision Agent 主机 (中转服务)
        画面帧 + 键鼠事件 ───>  录制 / 训练
        <─── 控制指令（Agent 操控）

协议（与 remote_capture_client 对应）:
    客户端 → 中转:
        文本 JSON: {"type":"meta", "fps":10, "width":1920, "height":1080}
        文本 JSON: {"type":"key_down", "time":1.23, "key":"a"} 等事件
        二进制:    8 字节时间戳 (big-endian double) + JPEG 图像字节
    中转 → 客户端:
        文本 JSON: {"cmd":"key_tap", "key":"a"} 等控制指令
"""

import asyncio
import json
import logging
import queue
import socket
import struct
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RemoteHub:
    """WebSocket 中转服务 — 接收远程客户端画面，提供 Agent 控制通道。

    用法:
        hub = RemoteHub(port=9876)
        hub.start()                    # 非阻塞启动
        # 等待客户端连接...
        frame = hub.get_frame()        # 获取最新画面
        events = hub.get_events()      # 获取待处理事件
        hub.send_command({"cmd":"key_tap", "key":"a"})  # 发送控制指令
        hub.stop()
    """

    def __init__(self, port: int = 9876, on_log=None):
        self.port = port
        self._on_log = on_log

        self._running = False
        self._server_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # 客户端连接状态
        self._client_ws = None
        self._client_connected = threading.Event()
        self._client_addr: str = ""
        self._meta: dict = {}

        # 数据缓冲
        self._latest_frame: tuple[float, np.ndarray] | None = None
        self._frame_lock = threading.Lock()
        self._event_queue: queue.Queue = queue.Queue(maxsize=10000)

    # ── 公开接口 ──

    def start(self):
        """启动中转服务（非阻塞）。"""
        if self._running:
            return
        self._running = True
        self._server_thread = threading.Thread(target=self._run, daemon=True)
        self._server_thread.start()

    def stop(self):
        """停止中转服务。"""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._server_thread:
            self._server_thread.join(timeout=5)
        self._client_ws = None
        self._client_connected.clear()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_client_connected(self) -> bool:
        return self._client_connected.is_set()

    @property
    def client_addr(self) -> str:
        return self._client_addr

    @property
    def client_meta(self) -> dict:
        return self._meta.copy()

    def get_frame(self) -> np.ndarray | None:
        """获取最新一帧画面（BGR numpy array）。"""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame[1]

    def get_frame_with_ts(self) -> tuple[float, np.ndarray] | None:
        """获取最新帧及时间戳。"""
        with self._frame_lock:
            return self._latest_frame

    def get_events(self) -> list[dict]:
        """取出所有待处理的键鼠事件。"""
        events = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def send_command(self, cmd: dict):
        """发送控制指令到远程客户端。"""
        if self._client_ws and self._loop and self._client_connected.is_set():
            try:
                self._loop.call_soon_threadsafe(
                    self._loop.create_task,
                    self._async_send(json.dumps(cmd)),
                )
            except RuntimeError:
                pass

    def send_key_tap(self, key: str):
        self.send_command({"cmd": "key_tap", "key": key})

    def send_key_press(self, key: str):
        self.send_command({"cmd": "key_press", "key": key})

    def send_key_release(self, key: str):
        self.send_command({"cmd": "key_release", "key": key})

    def send_mouse_click(self, x: int, y: int, button: str = "left"):
        self.send_command({"cmd": "mouse_click", "x": x, "y": y, "button": button})

    def send_mouse_move(self, x: int, y: int):
        self.send_command({"cmd": "mouse_move", "x": x, "y": y})

    def get_local_ip(self) -> str:
        """获取本机局域网 IP。"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "0.0.0.0"

    # ── 内部实现 ──

    async def _async_send(self, data: str):
        """异步发送消息到客户端。"""
        if self._client_ws:
            try:
                await self._client_ws.send(data)
            except Exception as e:
                logger.debug(f"发送指令失败: {e}")

    def _run(self):
        """服务线程入口。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            if self._running:
                self._log(f"[中转服务] 异常退出: {e}")
        finally:
            self._loop.close()
            self._loop = None

    async def _serve(self):
        import websockets

        local_ip = self.get_local_ip()
        self._log(f"[中转服务] 启动: ws://{local_ip}:{self.port}")
        self._log(f"[中转服务] 等待远程客户端连接...")

        async with websockets.serve(
            self._handler, "0.0.0.0", self.port, max_size=10_000_000
        ):
            while self._running:
                await asyncio.sleep(0.5)

    async def _handler(self, websocket):
        addr = websocket.remote_address
        addr_str = f"{addr[0]}:{addr[1]}" if addr else "unknown"

        # 只允许一个客户端
        if self._client_connected.is_set():
            await websocket.send(json.dumps({
                "type": "error", "msg": "已有客户端连接，拒绝新连接"
            }))
            await websocket.close()
            self._log(f"[中转服务] 拒绝新连接: {addr_str}（已有客户端）")
            return

        self._client_ws = websocket
        self._client_addr = addr_str
        self._client_connected.set()
        self._log(f"[中转服务] 客户端已连接: {addr_str}")

        try:
            async for msg in websocket:
                if not self._running:
                    break
                if isinstance(msg, bytes):
                    self._handle_frame(msg)
                elif isinstance(msg, str):
                    self._handle_text(msg)
        except Exception as e:
            self._log(f"[中转服务] 客户端断开: {e}")
        finally:
            self._client_ws = None
            self._client_addr = ""
            self._client_connected.clear()
            self._log(f"[中转服务] 客户端已断开: {addr_str}")

    def _handle_frame(self, msg: bytes):
        """处理二进制帧消息：8字节时间戳 + JPEG。"""
        if len(msg) < 9:
            return
        ts = struct.unpack(">d", msg[:8])[0]
        jpeg_bytes = msg[8:]
        frame = cv2.imdecode(
            np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is not None:
            with self._frame_lock:
                self._latest_frame = (ts, frame)

    def _handle_text(self, msg: str):
        """处理文本 JSON 消息。"""
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type", "")
        if msg_type == "meta":
            self._meta = data
            w, h = data.get("width", 0), data.get("height", 0)
            fps = data.get("fps", 0)
            window = data.get("window") or "全屏"
            self._log(f"[中转服务] 远程画面: {w}x{h} FPS={fps} 窗口={window}")
        else:
            # 键鼠事件入队
            try:
                self._event_queue.put_nowait(data)
            except queue.Full:
                try:
                    self._event_queue.get_nowait()
                    self._event_queue.put_nowait(data)
                except queue.Empty:
                    pass

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass
