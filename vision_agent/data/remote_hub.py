"""远程中转服务 — 运行在 Vision Agent 主机上。

接收远程客户端推送的画面和键鼠事件，提供给录制和 Agent 模块使用。
同时转发 Agent 的控制指令到远程客户端执行。

支持两种模式:
    直连模式（默认）:
        远程客户端  --ws-->  Vision Agent 主机 (WebSocket Server)
    中继模式:
        Vision Agent --ws-->  公网 Relay  <--ws-- 远程客户端

协议（与 remote_capture_client / Android App 对应）:
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
import secrets
import socket
import struct
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RemoteHub:
    """WebSocket 中转服务 — 接收远程客户端画面，提供 Agent 控制通道。

    用法（直连模式）:
        hub = RemoteHub(port=9876)
        hub.start()

    用法（中继模式）:
        hub = RemoteHub(relay_url="ws://my-server.com:9877", room_id="abc123")
        hub.start()

    公共接口:
        frame = hub.get_frame()
        events = hub.get_events()
        hub.send_command({"cmd":"key_tap", "key":"a"})
        hub.stop()
    """

    def __init__(
        self,
        port: int = 9876,
        relay_url: str = "",
        room_id: str = "",
        relay_token: str = "",
        on_log=None,
    ):
        self.port = port
        self.relay_url = relay_url.strip()
        self.room_id = room_id.strip() or secrets.token_hex(4)
        self.relay_token = relay_token.strip()
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

    @property
    def is_relay_mode(self) -> bool:
        return bool(self.relay_url)

    # ── 公开接口 ──

    def start(self):
        """启动（非阻塞）。"""
        if self._running:
            return
        self._running = True
        self._server_thread = threading.Thread(target=self._run, daemon=True)
        self._server_thread.start()

    def stop(self):
        """停止。"""
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
            msg = json.dumps(cmd)
            try:
                self._loop.call_soon_threadsafe(
                    lambda: self._loop.create_task(self._async_send(msg))
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

    def send_tap(self, x: int, y: int):
        """发送点击指令（兼容 PC mouse_click 和 Android tap）。"""
        self.send_command({"cmd": "mouse_click", "x": x, "y": y})

    def send_swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
        """发送滑动指令（Android 原生支持，PC 端可忽略）。"""
        self.send_command({
            "cmd": "swipe",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "duration": duration_ms,
        })

    @property
    def screen_size(self) -> tuple[int, int]:
        """从客户端 meta 获取屏幕分辨率 (width, height)。"""
        return self._meta.get("width", 0), self._meta.get("height", 0)

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
        """异步发送消息到客户端（直连）或通过中继转发。"""
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
            if self.is_relay_mode:
                self._loop.run_until_complete(self._relay_connect())
            else:
                self._loop.run_until_complete(self._serve())
        except Exception as e:
            if self._running:
                self._log(f"[中转服务] 异常退出: {e}")
        finally:
            self._loop.close()
            self._loop = None

    # ── 直连模式（本地 WebSocket Server） ──

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

    # ── 中继模式（连接公网 Relay） ──

    async def _relay_connect(self):
        """连接公网中继，自动重连。"""
        import websockets

        while self._running:
            try:
                self._log(f"[中继] 连接: {self.relay_url}  房间: {self.room_id}")
                async with websockets.connect(
                    self.relay_url, max_size=10_000_000, open_timeout=10,
                ) as ws:
                    # 注册为 hub
                    reg = {"cmd": "join", "room": self.room_id, "role": "hub"}
                    if self.relay_token:
                        reg["token"] = self.relay_token
                    await ws.send(json.dumps(reg))

                    # 等待确认
                    resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                    if resp.get("event") == "error":
                        self._log(f"[中继] 注册失败: {resp.get('msg')}")
                        return
                    if resp.get("event") != "joined":
                        self._log(f"[中继] 意外响应: {resp}")
                        return

                    self._log(f"[中继] 已加入房间 {self.room_id}，等待客户端...")
                    self._client_ws = ws
                    self._client_addr = f"relay:{self.room_id}"

                    # 消息循环
                    async for msg in ws:
                        if not self._running:
                            break
                        if isinstance(msg, str):
                            data = json.loads(msg)
                            evt = data.get("event", "")
                            if evt == "paired":
                                self._client_connected.set()
                                self._log(f"[中继] 客户端已连接（通过中继）")
                                continue
                            if evt == "peer_left":
                                self._client_connected.clear()
                                self._log(f"[中继] 客户端已断开")
                                continue
                            # 普通文本消息（meta / 键鼠事件）
                            self._handle_text(msg)
                        elif isinstance(msg, bytes):
                            self._handle_frame(msg)

            except Exception as e:
                if not self._running:
                    break
                self._log(f"[中继] 连接断开: {e}，5 秒后重连...")
                self._client_ws = None
                self._client_connected.clear()
                await asyncio.sleep(5)

    # ── 消息处理（直连 / 中继共用） ──

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
