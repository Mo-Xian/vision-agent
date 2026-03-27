"""远程 PC 画面采集服务端 — 在远程游戏 PC 上运行。

通过 WebSocket 实时推送屏幕画面和键鼠事件到本机 Vision Agent。
远程 PC 只需安装少量依赖即可运行，无需完整的 vision-agent 环境。

依赖（远程 PC）:
    pip install mss pynput websockets opencv-python-headless

用法:
    python remote_capture_server.py                        # 默认全屏 10fps 端口9876
    python remote_capture_server.py --port 9876 --fps 15
    python remote_capture_server.py --window "王者荣耀"    # 指定窗口

协议:
    文本消息 → JSON:
      {"type":"meta", "fps":10, "width":1920, "height":1080}   (连接时首条)
      {"type":"key_down",    "time":1.23, "key":"a"}
      {"type":"key_up",      "time":1.45, "key":"a"}
      {"type":"mouse_down",  "time":2.0,  "button":"left", "x":100, "y":200}
      {"type":"mouse_up",    "time":2.1,  "button":"left", "x":100, "y":200}
      {"type":"mouse_scroll","time":3.0,  "x":100, "y":200, "dx":0, "dy":-3}
    二进制消息 → 帧数据:
      8 字节时间戳 (big-endian double) + JPEG 图像字节
"""

import asyncio
import json
import signal
import struct
import sys
import threading
import time
from argparse import ArgumentParser


class RemoteCaptureServer:
    """远程画面采集 WebSocket 服务端。"""

    def __init__(
        self,
        fps: int = 10,
        window_title: str = "",
        port: int = 9876,
        jpeg_quality: int = 80,
    ):
        self.fps = fps
        self.window_title = window_title
        self.port = port
        self.jpeg_quality = jpeg_quality

        self._running = False
        self._start_time = 0.0
        self._frame_size = (0, 0)  # (width, height)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._send_queue: asyncio.Queue | None = None
        self._client_connected = threading.Event()
        self._active_client = None  # 当前连接的 websocket

    # ── 公开接口 ──

    def run(self):
        """启动服务（阻塞）。"""
        self._running = True
        self._start_time = time.time()
        # Windows 下确保 Ctrl+C 能正常退出
        if sys.platform == "win32":
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        asyncio.run(self._serve())

    def stop(self):
        self._running = False

    # ── WebSocket 服务 ──

    async def _serve(self):
        import websockets

        self._loop = asyncio.get_event_loop()
        self._send_queue = asyncio.Queue(maxsize=60)

        # 启动采集线程
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._keyboard_loop, daemon=True).start()
        threading.Thread(target=self._mouse_loop, daemon=True).start()

        # 等待首帧确定分辨率
        for _ in range(50):
            if self._frame_size != (0, 0):
                break
            await asyncio.sleep(0.1)

        local_ip = self._get_local_ip()
        print(f"[采集服务] 已启动: ws://{local_ip}:{self.port}")
        print(f"[采集服务] FPS={self.fps}  画面={self._frame_size[0]}x{self._frame_size[1]}  "
              f"窗口={'全屏' if not self.window_title else self.window_title}")
        print(f"[采集服务] 等待 Vision Agent 连接... (Ctrl+C 停止)")

        async with websockets.serve(
            self._handler, "0.0.0.0", self.port, max_size=10_000_000
        ):
            await asyncio.Future()

    async def _handler(self, websocket):
        addr = websocket.remote_address
        print(f"[采集服务] 客户端连接: {addr}")

        # 只允许一个客户端
        if self._active_client is not None:
            await websocket.send(json.dumps({
                "type": "error", "msg": "已有客户端连接，拒绝新连接"
            }))
            await websocket.close()
            return

        self._active_client = websocket

        # 清空旧数据
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # 发送元数据
        await websocket.send(json.dumps({
            "type": "meta",
            "fps": self.fps,
            "width": self._frame_size[0],
            "height": self._frame_size[1],
            "window": self.window_title,
        }))
        self._client_connected.set()

        try:
            while True:
                msg_type, data = await self._send_queue.get()
                if msg_type == "frame":
                    await websocket.send(data)
                else:
                    await websocket.send(data)
        except Exception as e:
            print(f"[采集服务] 客户端断开: {e}")
        finally:
            self._client_connected.clear()
            self._active_client = None

    def _enqueue(self, msg_type: str, data):
        """线程安全地将消息放入发送队列。"""
        if not self._client_connected.is_set() or self._loop is None:
            return

        def _safe_put():
            if self._send_queue.full():
                try:
                    self._send_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                self._send_queue.put_nowait((msg_type, data))
            except asyncio.QueueFull:
                pass

        try:
            self._loop.call_soon_threadsafe(_safe_put)
        except RuntimeError:
            pass

    # ── 截屏线程 ──

    def _capture_loop(self):
        import cv2
        import mss
        import numpy as np

        interval = 1.0 / self.fps

        with mss.mss() as sct:
            region = self._resolve_region(sct)
            monitor = region or sct.monitors[1]

            while self._running:
                t0 = time.time()
                ts = t0 - self._start_time

                try:
                    img = sct.grab(monitor)
                    frame = np.array(img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    h, w = frame.shape[:2]
                    if self._frame_size != (w, h):
                        self._frame_size = (w, h)

                    _, jpeg = cv2.imencode(
                        ".jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    )
                    header = struct.pack(">d", ts)
                    self._enqueue("frame", header + jpeg.tobytes())
                except Exception:
                    pass

                elapsed = time.time() - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    # ── 键盘监听 ──

    def _keyboard_loop(self):
        try:
            from pynput import keyboard
        except ImportError:
            print("[采集服务] 键盘监听需要 pynput: pip install pynput")
            return

        def on_press(key):
            if not self._running:
                return False
            ts = time.time() - self._start_time
            self._enqueue("event", json.dumps({
                "type": "key_down", "time": round(ts, 4),
                "key": self._key_name(key),
            }))

        def on_release(key):
            if not self._running:
                return False
            ts = time.time() - self._start_time
            self._enqueue("event", json.dumps({
                "type": "key_up", "time": round(ts, 4),
                "key": self._key_name(key),
            }))

        with keyboard.Listener(
            on_press=on_press, on_release=on_release
        ) as listener:
            while self._running:
                time.sleep(0.5)
            listener.stop()

    # ── 鼠标监听 ──

    def _mouse_loop(self):
        try:
            from pynput import mouse
        except ImportError:
            print("[采集服务] 鼠标监听需要 pynput: pip install pynput")
            return

        def on_click(x, y, button, pressed):
            if not self._running:
                return False
            ts = time.time() - self._start_time
            self._enqueue("event", json.dumps({
                "type": "mouse_down" if pressed else "mouse_up",
                "time": round(ts, 4),
                "button": button.name, "x": x, "y": y,
            }))

        def on_scroll(x, y, dx, dy):
            if not self._running:
                return False
            ts = time.time() - self._start_time
            self._enqueue("event", json.dumps({
                "type": "mouse_scroll", "time": round(ts, 4),
                "x": x, "y": y, "dx": dx, "dy": dy,
            }))

        with mouse.Listener(
            on_click=on_click, on_scroll=on_scroll
        ) as listener:
            while self._running:
                time.sleep(0.5)
            listener.stop()

    # ── 工具方法 ──

    def _resolve_region(self, sct) -> dict | None:
        if not self.window_title or sys.platform != "win32":
            return None
        try:
            import ctypes
            import ctypes.wintypes

            user32 = ctypes.windll.user32
            result = [None]
            title_lower = self.window_title.lower()

            def callback(hwnd, _):
                if not user32.IsWindowVisible(hwnd):
                    return True
                buf = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, buf, 256)
                if title_lower in buf.value.lower():
                    rect = ctypes.wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    result[0] = {
                        "left": rect.left, "top": rect.top,
                        "width": rect.right - rect.left,
                        "height": rect.bottom - rect.top,
                    }
                    return False
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(
                ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
            )
            user32.EnumWindows(WNDENUMPROC(callback), 0)

            if result[0]:
                print(f"[采集服务] 已定位窗口: {self.window_title} → {result[0]}")
            else:
                print(f"[采集服务] 未找到窗口 '{self.window_title}'，使用全屏")
            return result[0]
        except Exception:
            return None

    @staticmethod
    def _key_name(key) -> str:
        if hasattr(key, "char") and key.char:
            return key.char.lower()
        return key.name if hasattr(key, "name") else str(key)

    @staticmethod
    def _get_local_ip() -> str:
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "0.0.0.0"


# ── 独立运行 ──

if __name__ == "__main__":
    parser = ArgumentParser(description="远程 PC 画面采集服务端")
    parser.add_argument("--port", type=int, default=9876, help="WebSocket 端口 (默认 9876)")
    parser.add_argument("--fps", type=int, default=10, help="截屏帧率 (默认 10)")
    parser.add_argument("--window", type=str, default="", help="窗口标题（模糊匹配），留空为全屏")
    parser.add_argument("--quality", type=int, default=80, help="JPEG 质量 1-100 (默认 80)")
    args = parser.parse_args()

    server = RemoteCaptureServer(
        fps=args.fps,
        window_title=args.window,
        port=args.port,
        jpeg_quality=args.quality,
    )
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[采集服务] 已停止")
        server.stop()
