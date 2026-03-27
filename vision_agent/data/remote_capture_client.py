"""远程采集客户端 — 在远程游戏 PC 上运行。

连接到 Vision Agent 中转服务，推送屏幕画面和键鼠事件，
同时接收并执行 Agent 发来的控制指令。

远程 PC 只需安装少量依赖即可运行，无需完整的 vision-agent 环境。

依赖（远程 PC）:
    pip install mss pynput websockets opencv-python-headless numpy

用法:
    python remote_capture_client.py ws://192.168.1.100:9876
    python remote_capture_client.py ws://192.168.1.100:9876 --fps 15
    python remote_capture_client.py ws://192.168.1.100:9876 --window "王者荣耀"

协议:
    客户端 → 中转服务:
        文本 JSON: {"type":"meta", "fps":10, "width":1920, "height":1080}
        文本 JSON: {"type":"key_down", "time":1.23, "key":"a"} 等事件
        二进制:    8 字节时间戳 (big-endian double) + JPEG 图像字节
    中转服务 → 客户端:
        文本 JSON: {"cmd":"key_tap", "key":"a"} 等控制指令
"""

import asyncio
import json
import signal
import struct
import sys
import threading
import time
from argparse import ArgumentParser


class RemoteCaptureClient:
    """远程采集客户端 — 连接中转服务，推送画面/事件，执行控制指令。"""

    def __init__(
        self,
        server_url: str = "ws://192.168.1.100:9876",
        fps: int = 10,
        window_title: str = "",
        jpeg_quality: int = 80,
    ):
        self.server_url = server_url
        self.fps = fps
        self.window_title = window_title
        self.jpeg_quality = jpeg_quality

        self._running = False
        self._start_time = 0.0
        self._frame_size = (0, 0)  # (width, height)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._send_queue: asyncio.Queue | None = None
        self._connected = False

    # ── 公开接口 ──

    def run(self):
        """连接并开始采集（阻塞）。"""
        self._running = True
        self._start_time = time.time()
        if sys.platform == "win32":
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        asyncio.run(self._connect_loop())

    def stop(self):
        self._running = False

    # ── 连接与通信 ──

    async def _connect_loop(self):
        """自动重连循环。"""
        while self._running:
            try:
                await self._connect_and_run()
            except Exception as e:
                if not self._running:
                    break
                print(f"[采集客户端] 连接断开: {e}")
                print(f"[采集客户端] 5 秒后重连...")
                self._connected = False
                await asyncio.sleep(5)

    async def _connect_and_run(self):
        import websockets

        self._loop = asyncio.get_event_loop()
        self._send_queue = asyncio.Queue(maxsize=60)

        # 启动采集线程
        cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        ms_thread = threading.Thread(target=self._mouse_loop, daemon=True)
        cap_thread.start()
        kb_thread.start()
        ms_thread.start()

        # 等待首帧确定分辨率
        for _ in range(50):
            if self._frame_size != (0, 0):
                break
            await asyncio.sleep(0.1)

        print(f"[采集客户端] 连接中转服务: {self.server_url}")

        async with websockets.connect(
            self.server_url, max_size=10_000_000, open_timeout=10
        ) as ws:
            self._connected = True
            print(f"[采集客户端] 已连接!")
            print(f"[采集客户端] FPS={self.fps}  画面={self._frame_size[0]}x{self._frame_size[1]}  "
                  f"窗口={'全屏' if not self.window_title else self.window_title}")

            # 发送元数据
            await ws.send(json.dumps({
                "type": "meta",
                "fps": self.fps,
                "width": self._frame_size[0],
                "height": self._frame_size[1],
                "window": self.window_title,
            }))

            # 双向通信
            send_task = asyncio.create_task(self._send_loop(ws))
            recv_task = asyncio.create_task(self._recv_loop(ws))
            done, pending = await asyncio.wait(
                [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()

    async def _send_loop(self, ws):
        """持续发送帧和事件到中转服务。"""
        while self._running:
            msg_type, data = await self._send_queue.get()
            await ws.send(data)

    async def _recv_loop(self, ws):
        """接收中转服务的控制指令。"""
        async for msg in ws:
            if not isinstance(msg, str):
                continue
            try:
                data = json.loads(msg)
                if data.get("type") == "error":
                    print(f"[采集客户端] 服务端拒绝: {data.get('msg')}")
                    return
                cmd = data.get("cmd", "")
                if cmd:
                    self._execute_control(data)
            except Exception as e:
                print(f"[采集客户端] 控制指令错误: {e}")

    def _execute_control(self, data: dict):
        """在本机执行控制指令。"""
        cmd = data["cmd"]
        try:
            if cmd in ("key_press", "key_release", "key_tap"):
                from pynput.keyboard import Controller, Key
                kb = Controller()
                key_name = data.get("key", "")
                key = getattr(Key, key_name, None) or key_name
                if cmd == "key_press":
                    kb.press(key)
                elif cmd == "key_release":
                    kb.release(key)
                else:  # key_tap
                    kb.press(key)
                    kb.release(key)

            elif cmd == "mouse_click":
                from pynput.mouse import Controller, Button
                ms = Controller()
                x, y = int(data.get("x", 0)), int(data.get("y", 0))
                btn_name = data.get("button", "left")
                btn = getattr(Button, btn_name, Button.left)
                ms.position = (x, y)
                ms.click(btn)

            elif cmd == "mouse_move":
                from pynput.mouse import Controller
                ms = Controller()
                ms.position = (int(data.get("x", 0)), int(data.get("y", 0)))

        except Exception as e:
            print(f"[采集客户端] 执行控制失败: {cmd} -> {e}")

    def _enqueue(self, msg_type: str, data):
        """线程安全地将消息放入发送队列。"""
        if not self._connected or self._loop is None:
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
            print("[采集客户端] 键盘监听需要 pynput: pip install pynput")
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
            print("[采集客户端] 鼠标监听需要 pynput: pip install pynput")
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
                print(f"[采集客户端] 已定位窗口: {self.window_title} -> {result[0]}")
            else:
                print(f"[采集客户端] 未找到窗口 '{self.window_title}'，使用全屏")
            return result[0]
        except Exception:
            return None

    @staticmethod
    def _key_name(key) -> str:
        if hasattr(key, "char") and key.char:
            return key.char.lower()
        return key.name if hasattr(key, "name") else str(key)


# ── 独立运行 ──

if __name__ == "__main__":
    parser = ArgumentParser(description="远程采集客户端 — 连接 Vision Agent 中转服务")
    parser.add_argument(
        "server", nargs="?", default="ws://192.168.1.100:9876",
        help="中转服务地址，如 ws://192.168.1.100:9876",
    )
    parser.add_argument("--fps", type=int, default=10, help="截屏帧率 (默认 10)")
    parser.add_argument("--window", type=str, default="", help="窗口标题（模糊匹配），留空为全屏")
    parser.add_argument("--quality", type=int, default=80, help="JPEG 质量 1-100 (默认 80)")
    args = parser.parse_args()

    # 自动补全 ws:// 前缀
    server_url = args.server
    if not server_url.startswith("ws://") and not server_url.startswith("wss://"):
        server_url = f"ws://{server_url}"

    print("=" * 50)
    print("  Vision Agent 远程采集客户端")
    print("=" * 50)
    print(f"  中转服务: {server_url}")
    print(f"  帧率: {args.fps} FPS")
    print(f"  窗口: {args.window or '全屏'}")
    print(f"  JPEG 质量: {args.quality}")
    print("=" * 50)
    print()

    client = RemoteCaptureClient(
        server_url=server_url,
        fps=args.fps,
        window_title=args.window,
        jpeg_quality=args.quality,
    )
    try:
        client.run()
    except KeyboardInterrupt:
        print("\n[采集客户端] 已停止")
        client.stop()
