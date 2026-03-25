"""实时流视频源：支持 RTSP、HTTP-FLV、HLS(m3u8)、B站直播。

用法:
  - RTSP:       StreamSource("rtsp://admin:pass@192.168.1.100:554/stream")
  - HTTP-FLV:   StreamSource("https://xxx.bilivideo.com/live-bvc/xxx.flv")
  - B站直播间:  StreamSource("bilibili://12345")  或  StreamSource.from_bilibili(12345)
  - HLS:        StreamSource("https://xxx.m3u8")
"""

import logging
import re
import threading
import time

import cv2
import numpy as np
import requests

from .base import BaseSource

logger = logging.getLogger(__name__)


def get_bilibili_live_url(room_id: int | str) -> str | None:
    """获取B站直播间的真实流地址。

    Args:
        room_id: B站直播间号（短号或长号均可）

    Returns:
        HTTP-FLV 流地址，或 None（未开播/获取失败）
    """
    room_id = int(room_id)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://live.bilibili.com/",
    }

    try:
        # 1. 获取真实房间号（处理短号→长号）
        resp = requests.get(
            "https://api.live.bilibili.com/room/v1/Room/room_init",
            params={"id": room_id},
            headers=headers,
            timeout=10,
        )
        data = resp.json()
        if data.get("code") != 0:
            logger.error(f"获取房间信息失败: {data.get('message', 'unknown')}")
            return None

        room_info = data["data"]
        real_room_id = room_info["room_id"]
        live_status = room_info.get("live_status", 0)

        if live_status != 1:
            logger.warning(f"直播间 {room_id} 未开播 (status={live_status})")
            return None

        # 2. 获取直播流地址
        resp = requests.get(
            "https://api.live.bilibili.com/room/v1/Room/playUrl",
            params={
                "cid": real_room_id,
                "platform": "web",
                "quality": 4,  # 原画
                "qn": 10000,
            },
            headers=headers,
            timeout=10,
        )
        play_data = resp.json()
        if play_data.get("code") != 0:
            logger.error(f"获取直播流失败: {play_data.get('message', 'unknown')}")
            return None

        durl_list = play_data.get("data", {}).get("durl", [])
        if not durl_list:
            logger.error("直播流 durl 为空")
            return None

        stream_url = durl_list[0]["url"]
        logger.info(f"B站直播流获取成功: room={room_id}, real_room={real_room_id}")
        return stream_url

    except Exception as e:
        logger.error(f"获取B站直播流失败: {e}")
        return None


class StreamSource(BaseSource):
    """实时流视频源，支持 RTSP / HTTP-FLV / HLS / B站直播。

    特性:
      - 自动重连（断线后自动恢复）
      - 后台线程持续拉帧，避免缓冲区堆积
      - 丢帧策略：只保留最新帧，确保实时性
    """

    def __init__(self, url: str, reconnect_interval: float = 3.0,
                 max_reconnects: int = 10, on_log=None):
        """
        Args:
            url: 流地址。支持:
                 - rtsp://...
                 - http(s)://...flv / ...m3u8
                 - bilibili://<room_id>  (自动解析B站直播间)
            reconnect_interval: 断线重连间隔（秒）
            max_reconnects: 最大重连次数（0=无限）
            on_log: 日志回调
        """
        self._raw_url = url
        self._stream_url = None  # 解析后的实际流地址
        self._reconnect_interval = reconnect_interval
        self._max_reconnects = max_reconnects
        self._on_log = on_log

        self._cap = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._reconnect_count = 0

    def _log(self, msg: str):
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass

    @classmethod
    def from_bilibili(cls, room_id: int | str, **kwargs) -> "StreamSource":
        """从B站直播间号创建流源。"""
        return cls(f"bilibili://{room_id}", **kwargs)

    def _resolve_url(self) -> str | None:
        """解析流地址（处理 bilibili:// 协议）。"""
        url = self._raw_url.strip()

        # B站直播间协议
        m = re.match(r"bilibili://(\d+)", url)
        if m:
            room_id = m.group(1)
            self._log(f"解析B站直播间: {room_id}")
            stream_url = get_bilibili_live_url(room_id)
            if stream_url:
                self._log(f"直播流地址获取成功")
                return stream_url
            else:
                self._log(f"直播间 {room_id} 未开播或获取失败")
                return None

        # 直接使用 URL
        return url

    def start(self):
        """启动流拉取（后台线程）。"""
        self._stream_url = self._resolve_url()
        if not self._stream_url:
            raise RuntimeError(f"无法解析流地址: {self._raw_url}")

        self._running = True
        self._reconnect_count = 0
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._log(f"流源已启动: {self._raw_url}")

    def _open_capture(self) -> bool:
        """打开视频捕获。"""
        try:
            if self._cap is not None:
                self._cap.release()

            # RTSP 优化参数
            if self._stream_url.startswith("rtsp://"):
                self._cap = cv2.VideoCapture(self._stream_url, cv2.CAP_FFMPEG)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # 使用 TCP 传输（比 UDP 更稳定）
                self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            else:
                self._cap = cv2.VideoCapture(self._stream_url)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self._cap.isOpened():
                self._log(f"流连接成功")
                self._reconnect_count = 0
                return True
            else:
                self._log(f"流连接失败")
                return False

        except Exception as e:
            self._log(f"打开流异常: {e}")
            return False

    def _capture_loop(self):
        """后台持续拉帧线程。"""
        if not self._open_capture():
            # 首次连接失败，尝试重连
            if not self._reconnect():
                self._log("流源启动失败，停止拉取")
                self._running = False
                return

        consecutive_failures = 0

        while self._running:
            try:
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    with self._frame_lock:
                        self._frame = frame
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        self._log(f"连续 {consecutive_failures} 帧读取失败，尝试重连")
                        if not self._reconnect():
                            break
                        consecutive_failures = 0
                    else:
                        time.sleep(0.01)

            except Exception as e:
                self._log(f"拉帧异常: {e}")
                if not self._reconnect():
                    break

        self._log("流拉取线程退出")

    def _reconnect(self) -> bool:
        """断线重连。"""
        while self._running:
            self._reconnect_count += 1
            if self._max_reconnects > 0 and self._reconnect_count > self._max_reconnects:
                self._log(f"超过最大重连次数 ({self._max_reconnects})，停止")
                return False

            self._log(f"重连中... ({self._reconnect_count})")
            time.sleep(self._reconnect_interval)

            if not self._running:
                return False

            # B站直播流可能需要重新解析地址
            if self._raw_url.startswith("bilibili://"):
                new_url = self._resolve_url()
                if new_url:
                    self._stream_url = new_url

            if self._open_capture():
                return True

        return False

    def read(self) -> np.ndarray | None:
        """获取最新帧（非阻塞，总是返回最新的一帧）。"""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        """停止流拉取。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame = None
        self._log("流源已停止")

    @property
    def is_connected(self) -> bool:
        """是否已连接。"""
        return self._cap is not None and self._cap.isOpened()

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count
