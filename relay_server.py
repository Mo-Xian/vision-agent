"""轻量公网中继服务 — 房间配对 + 原样转发，不解析不缓存。

部署到公网服务器，Vision Agent 和远程客户端都主动连上来，
中继只负责按房间转发消息，对服务器性能要求极低。

依赖:
    pip install websockets

用法:
    python relay_server.py                          # 默认 0.0.0.0:9877
    python relay_server.py --port 8080              # 自定义端口
    python relay_server.py --port 8080 --token xyz  # 加 token 验证

协议:
    连接后第一条文本消息必须是注册:
        {"cmd":"join", "room":"abc123", "role":"hub"}     # Vision Agent 端
        {"cmd":"join", "room":"abc123", "role":"client"}  # 远程客户端

    注册成功后:
        {"event":"joined", "room":"abc123"}               # 确认加入
        {"event":"paired"}                                # 对端已就绪，开始转发

    之后所有消息（文本 / 二进制）原样转发给同房间对端。
"""

import asyncio
import json
import logging
import argparse
import secrets
import time

try:
    import websockets
except ImportError:
    print("需要安装: pip install websockets")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("relay")

# room_id -> {"hub": websocket, "client": websocket}
rooms: dict[str, dict[str, object]] = {}

# 统计
stats = {"total_connections": 0, "total_bytes": 0, "start_time": time.time()}


async def handler(ws, token: str = ""):
    """处理单个 WebSocket 连接。"""
    room_id = None
    role = None
    stats["total_connections"] += 1

    try:
        # ── 等待注册消息（10 秒超时） ──
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        if not isinstance(raw, str):
            await ws.close(4000, "First message must be JSON text")
            return

        reg = json.loads(raw)
        cmd = reg.get("cmd")
        if cmd != "join":
            await ws.close(4000, "First message must be {cmd:'join'}")
            return

        # token 验证
        if token and reg.get("token") != token:
            await ws.send(json.dumps({"event": "error", "msg": "Invalid token"}))
            await ws.close(4003, "Invalid token")
            return

        room_id = reg.get("room", "").strip()
        role = reg.get("role", "").strip()

        if not room_id or role not in ("hub", "client"):
            await ws.send(json.dumps({
                "event": "error",
                "msg": "Need room (non-empty) and role (hub|client)",
            }))
            await ws.close(4000, "Invalid registration")
            return

        # ── 加入房间 ──
        if room_id not in rooms:
            rooms[room_id] = {}

        room = rooms[room_id]

        if role in room:
            await ws.send(json.dumps({
                "event": "error",
                "msg": f"Role '{role}' already taken in room '{room_id}'",
            }))
            await ws.close(4001, "Role taken")
            return

        room[role] = ws
        other_role = "client" if role == "hub" else "hub"
        log.info(f"[{room_id}] {role} joined  ({ws.remote_address})")

        await ws.send(json.dumps({"event": "joined", "room": room_id}))

        # 如果对端已在，通知双方 paired
        peer = room.get(other_role)
        if peer:
            try:
                await peer.send(json.dumps({"event": "paired", "peer": role}))
            except Exception:
                pass
            await ws.send(json.dumps({"event": "paired", "peer": other_role}))
            log.info(f"[{room_id}] paired!")

        # ── 转发循环 ──
        async for msg in ws:
            peer = room.get(other_role)
            if peer:
                try:
                    await peer.send(msg)
                    if isinstance(msg, bytes):
                        stats["total_bytes"] += len(msg)
                except Exception:
                    pass

    except asyncio.TimeoutError:
        await ws.close(4000, "Registration timeout")
    except websockets.ConnectionClosed:
        pass
    except json.JSONDecodeError:
        await ws.close(4000, "Invalid JSON")
    except Exception as e:
        log.error(f"Unexpected error: {e}")
    finally:
        # 清理
        if room_id and role and room_id in rooms:
            rooms[room_id].pop(role, None)
            # 通知对端
            other_role = "client" if role == "hub" else "hub"
            peer = rooms[room_id].get(other_role)
            if peer:
                try:
                    await peer.send(json.dumps({
                        "event": "peer_left", "role": role,
                    }))
                except Exception:
                    pass
            # 空房间回收
            if not rooms[room_id]:
                del rooms[room_id]
            log.info(f"[{room_id}] {role} left  (rooms={len(rooms)})")


async def status_logger():
    """定期打印状态。"""
    while True:
        await asyncio.sleep(60)
        uptime = int(time.time() - stats["start_time"])
        mb = stats["total_bytes"] / (1024 * 1024)
        log.info(
            f"[stats] rooms={len(rooms)}  "
            f"conns={stats['total_connections']}  "
            f"forwarded={mb:.1f}MB  "
            f"uptime={uptime}s"
        )


async def main(host: str, port: int, token: str):
    log.info(f"Relay server starting on ws://{host}:{port}")
    if token:
        log.info(f"Token authentication enabled")
    else:
        log.info(f"No token (open access)")

    asyncio.create_task(status_logger())

    async with websockets.serve(
        lambda ws: handler(ws, token),
        host, port,
        max_size=4 * 1024 * 1024,  # 4MB per message
        ping_interval=20,
        ping_timeout=10,
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vision Agent Relay - lightweight WebSocket room forwarder"
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9877, help="bind port (default: 9877)")
    parser.add_argument("--token", default="", help="optional auth token (clients must match)")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.host, args.port, args.token))
    except KeyboardInterrupt:
        log.info("Relay server stopped")
