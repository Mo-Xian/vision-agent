"""PyInstaller 打包脚本 — 生成 RelayServer.exe。

公网中继服务，部署到服务器后 Vision Agent 和远程客户端都可以通过它转发。
极轻量：仅依赖 websockets，~160 行。

用法:
    python build_relay.py              # 单文件打包
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def build():
    entry = str(ROOT / "relay_server.py")
    name = "RelayServer"

    hidden = [
        "websockets", "websockets.legacy", "websockets.legacy.server",
        "websockets.legacy.protocol", "websockets.frames",
        "websockets.uri", "websockets.headers", "asyncio",
    ]

    excludes = [
        "torch", "torchvision", "ultralytics",
        "PySide6", "matplotlib", "tkinter", "IPython", "jupyter",
        "notebook", "pytest", "sphinx", "setuptools", "PIL",
        "sklearn", "scipy", "pandas", "cv2", "numpy",
        "pynput", "mss", "openai",
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--name", name,
        "--onefile",
        "--console",
    ]

    for h in hidden:
        cmd.extend(["--hidden-import", h])

    for e in excludes:
        cmd.extend(["--exclude-module", e])

    cmd.append(entry)

    print(f"[build-relay] Building {name}...")
    print(f"[build-relay] Entry: {entry}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[build-relay] FAILED")
        sys.exit(1)

    print(f"[build-relay] Done -> dist/{name}.exe")


if __name__ == "__main__":
    build()
