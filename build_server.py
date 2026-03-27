"""PyInstaller 打包脚本 — 生成 RemoteCaptureServer.exe。

远程 PC 上运行的轻量采集服务，打包为单文件 EXE，无需安装 Python。

用法:
    python build_server.py          # 正常打包（带控制台）
    python build_server.py --onefile  # 单文件模式
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def build(onefile: bool = False):
    entry = str(ROOT / "vision_agent" / "data" / "remote_capture_server.py")
    name = "RemoteCaptureServer"

    hidden = [
        "cv2", "numpy", "mss",
        "pynput", "pynput.keyboard", "pynput.mouse",
        "websockets", "websockets.legacy", "websockets.legacy.server",
    ]

    excludes = [
        "torch", "torchvision", "ultralytics",
        "PySide6", "matplotlib", "tkinter", "IPython", "jupyter",
        "notebook", "pytest", "sphinx", "setuptools", "PIL",
        "sklearn", "scipy", "pandas",
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--name", name,
        "--console",  # 采集服务需要控制台显示状态
    ]

    if onefile:
        cmd.append("--onefile")

    for h in hidden:
        cmd.extend(["--hidden-import", h])

    for e in excludes:
        cmd.extend(["--exclude-module", e])

    cmd.append(entry)

    print(f"[build-server] Building {name}...")
    print(f"[build-server] Entry: {entry}")
    print(f"[build-server] Mode: {'onefile' if onefile else 'onedir'}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[build-server] FAILED")
        sys.exit(1)

    if onefile:
        print(f"[build-server] Done -> dist/{name}.exe")
    else:
        print(f"[build-server] Done -> dist/{name}/")


if __name__ == "__main__":
    onefile = "--onefile" in sys.argv
    build(onefile=onefile)
