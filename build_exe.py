"""PyInstaller 打包脚本 — 生成 VisionAgent.exe。

用法:
    python build_exe.py          # 正常打包
    python build_exe.py --debug  # 带控制台的调试版
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def find_ultralytics_cfg() -> str:
    """查找 ultralytics/cfg 目录（PyInstaller 需要）。"""
    try:
        import ultralytics
        cfg = Path(ultralytics.__file__).parent / "cfg"
        if cfg.exists():
            return str(cfg)
    except ImportError:
        pass
    return ""


def build(debug: bool = False):
    entry = str(ROOT / "gui_app.py")
    name = "VisionAgent_debug" if debug else "VisionAgent"

    # 数据文件
    datas = [
        (str(ROOT / "config.yaml"), "."),
        (str(ROOT / "profiles"), "profiles"),
    ]
    cfg = find_ultralytics_cfg()
    if cfg:
        datas.append((cfg, "ultralytics/cfg"))

    # 隐式导入
    hidden = [
        "ultralytics", "ultralytics.nn", "ultralytics.engine",
        "ultralytics.models", "ultralytics.utils", "ultralytics.cfg",
        "torch", "torch.nn", "torchvision",
        "sklearn", "sklearn.ensemble", "joblib",
        "cv2", "numpy", "yaml", "mss",
        "pynput", "pynput.keyboard", "pynput.mouse",
        "websockets",
        "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
    ]

    # 排除不需要的包
    excludes = [
        "matplotlib", "tkinter", "IPython", "jupyter",
        "notebook", "pytest", "sphinx", "setuptools",
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--name", name,
        "--collect-all", "vision_agent",
    ]

    if debug:
        cmd.append("--console")
    else:
        cmd.append("--windowed")

    for src, dst in datas:
        cmd.extend(["--add-data", f"{src}{';' if sys.platform == 'win32' else ':'}{dst}"])

    for h in hidden:
        cmd.extend(["--hidden-import", h])

    for e in excludes:
        cmd.extend(["--exclude-module", e])

    cmd.append(entry)

    print(f"[build] {'Debug' if debug else 'Release'} build...")
    print(f"[build] Entry: {entry}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[build] FAILED")
        sys.exit(1)
    print(f"[build] Done -> dist/{name}/")


if __name__ == "__main__":
    debug = "--debug" in sys.argv
    build(debug=debug)
