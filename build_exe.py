"""打包 Vision Agent 为 Windows 可执行文件。

用法:
    1. 创建 venv 并安装依赖:
       python -m venv .venv
       .venv\Scripts\activate
       pip install -r requirements.txt
       pip install pyinstaller

    2. 运行打包:
       python build_exe.py

    3. 产出在 dist/VisionAgent/ 目录下
"""

import PyInstaller.__main__
import os
import sys

# 获取项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))

# ultralytics 需要打包其数据文件
try:
    import ultralytics
    ultra_path = os.path.dirname(ultralytics.__file__)
    ultra_data = [(os.path.join(ultra_path, "cfg"), "ultralytics/cfg")]
except ImportError:
    ultra_data = []

# PySide6 插件路径
try:
    import PySide6
    pyside_path = os.path.dirname(PySide6.__file__)
except ImportError:
    pyside_path = ""

args = [
    os.path.join(ROOT, "gui_app.py"),
    "--name=VisionAgent",
    "--windowed",                    # GUI 程序，不显示控制台
    "--noconfirm",                   # 覆盖已有输出
    f"--icon={os.path.join(ROOT, 'assets', 'icon.ico')}" if os.path.exists(os.path.join(ROOT, "assets", "icon.ico")) else "--icon=NONE",

    # 收集整个 vision_agent 包
    "--collect-all=vision_agent",

    # 关键依赖的隐式导入
    "--hidden-import=ultralytics",
    "--hidden-import=ultralytics.nn",
    "--hidden-import=ultralytics.engine",
    "--hidden-import=ultralytics.models",
    "--hidden-import=ultralytics.utils",
    "--hidden-import=ultralytics.cfg",
    "--hidden-import=torch",
    "--hidden-import=torch.nn",
    "--hidden-import=torchvision",
    "--hidden-import=sklearn",
    "--hidden-import=sklearn.ensemble",
    "--hidden-import=joblib",
    "--hidden-import=cv2",
    "--hidden-import=numpy",
    "--hidden-import=yaml",
    "--hidden-import=mss",
    "--hidden-import=pynput",
    "--hidden-import=pynput.keyboard",
    "--hidden-import=pynput.mouse",
    "--hidden-import=websockets",
    "--hidden-import=PySide6.QtCore",
    "--hidden-import=PySide6.QtGui",
    "--hidden-import=PySide6.QtWidgets",

    # 附加数据
    f"--add-data={os.path.join(ROOT, 'config.yaml')};.",
    f"--add-data={os.path.join(ROOT, 'profiles')};profiles",
]

# ultralytics 配置文件
for src, dst in ultra_data:
    args.append(f"--add-data={src};{dst}")

# 排除不需要的大型模块（减小体积）
excludes = [
    "matplotlib", "tkinter", "IPython", "jupyter",
    "notebook", "pytest", "sphinx", "setuptools",
]
for ex in excludes:
    args.append(f"--exclude-module={ex}")

print("=" * 60)
print("  Vision Agent - 打包为 Windows EXE")
print("=" * 60)
print(f"入口: gui_app.py")
print(f"输出: dist/VisionAgent/")
print()

PyInstaller.__main__.run(args)

print()
print("=" * 60)
print("  打包完成！")
print(f"  运行: dist\\VisionAgent\\VisionAgent.exe")
print("=" * 60)
