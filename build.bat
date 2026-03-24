@echo off
chcp 65001 >nul
echo ============================================================
echo   Vision Agent - 一键打包 EXE
echo ============================================================
echo.

:: 检查 Python
where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

:: 创建虚拟环境
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] 创建虚拟环境...
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 创建 venv 失败
        pause
        exit /b 1
    )
) else (
    echo [1/4] 虚拟环境已存在
)

:: 激活并安装依赖
echo [2/4] 安装依赖（首次较慢）...
call .venv\Scripts\activate.bat
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)

:: 安装 PyInstaller
pip install pyinstaller -q
if errorlevel 1 (
    echo [错误] PyInstaller 安装失败
    pause
    exit /b 1
)

:: 打包
echo [3/4] 开始打包...
python build_exe.py
if errorlevel 1 (
    echo [错误] 打包失败
    pause
    exit /b 1
)

echo.
echo [4/4] 打包完成!
echo.
echo   输出目录: dist\VisionAgent\
echo   运行文件: dist\VisionAgent\VisionAgent.exe
echo.

:: 复制配置文件
if not exist "dist\VisionAgent\profiles" mkdir "dist\VisionAgent\profiles"
copy /Y profiles\*.yaml "dist\VisionAgent\profiles\" >nul 2>&1
copy /Y config.yaml "dist\VisionAgent\" >nul 2>&1
echo   已复制配置文件到输出目录

echo.
pause
