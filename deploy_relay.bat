@echo off
chcp 65001 >nul
title Vision Agent Relay Server

echo ============================================
echo   Vision Agent 公网中继服务
echo ============================================
echo.

:: 配置区 - 按需修改
set PORT=9877
set TOKEN=vision2026

echo   端口: %PORT%
echo   Token: %TOKEN%
echo.
echo   客户端连接方式:
echo   中继地址: ws://本机公网IP:%PORT%
echo   房间号: 自行约定（如 myroom）
echo   Token: %TOKEN%
echo.
echo ============================================
echo.

:: 优先尝试 EXE
if exist RelayServer.exe (
    echo [启动] 使用 RelayServer.exe
    RelayServer.exe --port %PORT% --token %TOKEN%
    goto :end
)

:: 其次尝试 Python
where python >nul 2>&1
if %errorlevel%==0 (
    echo [启动] 使用 Python
    :: 检查 websockets
    python -c "import websockets" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [安装] pip install websockets ...
        pip install websockets
    )
    python relay_server.py --port %PORT% --token %TOKEN%
    goto :end
)

echo [错误] 未找到 RelayServer.exe 或 Python
echo.
echo 请选择以下方式之一:
echo   1. 从 GitHub Releases 下载 RelayServer.exe 放到本目录
echo   2. 安装 Python: https://python.org/downloads
echo.

:end
pause
