@echo off
chcp 65001 >nul 2>&1
title Vision Agent - Debug
cd /d "%~dp0"

echo Starting Vision Agent (debug mode)...
echo.
venv\Scripts\python.exe gui_app.py
echo.
echo Exit code: %errorlevel%
pause
