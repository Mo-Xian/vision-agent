@echo off
chcp 65001 >nul 2>&1
title Vision Agent
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv venv
    if errorlevel 1 (
        echo Failed to create venv. Please install Python 3.10+
        pause
        exit /b 1
    )
    echo Installing dependencies...
    venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
    echo Done.
)

start "" venv\Scripts\pythonw.exe gui_app.py
