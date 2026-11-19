@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app\server.py
) else (
    python app\server.py
)
pause
