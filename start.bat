@echo off
rem One-click Audio Dedup launcher. Opens the browser after a short delay so the server is up first.
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "HF_HOME=%~dp0.hf-cache"
start /b cmd /c "timeout /t 2 /nobreak >nul & start "" http://127.0.0.1:7861"
".venv\Scripts\python.exe" server.py
pause
