@echo off
rem 一键启动 Audio Dedup 服务器，并延迟打开浏览器（避免先弹浏览器连不上）
cd /d F:\whisper
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HOME=F:\whisper\.hf-cache
start /b cmd /c "timeout /t 2 /nobreak >nul & start "" http://127.0.0.1:7861"
".venv\Scripts\python.exe" server.py
pause
