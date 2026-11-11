@echo off
REM Start the DocuAsk chat app at http://127.0.0.1:7860  (close this window to stop it)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_PROGRESS_BARS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_OFFLINE=1
echo Starting DocuAsk... the first start takes about 10-20 seconds.
.venv\Scripts\python.exe app\app.py
pause
