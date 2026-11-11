@echo off
REM Deploy DocuAsk to Hugging Face Spaces. Needs HF_TOKEN (write) and GROQ_API_KEY in .env.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\deploy_space.py > deploy_log.txt 2>&1
type deploy_log.txt
echo.
echo Output saved in deploy_log.txt.
pause
