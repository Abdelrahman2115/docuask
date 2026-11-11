@echo off
REM DocuAsk Phase 7: tests + real interactive latency benchmark. Output in phase7_log.txt.
cd /d "%~dp0"
set LOG=phase7_log.txt
set VPY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_PROGRESS_BARS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_OFFLINE=1
echo Phase 7 started %DATE% %TIME% > %LOG%

echo [1/2] Running tests...
echo ===== PYTEST ===== >> %LOG%
%VPY% -m pytest -q -rs >> %LOG% 2>&1

echo [2/2] Measuring app latency on 8 questions (about 3 minutes)...
echo. >> %LOG%
echo ===== LATENCY BENCHMARK ===== >> %LOG%
%VPY% scripts\benchmark_latency.py >> %LOG% 2>&1

echo. >> %LOG%
echo Finished %DATE% %TIME% >> %LOG%
echo.
echo Done. Results are in phase7_log.txt. Now double-click start_app.bat to open the app.
pause
