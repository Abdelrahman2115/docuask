@echo off
REM DocuAsk Phase 4: demo questions through retrieval + LLM, then tests. Output in phase4_log.txt.
cd /d "%~dp0"
set LOG=phase4_log.txt
set VPY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_PROGRESS_BARS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_OFFLINE=1
echo Phase 4 started %DATE% %TIME% > %LOG%

echo [1/2] Asking 7 demo questions (about 1 minute; calls are rate-limited)...
echo ===== DEMO ===== >> %LOG%
%VPY% scripts\ask.py --demo --show-passages >> %LOG% 2>&1

echo [2/2] Running tests...
echo. >> %LOG%
echo ===== PYTEST ===== >> %LOG%
%VPY% -m pytest -q -rs >> %LOG% 2>&1

echo. >> %LOG%
echo Finished %DATE% %TIME% >> %LOG%
echo.
echo Done. Results are in phase4_log.txt. You can close this window.
pause
