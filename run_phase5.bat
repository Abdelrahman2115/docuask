@echo off
REM DocuAsk Phase 5b: probe the RAGAS judge, then score the baseline run with RAGAS.
cd /d "%~dp0"
set LOG=phase5b_log.txt
set VPY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_PROGRESS_BARS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_OFFLINE=1
echo Phase 5b started %DATE% %TIME% > %LOG%

echo [1/3] Testing RAGAS judges on a toy example...
echo ===== PROBE ===== >> %LOG%
%VPY% scripts\probe_eval.py >> %LOG% 2>&1

echo [2/3] Scoring 29 baseline answers with RAGAS (about 20-30 minutes, rate-limited)...
echo. >> %LOG%
echo ===== RAGAS BASELINE ===== >> %LOG%
%VPY% scripts\ragas_eval.py --name baseline >> %LOG% 2>&1

echo [3/3] Running tests...
echo. >> %LOG%
echo ===== PYTEST ===== >> %LOG%
%VPY% -m pytest -q -rs >> %LOG% 2>&1

echo. >> %LOG%
echo Finished %DATE% %TIME% >> %LOG%
echo.
echo Done. Results are in phase5b_log.txt. You can close this window.
pause
