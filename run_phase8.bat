@echo off
REM DocuAsk Phase 8: rebuild chunks + index after the cleaning fix, run all tests.
cd /d "%~dp0"
set LOG=phase8_log.txt
set VPY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_PROGRESS_BARS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_OFFLINE=1
echo Phase 8 started %DATE% %TIME% > %LOG%

echo [1/3] Re-chunking the documents...
echo ===== INGEST ===== >> %LOG%
%VPY% scripts\ingest.py >> %LOG% 2>&1

echo [2/3] Rebuilding the e5 index used by the app...
echo. >> %LOG%
echo ===== INDEX ===== >> %LOG%
%VPY% scripts\build_index.py --quiet >> %LOG% 2>&1

echo [3/3] Running tests...
echo. >> %LOG%
echo ===== PYTEST ===== >> %LOG%
%VPY% -m pytest -q -rs >> %LOG% 2>&1

echo. >> %LOG%
echo Finished %DATE% %TIME% >> %LOG%
echo.
echo Done. Results are in phase8_log.txt.
pause
