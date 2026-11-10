@echo off
REM DocuAsk Phase 2: download documents, ingest + chunk, run tests. Output in phase2_log.txt.
cd /d "%~dp0"
set LOG=phase2_log.txt
set VPY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
echo Phase 2 started %DATE% %TIME% > %LOG%

echo [1/3] Downloading 40 FCC documents (about 1 minute)...
echo ===== DOWNLOAD ===== >> %LOG%
%VPY% scripts\download_docs.py >> %LOG% 2>&1

echo [2/3] Cleaning and chunking...
echo. >> %LOG%
echo ===== INGEST ===== >> %LOG%
%VPY% scripts\ingest.py >> %LOG% 2>&1

echo [3/3] Running tests...
echo. >> %LOG%
echo ===== PYTEST ===== >> %LOG%
%VPY% -m pytest -q >> %LOG% 2>&1

echo. >> %LOG%
echo Finished %DATE% %TIME% >> %LOG%
echo.
echo Done. Results are in phase2_log.txt. You can close this window.
pause
