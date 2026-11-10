@echo off
REM DocuAsk Phase 3: build the vector index, run a sample search, run tests. Output in phase3_log.txt.
cd /d "%~dp0"
set LOG=phase3_log.txt
set VPY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_PROGRESS_BARS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TOKENIZERS_PARALLELISM=false
echo Phase 3 started %DATE% %TIME% > %LOG%

echo [1/3] Embedding 305 chunks and building the Chroma index (first run downloads ~90 MB)...
echo ===== BUILD INDEX ===== >> %LOG%
%VPY% scripts\build_index.py >> %LOG% 2>&1

echo [2/3] Sample search...
echo. >> %LOG%
echo ===== SEARCH ===== >> %LOG%
%VPY% scripts\search.py "my bill has charges for services I never ordered" >> %LOG% 2>&1

echo [3/3] Running tests...
echo. >> %LOG%
echo ===== PYTEST ===== >> %LOG%
%VPY% -m pytest -q -rs >> %LOG% 2>&1

echo. >> %LOG%
echo Finished %DATE% %TIME% >> %LOG%
echo.
echo Done. Results are in phase3_log.txt. You can close this window.
pause
