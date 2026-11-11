@echo off
REM DocuAsk Phase 6: retrieval experiments (no LLM calls). Output in phase6_log.txt.
REM First run downloads ~1.2 GB of models (3 embedding models + 1 reranker).
cd /d "%~dp0"
set LOG=phase6_log.txt
set VPY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_PROGRESS_BARS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TOKENIZERS_PARALLELISM=false
echo Phase 6 started %DATE% %TIME% > %LOG%

echo [1/2] Running 11 retrieval experiments (15-30 minutes on first run: model downloads + indexing)...
echo ===== EXPERIMENTS ===== >> %LOG%
%VPY% scripts\run_experiments.py >> %LOG% 2>&1

echo [2/2] Running tests...
echo. >> %LOG%
echo ===== PYTEST ===== >> %LOG%
%VPY% -m pytest -q -rs >> %LOG% 2>&1

echo. >> %LOG%
echo Finished %DATE% %TIME% >> %LOG%
echo.
echo Done. Results are in phase6_log.txt. You can close this window.
pause
