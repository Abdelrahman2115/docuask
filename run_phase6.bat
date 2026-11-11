@echo off
REM DocuAsk Phase 6b: two more experiments, pick the winner by a fixed rule,
REM then a full generation + RAGAS run on it and a comparison with the baseline.
cd /d "%~dp0"
set LOG=phase6b_log.txt
set VPY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_PROGRESS_BARS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TOKENIZERS_PARALLELISM=false
echo Phase 6b started %DATE% %TIME% > %LOG%

echo [1/5] Two more retrieval experiments (e5 + hybrid, e5 + rerank)...
echo ===== EXPERIMENTS ===== >> %LOG%
%VPY% scripts\run_experiments.py --only hybrid_e5 rerank_e5 >> %LOG% 2>&1

echo [2/5] Picking the final configuration...
echo. >> %LOG%
echo ===== PICK BEST ===== >> %LOG%
%VPY% scripts\pick_best.py >> %LOG% 2>&1

echo [3/5] Answering 40 questions with the final configuration (about 10 minutes)...
echo. >> %LOG%
echo ===== FINAL RUN ===== >> %LOG%
%VPY% scripts\run_eval.py --name final --from-choice --rpm 6 >> %LOG% 2>&1

echo [4/5] RAGAS scoring (about 15-25 minutes)...
echo. >> %LOG%
echo ===== RAGAS FINAL ===== >> %LOG%
%VPY% scripts\ragas_eval.py --name final >> %LOG% 2>&1

echo [5/5] Comparing with the baseline and running tests...
echo. >> %LOG%
echo ===== COMPARISON ===== >> %LOG%
%VPY% scripts\compare_runs.py baseline final >> %LOG% 2>&1
echo. >> %LOG%
echo ===== PYTEST ===== >> %LOG%
%VPY% -m pytest -q -rs >> %LOG% 2>&1

echo. >> %LOG%
echo Finished %DATE% %TIME% >> %LOG%
echo.
echo Done. Results are in phase6b_log.txt. You can close this window.
pause
