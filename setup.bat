@echo off
REM DocuAsk Phase 1 setup. Double-click to run. Output is saved to setup_log.txt.
cd /d "%~dp0"
set LOG=setup_log.txt
echo DocuAsk setup started %DATE% %TIME% > %LOG%

where py >nul 2>&1 && (set PY=py -3.13) || (set PY=python)
echo Using: %PY% >> %LOG%
%PY% --version >> %LOG% 2>&1

echo [1/6] Creating virtual environment...
if not exist .venv (%PY% -m venv .venv >> %LOG% 2>&1)
if not exist .venv\Scripts\python.exe (echo FAILED: could not create .venv >> %LOG% & goto end)
set VPY=.venv\Scripts\python.exe

echo [2/6] Upgrading pip...
%VPY% -m pip install --upgrade pip >> %LOG% 2>&1

echo [3/6] Installing packages (about 1 GB, this takes a while)...
%VPY% -m pip install -r requirements-dev.txt >> %LOG% 2>&1
if errorlevel 1 (echo FAILED: pip install requirements >> %LOG% & goto end)
%VPY% -m pip install -e . >> %LOG% 2>&1

echo [4/6] Creating .env from template if missing...
if not exist .env (copy .env.example .env >> %LOG% 2>&1)

echo [5/6] Running tests...
echo. >> %LOG%
echo ===== PYTEST ===== >> %LOG%
%VPY% -m pytest -q >> %LOG% 2>&1

echo [6/6] Running setup check...
echo. >> %LOG%
echo ===== CHECK_SETUP ===== >> %LOG%
%VPY% scripts\check_setup.py >> %LOG% 2>&1

%VPY% -m pip freeze --exclude-editable > requirements.lock.txt

:end
echo. >> %LOG%
echo Finished %DATE% %TIME% >> %LOG%
echo.
echo Done. Results are in setup_log.txt. You can close this window.
pause
