@echo off
REM OPTIONAL: only if Docker Desktop is installed. Builds the image and runs it at http://localhost:7860
cd /d "%~dp0"
docker build -t docuask . || goto end
docker run --rm -p 7860:7860 --env-file .env docuask
:end
pause
