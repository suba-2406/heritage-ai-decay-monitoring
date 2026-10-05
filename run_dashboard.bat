@echo off
echo =====================================================================
echo  Starting Heritage AI Monument Decay Monitoring Server...
echo =====================================================================

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo [OK] Using virtual environment Python...
    ".venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8008 --reload
) else (
    echo [OK] Using system Python...
    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8008 --reload
)

pause
