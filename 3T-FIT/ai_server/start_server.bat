@echo off
echo Starting OmniMer Health AI Server...
echo.

REM Change to the 3T-FIT directory (parent of ai_server)
cd /d "%~dp0\.."

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Set PYTHONPATH to include the app directory
set PYTHONPATH=%cd%\ai_server\app

REM Run uvicorn from the 3T-FIT directory
echo Starting uvicorn server...
echo Running from: %cd%
uvicorn ai_server.app.main:app --host 0.0.0.0 --port 8888 --reload

pause
