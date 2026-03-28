@echo off
chcp 65001 >nul
echo ============================================
echo   CANBERRA VISION - UNIFIED DETECTION
echo ============================================
echo.
echo Starting Unified Detection Web Application...
echo.

REM Change to project root
cd /d "%~dp0.."

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    pause
    exit /b 1
)

REM Activate virtual environment if exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [INFO] Virtual environment activated
) else if exist ".venv_gpu\Scripts\activate.bat" (
    call .venv_gpu\Scripts\activate.bat
    echo [INFO] GPU Virtual environment activated
)

REM Set environment variables
set PYTHONPATH=%CD%
set CUDA_VISIBLE_DEVICES=0
set OPENCV_VIDEOIO_PRIORITY_MSMF=0
set OPENCV_VIDEOIO_PRIORITY_DSHOW=0

echo.
echo [INFO] Launching Unified Detection App...
echo [INFO] URL: http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the app
python apps\unified_detection_app.py

pause
