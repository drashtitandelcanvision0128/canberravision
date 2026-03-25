@echo off
echo ========================================
echo   YOLO26 Smart Parking Detection System
echo ========================================
echo.
echo Starting improved parking detection system with:
echo - Green boxes for occupied slots
echo - Red boxes for empty slots  
echo - Larger bounding boxes
echo - Better label positioning
echo - JSON slot counting output
echo.
echo [INFO] Fixed permission issues and file handling
echo [INFO] Enhanced visualization applied
echo.

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Start the application
echo [INFO] Starting YOLO26 Parking Detection App...
python apps\app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application failed to start
    echo [INFO] Check the error message above for details
    pause
)

echo.
echo [INFO] Application stopped
pause
