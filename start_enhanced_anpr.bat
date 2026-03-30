@echo off
echo ========================================
echo    Enhanced ANPR System Startup
echo ========================================
echo.

echo [INFO] Starting Enhanced ANPR System...
echo [INFO] This system provides comprehensive vehicle detection and analysis
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo [INFO] Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if required modules are available
echo [INFO] Checking dependencies...
python -c "import torch, cv2, gradio, ultralytics" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Some dependencies may be missing
    echo [INFO] Installing required packages...
    pip install torch torchvision ultralytics opencv-python gradio scikit-learn
)

REM Create necessary directories
if not exist "database" mkdir database
if not exist "logs" mkdir logs
if not exist "outputs" mkdir outputs

echo [INFO] Dependencies checked
echo [INFO] Starting ANPR system on http://localhost:7865
echo.

REM Run the enhanced ANPR system
python apps/enhanced_anpr_system.py

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start ANPR system
    echo [INFO] Please check the error messages above
    pause
)

echo.
echo [INFO] ANPR system stopped
pause
