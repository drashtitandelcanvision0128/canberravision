@echo off
echo ========================================
echo       YOLO26 Video Detection
echo ========================================
echo.
echo Starting YOLO26 Object Detection Application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import gradio, cv2, torch, ultralytics" >nul 2>&1
if errorlevel 1 (
    echo Installing missing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Check if model files exist
if not exist "models\yolo26n.pt" (
    echo ERROR: yolo26n.pt model file not found
    echo Please ensure YOLO26 model files are in the models directory
    pause
    exit /b 1
)

echo.
echo ✅ All checks passed!
echo.
echo Starting web application...
echo The application will open in your default browser
echo.
echo URL: http://127.0.0.1:7866
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the application
python apps\app.py

echo.
echo Application stopped.
pause
