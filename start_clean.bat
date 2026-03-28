@echo off
echo 🧹 Cleaning up temp files and starting app...
echo.

REM Kill any existing Python processes
taskkill /f /im python.exe 2>nul
timeout /t 2 >nul

REM Clean temp directories
echo Cleaning temp files...
if exist "%TEMP%\gradio" (
    echo Deleting Gradio temp files...
    rmdir /s /q "%TEMP%\gradio" 2>nul
)

if exist "inputs" (
    echo Cleaning inputs folder...
    del /q "inputs\*.mp4" 2>nul
    del /q "inputs\*.avi" 2>nul
    del /q "inputs\*.mov" 2>nul
)

if exist "outputs" (
    echo Cleaning outputs folder...
    del /q "outputs\*.mp4" 2>nul
    del /q "outputs\*.avi" 2>nul
    del /q "outputs\*.mov" 2>nul
)

REM Create fresh directories
echo Creating fresh directories...
if not exist "inputs" mkdir inputs
if not exist "outputs" mkdir outputs
if not exist "processed_images" mkdir processed_images
if not exist "processed_videos" mkdir processed_videos

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo ⚠️ Virtual environment not found, using system Python
)

REM Check CUDA
echo.
echo 🔍 Checking CUDA availability...
python check_cuda.py

echo.
echo 🚀 Starting YOLO26 App with Smart Video Processing...
echo 💡 Features:
echo    - Automatic optimization for blurry videos
echo    - Fast processing for lengthy videos  
echo    - CUDA GPU acceleration enabled
echo    - Smart memory management
echo.

REM Start the app
python apps\app.py

pause
