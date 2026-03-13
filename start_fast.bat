@echo off
echo 🚀 Starting Fast Video Processing App...
echo.

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo 📦 Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo ⚠️ Virtual environment not found, using system Python
)

REM Check CUDA availability
echo 🔍 Checking CUDA availability...
python check_cuda.py

echo.
echo 🎬 Starting Fast Video Processing App...
echo 💡 Tips for maximum speed:
echo    - Use "ultra_fast" mode for quick previews
echo    - Use "fast" mode for normal processing  
echo    - Skip frames (2-3) for 2-3x speedup
echo    - Smaller image size (320px) for faster processing
echo.

REM Start the optimized app
python apps\app_gpu.py

pause
