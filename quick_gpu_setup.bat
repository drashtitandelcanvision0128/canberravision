@echo off
echo ==========================================
echo QUICK GPU SETUP FOR RTX 4050
echo ==========================================
echo.

echo Your GPU: NVIDIA GeForce RTX 4050 Laptop GPU
echo Status: Drivers not installed
echo.

echo Step 1: Installing NVIDIA Driver...
echo Opening download page...
start "" "https://www.nvidia.com/download/driverResults.aspx/207447/en-us/"

echo.
echo PLEASE FOLLOW THESE STEPS:
echo 1. Download the driver from browser
echo 2. Run the installer
echo 3. Choose "Express Installation"
echo 4. Restart PC when asked
echo.

echo After restart, run this file again...
echo It will automatically detect GPU and setup everything!

pause

echo Checking if NVIDIA driver is installed...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ NVIDIA driver detected!
    echo Installing remaining components...
    
    echo Installing PyTorch GPU...
    py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    echo Installing other requirements...
    py -3.11 -m pip install gradio==6.3.0 ultralytics opencv-python numpy pillow scipy scikit-learn pytesseract transformers timm
    
    echo Verifying GPU setup...
    py -3.11 -c "import torch; print(f'✅ CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
    
    echo.
    echo 🎉 GPU SETUP COMPLETE!
    echo Starting your application on GPU...
    py -3.11 app.py
    
) else (
    echo ❌ NVIDIA driver not found
    echo Please install the driver first
    echo Then run this script again
)

pause
