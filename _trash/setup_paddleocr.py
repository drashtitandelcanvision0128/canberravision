#!/usr/bin/env python3
"""
Setup script for PaddleOCR with GPU support for RTX 4050
Follow the installation steps from the user's instructions
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED!")
        print("Error:", e.stderr)
        return False

def main():
    print("🚀 Setting up PaddleOCR with GPU support for RTX 4050")
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("⚠️  WARNING: Not in a virtual environment!")
        print("Please run: python -m venv alpr_env && alpr_env\\Scripts\\activate")
        return False
    
    # Installation steps
    steps = [
        ("pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu124/", 
         "Install PaddlePaddle GPU for CUDA 12.4"),
        ("pip install \"paddleocr>=3.4.0\"", 
         "Install PaddleOCR toolkit with PP-OCRv5"),
        ("pip install ultralytics[export]", 
         "Install YOLO support with export capabilities"),
        ("pip install nvidia-tensorrt", 
         "Install TensorRT for RTX 4050 acceleration")
    ]
    
    success_count = 0
    for command, description in steps:
        if run_command(command, description):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"INSTALLATION SUMMARY: {success_count}/{len(steps)} steps completed")
    print(f"{'='*50}")
    
    if success_count == len(steps):
        print("\n🎉 All installations completed successfully!")
        print("\nNow let's verify the installation...")
        verify_command = "python -c \"import torch; import paddle; print('PyTorch GPU:', torch.cuda.is_available()); print('Paddle GPU:', paddle.is_compiled_with_cuda())\""
        run_command(verify_command, "Verify GPU support")
        
        print("\n🧪 Testing PaddleOCR initialization...")
        test_command = "python -c \"from paddleocr import PaddleOCR; ocr = PaddleOCR(use_gpu=True, version='PP-OCRv5', lang='en'); print('✅ PP-OCRv5 is ready for RTX 4050')\""
        run_command(test_command, "Test PaddleOCR initialization")
        
        return True
    else:
        print("\n❌ Some installations failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    main()
