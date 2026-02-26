"""
GPU PaddleOCR Setup Script
Automatically detects and installs the appropriate PaddleOCR version for your system.
"""

import subprocess
import sys
import os
import platform

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_system_info():
    """Get system information"""
    info = {
        'platform': platform.system(),
        'python_version': sys.version,
        'cuda_available': check_cuda()
    }
    
    if info['cuda_available']:
        try:
            import torch
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        except:
            pass
    
    return info

def install_paddleocr_gpu():
    """Install GPU-enabled PaddleOCR"""
    print("🚀 Installing GPU-enabled PaddleOCR...")
    
    try:
        # Install paddlepaddle-gpu
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "paddlepaddle-gpu==2.6.2"
        ])
        
        # Install paddleocr
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "paddleocr>=3.4.0"
        ])
        
        print("✅ GPU-enabled PaddleOCR installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install GPU PaddleOCR: {e}")
        return False

def install_paddleocr_cpu():
    """Install CPU-only PaddleOCR as fallback"""
    print("💻 Installing CPU-only PaddleOCR...")
    
    try:
        # Uninstall GPU version if exists
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", 
            "paddlepaddle-gpu", "-y"
        ], check=False)
        
        # Install CPU version
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "paddlepaddle==2.6.2"
        ])
        
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "paddleocr>=3.4.0"
        ])
        
        print("✅ CPU-only PaddleOCR installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install CPU PaddleOCR: {e}")
        return False

def test_paddleocr():
    """Test PaddleOCR installation"""
    print("🧪 Testing PaddleOCR installation...")
    
    try:
        from paddleocr import PaddleOCR
        import cv2
        import numpy as np
        
        # Create test image
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "TEST123", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Initialize PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        # Test OCR
        result = ocr.ocr(test_image, cls=True)
        
        if result and len(result) > 0 and result[0]:
            print("✅ PaddleOCR test successful!")
            return True
        else:
            print("⚠️ PaddleOCR installed but test returned empty results")
            return False
            
    except Exception as e:
        print(f"❌ PaddleOCR test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🔧 YOLO26 PaddleOCR GPU Setup")
    print("=" * 50)
    
    # Get system information
    info = get_system_info()
    print(f"📋 System Information:")
    print(f"   Platform: {info['platform']}")
    print(f"   Python: {info['python_version'].split()[0]}")
    print(f"   CUDA Available: {'Yes' if info['cuda_available'] else 'No'}")
    
    if info['cuda_available']:
        print(f"   CUDA Version: {info.get('cuda_version', 'Unknown')}")
        print(f"   GPU: {info.get('gpu_name', 'Unknown')}")
        print(f"   GPU Memory: {info.get('gpu_memory', 'Unknown'):.1f} GB")
    
    print()
    
    # Choose installation method
    if info['cuda_available']:
        print("🎯 CUDA GPU detected - Installing GPU-enabled PaddleOCR")
        success = install_paddleocr_gpu()
        
        if not success:
            print("⚠️ GPU installation failed, trying CPU fallback...")
            success = install_paddleocr_cpu()
    else:
        print("💻 No CUDA GPU detected - Installing CPU-only PaddleOCR")
        success = install_paddleocr_cpu()
    
    if success:
        print("\n🧪 Testing installation...")
        test_success = test_paddleocr()
        
        if test_success:
            print("\n🎉 Setup completed successfully!")
            print("\n📖 Next steps:")
            print("   1. Run: python app.py")
            print("   2. Open your browser to the provided URL")
            print("   3. Upload an image to test object detection + text extraction")
            
            if info['cuda_available']:
                print("   4. 🚀 GPU acceleration is active for maximum speed!")
            else:
                print("   4. 💻 Consider installing CUDA for GPU acceleration")
        else:
            print("\n⚠️ Installation completed but test failed")
            print("   Please check the error messages above")
    else:
        print("\n❌ Setup failed!")
        print("   Please check the error messages and try manually installing:")
        print("   pip install paddlepaddle-gpu==2.6.2 paddleocr>=3.4.0")

if __name__ == "__main__":
    main()
