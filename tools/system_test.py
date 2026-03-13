"""
Test Script for Car and License Plate Detection System
Quick verification that all components are working
"""

import os
import sys
import cv2
import numpy as np

def test_imports():
    """Test all required imports"""
    print("🔍 Testing Imports...")
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
    except ImportError:
        print("❌ Ultralytics not available")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError:
        print("❌ OpenCV not available")
        return False
    
    try:
        from car_plate_video_processor import CarPlateVideoProcessor
        print("✅ Car Plate Video Processor imported successfully")
    except ImportError as e:
        print(f"❌ Car Plate Video Processor import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test YOLO model loading"""
    print("\n🚗 Testing Model Loading...")
    
    try:
        from car_plate_video_processor import CarPlateVideoProcessor
        
        # Try to initialize processor
        processor = CarPlateVideoProcessor(model_path="yolo26n.pt", use_gpu=True)
        
        if processor.model:
            print("✅ YOLO model loaded successfully")
            print(f"   Model: {processor.model_path}")
            print(f"   Device: {processor.device}")
            return True
        else:
            print("❌ Model failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def check_model_files():
    """Check if YOLO model files exist"""
    print("\n📁 Checking Model Files...")
    
    model_files = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolov8s.pt"]
    found_models = []
    
    for model in model_files:
        if os.path.exists(model):
            found_models.append(model)
            print(f"✅ Found: {model}")
        else:
            print(f"❌ Missing: {model}")
    
    if found_models:
        print(f"✅ Found {len(found_models)} model files")
        return True
    else:
        print("❌ No model files found")
        return False

def main():
    """Run basic tests"""
    print("🧪 Car & License Plate Detection System Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Model Files", check_model_files),
        ("Model Loading", test_model_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n📖 Next Steps:")
        print("1. Run 'python demo_car_plate_detection.py' for interactive demo")
        print("2. Run 'python gradio_car_plate_app.py' for web interface")
        print("3. Use the processor in your own code")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print("\n🔧 Common fixes:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Download YOLO model files")
        print("- Check GPU drivers for CUDA support")

if __name__ == "__main__":
    main()
