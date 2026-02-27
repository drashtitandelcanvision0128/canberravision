#!/usr/bin/env python3
"""
🎨 COLOR DETECTION TEST SCRIPT
Test advanced color detection functionality in YOLO26
"""

import cv2
import numpy as np
import time
from modules.utils import _classify_color_bgr

def test_color_detection():
    """Test color detection with sample colors"""
    print("🎨 YOLO26 COLOR DETECTION TEST")
    print("=" * 50)
    
    # Test colors with their BGR values
    test_colors = {
        'Red': (0, 0, 255),
        'Blue': (255, 0, 0),
        'Green': (0, 255, 0),
        'Yellow': (0, 255, 255),
        'Orange': (0, 165, 255),
        'Purple': (255, 0, 255),
        'Black': (0, 0, 0),
        'White': (255, 255, 255),
        'Gray': (128, 128, 128),
        'Brown': (42, 42, 165),
        'Pink': (203, 192, 255),
        'Cyan': (255, 255, 0)
    }
    
    print("Testing color detection with sample colors...")
    print()
    
    correct_detections = 0
    total_tests = len(test_colors)
    
    for color_name, bgr_value in test_colors.items():
        # Create a test image with the color
        test_image = np.full((100, 100, 3), bgr_value, dtype=np.uint8)
        
        # Add some noise to make it realistic
        noise = np.random.randint(-20, 20, test_image.shape, dtype=np.int16)
        test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Test color detection
        start_time = time.time()
        detected_color = _classify_color_bgr(test_image)
        detection_time = time.time() - start_time
        
        # Check if detection is correct (allowing for some variations)
        is_correct = False
        if color_name.lower() in detected_color.lower():
            is_correct = True
        elif detected_color.lower() in color_name.lower():
            is_correct = True
        
        if is_correct:
            correct_detections += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} {color_name:8} -> {detected_color:15} ({detection_time*1000:.1f}ms)")
    
    print()
    print(f"🎯 Color Detection Results:")
    print(f"   ✅ Correct: {correct_detections}/{total_tests}")
    print(f"   📊 Accuracy: {(correct_detections/total_tests)*100:.1f}%")
    
    if correct_detections >= total_tests * 0.8:
        print("   🚀 Color detection is working excellently!")
    else:
        print("   ⚠️ Color detection needs improvement")
    
    return correct_detections >= total_tests * 0.8

if __name__ == "__main__":
    print("🚀 Starting YOLO26 Color Detection Test...")
    print()
    
    # Test basic color detection
    success = test_color_detection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 COLOR DETECTION TEST PASSED!")
        print("✅ Ready for enhanced video processing with color detection!")
    else:
        print("⚠️ COLOR DETECTION TEST NEEDS IMPROVEMENT")
        print("🔧 Check color detection algorithms")
    
    print("\n💡 Next Steps:")
    print("   1. Run the main app: python app.py")
    print("   2. Upload a video to test color detection in action")
    print("   3. Check the enhanced labels with color information")
    print("   4. Review the detection summary with color analytics")
