"""
Test Japanese License Plate Detection
Specially designed to detect Japanese license plates like "日本 337 BR45IL"
"""

import cv2
import numpy as np
import os
from enhanced_plate_detector import EnhancedPlateDetector

def test_japanese_plate_detection():
    """Test Japanese plate detection with sample video"""
    
    print("🇯🇵 Testing Japanese License Plate Detection")
    print("=" * 50)
    
    # Find a video file
    video_files = [f for f in os.listdir('.') if f.lower().endswith('.mp4')]
    
    if not video_files:
        print("❌ No video files found")
        return
    
    video_path = video_files[0]
    print(f"🎬 Testing with video: {video_path}")
    
    # Initialize enhanced detector
    detector = EnhancedPlateDetector()
    
    # Test single frame extraction
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return
    
    # Read a few frames to find one with cars
    frame_count = 0
    test_frames = []
    
    while frame_count < 100:  # Check first 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        test_frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    print(f"📹 Extracted {len(test_frames)} frames for testing")
    
    # Test each frame
    for i, frame in enumerate(test_frames[::10]):  # Test every 10th frame
        print(f"\n🔍 Testing frame {i*10}:")
        
        # Convert to grayscale for plate detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for plate-like rectangles
        plate_candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # License plates typically have aspect ratio between 2:1 and 5:1
            if 1.5 < aspect_ratio < 6.0 and w > 60 and h > 20:
                plate_candidates.append((x, y, w, h))
        
        print(f"   Found {len(plate_candidates)} potential plate regions")
        
        # Test OCR on each candidate
        for j, (x, y, w, h) in enumerate(plate_candidates[:5]):  # Test top 5 candidates
            plate_region = frame[y:y+h, x:x+w]
            
            # Preprocess
            processed = detector.preprocess_plate_region(plate_region)
            
            # Extract text
            text_results = detector.extract_text_from_image(processed)
            
            print(f"   Region {j+1}: {len(text_results)} text results")
            
            for result in text_results:
                text = result['text']
                confidence = result['confidence']
                engine = result['engine']
                
                print(f"      Text: '{text}' (conf: {confidence:.2f}, engine: {engine})")
                
                if detector.is_license_plate(text):
                    print(f"      ✅ VALID LICENSE PLATE: {text}")

def create_japanese_plate_patterns():
    """Create and test Japanese plate patterns"""
    
    print("\n🇯🇵 Japanese Plate Pattern Testing")
    print("=" * 40)
    
    # Test Japanese plate patterns
    test_plates = [
        "日本 337 BR45IL",  # Your example
        "東京 123 あ4567",  # Tokyo plate
        "大阪 456 い8901",  # Osaka plate
        "品川 789 う2345",  # Shinagawa plate
        "横浜 321 え6789",  # Yokohama plate
        "123-456",         # Simple format
        "ABC 123",         # English format
        "日本1234",         # Japanese numbers only
    ]
    
    detector = EnhancedPlateDetector()
    
    print("Testing plate validation:")
    for plate in test_plates:
        is_valid = detector.is_license_plate(plate)
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"   '{plate}' -> {status}")

if __name__ == "__main__":
    print("🇯🇵 Japanese License Plate Detection Test Suite")
    print("=" * 60)
    
    # Test 1: Pattern validation
    create_japanese_plate_patterns()
    
    # Test 2: Video processing
    test_japanese_plate_detection()
    
    print("\n✅ Testing completed!")
