"""
Diagnose why license plates are not being detected
"""

import cv2
import os
import numpy as np

# Find a video with cars
video_files = [f for f in os.listdir('.') if 'compatible' in f.lower() and f.endswith('.mp4')]

if not video_files:
    print("❌ No compatible videos found")
else:
    video_path = video_files[0]
    print(f"🎬 Testing with: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if cap.isOpened():
        # Read first 10 frames
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame
            frame_file = f"test_frame_{i}.jpg"
            cv2.imwrite(frame_file, frame)
            print(f"📷 Saved: {frame_file}")
        
        cap.release()
        
        # Now try OCR on saved frames
        print("\n🔍 Testing OCR on frames...")
        
        try:
            import pytesseract
            if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            
            for i in range(10):
                frame_file = f"test_frame_{i}.jpg"
                if os.path.exists(frame_file):
                    img = cv2.imread(frame_file)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Try OCR
                    text = pytesseract.image_to_string(gray, config='--psm 6')
                    text = text.strip()
                    
                    if text:
                        print(f"   Frame {i}: '{text[:50]}...' " if len(text) > 50 else f"   Frame {i}: '{text}'")
                    
                    # Clean up
                    os.remove(frame_file)
        
        except Exception as e:
            print(f"❌ OCR failed: {e}")
    else:
        print("❌ Cannot open video")
