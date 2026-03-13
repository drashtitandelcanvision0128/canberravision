"""
Simple working license plate detector
Uses direct approach with Tesseract OCR
"""

import cv2
import numpy as np
import os
import re

def simple_detect_plates(image_path):
    """Simple plate detection on single image"""
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Cannot load image")
        return []
    
    print(f"📷 Processing image: {img.shape}")
    
    # Try to import Tesseract
    try:
        import pytesseract
        if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    except:
        print("❌ Tesseract not available")
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find potential plate regions
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / h if h > 0 else 0
        
        # Plate-like shape
        if 1.5 < aspect < 6.0 and w > 50 and h > 15:
            # Extract region
            region = img[y:y+h, x:x+w]
            
            # OCR
            region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(region_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            text = pytesseract.image_to_string(binary, config='--psm 7')
            text = text.strip().replace('\n', ' ')
            
            if text and len(text) >= 3:
                # Validate plate
                if is_plate(text):
                    plates.append({
                        'text': text,
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.7
                    })
                    
                    # Draw on image
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save result
    cv2.imwrite("plate_detection_result.jpg", img)
    print(f"📁 Result saved: plate_detection_result.jpg")
    
    return plates

def is_plate(text):
    """Check if text looks like license plate"""
    text = text.upper().strip()
    
    # Has letters and numbers
    has_letters = bool(re.search(r'[A-Z]', text))
    has_numbers = bool(re.search(r'[0-9]', text))
    
    return has_letters and has_numbers and len(text) >= 4

if __name__ == "__main__":
    # Find an image to test
    for file in os.listdir('.'):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            print(f"\n🧪 Testing with: {file}")
            plates = simple_detect_plates(file)
            
            if plates:
                print(f"✅ Found {len(plates)} plates:")
                for p in plates:
                    print(f"   📋 {p['text']}")
            else:
                print("❌ No plates found")
            break
