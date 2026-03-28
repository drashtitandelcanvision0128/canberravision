"""
TESSERACT-BASED License Plate Detection
Bypasses PaddleOCR issues, uses Tesseract directly
"""

import cv2
import numpy as np
import os
import re
import json
from datetime import datetime

def detect_with_tesseract(video_path, output_path=None):
    """
    Detect cars and license plates using Tesseract OCR
    Guaranteed to work since Tesseract is already installed
    """
    
    print("=" * 70)
    print("🚗 TESSERACT License Plate Detection")
    print("=" * 70)
    print(f"📁 Video: {video_path}")
    
    if not os.path.exists(video_path):
        print("❌ Video not found")
        return None
    
    # Setup Tesseract
    try:
        import pytesseract
        if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        print("✅ Tesseract OCR ready")
    except Exception as e:
        print(f"❌ Tesseract error: {e}")
        return None
    
    # Load YOLO
    try:
        from ultralytics import YOLO
        model = YOLO("yolo26n.pt")
        print("✅ YOLO model loaded")
    except Exception as e:
        print(f"❌ YOLO error: {e}")
        return None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    
    # Output video
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"PLATE_DETECTED_{timestamp}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Stats
    frame_count = 0
    cars_total = 0
    plates_list = []
    unique_plates = set()
    
    print("\n🔄 Processing...")
    print("-" * 70)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        display = frame.copy()
        
        # Detect cars with YOLO
        results = model.predict(frame, conf=0.4, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    name = result.names.get(cls, 'unknown')
                    
                    # Vehicle check
                    if name.lower() in ['car', 'truck', 'bus', 'van', 'vehicle']:
                        cars_total += 1
                        cx1, cy1, cx2, cy2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Draw car box (GREEN)
                        cv2.rectangle(display, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                        cv2.putText(display, f"CAR {conf:.2f}", 
                                   (cx1, cy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Calculate plate region coordinates
                        car_height = cy2 - cy1
                        plate_y1 = cy1 + int(car_height * 0.65)
                        plate_y2 = cy2 - 5
                        plate_x1 = cx1 + 20
                        plate_x2 = cx2 - 20
                        
                        # EXTRACT PLATE using Tesseract
                        plate_text = extract_plate_tesseract(frame, plate_x1, plate_y1, plate_x2, plate_y2, pytesseract)
                        
                        if plate_text:
                            plates_list.append({
                                'text': plate_text,
                                'frame': frame_count,
                                'car': [cx1, cy1, cx2, cy2]
                            })
                            unique_plates.add(plate_text)
                            
                            # Draw plate text (BIG + CLEAR) - On car, not below
                            label = f"PLATE: {plate_text}"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                            
                            # Draw INSIDE car box at top-left (like the user image shows)
                            # Background - Yellow for visibility
                            text_x = cx1 + 5
                            text_y = cy1 + th + 10
                            cv2.rectangle(display, 
                                         (text_x - 3, text_y - th - 5), 
                                         (text_x + tw + 5, text_y + 5), 
                                         (0, 255, 255), -1)  # Yellow background
                            
                            # Text - Black for contrast
                            cv2.putText(display, label, 
                                       (text_x, text_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)  # BOLD black text
                            
                            # Also draw small box around detected plate region
                            cv2.rectangle(display, 
                                         (plate_x1, plate_y1), 
                                         (plate_x2, plate_y2), 
                                         (255, 0, 0), 2)  # Blue plate box
                            
                            print(f"   📋 Frame {frame_count}: {plate_text}")
        
        # Frame info
        info = f"Frame: {frame_count} | Cars: {cars_total} | Plates: {len(unique_plates)}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(display)
        
        # Progress
        if frame_count % 100 == 0:
            pct = (frame_count / total_frames) * 100
            print(f"Progress: {pct:.1f}% ({frame_count}/{total_frames})")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Results
    results_data = {
        'video': video_path,
        'output': output_path,
        'frames': frame_count,
        'cars': cars_total,
        'plates_found': len(plates_list),
        'unique_plates': sorted(list(unique_plates)),
        'all_plates': plates_list
    }
    
    # Save JSON
    json_path = output_path.replace('.mp4', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"📁 Video: {output_path}")
    print(f"📁 JSON: {json_path}")
    print(f"🚗 Cars: {cars_total}")
    print(f"📋 Plates: {len(plates_list)}")
    print(f"🔢 Unique: {len(unique_plates)}")
    
    if unique_plates:
        print("\n📋 DETECTED PLATES:")
        for i, p in enumerate(sorted(unique_plates), 1):
            print(f"   {i}. {p}")
    
    return results_data

def fix_ocr_errors(text):
    """Fix common OCR mistakes where numbers/letters are confused"""
    import re
    
    if not text:
        return text
    
    # Common OCR confusions in license plates
    corrections = {
        'O': '0',  # Letter O -> Number 0 (in middle of plate)
        'I': '1',  # Letter I -> Number 1
        'L': '1',  # Letter L -> Number 1
        'S': '5',  # Letter S -> Number 5
        'A': '4',  # Letter A -> Number 4 (if looks like 4)
        'B': '8',  # Letter B -> Number 8
        'Z': '2',  # Letter Z -> Number 2
        'G': '6',  # Letter G -> Number 6
        'T': '7',  # Letter T -> Number 7 (sometimes)
    }
    
    # For license plates, pattern is typically: LETTERS-NUMBERS or LETTERS-NUMBERS-LETTERS
    # Example: BR45SIL -> should be detected properly
    
    # First, try to identify the pattern
    text_upper = text.upper().strip()
    
    # Remove spaces and special chars
    clean = re.sub(r'[^A-Z0-9]', '', text_upper)
    
    # Apply corrections based on position
    # Numbers usually come AFTER letters in most plates
    result = ""
    for i, char in enumerate(clean):
        # Heuristic: if char is a letter that looks like a number
        # and it's in the middle/end where numbers should be
        if char in corrections:
            # Check context - if surrounded by numbers, likely a number
            neighbors = ""
            if i > 0:
                neighbors += clean[i-1]
            if i < len(clean) - 1:
                neighbors += clean[i+1]
            
            # If neighbors are numbers, this is probably a number too
            if any(n.isdigit() for n in neighbors):
                result += corrections[char]
            else:
                result += char
        else:
            result += char
    
    return result

def extract_plate_tesseract(frame, plate_x1, plate_y1, plate_x2, plate_y2, pytesseract):
    """Extract license plate using Tesseract with improved accuracy"""
    
    if plate_y2 <= plate_y1 or plate_x2 <= plate_x1:
        return None
    
    # Extract region
    plate_region = frame[plate_y1:plate_y2, plate_x1:plate_x2]
    if plate_region.size == 0:
        return None
    
    # Resize small regions for better OCR
    h, w = plate_region.shape[:2]
    if w < 200:
        plate_region = cv2.resize(plate_region, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    
    # Enhance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Binarize
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Try multiple OCR configs with WHITELIST for alphanumeric
    configs = [
        '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Single line
        '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Single word
        '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Uniform block
    ]
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(binary, config=config)
            text = text.strip().upper()
            
            # Apply error correction
            text = fix_ocr_errors(text)
            
            if text and 4 <= len(text) <= 12:
                # Must have letters AND numbers for a valid plate
                has_letters = bool(re.search(r'[A-Z]', text))
                has_numbers = bool(re.search(r'[0-9]', text))
                
                if has_letters and has_numbers:
                    return text
        except:
            pass
    
    return None

if __name__ == "__main__":
    # Use latest video
    video = "processed_video_1773296566.mp4"
    
    if os.path.exists(video):
        detect_with_tesseract(video)
    else:
        # Find any video
        import glob
        videos = glob.glob('compatible_video_*.mp4')
        if videos:
            print(f"Using: {videos[0]}")
            detect_with_tesseract(videos[0])
        else:
            print("No videos found")
