"""
WORKING License Plate Detector
Detects cars, finds license plates, extracts text, displays on video
"""

import cv2
import numpy as np
import os
import re
import json
from datetime import datetime

def process_video_detect_plates(video_path):
    """
    Main function: Detect cars and license plates
    """
    print("=" * 60)
    print("🚗 WORKING License Plate Detector")
    print("=" * 60)
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    # Load YOLO
    try:
        from ultralytics import YOLO
        model = YOLO("yolo26n.pt")
        print("✅ YOLO loaded")
    except Exception as e:
        print(f"❌ YOLO error: {e}")
        return None
    
    # Load Tesseract
    try:
        import pytesseract
        if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        print("✅ Tesseract ready")
    except Exception as e:
        print(f"❌ Tesseract error: {e}")
        return None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return None
    
    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height} @ {fps:.1f}fps")
    print(f"   Total frames: {total_frames}")
    
    # Output video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"PLATE_DETECTED_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Stats
    frame_count = 0
    cars_total = 0
    plates_list = []
    
    print("\n🔄 Processing...")
    print("-" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        display = frame.copy()
        frame_plates = []
        
        # Detect objects
        results = model.predict(frame, conf=0.4, iou=0.5, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    name = result.names.get(cls, 'unknown')
                    
                    # Check if vehicle
                    if name.lower() in ['car', 'truck', 'bus', 'van', 'vehicle', 'automobile']:
                        cars_total += 1
                        
                        cx1, cy1, cx2, cy2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Draw car box (GREEN)
                        cv2.rectangle(display, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                        cv2.putText(display, f"CAR", (cx1, cy1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # ========== LICENSE PLATE DETECTION ==========
                        # Focus on lower part of car where plates usually are
                        car_height = cy2 - cy1
                        plate_y1 = cy1 + int(car_height * 0.65)  # Lower 35% of car
                        plate_y2 = cy2 - 5
                        plate_x1 = cx1 + 10
                        plate_x2 = cx2 - 10
                        
                        # Extract plate region
                        plate_region = frame[plate_y1:plate_y2, plate_x1:plate_x2]
                        
                        if plate_region.size > 0:
                            # OCR the plate region
                            plate_text = extract_text_from_region(plate_region)
                            
                            if plate_text:
                                frame_plates.append(plate_text)
                                plates_list.append({
                                    'text': plate_text,
                                    'frame': frame_count,
                                    'car': [cx1, cy1, cx2, cy2]
                                })
                                
                                # Draw plate box (RED)
                                cv2.rectangle(display, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 0, 255), 2)
                                
                                # Draw plate text (BIG + CLEAR)
                                text = f"PLATE: {plate_text}"
                                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                                
                                # Background
                                cv2.rectangle(display, 
                                             (cx1, cy2+5), 
                                             (cx1+tw+10, cy2+th+15), 
                                             (0, 0, 255), -1)
                                
                                # Text
                                cv2.putText(display, text, 
                                           (cx1+5, cy2+th+10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                
                                print(f"   📋 Frame {frame_count}: {plate_text}")
        
        # Frame info at top
        info = f"Frame: {frame_count} | Cars: {cars_total} | Plates: {len(plates_list)}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write output
        out.write(display)
        
        # Progress
        if frame_count % 100 == 0:
            pct = (frame_count / total_frames) * 100
            print(f"Progress: {pct:.1f}% ({frame_count}/{total_frames})")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Results
    unique_plates = list(set([p['text'] for p in plates_list]))
    
    results = {
        'video': video_path,
        'output': output_path,
        'frames': frame_count,
        'cars': cars_total,
        'plates_found': len(plates_list),
        'unique_plates': unique_plates,
        'all_plates': plates_list
    }
    
    # Save JSON
    json_path = output_path.replace('.mp4', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📁 Output video: {output_path}")
    print(f"📁 Results JSON: {json_path}")
    print(f"🚗 Total cars: {cars_total}")
    print(f"📋 Total plates: {len(plates_list)}")
    print(f"🔢 Unique plates: {len(unique_plates)}")
    
    if unique_plates:
        print("\n📋 DETECTED LICENSE PLATES:")
        for i, plate in enumerate(sorted(unique_plates), 1):
            print(f"   {i}. {plate}")
    
    print("=" * 60)
    
    return results

def extract_text_from_region(region):
    """Extract text from license plate region"""
    import pytesseract
    
    if region.size == 0:
        return None
    
    # Resize for better OCR
    h, w = region.shape[:2]
    if w < 150:
        scale = 3
        region = cv2.resize(region, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Enhance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Binarize
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR configs to try
    configs = ['--psm 7', '--psm 8', '--psm 6']
    
    texts = []
    for config in configs:
        try:
            text = pytesseract.image_to_string(binary, config=config)
            text = text.strip().upper()
            text = re.sub(r'[^A-Z0-9\s-]', '', text)
            text = text.strip()
            
            if text and 4 <= len(text) <= 12:
                # Must contain letters or numbers
                if re.search(r'[A-Z0-9]', text):
                    texts.append(text)
        except:
            pass
    
    # Return most common or first valid
    if texts:
        from collections import Counter
        most_common = Counter(texts).most_common(1)[0][0]
        return most_common
    
    return None

# Run if called directly
if __name__ == "__main__":
    # Find video
    videos = [f for f in os.listdir('.') if f.endswith('.mp4') and 'compatible' in f.lower()]
    
    if videos:
        print(f"Found {len(videos)} videos")
        process_video_detect_plates(videos[0])
    else:
        print("No videos found. Usage:")
        print("  python working_plate_detector.py")
        print("Or in Python:")
        print("  from working_plate_detector import process_video_detect_plates")
        print("  process_video_detect_plates('your_video.mp4')")
