"""
Quick Fix for License Plate Detection
Immediately working solution for Japanese and international plates
"""

import cv2
import numpy as np
import os
import time
import json
import re

# Try to import Tesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    # Set Tesseract path for Windows
    if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    print("✅ Tesseract initialized")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("❌ Tesseract not available")

def quick_plate_detection(video_path, output_path=None, show_realtime=True):
    """Quick and effective license plate detection"""
    
    print("🚗 Quick License Plate Detection")
    print(f"📁 Input: {video_path}")
    
    if not os.path.exists(video_path):
        return {'error': 'Video not found'}
    
    if output_path is None:
        timestamp = int(time.time())
        output_path = f"quick_plate_detection_{timestamp}.mp4"
    
    try:
        # Try to import YOLO
        from ultralytics import YOLO
        model = YOLO("yolo26n.pt")
        print("✅ YOLO model loaded")
    except ImportError:
        print("❌ YOLO not available, using fallback")
        return fallback_detection(video_path, output_path, show_realtime)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': 'Cannot open video'}
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height} @ {fps:.1f} FPS")
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Processing variables
    frame_count = 0
    all_plates = []
    unique_plates = set()
    cars_detected = 0
    
    print("🔄 Starting detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        annotated_frame = frame.copy()
        
        # Detect cars
        try:
            results = model.predict(frame, conf=0.4, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = result.names.get(class_id, f"class_{class_id}")
                        
                        # Check if it's a vehicle
                        if class_name.lower() in ['car', 'truck', 'bus', 'motorcycle', 'van']:
                            cars_detected += 1
                            
                            # Draw car box
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"{class_name}", 
                                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # Extract license plates from car region
                            plates = extract_plates_from_car_region(frame, int(x1), int(y1), int(x2), int(y2))
                            
                            for plate_text in plates:
                                if plate_text and len(plate_text.strip()) > 2:
                                    # Add to results
                                    all_plates.append({
                                        'text': plate_text,
                                        'frame_number': frame_count,
                                        'car_bbox': [int(x1), int(y1), int(x2), int(y2)]
                                    })
                                    unique_plates.add(plate_text)
                                    
                                    # Display plate text next to car
                                    plate_label = f"📋 {plate_text}"
                                    cv2.putText(annotated_frame, plate_label, 
                                               (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        except Exception as e:
            print(f"Frame {frame_count} error: {e}")
        
        # Add frame info
        info_text = f"Frame: {frame_count} | Cars: {cars_detected} | Plates: {len(unique_plates)}"
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(annotated_frame)
        
        # Show real-time
        if show_realtime:
            cv2.imshow('Quick Plate Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processed {frame_count}/{total_frames} ({progress:.1f}%) - Found {len(unique_plates)} plates")
    
    # Cleanup
    cap.release()
    out.release()
    if show_realtime:
        cv2.destroyAllWindows()
    
    # Results
    results = {
        'cars_detected': cars_detected,
        'plates_found': len(all_plates),
        'unique_plates': list(unique_plates),
        'all_plates': all_plates,
        'video_info': {
            'input_path': video_path,
            'output_path': output_path,
            'frames_processed': frame_count
        }
    }
    
    print(f"✅ Processing completed!")
    print(f"📁 Output: {output_path}")
    print(f"🚗 Cars: {cars_detected}")
    print(f"📋 Plates: {len(unique_plates)}")
    
    if unique_plates:
        print("\n📋 Detected Plates:")
        for i, plate in enumerate(unique_plates, 1):
            print(f"   {i}. {plate}")
    
    return results

def extract_plates_from_car_region(frame, x1, y1, x2, y2):
    """Extract license plates from a car region"""
    plates = []
    
    # Extract car region
    car_region = frame[y1:y2, x1:x2]
    
    if car_region.size == 0:
        return plates
    
    # Method 1: Find rectangular regions (potential plates)
    gray = cv2.cvtColor(car_region, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple preprocessing techniques
    techniques = [
        ("original", gray),
        ("blurred", cv2.GaussianBlur(gray, (5, 5), 0)),
        ("threshold", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
    ]
    
    for tech_name, processed in techniques:
        # Edge detection
        edges = cv2.Canny(processed, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for plate-like shapes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # License plate characteristics
            if (2.0 < aspect_ratio < 6.0 and 
                w > 40 and h > 15 and 
                w < car_region.shape[1] * 0.8 and 
                h < car_region.shape[0] * 0.3):
                
                # Extract plate region
                plate_region = car_region[y:y+h, x:x+w]
                
                # OCR on plate region
                plate_text = ocr_plate_region(plate_region)
                
                if plate_text and is_valid_plate(plate_text):
                    plates.append(plate_text)
    
    # Method 2: Full region OCR if no plates found
    if not plates:
        full_text = ocr_plate_region(car_region)
        if full_text and is_valid_plate(full_text):
            plates.append(full_text)
    
    return list(set(plates))  # Remove duplicates

def ocr_plate_region(plate_region):
    """Extract text from plate region using Tesseract"""
    if not TESSERACT_AVAILABLE:
        return ""
    
    try:
        # Preprocess plate region
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY) if len(plate_region.shape) == 3 else plate_region
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Try multiple Tesseract configurations
        configs = [
            r'--oem 3 --psm 7',  # Single text line
            r'--oem 3 --psm 8',  # Single word
            r'--oem 3 --psm 6',  # Uniform block
            r'--oem 3 --psm 13'  # Raw line
        ]
        
        all_texts = []
        
        for config in configs:
            try:
                # Try with different languages
                for lang in ['eng', 'jpn', 'eng+jpn']:
                    try:
                        text = pytesseract.image_to_string(binary, lang=lang, config=config)
                        text = text.strip()
                        if text and len(text) > 2:
                            all_texts.append(text)
                    except:
                        continue
            except:
                continue
        
        # Return the most common result
        if all_texts:
            from collections import Counter
            most_common = Counter(all_texts).most_common(1)[0][0]
            return most_common
        
    except Exception as e:
        pass
    
    return ""

def is_valid_plate(text):
    """Check if text looks like a license plate"""
    if not text or len(text.strip()) < 3:
        return False
    
    text = text.strip().upper()
    
    # Remove common OCR errors
    text = re.sub(r'[^\w\s]', '', text)
    
    # Japanese and international patterns
    patterns = [
        r'^日本\s*\d{1,4}\s*[A-Z0-9]{2,6}$',  # Japanese format
        r'^[A-Z0-9]{3,8}$',                    # Simple alphanumeric
        r'^[A-Z]{2,3}\s*\d{2,4}\s*[A-Z]{0,3}$',  # International
        r'^\d{2,4}\s*[A-Z]{2,3}\s*\d{0,2}$',    # Reverse format
        r'^[A-Z0-9]{2,6}\s*[A-Z0-9]{2,6}$',      # Split format
    ]
    
    for pattern in patterns:
        if re.match(pattern, text):
            return True
    
    # Check for mixed alphanumeric (common in plates)
    has_letters = bool(re.search(r'[A-Z]', text))
    has_numbers = bool(re.search(r'[0-9]', text))
    has_japanese = bool(re.search(r'[日本]', text))
    
    return (has_letters and has_numbers) or has_japanese

def fallback_detection(video_path, output_path, show_realtime):
    """Fallback detection without YOLO"""
    print("🔄 Using fallback detection (no car detection)")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': 'Cannot open video'}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    all_plates = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Try to find plates directly in frame
        plates = extract_plates_from_car_region(frame, 0, 0, width, height)
        all_plates.extend(plates)
        
        # Draw info
        annotated = frame.copy()
        info_text = f"Frame: {frame_count} | Plates found: {len(set(all_plates))}"
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if plates:
            plate_text = ", ".join(set(plates))
            cv2.putText(annotated, f"Plates: {plate_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        out.write(annotated)
        
        if show_realtime:
            cv2.imshow('Fallback Detection', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    out.release()
    if show_realtime:
        cv2.destroyAllWindows()
    
    return {
        'cars_detected': 0,
        'plates_found': len(all_plates),
        'unique_plates': list(set(all_plates)),
        'all_plates': all_plates,
        'video_info': {
            'input_path': video_path,
            'output_path': output_path,
            'frames_processed': frame_count
        }
    }

if __name__ == "__main__":
    # Test with available video
    video_files = [f for f in os.listdir('.') if f.lower().endswith('.mp4')]
    
    if video_files:
        print(f"🎬 Testing with: {video_files[0]}")
        results = quick_plate_detection(video_files[0], show_realtime=False)
        
        print(f"\n📊 Results:")
        print(f"Cars detected: {results.get('cars_detected', 0)}")
        print(f"Plates found: {results.get('plates_found', 0)}")
        print(f"Unique plates: {results.get('unique_plates', [])}")
    else:
        print("❌ No video files found")
