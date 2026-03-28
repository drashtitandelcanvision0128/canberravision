"""
Fixed License Plate Video Processor
Properly detects license plates on cars and displays the text
"""

import cv2
import numpy as np
import os
import time
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional

def process_video_with_plate_detection(video_path, output_path=None, show_realtime=True):
    """
    Process video to detect cars and EXTRACT LICENSE PLATE TEXT properly
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
        show_realtime: Whether to show real-time processing
        
    Returns:
        Dictionary with detection results including license plates
    """
    
    print("🚗 Starting PROPER License Plate Detection")
    print(f"📁 Input video: {video_path}")
    
    if not os.path.exists(video_path):
        return {'error': f'Video file not found: {video_path}'}
    
    if output_path is None:
        timestamp = int(time.time())
        output_path = f"proper_plate_detection_{timestamp}.mp4"
    
    try:
        # Import YOLO for car detection
        from ultralytics import YOLO
        model = YOLO("yolo26n.pt")
        print("✅ YOLO model loaded")
    except Exception as e:
        print(f"❌ YOLO failed: {e}")
        return {'error': 'YOLO not available'}
    
    # Try to import OCR
    try:
        from optimized_paddleocr_gpu import extract_text_optimized
        PADDLE_AVAILABLE = True
        print("✅ PaddleOCR available")
    except:
        PADDLE_AVAILABLE = False
        print("⚠️ PaddleOCR not available, using Tesseract fallback")
    
    try:
        import pytesseract
        TESSERACT_AVAILABLE = True
        if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    except:
        TESSERACT_AVAILABLE = False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': 'Cannot open video'}
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Processing variables
    frame_count = 0
    all_cars = []
    all_plates = []
    unique_plates = set()
    
    print("🔄 Processing...")
    
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
                        if class_name.lower() in ['car', 'truck', 'bus', 'motorcycle', 'van', 'vehicle']:
                            car_bbox = [int(x1), int(y1), int(x2), int(y2)]
                            
                            # Store car info
                            car_info = {
                                'frame': frame_count,
                                'bbox': car_bbox,
                                'class': class_name,
                                'confidence': confidence
                            }
                            all_cars.append(car_info)
                            
                            # Draw car box
                            cv2.rectangle(annotated_frame, (car_bbox[0], car_bbox[1]), 
                                         (car_bbox[2], car_bbox[3]), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"{class_name} ({confidence:.2f})", 
                                       (car_bbox[0], car_bbox[1] - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # EXTRACT LICENSE PLATE from car region
                            plates = extract_plates_from_car(frame, car_bbox, PADDLE_AVAILABLE, TESSERACT_AVAILABLE)
                            
                            for plate in plates:
                                plate_text = plate['text']
                                plate_conf = plate['confidence']
                                
                                # Add to results
                                plate_info = {
                                    'text': plate_text,
                                    'confidence': plate_conf,
                                    'frame': frame_count,
                                    'car_bbox': car_bbox,
                                    'engine': plate.get('engine', 'unknown')
                                }
                                all_plates.append(plate_info)
                                unique_plates.add(plate_text)
                                
                                # DRAW PLATE TEXT ON FRAME - NEXT TO CAR
                                text_x = car_bbox[0]
                                text_y = car_bbox[3] + 30  # Below the car box
                                
                                plate_label = f"📋 {plate_text}"
                                
                                # Draw background for text
                                (text_w, text_h), _ = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                cv2.rectangle(annotated_frame, 
                                             (text_x, text_y - text_h - 5), 
                                             (text_x + text_w + 10, text_y + 5), 
                                             (0, 0, 255), -1)
                                
                                # Draw plate text
                                cv2.putText(annotated_frame, plate_label, 
                                           (text_x + 5, text_y), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                
                                # Also draw small bbox for plate region
                                if 'plate_bbox' in plate:
                                    pb = plate['plate_bbox']
                                    cv2.rectangle(annotated_frame, 
                                                 (pb[0], pb[1]), (pb[2], pb[3]), 
                                                 (255, 0, 0), 2)
        
        except Exception as e:
            print(f"Frame {frame_count} error: {e}")
        
        # Add frame info at top
        info_text = f"Frame: {frame_count} | Cars: {len(all_cars)} | Plates: {len(unique_plates)}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(annotated_frame)
        
        # Show real-time
        if show_realtime:
            display_frame = cv2.resize(annotated_frame, (960, 540))
            cv2.imshow('License Plate Detection', display_frame)
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
        'total_frames': frame_count,
        'cars_detected': len(all_cars),
        'plates_found': len(all_plates),
        'unique_plates': list(unique_plates),
        'all_plates': all_plates,
        'all_cars': all_cars,
        'video_info': {
            'input_path': video_path,
            'output_path': output_path
        }
    }
    
    # Save JSON
    json_path = output_path.replace('.mp4', '_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Processing completed!")
    print(f"📁 Output video: {output_path}")
    print(f"📁 Results JSON: {json_path}")
    print(f"🚗 Cars detected: {len(all_cars)}")
    print(f"📋 Total plates: {len(all_plates)}")
    print(f"🔢 Unique plates: {len(unique_plates)}")
    
    if unique_plates:
        print("\n📋 DETECTED LICENSE PLATES:")
        for i, plate in enumerate(sorted(unique_plates), 1):
            print(f"   {i}. {plate}")
    
    return results

def extract_plates_from_car(frame, car_bbox, paddle_available, tesseract_available):
    """Extract license plates from a car region"""
    plates = []
    
    x1, y1, x2, y2 = car_bbox
    
    # Expand car region slightly to capture plates
    margin = 20
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(frame.shape[1], x2 + margin)
    y2 = min(frame.shape[0], y2 + margin)
    
    # Extract car region
    car_region = frame[y1:y2, x1:x2]
    
    if car_region.size == 0:
        return plates
    
    # Method 1: Look for rectangular plate-like regions
    gray = cv2.cvtColor(car_region, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple preprocessing techniques
    preprocessed_images = []
    
    # Original grayscale
    preprocessed_images.append(("gray", gray))
    
    # Blurred
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    preprocessed_images.append(("blurred", blurred))
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    preprocessed_images.append(("clahe", enhanced))
    
    # Edge detection for plate finding
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for plate-like regions
    plate_regions = []
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        aspect_ratio = cw / ch if ch > 0 else 0
        
        # License plate characteristics
        if (2.0 < aspect_ratio < 6.0 and 
            cw > 60 and ch > 15 and 
            cw < car_region.shape[1] * 0.8):
            
            plate_regions.append((cx, cy, cw, ch))
    
    # If no plate regions found, try full car region
    if not plate_regions:
        plate_regions = [(0, 0, car_region.shape[1], car_region.shape[0])]
    
    # OCR on each plate region
    for px, py, pw, ph in plate_regions[:3]:  # Limit to top 3 regions
        plate_crop = car_region[py:py+ph, px:px+pw]
        
        if plate_crop.size == 0:
            continue
        
        # Try PaddleOCR first
        if paddle_available:
            try:
                from optimized_paddleocr_gpu import extract_text_optimized
                
                result = extract_text_optimized(
                    plate_crop, 
                    confidence_threshold=0.3,
                    lang='en',
                    use_gpu=None,
                    use_cache=False,
                    preprocess=True
                )
                
                if result and result.get('text'):
                    text = result['text'].strip()
                    conf = result.get('confidence', 0)
                    
                    if text and len(text) >= 3:
                        plates.append({
                            'text': text,
                            'confidence': conf,
                            'engine': 'paddleocr',
                            'plate_bbox': [x1+px, y1+py, x1+px+pw, y1+py+ph]
                        })
            except Exception as e:
                pass
        
        # Fallback to Tesseract
        if tesseract_available and (not plates or len(plates) == 0):
            try:
                import pytesseract
                
                # Preprocess for Tesseract
                plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # OCR
                custom_config = r'--oem 3 --psm 7'
                text = pytesseract.image_to_string(binary, config=custom_config)
                
                if text and len(text.strip()) >= 3:
                    plates.append({
                        'text': text.strip(),
                        'confidence': 0.6,
                        'engine': 'tesseract',
                        'plate_bbox': [x1+px, y1+py, x1+px+pw, y1+py+ph]
                    })
            except Exception as e:
                pass
    
    # Remove duplicates
    seen_texts = set()
    unique_plates = []
    for plate in plates:
        text = plate['text']
        if text not in seen_texts:
            seen_texts.add(text)
            unique_plates.append(plate)
    
    return unique_plates

if __name__ == "__main__":
    # Test with available video
    video_files = [f for f in os.listdir('.') if f.lower().endswith('.mp4')]
    
    if video_files:
        print(f"Testing with: {video_files[0]}")
        results = process_video_with_plate_detection(video_files[0], show_realtime=False)
        print(f"\nResults: {results['unique_plates']}")
    else:
        print("No video files found")
