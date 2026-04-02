"""
Video Processing with License Plate Detection - FIXED VERSION
Uses dedicated license plate detection model or region-based detection with improved OCR
"""

import cv2
import numpy as np
import torch
import os
import sys
import time
import re
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# COCO class IDs with names
VEHICLE_CLASSES = {
    2: 'Car',
    3: 'Motorcycle', 
    5: 'Bus',
    7: 'Truck'
}
LICENSE_PLATE_CLASS = None  # Will be set if model has license plate class


def get_vehicle_type(cls_id):
    """Get vehicle type name from class ID"""
    return VEHICLE_CLASSES.get(cls_id, 'Unknown')


def detect_vehicle_color(vehicle_crop):
    """
    Detect dominant color of vehicle
    Returns color name like 'Silver', 'Black', 'White', etc.
    """
    try:
        if vehicle_crop is None or vehicle_crop.size == 0:
            return 'Unknown'
        
        # Resize for faster processing
        resized = cv2.resize(vehicle_crop, (100, 100))
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV
        color_ranges = {
            'Black': [(0, 0, 0), (180, 255, 30)],
            'White': [(0, 0, 200), (180, 30, 255)],
            'Gray/Silver': [(0, 0, 30), (180, 30, 200)],
            'Red': [(0, 100, 100), (10, 255, 255)],
            'Red2': [(160, 100, 100), (180, 255, 255)],  # Red wraps around
            'Orange': [(10, 100, 100), (25, 255, 255)],
            'Yellow': [(25, 100, 100), (35, 255, 255)],
            'Green': [(35, 100, 100), (85, 255, 255)],
            'Blue': [(85, 100, 100), (125, 255, 255)],
            'Purple': [(125, 100, 100), (145, 255, 255)],
            'Pink': [(145, 100, 100), (160, 255, 255)],
            'Brown': [(10, 100, 50), (20, 200, 150)],
        }
        
        color_counts = {}
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            count = np.sum(mask > 0)
            
            # Combine both red ranges
            if color_name in ['Red', 'Red2']:
                color_counts['Red'] = color_counts.get('Red', 0) + count
            elif color_name == 'Gray/Silver':
                color_counts['Silver'] = count
            else:
                color_counts[color_name] = count
        
        # Get dominant color
        if color_counts:
            dominant = max(color_counts.items(), key=lambda x: x[1])
            return dominant[0] if dominant[1] > 100 else 'Unknown'
        return 'Unknown'
        
    except Exception as e:
        return 'Unknown'


def draw_info_panel(frame, detections, current_time):
    """
    Draw ANPR-style info panel on the right side of frame
    Shows: Plate, Vehicle Type, Color, Time, Date
    """
    try:
        h, w = frame.shape[:2]
        
        # Panel dimensions
        panel_width = 350
        panel_x = w - panel_width - 10
        panel_y = 10
        line_height = 35
        
        # Calculate panel height based on number of detections
        panel_height = max(150, 60 + len(detections) * 100)
        
        # Draw semi-transparent black background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (255, 165, 0), 2)  # Orange border
        
        # Title
        title = "ANPR SYSTEM"
        cv2.putText(frame, title, (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        
        # Current time and date
        time_str = current_time.strftime("%H:%M:%S")
        date_str = current_time.strftime("%d/%m/%Y")
        cv2.putText(frame, f"Time: {time_str}", (panel_x + 10, panel_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Date: {date_str}", (panel_x + 10, panel_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Separator line
        cv2.line(frame, (panel_x + 10, panel_y + 85), 
                (panel_x + panel_width - 10, panel_y + 85), (255, 165, 0), 1)
        
        # Display detections
        y_offset = panel_y + 110
        
        for i, det in enumerate(detections[-3:]):  # Show last 3 detections
            # Plate number (highlighted)
            plate_text = det.get('plate', 'Unknown')
            color = (0, 255, 0) if plate_text != 'Unknown' else (255, 0, 0)
            cv2.putText(frame, f"Plate: {plate_text}", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
            
            # Vehicle info
            vehicle_type = det.get('vehicle_type', 'Unknown')
            vehicle_color = det.get('vehicle_color', 'Unknown')
            cv2.putText(frame, f"Type: {vehicle_type}", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            cv2.putText(frame, f"Color: {vehicle_color}", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            # Confidence
            conf = det.get('conf', 0)
            cv2.putText(frame, f"Conf: {conf:.1%}", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 35  # Space between detections
            
            # Separator
            if i < len(detections) - 1 and i < 2:
                cv2.line(frame, (panel_x + 10, y_offset - 15), 
                        (panel_x + panel_width - 10, y_offset - 15), (100, 100, 100), 1)
        
        return frame
        
    except Exception as e:
        return frame


def detect_vehicles_and_plates(frame, model):
    """
    Detect vehicles and find license plates with vehicle type and color
    Returns list of detection dicts with all vehicle info
    """
    try:
        from datetime import datetime
        
        # Run detection
        results = model.predict(
            source=frame,
            conf=0.25,
            iou=0.4,
            imgsz=640,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False
        )
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            return []
        
        detected_vehicles = []
        boxes = result.boxes
        current_time = datetime.now()
        
        # First pass: find all vehicles and their plates
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            
            if cls in VEHICLE_CLASSES:
                vx1, vy1, vx2, vy2 = map(int, boxes.xyxy[i])
                
                # Crop vehicle region
                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                if vehicle_crop.size == 0:
                    continue
                
                # Detect vehicle color
                vehicle_color = detect_vehicle_color(vehicle_crop)
                
                # Get vehicle type
                vehicle_type = get_vehicle_type(cls)
                
                # Find license plate within vehicle
                plate_regions = find_license_plate_in_crop(vehicle_crop)
                
                plates_found = []
                for (px1, py1, px2, py2) in plate_regions[:1]:  # Take best plate only
                    # Convert to frame coordinates
                    fx1, fy1 = vx1 + px1, vy1 + py1
                    fx2, fy2 = vx1 + px2, vy1 + py2
                    
                    plate_crop = frame[fy1:fy2, fx1:fx2]
                    if plate_crop.size > 0 and plate_crop.shape[0] > 15 and plate_crop.shape[1] > 40:
                        
                        # Get plate text
                        preprocessed = preprocess_for_ocr(plate_crop)
                        plate_text = extract_plate_text(preprocessed)
                        
                        plates_found.append({
                            'bbox': (fx1, fy1, fx2, fy2),
                            'crop': plate_crop,
                            'text': plate_text
                        })
                
                detected_vehicles.append({
                    'vehicle_bbox': (vx1, vy1, vx2, vy2),
                    'vehicle_type': vehicle_type,
                    'vehicle_color': vehicle_color,
                    'conf': float(boxes.conf[i]),
                    'plates': plates_found,
                    'timestamp': current_time
                })
        
        return detected_vehicles
        
    except Exception as e:
        print(f"⚠️ Detection error: {e}")
        return []


def find_license_plate_in_crop(vehicle_crop):
    """
    Smart license plate detection within vehicle crop
    Uses edge detection and MSER to find actual plate region
    Focuses on bumper area where plates actually are
    """
    try:
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []
        
        h, w = vehicle_crop.shape[:2]
        all_candidates = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Strategy 1: Edge-based detection with focus on plate-like regions
        for low_thresh in [30, 50, 70]:
            edges = cv2.Canny(gray, low_thresh, low_thresh * 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                if cw <= 0 or ch <= 0:
                    continue
                
                area = cw * ch
                min_area = (w * h) * 0.001  # Reduced to catch smaller plates
                if area < min_area:
                    continue
                
                # Aspect ratio check (license plates are wide and flat)
                ar = cw / float(ch)
                if ar < 2.5 or ar > 6.0:  # Tighter range for actual plates
                    continue
                
                # Height check - plates are relatively thin
                if ch < 15 or ch > h * 0.25:  # Too small or too tall
                    continue
                
                # Position check - LICENSE PLATES ARE AT BUMPER LEVEL (35-60% from top)
                y_center = y + ch / 2
                y_ratio = y_center / h
                
                # Strong preference for bumper area
                position_score = 0.0
                if 0.35 <= y_ratio <= 0.55:  # Perfect plate zone (bumper level)
                    position_score = 2.0
                elif 0.30 <= y_ratio <= 0.65:  # Acceptable
                    position_score = 1.0
                elif y_ratio < 0.20:  # Too high (grille/emblem)
                    position_score = 0.1
                elif y_ratio > 0.75:  # Too low (wheels/ground)
                    position_score = 0.1
                else:
                    position_score = 0.3
                
                # Skip very low scores
                if position_score < 0.5:
                    continue
                
                # Padding around detected region
                padding_x = int(cw * 0.15)
                padding_y = int(ch * 0.15)
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(w, x + cw + padding_x)
                y2 = min(h, y + ch + padding_y)
                
                weighted_area = area * position_score
                all_candidates.append((x1, y1, x2, y2, weighted_area, y_ratio))
        
        # Strategy 2: MSER for text regions (focused on bumper area)
        try:
            mser = cv2.MSER_create()
            mser.set_delta(5)
            mser.set_min_area(int((w * h) * 0.0005))
            mser.set_max_area(int((w * h) * 0.1))
            
            regions, _ = mser.detectRegions(gray)
            for region in regions:
                x, y, cw, ch = cv2.boundingRect(region)
                
                # Only look in bumper area (35-60% from top)
                y_center = y + ch / 2
                y_ratio = y_center / h
                if y_ratio < 0.35 or y_ratio > 0.65:
                    continue
                
                area = cw * ch
                if area < (w * h) * 0.0005:
                    continue
                
                ar = cw / float(ch)
                if ar < 2.0 or ar > 8.0:
                    continue
                
                padding_x = int(cw * 0.2)
                padding_y = int(ch * 0.2)
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(w, x + cw + padding_x)
                y2 = min(h, y + ch + padding_y)
                
                # High score for bumper-area text
                position_score = 1.5 if 0.40 <= y_ratio <= 0.55 else 0.8
                weighted_area = area * position_score
                all_candidates.append((x1, y1, x2, y2, weighted_area, y_ratio))
        except:
            pass
        
        # Remove duplicates and sort
        unique_candidates = []
        seen = set()
        
        for candidate in all_candidates:
            x1, y1, x2, y2, weighted_area, y_ratio = candidate
            key = (x1//10, y1//10, x2//10, y2//10)
            if key not in seen:
                seen.add(key)
                unique_candidates.append((x1, y1, x2, y2, weighted_area))
        
        # Sort by weighted area
        unique_candidates.sort(key=lambda t: t[4], reverse=True)
        
        # Return top 3 candidates
        return [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in unique_candidates[:3]]
        
    except Exception as e:
        print(f"⚠️ Plate detection error: {e}")
        return []


def preprocess_for_ocr(image):
    """
    Advanced preprocessing for license plate OCR
    Uses multiple techniques to enhance text readability
    """
    try:
        if image is None or image.size == 0:
            return None
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize with aspect ratio preservation (license plates are wider)
        h, w = gray.shape
        target_height = 100
        aspect = w / h
        target_width = int(target_height * aspect * 1.5)
        target_width = max(target_width, 150)  # Minimum width for better OCR
        
        resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Try multiple preprocessing approaches and return the best one
        preprocessed_versions = []
        
        # Version 1: CLAHE + Bilateral
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced1 = clahe.apply(resized)
        filtered1 = cv2.bilateralFilter(enhanced1, 11, 17, 17)
        _, binary1 = cv2.threshold(filtered1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_versions.append(binary1)
        
        # Version 2: High contrast + sharpening
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(resized, -1, kernel_sharpen)
        _, binary2 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_versions.append(binary2)
        
        # Version 3: Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        preprocessed_versions.append(adaptive)
        
        # Version 4: Inverted (for dark text on light background)
        inverted = cv2.bitwise_not(binary1)
        preprocessed_versions.append(inverted)
        
        # Return all versions for OCR to try
        return preprocessed_versions
        
    except Exception as e:
        return None


def is_valid_plate_text(text):
    """
    Validate if text looks like a real license plate
    Rejects garbage like 'SAGO', 'WPAWNSO'
    """
    if not text or len(text) < 4 or len(text) > 10:
        return False
    
    # Clean text
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Check for minimum requirements
    if len(clean) < 4:
        return False
    
    # Count digits and letters
    digits = sum(c.isdigit() for c in clean)
    letters = sum(c.isalpha() for c in clean)
    
    # License plates should have both letters and digits
    if digits == 0 or letters == 0:
        return False
    
    # Check for common patterns
    # Pattern 1: ABC123 (letters followed by digits)
    pattern1 = re.match(r'^[A-Z]{2,4}[0-9]{2,4}$', clean)
    # Pattern 2: 123ABC (digits followed by letters)
    pattern2 = re.match(r'^[0-9]{2,4}[A-Z]{2,4}$', clean)
    # Pattern 3: AB12CD (mixed)
    pattern3 = re.match(r'^[A-Z0-9]{4,8}$', clean)
    
    if not (pattern1 or pattern2 or pattern3):
        return False
    
    # Reject if too many repeated characters (likely noise)
    repeats = sum(1 for i in range(1, len(clean)) if clean[i] == clean[i-1])
    if repeats > 2:
        return False
    
    # Reject common OCR errors
    ocr_errors = ['SAGO', 'WPAW', 'ANVY', 'MESH', 'RESI', 'MESS', 'MEW', 'SESS', 'BEE', 
                  'SSSE', 'SSST', 'SSSY', 'SST', 'SSWS', 'SZT', 'RTNS', 'STNS', 'ASUN',
                  'BOP', 'PSM', 'SMS', 'SNST', 'SSM', 'BO', 'PO', 'WO', 'QO', 'ZO']
    for error in ocr_errors:
        if error in clean:
            return False
    
    return True


def extract_plate_text(preprocessed_images):
    """
    Extract license plate text using multiple OCR attempts
    Returns None if text doesn't look like a valid plate
    """
    if preprocessed_images is None:
        return None
    
    if not isinstance(preprocessed_images, list):
        preprocessed_images = [preprocessed_images]
    
    all_texts = []
    
    # Try Tesseract with different configs on each preprocessed version
    try:
        import pytesseract
        
        for img in preprocessed_images:
            # PSM 7 = single line
            config7 = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text7 = pytesseract.image_to_string(img, config=config7).strip().upper()
            text7 = re.sub(r'[^A-Z0-9]', '', text7)
            if is_valid_plate_text(text7):
                all_texts.append(('t7', text7))
            
            # PSM 8 = single word
            config8 = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text8 = pytesseract.image_to_string(img, config=config8).strip().upper()
            text8 = re.sub(r'[^A-Z0-9]', '', text8)
            if is_valid_plate_text(text8):
                all_texts.append(('t8', text8))
            
            # PSM 6 = uniform text block
            config6 = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text6 = pytesseract.image_to_string(img, config=config6).strip().upper()
            text6 = re.sub(r'[^A-Z0-9]', '', text6)
            if is_valid_plate_text(text6):
                all_texts.append(('t6', text6))
    except:
        pass
    
    # Try EasyOCR if available
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
        for img in preprocessed_images[:2]:  # Try first 2 versions
            result = reader.readtext(img)
            if result:
                text = ''.join([r[1] for r in result])
                text = re.sub(r'[^A-Z0-9]', '', text.upper())
                if is_valid_plate_text(text):
                    all_texts.append(('easy', text))
    except:
        pass
    
    if not all_texts:
        return None
    
    # Score and select best text
    best_score = 0
    best_text = None
    
    for source, text in all_texts:
        score = 0
        
        # Prefer longer texts (but not too long)
        if 5 <= len(text) <= 8:
            score += 10
        elif 4 <= len(text) <= 9:
            score += 5
        
        # Prefer balanced letter/digit ratio
        digits = sum(c.isdigit() for c in text)
        letters = sum(c.isalpha() for c in text)
        
        if digits >= 2 and letters >= 2:
            score += 5
        
        # Prefer certain patterns
        if re.match(r'^[A-Z]{3}[0-9]{3,4}$', text):  # ABC123 or ABC1234
            score += 10
        elif re.match(r'^[A-Z]{2,4}[0-9]{2,4}$', text):  # AB12 or AB1234
            score += 8
        
        if score > best_score:
            best_score = score
            best_text = text
    
    return best_text if best_score >= 10 else None


def process_video_with_license_plates(video_path, model_name="yolo26n", mode="ultra_fast"):
    """
    Process video with improved license plate detection
    """
    try:
        print(f"🚀 Starting video processing with LICENSE PLATE detection: {mode} mode")
        start_time = time.time()
        
        # Validate video
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return None, "Error: Video file not found"
        
        # Import YOLO
        from ultralytics import YOLO
        
        # Load model
        model_path = f"models/{model_name}.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None, f"Error: Model not found: {model_path}"
        
        print(f"🤖 Loading model: {model_path}")
        model = YOLO(model_path)
        
        # Check model classes
        print(f"📋 Model classes: {model.names if hasattr(model, 'names') else 'Unknown'}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 Device: {device}")
        
        # Mode settings
        settings = {
            'ultra_fast': {'conf': 0.4, 'imgsz': 256, 'skip': 3},
            'fast': {'conf': 0.35, 'imgsz': 320, 'skip': 2},
            'balanced': {'conf': 0.3, 'imgsz': 416, 'skip': 1}
        }
        cfg = settings.get(mode, settings['fast'])
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Error: Cannot open video file"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Output
        timestamp = int(time.time())
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"video_plates_{mode}_{timestamp}.mp4")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return None, "Error: Cannot create output video"
        
        # Create directory for cropped plates
        plates_dir = os.path.join(output_dir, f"plates_{timestamp}")
        os.makedirs(plates_dir, exist_ok=True)
        
        # Processing
        processed_count = 0
        actual_processed = 0
        total_vehicles = 0
        plates_with_text = 0
        all_detections = []  # For info panel display
        seen_plates = set()
        
        print("🎬 Processing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_count += 1
            
            if processed_count % cfg['skip'] != 0:
                continue
            
            actual_processed += 1
            current_time = datetime.now()
            
            # Progress
            if actual_processed % 30 == 0:
                elapsed = time.time() - start_time
                fps_proc = actual_processed / elapsed if elapsed > 0 else 0
                pct = (processed_count / total_frames) * 100
                eta = ((total_frames - processed_count) / (fps_proc * cfg['skip'] + 1)) / 60
                print(f"📊 {pct:.1f}% | {fps_proc:.1f} FPS | ETA: {eta:.1f}min | Vehicles: {total_vehicles}")
            
            try:
                # Detect vehicles with plates
                vehicles = detect_vehicles_and_plates(frame, model)
                
                annotated_frame = frame.copy()
                frame_detections = []  # Detections for this frame's info panel
                
                if vehicles:
                    total_vehicles += len(vehicles)
                    
                    for vehicle in vehicles:
                        # Draw vehicle bounding box
                        vx1, vy1, vx2, vy2 = vehicle['vehicle_bbox']
                        cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (0, 255, 255), 2)
                        
                        # Draw vehicle type label
                        vtype = vehicle['vehicle_type']
                        vcolor = vehicle['vehicle_color']
                        label = f"{vtype} | {vcolor}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (vx1, vy1-th-10), (vx1+tw, vy1), (0, 255, 255), -1)
                        cv2.putText(annotated_frame, label, (vx1, vy1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        # Process plates for this vehicle
                        plate_text = None
                        for plate in vehicle['plates']:
                            px1, py1, px2, py2 = plate['bbox']
                            
                            # Draw plate box
                            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                            
                            if plate['text']:
                                plate_text = plate['text']
                                plates_with_text += 1
                                
                                # Draw plate text
                                (tw, th), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                cv2.rectangle(annotated_frame, (px1, py1-th-10), (px1+tw, py1), (0, 0, 0), -1)
                                cv2.putText(annotated_frame, plate_text, (px1, py1-5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                
                                # Save plate
                                if plate_text not in seen_plates:
                                    seen_plates.add(plate_text)
                                    safe_text = re.sub(r'[^A-Z0-9]', '_', plate_text)
                                    plate_filename = f"plate_{safe_text}_f{processed_count}.jpg"
                                    plate_path = os.path.join(plates_dir, plate_filename)
                                    cv2.imwrite(plate_path, plate['crop'])
                                    print(f"🚗 Frame {processed_count}: {vtype} {vcolor} | Plate: {plate_text}")
                        
                        # Add to frame detections for info panel
                        detection_info = {
                            'plate': plate_text if plate_text else 'Unknown',
                            'vehicle_type': vehicle['vehicle_type'],
                            'vehicle_color': vehicle['vehicle_color'],
                            'conf': vehicle['conf']
                        }
                        frame_detections.append(detection_info)
                        all_detections.append(detection_info)
                
                # Draw ANPR info panel
                annotated_frame = draw_info_panel(annotated_frame, all_detections, current_time)
                
                out.write(annotated_frame)
                
            except Exception as e:
                print(f"⚠️ Frame {processed_count} error: {e}")
                out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Stats
        total_time = time.time() - start_time
        avg_fps = actual_processed / total_time if total_time > 0 else 0
        
        print(f"\n✅ Processing Complete!")
        print(f"⏱️  Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"🚀 Speed: {avg_fps:.1f} FPS")
        print(f"📊 Frames: {actual_processed}/{total_frames}")
        print(f"� Total Vehicles: {total_vehicles}")
        print(f"📝 Plates with Text: {plates_with_text}")
        print(f"🎯 Unique Plates: {len(seen_plates)}")
        
        # Show detected plates with vehicle info
        if seen_plates:
            print(f"\n🚗 Detected License Plates:")
            for plate in sorted(seen_plates):
                print(f"   • {plate}")
        
        print(f"\n💾 Output: {output_path}")
        print(f"📁 Crops: {plates_dir}")
        
        # Summary with vehicle info
        summary = f"""🎥 **ANPR Video Processing Complete!**

📊 **Statistics:**
• Mode: {mode.upper()}
• Time: {total_time:.1f}s ({total_time/60:.1f} min)
• Speed: {avg_fps:.1f} FPS
• Frames: {actual_processed}/{total_frames}

� **Vehicle Detection:**
• Total Vehicles: {total_vehicles}
• Plates Detected: {plates_with_text}
• Unique Plates: {len(seen_plates)}

� **Detected Plates:**
"""
        for plate in sorted(seen_plates)[:15]:
            summary += f"• {plate}\n"
        
        summary += f"""
💾 **Output:**
• Video: {output_path}
• Crops: {plates_dir}

✅ ANPR Processing finished! Video includes:
   - Vehicle bounding boxes (yellow)
   - Vehicle type & color labels
   - License plate boxes (green)
   - Plate text detection
   - ANPR info panel (right side)
   - Real-time date & time
"""
        
        return output_path, summary
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return None, f"Error: {str(e)}"


# Backward compatibility
def process_video_simple(video_path, model_name="yolo26n", mode="ultra_fast"):
    return process_video_with_license_plates(video_path, model_name, mode)


def process_video_ultra_fast(video_path, model_name="yolo26n", mode="ultra_fast"):
    return process_video_with_license_plates(video_path, model_name, mode)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='Video path')
    parser.add_argument('--model', default='yolo26n')
    parser.add_argument('--mode', default='fast', choices=['ultra_fast', 'fast', 'balanced'])
    
    args = parser.parse_args()
    
    output, summary = process_video_with_license_plates(args.video, args.model, args.mode)
    
    if output:
        print(f"\n✅ Success: {output}")
        print(f"\n{summary}")
    else:
        print(f"\n❌ Failed: {summary}")
