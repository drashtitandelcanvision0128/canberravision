"""
Standalone Unified Detection Module
This module provides the unified detection functionality for the main app
"""

import cv2
import numpy as np
import json
import time
import re
from datetime import datetime
from typing import Tuple, Dict, Any, List
from PIL import Image as PILImage

# Vehicle color ranges for detection (HSV)
UNIFIED_COLOR_RANGES = {
    'white': ((0, 0, 200), (180, 30, 255)),
    'black': ((0, 0, 0), (180, 255, 50)),
    'gray': ((0, 0, 50), (180, 30, 200)),
    'red': ((0, 100, 100), (10, 255, 255)),
    'red2': ((160, 100, 100), (180, 255, 255)),
    'blue': ((100, 100, 100), (140, 255, 255)),
    'green': ((40, 100, 100), (80, 255, 255)),
    'yellow': ((20, 100, 100), (40, 255, 255)),
    'orange': ((10, 100, 100), (20, 255, 255)),
    'purple': ((140, 100, 100), (160, 255, 255)),
    'brown': ((10, 50, 50), (30, 150, 150)),
    'silver': ((0, 0, 150), (180, 20, 220)),
}


def detect_unified_vehicle_color(vehicle_region: np.ndarray) -> str:
    """Detect the dominant color of a vehicle"""
    if vehicle_region.size == 0:
        return 'unknown'
    try:
        hsv = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2HSV)
        color_scores = {}
        for color_name, (lower, upper) in UNIFIED_COLOR_RANGES.items():
            if color_name == 'red2':
                continue
            lower = np.array(lower)
            upper = np.array(upper)
            if color_name == 'red':
                lower2 = np.array(UNIFIED_COLOR_RANGES['red2'][0])
                upper2 = np.array(UNIFIED_COLOR_RANGES['red2'][1])
                mask1 = cv2.inRange(hsv, lower, upper)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, lower, upper)
            color_score = np.sum(mask > 0)
            base_color = color_name.replace('2', '')
            color_scores[base_color] = color_scores.get(base_color, 0) + color_score
        if color_scores:
            dominant_color = max(color_scores, key=color_scores.get)
            total_score = sum(color_scores.values())
            if total_score > 0 and color_scores[dominant_color] / total_score > 0.1:
                return dominant_color
        return 'unknown'
    except:
        return 'unknown'


def clean_plate_text(text):
    """Clean and format license plate text - remove spaces and special chars"""
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return cleaned


def is_valid_plate_pattern(text):
    """Check if text matches Indian license plate patterns"""
    if not text or len(text) < 4:
        return False
    
    # Pattern 1: XX00XX0000 (most common Indian format e.g., MH14BN7077)
    pattern1 = r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{1,4}$'
    if re.match(pattern1, text):
        return True
    
    # Pattern 2: XX00X000 (e.g., GJ1A1234)
    pattern2 = r'^[A-Z]{2}\d{1,2}[A-Z]\d{1,4}$'
    if re.match(pattern2, text):
        return True
    
    # Pattern 3: At least 2 letters and 2 digits, 4-10 chars
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    if letters >= 2 and digits >= 2 and 4 <= len(text) <= 10:
        return True
    
    return False


def extract_plate_text_advanced(plate_region, pytesseract):
    """Extract text from plate region using multiple advanced preprocessing methods"""
    best_text = ""
    best_conf = 0
    
    try:
        methods = [
            lambda img: cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda img: cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            lambda img: cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            lambda img: cv2.threshold(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (img.shape[1]*3, img.shape[0]*3), interpolation=cv2.INTER_CUBIC), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda img: cv2.threshold(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (img.shape[1]*2, img.shape[0]*2), interpolation=cv2.INTER_CUBIC), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        ]
        
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        for method in methods:
            try:
                processed = method(plate_region)
                text = pytesseract.image_to_string(processed, config=config).strip().upper()
                cleaned = clean_plate_text(text)
                
                if cleaned and len(cleaned) >= 4:
                    if is_valid_plate_pattern(cleaned):
                        conf = 0.7 if len(cleaned) >= 8 else 0.6
                        if len(cleaned) > len(best_text) or conf > best_conf:
                            best_text = cleaned
                            best_conf = conf
            except:
                continue
                
    except Exception as e:
        print(f"[DEBUG] Plate text extraction failed: {e}")
    
    return best_text


def extract_plates_from_full_image_advanced(image, pytesseract):
    """Try to detect plates from full image using comprehensive text extraction"""
    plates = []
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        configs = [
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
            r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        ]
        
        all_texts = []
        for config in configs:
            try:
                text = pytesseract.image_to_string(gray, config=config)
                all_texts.append(text.upper())
            except:
                continue
        
        full_text = ' '.join(all_texts)
        full_text = re.sub(r'[^A-Z0-9\s]', '', full_text)
        
        pattern1 = r'[A-Z]{2}\d{2}[A-Z]{1,2}\d{1,4}'
        matches1 = re.findall(pattern1, full_text.replace(' ', ''))
        
        pattern2 = r'[A-Z]{2}\d{1,2}[A-Z]\d{1,4}'
        matches2 = re.findall(pattern2, full_text.replace(' ', ''))
        
        words = re.findall(r'\b[A-Z0-9]{4,10}\b', full_text)
        
        all_candidates = matches1 + matches2 + words
        
        for candidate in all_candidates:
            cleaned = clean_plate_text(candidate)
            if is_valid_plate_pattern(cleaned):
                is_new = True
                for existing in plates:
                    if cleaned == existing['text'] or (len(cleaned) > 4 and len(existing['text']) > 4 and (cleaned in existing['text'] or existing['text'] in cleaned)):
                        is_new = False
                        break
                
                if is_new and cleaned not in [p['text'] for p in plates]:
                    plates.append({
                        "text": cleaned,
                        "confidence": 0.65,
                        "bbox": [0, 0, 0, 0],
                        "method": "full_image_ocr"
                    })
                    
    except Exception as e:
        print(f"[DEBUG] Full image plate detection failed: {e}")
    
    return plates


def detect_license_plates(image, boxes, detection, tesseract_available):
    """Comprehensive license plate detection - from vehicle regions AND full image"""
    plates = []
    
    if not tesseract_available:
        return plates
    
    try:
        import pytesseract
        
        # First: Check YOLO detected plates
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = detection.names.get(class_id, f"class_{class_id}").lower()
            
            if 'license' in class_name or 'plate' in class_name:
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                plate_region = image[int(y1):int(y2), int(x1):int(x2)]
                
                if plate_region.size > 0:
                    plate_text = extract_plate_text_advanced(plate_region, pytesseract)
                    
                    if plate_text and len(plate_text) >= 4:
                        plates.append({
                            "text": plate_text,
                            "confidence": 0.85,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "method": "yolo_plate_detection"
                        })
                        print(f"[DEBUG] Found plate via YOLO: {plate_text}")
        
        # Second: Extract from vehicle regions
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = detection.names.get(class_id, f"class_{class_id}").lower()
            
            vehicle_types = ['car', 'truck', 'bus', 'motorcycle', 'bike', 'van', 'suv']
            is_vehicle = any(vtype in class_name for vtype in vehicle_types)
            
            if is_vehicle:
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                vehicle_region = image[int(y1):int(y2), int(x1):int(x2)]
                
                if vehicle_region.size > 0:
                    plate_text = extract_plate_text_advanced(vehicle_region, pytesseract)
                    
                    if plate_text and len(plate_text) >= 4:
                        is_new = True
                        for existing_plate in plates:
                            if plate_text == existing_plate['text'] or (abs(existing_plate['bbox'][0] - float(x1)) < 50 and abs(existing_plate['bbox'][1] - float(y1)) < 50):
                                is_new = False
                                break
                        
                        if is_new:
                            plates.append({
                                "text": plate_text,
                                "confidence": 0.75,
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "method": "vehicle_region_ocr"
                            })
                            print(f"[DEBUG] Found plate in vehicle region: {plate_text}")
        
        # Third: Full image OCR if needed
        if len(plates) < 2:
            full_image_plates = extract_plates_from_full_image_advanced(image, pytesseract)
            for plate in full_image_plates:
                is_new = True
                for existing_plate in plates:
                    if plate['text'] == existing_plate['text']:
                        is_new = False
                        break
                
                if is_new:
                    plates.append(plate)
                    print(f"[DEBUG] Found plate via full image OCR: {plate['text']}")
            
    except Exception as e:
        print(f"[WARNING] License plate detection failed: {e}")
    
    return plates


def process_unified_detection_simple(image, conf_threshold=0.5, get_model_func=None, tesseract_available=False, parking_available=False):
    """Simplified unified detection that doesn't depend on external dataclasses"""
    if image is None:
        return None, "{}", "Please upload an image"
    
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, PILImage.Image):
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        start_time = time.time()
        
        if get_model_func is None:
            return None, "{}", "Model function not provided"
        
        model = get_model_func("yolo26n")
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
        
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=0.5,
            device=device,
            verbose=False
        )
        
        detections = {
            "objects": [],
            "vehicles": [],
            "number_plates": [],
            "ppe": [],
            "parking": []
        }
        
        annotated = image.copy()
        detection = results[0] if results else None
        
        if detection and hasattr(detection, 'boxes') and detection.boxes is not None:
            boxes = detection.boxes
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = detection.names.get(class_id, f"class_{class_id}").lower()
                
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                
                vehicle_types = ['car', 'truck', 'bus', 'motorcycle', 'bike', 'van', 'suv']
                is_vehicle = any(vtype in class_name for vtype in vehicle_types)
                
                if is_vehicle:
                    vehicle_region = image[int(y1):int(y2), int(x1):int(x2)]
                    color = detect_unified_vehicle_color(vehicle_region)
                    
                    vtype = 'car'
                    if 'truck' in class_name:
                        vtype = 'truck'
                    elif 'bus' in class_name:
                        vtype = 'bus'
                    elif 'bike' in class_name or 'motorcycle' in class_name:
                        vtype = 'bike'
                    
                    vehicle_info = {
                        "id": f"VEH_{i+1:04d}",
                        "type": vtype,
                        "color": color,
                        "confidence": round(confidence, 2),
                        "bbox": bbox
                    }
                    detections["vehicles"].append(vehicle_info)
                    
                    color_map = {'bike': (0, 255, 255), 'car': (0, 255, 0), 'truck': (255, 0, 0), 'bus': (255, 255, 0)}
                    box_color = color_map.get(vtype, (128, 128, 128))
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                    label = f"{vtype.upper()} | {color} | {confidence:.2f}"
                    cv2.putText(annotated, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                elif 'person' in class_name:
                    person_info = {
                        "person_id": f"PER_{i+1:04d}",
                        "helmet": False,
                        "seatbelt": False,
                        "vest": False,
                        "confidence": round(confidence, 2),
                        "bbox": bbox,
                        "vehicle_type": "unknown"
                    }
                    detections["ppe"].append(person_info)
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
                    cv2.putText(annotated, f"PERSON {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                
                obj_info = {
                    "id": f"OBJ_{i+1:04d}",
                    "label": class_name,
                    "confidence": round(confidence, 2),
                    "bbox": bbox
                }
                detections["objects"].append(obj_info)
        
        # Detect license plates
        if tesseract_available:
            plates = detect_license_plates(image, boxes, detection, tesseract_available)
            detections["number_plates"] = plates
            
            for idx, plate in enumerate(plates):
                if plate['bbox'] != [0, 0, 0, 0]:
                    x1, y1, x2, y2 = plate['bbox']
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                cv2.putText(annotated, f"PLATE: {plate['text']}", (10, 30 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Add parking detection
        if parking_available:
            try:
                from modules.parking_detection import ParkingDetector
                parking_detector = ParkingDetector()
                parking_result = parking_detector.detect(image)
                if hasattr(parking_result, 'slots'):
                    for idx, slot in enumerate(parking_result.slots):
                        slot_info = {
                            "slot_id": idx + 1,
                            "occupied": slot.get('occupied', False),
                            "confidence": slot.get('confidence', 0.5),
                            "bbox": slot.get('bbox', [0, 0, 0, 0])
                        }
                        detections["parking"].append(slot_info)
                        bbox = slot.get('bbox', [0, 0, 0, 0])
                        color = (0, 0, 255) if slot.get('occupied', False) else (0, 255, 0)
                        status = "OCCUPIED" if slot.get('occupied', False) else "EMPTY"
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f"Slot {idx+1}: {status}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            except:
                pass
        
        output = {
            "source_type": "IMAGE",
            "timestamp": datetime.now().isoformat(),
            "frame_id": "0",
            "detections": detections,
            "metadata": {
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "total_detections": sum(len(v) for v in detections.values())
            }
        }
        
        json_output = json.dumps(output, indent=2, ensure_ascii=False)
        
        summary_lines = [
            "🎯 UNIFIED DETECTION RESULTS",
            "=" * 40,
            f"\n🚗 Vehicles: {len(detections['vehicles'])}",
            f"👥 Persons: {len(detections['ppe'])}",
            f"📋 Plates: {len(detections['number_plates'])}",
            f"🅿️ Parking: {len(detections['parking'])} slots",
            f"\n⚡ Processing: {output['metadata']['processing_time_ms']:.1f}ms"
        ]
        
        if detections['vehicles']:
            summary_lines.append("\nVehicle Details:")
            for v in detections['vehicles']:
                summary_lines.append(f"  • {v['type'].upper()} ({v['color']}) - {v['confidence']:.2f}")
        
        if detections['number_plates']:
            summary_lines.append("\nLicense Plates:")
            for p in detections['number_plates']:
                summary_lines.append(f"  • {p['text']}")
        
        summary = "\n".join(summary_lines)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        return annotated_rgb, json_output, summary
        
    except Exception as e:
        error_msg = f"Error in unified detection: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return None, "{}", error_msg
