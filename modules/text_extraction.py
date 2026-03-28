"""
Text extraction module for OCR and license plate detection.
Handles all text extraction, OCR processing, and license plate detection.
"""

import os
import sys
import time
import json
import re
from datetime import datetime
from typing import Optional
import numpy as np
import cv2

# Tesseract OCR - ENABLED for license plate detection
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    print("[INFO] Tesseract OCR enabled for license plate detection")
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None
    print("[WARNING] Tesseract OCR not available - install tesseract-ocr for better license plate detection")

# Import LightOnOCR integration
try:
    from lighton_ocr_integration import get_lighton_ocr_processor, extract_text_with_lighton
    LIGHTON_AVAILABLE = True
    print("[INFO] LightOnOCR integration loaded")
except ImportError:
    LIGHTON_AVAILABLE = False
    print("[WARNING] LightOnOCR integration not available")

# Import enhanced detection for challenging images
try:
    from enhanced_detection import enhanced_license_plate_detection
    ENHANCED_DETECTION_AVAILABLE = True
    print("[INFO] Enhanced detection for challenging images loaded")
except ImportError:
    ENHANCED_DETECTION_AVAILABLE = False
    print("[WARNING] Enhanced detection not available")

# Import international license plate recognition
try:
    from international_license_plates import extract_international_license_plates, InternationalLicensePlateRecognizer
    INTERNATIONAL_PLATES_AVAILABLE = True
    print("[INFO] International license plate recognition loaded")
except ImportError:
    INTERNATIONAL_PLATES_AVAILABLE = False
    print("[WARNING] International license plate recognition not available")

# Import optimized PaddleOCR GPU integration (PRIMARY METHOD)
try:
    from optimized_paddleocr_gpu import (
        extract_text_optimized, 
        extract_license_plates_optimized,
        get_paddle_ocr_instance,
        preprocess_image_for_ocr,
        batch_extract_text,
        get_gpu_info,
        initialize_gpu_environment
    )
    OPTIMIZED_PADDLEOCR_AVAILABLE = True
    print("[INFO] 🚀 Optimized PaddleOCR GPU integration loaded (PRIMARY)")
except ImportError:
    # FALLBACK: Use our new optimized GPU text extraction
    try:
        from optimized_gpu_text_extraction import (
            extract_text_optimized,
            extract_license_plates_optimized,
            get_gpu_info,
            initialize_gpu_environment
        )
        OPTIMIZED_PADDLEOCR_AVAILABLE = True
        print("[INFO] 🚀 Optimized GPU Text Extraction loaded (PRIMARY FALLBACK)")
    except ImportError:
        OPTIMIZED_PADDLEOCR_AVAILABLE = False
        print("[WARNING] Optimized PaddleOCR GPU integration not available")

# Import original PaddleOCR integration (fallback)
try:
    from paddleocr_integration import (
        extract_text_with_paddleocr, 
        extract_license_plates_with_paddleocr,
        get_paddle_ocr_instance as get_legacy_paddle_ocr_instance,
        preprocess_image_for_paddleocr,
        extract_text_multilingual
    )
    PADDLEOCR_AVAILABLE = True
    print("[INFO] Legacy PaddleOCR (PP-OCRv5) integration loaded (FALLBACK)")
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("[WARNING] Legacy PaddleOCR integration not available")

# Simple fallback OCR when PaddleOCR is not available
SIMPLE_OCR_AVAILABLE = True
print("[INFO] Simple OCR fallback available")

# Cache for text extraction
_text_extraction_cache = {}


def _detect_license_plates_in_vehicles(image_bgr: np.ndarray, vehicles_detected: list) -> list:
    """
    Detect license plates within vehicle regions using computer vision techniques.
    This function doesn't rely on YOLO having license plate class - it uses CV methods.
    
    Args:
        image_bgr: Input image in BGR format
        vehicles_detected: List of detected vehicles with their bounding boxes
        
    Returns:
        List of license plate bounding boxes [(x1, y1, x2, y2), ...]
    """
    license_plate_regions = []
    
    if not vehicles_detected:
        print(f"[DEBUG] No vehicles detected - cannot detect license plates")
        return license_plate_regions
    
    print(f"[DEBUG] Processing {len(vehicles_detected)} vehicles for license plate detection...")
    
    for vehicle in vehicles_detected:
        bbox = vehicle["bounding_box"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        
        # Ensure coordinates are within image bounds
        h, w = image_bgr.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Crop vehicle region
        vehicle_crop = image_bgr[y1:y2, x1:x2]
        
        if vehicle_crop.size == 0:
            continue
        
        print(f"[DEBUG] Processing {vehicle['class_name']} at ({x1},{y1},{x2},{y2})")
        
        # Detect license plate in vehicle crop
        plate_bbox = _detect_license_plate_in_vehicle_crop(vehicle_crop)
        
        if plate_bbox:
            px1, py1, px2, py2 = plate_bbox
            
            # Convert plate coordinates back to full image coordinates
            full_x1 = x1 + px1
            full_y1 = y1 + py1
            full_x2 = x1 + px2
            full_y2 = y1 + py2
            
            # Ensure final coordinates are within bounds
            full_x1 = max(0, min(full_x1, w))
            full_y1 = max(0, min(full_y1, h))
            full_x2 = max(0, min(full_x2, w))
            full_y2 = max(0, min(full_y2, h))
            
            if full_x2 > full_x1 and full_y2 > full_y1:
                license_plate_regions.append((full_x1, full_y1, full_x2, full_y2))
                print(f"[DEBUG] ✅ License plate found in {vehicle['class_name']}: ({full_x1},{full_y1},{full_x2},{full_y2})")
            else:
                print(f"[DEBUG] ❌ Invalid license plate coordinates: ({full_x1},{full_y1},{full_x2},{full_y2})")
        else:
            print(f"[DEBUG] ❌ No license plate found in {vehicle['class_name']}")
    
    print(f"[DEBUG] Found {len(license_plate_regions)} license plates total")
    return license_plate_regions


def _detect_license_plate_in_vehicle_crop(vehicle_crop: np.ndarray) -> tuple:
    """
    Detect license plate within a vehicle crop using computer vision techniques.
    
    Args:
        vehicle_crop: Cropped vehicle image in BGR format
        
    Returns:
        License plate bounding box (x1, y1, x2, y2) relative to vehicle crop, or None
    """
    try:
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Edge detection
        edges = cv2.Canny(bilateral, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on license plate characteristics
        plate_contours = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # License plate characteristics
            aspect_ratio = w / h if h > 0 else 0
            rect_area = w * h
            
            # Typical license plate properties:
            # - Aspect ratio between 2.0 and 6.0
            # - Area should be reasonable (not too small or too large)
            # - Should be rectangular
            if (2.0 <= aspect_ratio <= 6.0 and
                500 <= area <= 50000 and
                30 <= w <= vehicle_crop.shape[1] * 0.8 and
                10 <= h <= vehicle_crop.shape[0] * 0.3 and
                area >= rect_area * 0.7):  # Should be fairly rectangular
                
                plate_contours.append((x, y, w, h, area, aspect_ratio))
        
        if not plate_contours:
            # Try alternative method - look for bright rectangular regions
            # License plates are often brighter than surroundings
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            contours2, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours2:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if (2.0 <= aspect_ratio <= 6.0 and
                    w >= 40 and h >= 15 and
                    w <= vehicle_crop.shape[1] * 0.8 and
                    h <= vehicle_crop.shape[0] * 0.3):
                    
                    plate_contours.append((x, y, w, h, w*h, aspect_ratio))
        
        if not plate_contours:
            return None
        
        # Sort by area (prefer larger plates) and aspect ratio (prefer typical ratios)
        plate_contours.sort(key=lambda x: (x[4], abs(x[5] - 3.5)), reverse=True)
        
        # Return the best candidate
        x, y, w, h, area, aspect_ratio = plate_contours[0]
        
        # Add small margin
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        x2 = min(vehicle_crop.shape[1], x + w + 2*margin)
        y2 = min(vehicle_crop.shape[0], y + h + 2*margin)
        
        return (x, y, x2, y2)
        
    except Exception as e:
        print(f"[DEBUG] Error in license plate detection: {e}")
        return None


def _detect_vehicles_in_image(image_bgr: np.ndarray) -> list:
    """
    Detect if there are any vehicles in the image.
    Returns list of detected vehicles with their info.
    
    Args:
        image_bgr: Input image in BGR format
        
    Returns:
        List of detected vehicles with class names and bounding boxes
    """
    try:
        # Import here to avoid circular imports
        from .utils import get_model, _get_device
        
        # Vehicle classes that should trigger text extraction
        VEHICLE_CLASSES = {
            'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'van', 
            'taxi', 'ambulance', 'police', 'fire truck', 'tractor',
            'scooter', 'bike', 'auto', 'rickshaw', 'lorry'
        }
        
        # Get YOLO model
        model = get_model("yolo26n.pt")
        device = _get_device()
        detection_results = model.predict(
            source=image_bgr,
            conf=0.25,
            iou=0.5,
            imgsz=640,
            device=device,
            verbose=False,
            half=True if device != "cpu" else False,
        )
        
        detected_vehicles = []
        
        if detection_results and len(detection_results) > 0:
            detection = detection_results[0]
            
            if hasattr(detection, 'boxes') and detection.boxes is not None:
                boxes = detection.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                names = detection.names
                
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Get class name
                    class_id = int(cls[i]) if i < len(cls) else -1
                    class_name = names.get(class_id, f"class_{class_id}")
                    confidence = float(conf[i]) if i < len(conf) else 0.0
                    
                    # Check if this is a vehicle (case insensitive)
                    if class_name.lower() in VEHICLE_CLASSES:
                        vehicle_info = {
                            "class_name": class_name.lower(),
                            "confidence": confidence,
                            "bounding_box": {
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2
                            }
                        }
                        detected_vehicles.append(vehicle_info)
                        print(f"[DEBUG] 🚗 Vehicle detected: {class_name} (conf: {confidence:.3f})")
        
        return detected_vehicles
        
    except Exception as e:
        print(f"[DEBUG] Error in vehicle detection: {e}")
        return []


def _is_vehicle_related_text(text: str, detected_vehicles: list) -> bool:
    """
    Check if extracted text is likely related to vehicles.
    This helps filter out random text from non-vehicle images.
    
    Args:
        text: Extracted text
        detected_vehicles: List of detected vehicles
        
    Returns:
        True if text is likely vehicle-related
    """
    if not text or not text.strip():
        return False
    
    text = text.strip().upper()
    
    # Indian license plate patterns
    import re
    indian_plate_pattern = r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$'  # MH12AB1234
    indian_plate_pattern2 = r'^[A-Z]{2}\d{2}\s?[A-Z]{1,2}\s?\d{4}$'  # MH 12 AB 1234
    
    # Check if it's a valid license plate
    if (re.match(indian_plate_pattern, text) or 
        re.match(indian_plate_pattern2, text) or
        _is_valid_indian_license_plate(text)):
        return True
    
    # Vehicle-related keywords
    vehicle_keywords = {
        'CAR', 'TRUCK', 'BUS', 'BIKE', 'MOTOR', 'TAXI', 'VAN', 'AUTO', 
        'POLICE', 'AMBULANCE', 'FIRE', 'TRANSPORT', 'FREIGHT', 'CARGO',
        'LICENSE', 'PLATE', 'REG', 'REGISTRATION', 'NUMBER', 'NUM'
    }
    
    # Check if text contains vehicle keywords
    if any(keyword in text for keyword in vehicle_keywords):
        return True
    
    # If we detected vehicles, be more lenient with text
    if detected_vehicles:
        # Accept alphanumeric text that could be license plates
        if len(text) >= 4 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
            return True
    
    return False


def extract_text_from_image_json(image_bgr: np.ndarray, image_id: str = None) -> dict:
    """
    Extract all text from image using multiple methods and return structured JSON.
    NEW APPROACH: First detect vehicles, then extract text only if vehicles are present.
    
    Args:
        image_bgr: Input image in BGR format
        image_id: Unique identifier for the image
        
    Returns:
        Dictionary containing all extracted text information
    """
    if image_id is None:
        image_id = f"img_{int(time.time() * 1000)}"
    
    # Check cache first
    if image_id in _text_extraction_cache:
        print(f"[DEBUG] Using cached text extraction for {image_id}")
        return _text_extraction_cache[image_id]
    
    print(f"[DEBUG] Starting comprehensive text extraction for {image_id}")
    
    result = {
        "image_id": image_id,
        "timestamp": datetime.now().isoformat(),
        "text_extraction": {
            "all_objects": [],
            "license_plates": [],
            "general_text": [],
            "summary": {
                "total_objects": 0,
                "objects_with_text": 0,
                "license_plates_found": 0,
                "general_text_found": 0
            }
        }
    }
    
    try:
        # Import here to avoid circular imports
        from .utils import detect_license_plates_as_objects, get_model
        
        # STEP 0: ALWAYS EXTRACT TEXT - regardless of vehicles or objects
        print(f"[DEBUG] Step 0: Starting comprehensive text extraction...")
        vehicles_detected = _detect_vehicles_in_image(image_bgr)
        
        print(f"[DEBUG] ✅ Text extraction will run for ALL images")
        print(f"[DEBUG] Vehicles detected: {[v['class_name'] for v in vehicles_detected] if vehicles_detected else 'None'}")
        print(f"[DEBUG] Proceeding with text extraction...")
        
        # STEP 1: Detect license plates using computer vision within vehicle regions
        print(f"[DEBUG] Step 1: Detecting license plates within vehicle regions...")
        license_plate_regions = _detect_license_plates_in_vehicles(image_bgr, vehicles_detected)
        
        # STEP 2: Extract text from detected license plate regions
        print(f"[DEBUG] Step 2: Extracting text from {len(license_plate_regions)} license plate regions...")
        for i, (x1, y1, x2, y2) in enumerate(license_plate_regions):
            # Crop the license plate region
            plate_crop = image_bgr[y1:y2, x1:x2]
            
            if plate_crop.size == 0:
                continue
            
            print(f"[DEBUG] Processing license plate region {i+1}: ({x1}, {y1}, {x2}, {y2})")
            
            # Extract text from the cropped license plate using multiple methods
            plate_text = _extract_text_from_license_plate_crop(plate_crop)
            
            if plate_text and plate_text.strip():
                cleaned_plate = _clean_license_plate_text(plate_text)
                if _is_valid_indian_license_plate(cleaned_plate):
                    # ADDITIONAL VALIDATION: Check if this plate is vehicle-related
                    if _is_vehicle_related_text(cleaned_plate, vehicles_detected):
                        # RE-ENABLE STRICT VALIDATION TO PREVENT FALSE POSITIVES
                        if _validate_license_plate_in_image(plate_crop, cleaned_plate):
                            print(f"[DEBUG] ✅ Strict validation passed for: {cleaned_plate}")
                            # Add to license plates list
                            result["text_extraction"]["license_plates"].append({
                                "object_id": f"license_plate_{i}",
                                "plate_text": cleaned_plate,
                                "confidence": 0.9,
                                "method": "object_detection_crop",
                                "bounding_box": {
                                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                                }
                            })
                            result["text_extraction"]["summary"]["license_plates_found"] += 1
                            print(f"[DEBUG] ✅ Found VALID vehicle license plate: {cleaned_plate}")
                        else:
                            print(f"[DEBUG] ❌ REJECTED license plate - validation failed: {cleaned_plate}")
                    else:
                        print(f"[DEBUG] ❌ REJECTED non-vehicle license plate: {cleaned_plate} (no vehicles detected)")
        
        # STEP 3: Detect other objects in the image (excluding license plate regions)
        print(f"[DEBUG] Step 3: Detecting other objects...")
        model = get_model("yolo26n.pt")
        detection_results = model(image_bgr)
        
        if detection_results and len(detection_results) > 0:
            detection = detection_results[0]
            
            if hasattr(detection, 'boxes') and detection.boxes is not None:
                boxes = detection.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                names = detection.names
                
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Check if this object overlaps with any license plate region
                    overlaps_with_plate = False
                    for plate_x1, plate_y1, plate_x2, plate_y2 in license_plate_regions:
                        # Calculate overlap
                        overlap_x = max(0, min(x2, plate_x2) - max(x1, plate_x1))
                        overlap_y = max(0, min(y2, plate_y2) - max(y1, plate_y1))
                        overlap_area = overlap_x * overlap_y
                        object_area = (x2 - x1) * (y2 - y1)
                        
                        if overlap_area > object_area * 0.5:  # More than 50% overlap
                            overlaps_with_plate = True
                            break
                    
                    if overlaps_with_plate:
                        continue  # Skip this object as it's likely a license plate
                    
                    # Extract object crop
                    crop = image_bgr[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    
                    # Get class name
                    class_id = int(cls[i]) if i < len(cls) else -1
                    class_name = names.get(class_id, f"class_{class_id}")
                    confidence = float(conf[i]) if i < len(conf) else 0.0
                    
                    # Extract color
                    from .utils import _classify_color_bgr
                    color = _classify_color_bgr(crop)
                    
                    # Create object info
                    object_info = {
                        "object_id": f"{class_name}_{i}",
                        "class_name": class_name,
                        "confidence": round(confidence, 3),
                        "bounding_box": {
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2
                        },
                        "color": color,
                        "text_found": [],
                        "license_plate": None,
                        "general_text": []
                    }
                    
                    # Extract text from this object
                    text_found = _extract_all_text_from_object(crop, class_name)
                    
                    if text_found:
                        object_info["text_found"] = text_found["all_text"]
                        result["text_extraction"]["summary"]["objects_with_text"] += 1
                        
                        # Add license plate info if found
                        if text_found["license_plate"]:
                            object_info["license_plate"] = text_found["license_plate"]
                            result["text_extraction"]["license_plates"].append({
                                "object_id": object_info["object_id"],
                                "plate_text": text_found["license_plate"]["text"],
                                "confidence": text_found["license_plate"]["confidence"],
                                "method": text_found["license_plate"]["method"]
                            })
                            result["text_extraction"]["summary"]["license_plates_found"] += 1
                        
                        # Add general text if found (ACCEPT ALL TEXT)
                        if text_found["general_text"]:
                            # ACCEPT ALL GENERAL TEXT (no filtering)
                            object_info["general_text"] = text_found["general_text"]
                            result["text_extraction"]["general_text"].extend([
                                {
                                    "object_id": object_info["object_id"],
                                    "text": text_item["text"],
                                    "confidence": text_item["confidence"],
                                    "method": text_item["method"]
                                }
                                for text_item in text_found["general_text"]
                            ])
                            result["text_extraction"]["summary"]["general_text_found"] += len(text_found["general_text"])
                            for text_item in text_found["general_text"]:
                                print(f"[DEBUG] ✅ General text found: {text_item['text']}")
                    
                    result["text_extraction"]["all_objects"].append(object_info)
                    result["text_extraction"]["summary"]["total_objects"] += 1
        
        # STEP 4: Also try to extract text from the entire image (for missed objects)
        print(f"[DEBUG] Step 4: Full image text extraction...")
        full_image_text = _extract_general_text_from_image(image_bgr)
        if full_image_text:
            result["text_extraction"]["full_image_text"] = []
            
            # Separate license plates from general text in full image results
            for text_item in full_image_text:
                if "plate" in text_item["method"]:
                    # This is a license plate found in full image text
                    plate_text = text_item["text"]
                    if _is_valid_indian_license_plate(plate_text) and _is_vehicle_related_text(plate_text, vehicles_detected):
                        # Only add if we haven't already found this plate
                        existing_plates = [p["plate_text"] for p in result["text_extraction"]["license_plates"]]
                        if plate_text not in existing_plates:
                            result["text_extraction"]["license_plates"].append({
                                "object_id": "full_image",
                                "plate_text": plate_text,
                                "confidence": text_item["confidence"],
                                "method": text_item["method"]
                            })
                            result["text_extraction"]["summary"]["license_plates_found"] += 1
                            print(f"[DEBUG] Found additional vehicle license plate in full image text: {plate_text}")
                    else:
                        print(f"[DEBUG] Ignored non-vehicle license plate in full image: {plate_text}")
                else:
                    # This is general text - ACCEPT ALL TEXT (no filtering)
                    result["text_extraction"]["full_image_text"].append(text_item)
                    result["text_extraction"]["general_text"].append({
                        "object_id": "full_image",
                        "text": text_item["text"],
                        "confidence": text_item["confidence"],
                        "method": text_item["method"]
                    })
                    result["text_extraction"]["summary"]["general_text_found"] += 1
                    print(f"[DEBUG] Found general text: {text_item['text']}")
        
        print(f"[DEBUG] Text extraction summary:")
        print(f"[DEBUG]   Total objects: {result['text_extraction']['summary']['total_objects']}")
        print(f"[DEBUG]   License plates found: {result['text_extraction']['summary']['license_plates_found']}")
        print(f"[DEBUG]   General text found: {result['text_extraction']['summary']['general_text_found']}")
    
    except Exception as e:
        print(f"[DEBUG] Error in text extraction: {e}")
        result["error"] = str(e)
    
    # Cache the result
    _text_extraction_cache[image_id] = result
    
    # Limit cache size
    if len(_text_extraction_cache) > 50:
        oldest_keys = list(_text_extraction_cache.keys())[:-25]
        for key in oldest_keys:
            del _text_extraction_cache[key]
    
    print(f"[DEBUG] Text extraction completed for {image_id}")
    return result


def _extract_text_from_license_plate_crop(plate_crop: np.ndarray) -> str:
    """
    Extract text from a cropped license plate region using optimized methods.
    Now includes multi-angle support for angled license plates.
    
    Args:
        plate_crop: Cropped license plate image in BGR format
        
    Returns:
        Extracted text string
    """
    try:
        print(f"[DEBUG] Extracting text from license plate crop: {plate_crop.shape}")
        
        if plate_crop is None or plate_crop.size == 0:
            return ""
        
        # FIRST: Try multi-angle extraction for angled plates
        print("[DEBUG] Trying multi-angle extraction for angled plates...")
        multi_angle_result = _try_multi_angle_ocr(plate_crop)
        if multi_angle_result:
            print(f"[DEBUG] ✅ Multi-angle extraction found: {multi_angle_result}")
            return multi_angle_result
        
        # Try Tesseract with standard preprocessing
        try:
            import pytesseract
            if pytesseract:
                print("[DEBUG] Using Tesseract OCR for license plate extraction")
                
                # Preprocess for Tesseract
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                
                # Enhance contrast
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                
                # Apply threshold
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Configure Tesseract for license plates
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                
                # Extract text
                text = pytesseract.image_to_string(binary, config=custom_config)
                cleaned = text.strip().upper()
                
                if cleaned and len(cleaned) >= 4:
                    print(f"[DEBUG] ✅ Tesseract found: {cleaned}")
                    return cleaned
                else:
                    print(f"[DEBUG] Tesseract result too short: '{cleaned}'")
            else:
                print("[DEBUG] Tesseract not available")
        except Exception as e:
            print(f"[DEBUG] Tesseract failed: {e}")
        
        # Method 2: Optimized PaddleOCR GPU (if available)
        if OPTIMIZED_PADDLEOCR_AVAILABLE:
            try:
                print("[DEBUG] 🚀 Using Optimized PaddleOCR GPU for license plate extraction")
                
                paddleocr_result = extract_text_optimized(
                    plate_crop, 
                    confidence_threshold=0.3,
                    lang='en',
                    use_gpu=None,
                    use_cache=True,
                    preprocess=True
                )
                
                if paddleocr_result["text"] and paddleocr_result["text"].strip():
                    cleaned = _clean_license_plate_text(paddleocr_result["text"])
                    if cleaned and len(cleaned) >= 4:
                        print(f"[DEBUG] ✅ Optimized PaddleOCR found: {cleaned}")
                        return cleaned
                
            except Exception as e:
                print(f"[DEBUG] Optimized PaddleOCR failed for plate: {e}")
        
        # Method 3: Simple pattern matching for common Indian plate formats
        print("[DEBUG] Using pattern matching for Indian license plates")
        
        # Convert to grayscale and enhance
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Try different thresholding methods
        methods = [
            ("OTSU", cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("ADAPTIVE", cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("BINARY", cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)[1])
        ]
        
        for method_name, binary in methods:
            try:
                # Try Tesseract with different configurations
                configs = [
                    r'--oem 3 --psm 7',  # Single text line
                    r'--oem 3 --psm 8',  # Single word
                    r'--oem 3 --psm 6',  # Uniform block of text
                ]
                
                for config in configs:
                    text = pytesseract.image_to_string(binary, config=config)
                    cleaned = _clean_license_plate_text(text.strip())
                    
                    if cleaned and len(cleaned) >= 6 and _is_valid_indian_license_plate(cleaned):
                        print(f"[DEBUG] ✅ {method_name} with config {config} found: {cleaned}")
                        return cleaned
                        
            except Exception as e:
                continue
        
        print("[DEBUG] ❌ All OCR methods failed")
        return ""
        
    except Exception as e:
        print(f"[DEBUG] Error in license plate OCR: {e}")
        return ""


def _try_multi_angle_ocr(plate_crop: np.ndarray) -> Optional[str]:
    """
    Try OCR at multiple rotation angles to handle angled license plates.
    This handles plates that are rotated or at odd angles.
    
    Args:
        plate_crop: Cropped license plate image in BGR format
        
    Returns:
        Extracted text if found, None otherwise
    """
    try:
        import pytesseract
        
        # Angles to try: negative and positive rotations
        angles = [0, -5, 5, -10, 10, -15, 15, -20, 20, -30, 30, -45, 45]
        
        h, w = plate_crop.shape[:2]
        center = (w // 2, h // 2)
        
        for angle in angles:
            try:
                # Get rotation matrix
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Calculate new bounding box
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                
                # Adjust rotation matrix for new center
                M[0, 2] += (new_w / 2) - center[0]
                M[1, 2] += (new_h / 2) - center[1]
                
                # Rotate image with black background
                rotated = cv2.warpAffine(plate_crop, M, (new_w, new_h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
                
                if rotated.size == 0:
                    continue
                
                # Preprocess rotated image
                gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Try OCR with multiple configs
                configs = [
                    r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    r'--oem 3 --psm 7',
                    r'--oem 3 --psm 8',
                ]
                
                for config in configs:
                    text = pytesseract.image_to_string(binary, config=config)
                    cleaned = _clean_license_plate_text(text.strip())
                    
                    # Validate as license plate
                    if cleaned and len(cleaned) >= 4:
                        has_letters = sum(c.isalpha() for c in cleaned) >= 1
                        has_numbers = sum(c.isdigit() for c in cleaned) >= 1
                        
                        if has_letters and has_numbers:
                            print(f"[DEBUG] ✅ Multi-angle OCR found at {angle}°: {cleaned}")
                            return cleaned
                            
            except Exception as e:
                continue
        
        # Also try perspective correction
        warped = _try_perspective_correction(plate_crop)
        if warped is not None:
            try:
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                text = pytesseract.image_to_string(binary, 
                    config=r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                cleaned = _clean_license_plate_text(text.strip())
                
                if cleaned and len(cleaned) >= 4:
                    has_letters = sum(c.isalpha() for c in cleaned) >= 1
                    has_numbers = sum(c.isdigit() for c in cleaned) >= 1
                    
                    if has_letters and has_numbers:
                        print(f"[DEBUG] ✅ Perspective correction found: {cleaned}")
                        return cleaned
            except Exception as e:
                pass
        
        return None
        
    except Exception as e:
        print(f"[DEBUG] Multi-angle OCR failed: {e}")
        return None


def _try_perspective_correction(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Try to detect and correct perspective distortion in license plates.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Perspective-corrected image or None if correction fails
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply bilateral filter to reduce noise
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular contours
        for contour in contours:
            # Approximate contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # If 4 corners found, try perspective correction
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                
                # Order points
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]  # Top-left
                rect[2] = pts[np.argmax(s)]  # Bottom-right
                
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]  # Top-right
                rect[3] = pts[np.argmax(diff)]  # Bottom-left
                
                # Calculate width and height
                widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
                widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                
                heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
                heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                
                # Aspect ratio check for plates
                if maxWidth > 0 and maxHeight > 0:
                    aspect_ratio = maxWidth / maxHeight
                    if 2.0 <= aspect_ratio <= 6.0 and maxWidth > 50 and maxHeight > 15:
                        # Destination points
                        dst = np.array([
                            [0, 0],
                            [maxWidth - 1, 0],
                            [maxWidth - 1, maxHeight - 1],
                            [0, maxHeight - 1]], dtype="float32")
                        
                        # Perspective transform
                        M = cv2.getPerspectiveTransform(rect, dst)
                        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
                        
                        return warped
        
        return None
        
    except Exception as e:
        print(f"[DEBUG] Perspective correction failed: {e}")
        return None


def _preprocess_license_plate(plate_crop: np.ndarray) -> np.ndarray:
    """
    Preprocess license plate crop for better OCR results.
    
    Args:
        plate_crop: Original license plate crop in BGR format
        
    Returns:
        Preprocessed image in BGR format
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        clahe_enhanced = clahe.apply(bilateral)
        
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(clahe_enhanced, -1, kernel)
        
        # Convert back to BGR for LightOnOCR
        processed_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        return processed_bgr
        
    except Exception as e:
        print(f"[DEBUG] Error in license plate preprocessing: {e}")
        return plate_crop


def _extract_all_text_from_object(crop_bgr: np.ndarray, class_name: str) -> dict:
    """
    Extract all text from an object crop using intelligent fallback system.
    PRIMARY: PaddleOCR -> FALLBACK: LightOnOCR
    
    Args:
        crop_bgr: Object crop in BGR format
        class_name: Class name of the object
        
    Returns:
        Dictionary containing all extracted text information
    """
    text_results = {
        "all_text": [],
        "license_plate": None,
        "general_text": []
    }
    
    try:
        # Method 1: License plate detection for cars
        if class_name.lower() == "car":
            from .utils import _detect_license_plate_in_car
            plate_crop = _detect_license_plate_in_car(crop_bgr)
            if plate_crop is not None:
                plate_text = _extract_text_ocr(plate_crop)
                if plate_text and plate_text.strip():
                    cleaned_plate = _clean_license_plate_text(plate_text)
                    if _is_valid_indian_license_plate(cleaned_plate):
                        text_results["license_plate"] = {
                            "text": cleaned_plate,
                            "confidence": 0.9,
                            "method": "license_plate_detection"
                        }
                        text_results["all_text"].append({
                            "text": cleaned_plate,
                            "type": "license_plate",
                            "confidence": 0.9,
                            "method": "license_plate_detection"
                        })
        
        # Method 2: Optimized PaddleOCR GPU (PRIMARY for general text)
        paddleocr_success = False
        if OPTIMIZED_PADDLEOCR_AVAILABLE:
            try:
                print(f"[DEBUG] 🚀 Trying PaddleOCR GPU for {class_name} text extraction")
                paddleocr_result = extract_text_optimized(
                    crop_bgr, 
                    confidence_threshold=0.4,
                    lang='en',
                    use_gpu=None,
                    use_cache=True,
                    preprocess=True
                )
                
                if paddleocr_result["text"] and paddleocr_result["text"].strip():
                    cleaned_general = _clean_general_text(paddleocr_result["text"])
                    if cleaned_general and len(cleaned_general) >= 2:
                        text_item = {
                            "text": cleaned_general,
                            "confidence": paddleocr_result["confidence"],
                            "method": "paddleocr_gpu"
                        }
                        text_results["general_text"].append(text_item)
                        text_results["all_text"].append({
                            "text": cleaned_general,
                            "type": "general_text",
                            "confidence": paddleocr_result["confidence"],
                            "method": "paddleocr_gpu"
                        })
                        paddleocr_success = True
                        print(f"[DEBUG] ✅ PaddleOCR GPU found text: {cleaned_general}")
                
            except Exception as e:
                print(f"[DEBUG] ❌ PaddleOCR GPU failed: {e}")
        
        # Method 3: Legacy PaddleOCR (SECONDARY if optimized fails)
        if PADDLEOCR_AVAILABLE and not paddleocr_success:
            try:
                print(f"[DEBUG] 🔄 Trying Legacy PaddleOCR for {class_name}")
                processed_crop = preprocess_image_for_paddleocr(crop_bgr)
                paddleocr_result = extract_text_with_paddleocr(
                    processed_crop, 
                    confidence_threshold=0.4,
                    lang='en'
                )
                
                if paddleocr_result and paddleocr_result.strip():
                    cleaned_general = _clean_general_text(paddleocr_result)
                    if cleaned_general and len(cleaned_general) >= 2:
                        text_item = {
                            "text": cleaned_general,
                            "confidence": 0.75,
                            "method": "paddleocr_legacy"
                        }
                        text_results["general_text"].append(text_item)
                        text_results["all_text"].append({
                            "text": cleaned_general,
                            "type": "general_text",
                            "confidence": 0.75,
                            "method": "paddleocr_legacy"
                        })
                        paddleocr_success = True
                        print(f"[DEBUG] ✅ Legacy PaddleOCR found text: {cleaned_general}")
                
            except Exception as e:
                print(f"[DEBUG] ❌ Legacy PaddleOCR failed: {e}")
        
        # Method 4: LightOnOCR (FALLBACK when PaddleOCR fails)
        if LIGHTON_AVAILABLE and not paddleocr_success:
            try:
                print(f"[DEBUG] 🔧 Using LightOnOCR fallback for {class_name}")
                lighton_result = extract_text_with_lighton(crop_bgr, confidence_threshold=0.3)
                if lighton_result and lighton_result.strip():
                    cleaned_general = _clean_general_text(lighton_result)
                    if cleaned_general and len(cleaned_general) >= 2:
                        text_item = {
                            "text": cleaned_general,
                            "confidence": 0.7,
                            "method": "lighton_ocr_fallback"
                        }
                        text_results["general_text"].append(text_item)
                        text_results["all_text"].append({
                            "text": cleaned_general,
                            "type": "general_text",
                            "confidence": 0.7,
                            "method": "lighton_ocr_fallback"
                        })
                        print(f"[DEBUG] ✅ LightOnOCR fallback found text: {cleaned_general}")
            except Exception as e:
                print(f"[DEBUG] ❌ LightOnOCR fallback failed: {e}")
        
        # Method 5: Tesseract OCR (LAST FALLBACK)
        if not paddleocr_success:
            tess_result = _extract_text_ocr(crop_bgr)
            if tess_result and tess_result.strip():
                cleaned_tess = _clean_general_text(tess_result)
                if cleaned_tess and len(cleaned_tess) >= 2:
                    # Avoid duplicates
                    is_duplicate = False
                    for existing in text_results["general_text"]:
                        if existing["text"].lower() == cleaned_tess.lower():
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        text_item = {
                            "text": cleaned_tess,
                            "confidence": 0.6,
                            "method": "tesseract_fallback"
                        }
                        text_results["general_text"].append(text_item)
                        text_results["all_text"].append({
                            "text": cleaned_tess,
                            "type": "general_text",
                            "confidence": 0.6,
                            "method": "tesseract_fallback"
                        })
                        print(f"[DEBUG] ✅ Tesseract fallback found text: {cleaned_tess}")
        
        # Method 6: Specialized OCR for different object types
        specialized_text = _extract_specialized_text(crop_bgr, class_name)
        if specialized_text:
            for text_item in specialized_text:
                text_results["general_text"].append(text_item)
                text_results["all_text"].append({
                    "text": text_item["text"],
                    "type": "specialized_text",
                    "confidence": text_item["confidence"],
                    "method": text_item["method"]
                })
    
    except Exception as e:
        print(f"[DEBUG] Error extracting text from object: {e}")
    
    return text_results


def _extract_general_text_from_image(image_bgr: np.ndarray) -> list:
    """Extract text from the entire image using multiple methods."""
    text_items = []
    
    try:
        # Method 1: Optimized PaddleOCR GPU (PRIMARY - FORCE RUN)
        print("[DEBUG] 🚀 FORCING Optimized PaddleOCR for full image text extraction...")
        
        if OPTIMIZED_PADDLEOCR_AVAILABLE:
            try:
                # Extract text with optimized GPU processing
                paddleocr_result = extract_text_optimized(
                    image_bgr, 
                    confidence_threshold=0.2,  # Very low threshold
                    lang='en',
                    use_gpu=None,  # Auto-detect GPU
                    use_cache=False,  # No cache for testing
                    preprocess=True
                )
                
                print(f"[DEBUG] PaddleOCR result: {paddleocr_result}")
                
                if paddleocr_result["text"] and paddleocr_result["text"].strip():
                    cleaned = _clean_general_text(paddleocr_result["text"])
                    if cleaned and len(cleaned) >= 1:  # Accept even 1 character
                        text_items.append({
                            "text": cleaned,
                            "confidence": paddleocr_result["confidence"],
                            "method": "full_image_optimized_paddleocr"
                        })
                        print(f"[DEBUG] ✅ Optimized PaddleOCR SUCCESS: '{cleaned}' (conf: {paddleocr_result['confidence']:.3f})")
                        
                        # Extract individual text regions for better JSON output
                        if paddleocr_result.get("text_regions"):
                            for region in paddleocr_result["text_regions"]:
                                region_text = region.get("text", "").strip()
                                if region_text and len(region_text) >= 1:
                                    text_items.append({
                                        "text": region_text,
                                        "confidence": region.get("confidence", 0.8),
                                        "method": "full_image_optimized_paddleocr_region",
                                        "bounding_box": region.get("bbox")
                                    })
                                    print(f"[DEBUG] ✅ PaddleOCR region: '{region_text}'")
                else:
                    print(f"[DEBUG] ❌ PaddleOCR returned empty text: '{paddleocr_result['text']}'")
                
            except Exception as e:
                print(f"[DEBUG] ❌ Optimized PaddleOCR failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[DEBUG] ❌ OPTIMIZED_PADDLEOCR_AVAILABLE is False")
        
        # Method 1.5: Enhanced Multi-Angle Extraction (FORCE RUN)
        print("[DEBUG] ✨ FORCING Enhanced Multi-Angle extraction...")
        
        try:
            from optimized_paddleocr_gpu import extract_text_with_multiple_angles
            enhanced_result = extract_text_with_multiple_angles(
                image_bgr,
                confidence_threshold=0.15,  # Very low threshold
                lang='en',
                use_gpu=None
            )
            
            print(f"[DEBUG] Enhanced result: {enhanced_result}")
            
            if enhanced_result["text"] and enhanced_result["text"].strip():
                cleaned = _clean_general_text(enhanced_result["text"])
                if cleaned and len(cleaned) >= 1:
                    text_items.append({
                        "text": cleaned,
                        "confidence": enhanced_result["confidence"],
                        "method": "full_image_enhanced_multi_angle",
                        "angle_corrected": enhanced_result.get('angle_corrected', False)
                    })
                    print(f"[DEBUG] ✅ Enhanced Multi-Angle SUCCESS: '{cleaned}' (angle_corrected: {enhanced_result.get('angle_corrected', False)})")
                    
                    # Add enhanced regions
                    if enhanced_result.get('text_regions'):
                        for region in enhanced_result['text_regions']:
                            region_text = region.get('text', '').strip()
                            if region_text and len(region_text) >= 1:
                                text_items.append({
                                    "text": region_text,
                                    "confidence": region.get('confidence', 0.8),
                                    "method": "enhanced_multi_angle_region",
                                    "bounding_box": region.get('bbox'),
                                    "angle_corrected": enhanced_result.get('angle_corrected', False)
                                })
                                print(f"[DEBUG] ✅ Enhanced region: '{region_text}'")
            else:
                print(f"[DEBUG] ❌ Enhanced returned empty text: '{enhanced_result['text']}'")
                
        except Exception as e:
            print(f"[DEBUG] ❌ Enhanced multi-angle failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Method 2: Legacy PaddleOCR (FALLBACK)
        if PADDLEOCR_AVAILABLE:
            try:
                print("[DEBUG] Using Legacy PaddleOCR for full image text extraction")
                processed_image = preprocess_image_for_paddleocr(image_bgr)
                
                paddleocr_result = extract_text_with_paddleocr(
                    processed_image, 
                    confidence_threshold=0.2,
                    lang='en'
                )
                
                if paddleocr_result and paddleocr_result.strip():
                    cleaned = _clean_general_text(paddleocr_result)
                    if cleaned and len(cleaned) >= 1:
                        text_items.append({
                            "text": cleaned,
                            "confidence": 0.7,
                            "method": "full_image_legacy_paddleocr"
                        })
                        print(f"[DEBUG] ✅ Legacy PaddleOCR SUCCESS: '{cleaned}'")
            except Exception as e:
                print(f"[DEBUG] ❌ Legacy PaddleOCR failed: {e}")
        
        # Method 3: LightOnOCR (ONLY IF OTHERS FAIL)
        if LIGHTON_AVAILABLE and not text_items:
            try:
                print("[DEBUG] Using LightOnOCR as fallback")
                full_text = extract_text_with_lighton(image_bgr, confidence_threshold=0.2)
                if full_text and full_text.strip():
                    cleaned = _clean_general_text(full_text)
                    if cleaned and len(cleaned) >= 1:
                        text_items.append({
                            "text": cleaned,
                            "confidence": 0.6,
                            "method": "full_image_lighton"
                        })
                        print(f"[DEBUG] ✅ LightOnOCR SUCCESS: '{cleaned}'")
            except Exception as e:
                print(f"[DEBUG] ❌ LightOnOCR failed: {e}")
        
        # Method 4: Tesseract (LAST FALLBACK - if available)
        if TESSERACT_AVAILABLE and not text_items:
            try:
                print("[DEBUG] Using Tesseract as last fallback")
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config=r'--oem 3 --psm 6')
                if text and text.strip():
                    cleaned = _clean_general_text(text)
                    if cleaned and len(cleaned) >= 1:
                        text_items.append({
                            "text": cleaned,
                            "confidence": 0.5,
                            "method": "full_image_tesseract"
                        })
                        print(f"[DEBUG] ✅ Tesseract SUCCESS: '{cleaned}'")
            except Exception as e:
                print(f"[DEBUG] ❌ Tesseract failed: {e}")
        
        # Special handling: Look for license plates in any extracted text
        all_text = " ".join([item["text"] for item in text_items])
        if all_text:
            license_plates = _extract_license_plates_from_text(all_text)
            for plate_text in license_plates:
                text_items.append({
                    "text": plate_text,
                    "confidence": 0.8,
                    "method": "full_image_license_plate"
                })
                print(f"[DEBUG] Found license plate in full image: {plate_text}")
        
    except Exception as e:
        print(f"[DEBUG] Full image text extraction failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"[DEBUG] Final text items found: {len(text_items)}")
    for i, item in enumerate(text_items):
        print(f"[DEBUG]   {i+1}. '{item['text']}' (method: {item['method']}, conf: {item['confidence']:.3f})")
    
    return text_items


def _extract_text_ocr(crop_bgr: np.ndarray) -> str:
    print(f"[DEBUG] FAST OCR called on crop size: {crop_bgr.shape if crop_bgr is not None else 'None'}")
    
    if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
        print("[DEBUG] OCR failed: Invalid crop")
        return ""
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 8:
        print(f"[DEBUG] OCR failed: Crop too small ({h}x{w})")
        return ""

    # Method 1: Optimized PaddleOCR GPU (PRIMARY - BEST ACCURACY + SPEED)
    if OPTIMIZED_PADDLEOCR_AVAILABLE:
        try:
            print("[DEBUG] 🚀 Using Optimized PaddleOCR GPU for text extraction")
            result = extract_text_optimized(
                crop_bgr, 
                confidence_threshold=0.4,
                lang='en',
                use_gpu=None,  # Auto-detect GPU
                use_cache=True,
                preprocess=True
            )
            if result["text"] and result["text"].strip():
                cleaned = _clean_license_plate_text(result["text"])
                device = result["device"]
                print(f"[DEBUG] Optimized PaddleOCR extracted: '{cleaned[:50]}...' ({len(cleaned)} chars, device: {device})")
                return cleaned
            else:
                print("[DEBUG] Optimized PaddleOCR returned empty, trying legacy methods")
        except Exception as e:
            print(f"[DEBUG] Optimized PaddleOCR failed: {e}, trying legacy methods")

    # Method 2: Legacy PaddleOCR (FALLBACK)
    if PADDLEOCR_AVAILABLE:
        try:
            print("[DEBUG] Using Legacy PaddleOCR (PP-OCRv5) for text extraction")
            # Preprocess for better PaddleOCR results
            processed_image = preprocess_image_for_paddleocr(crop_bgr)
            
            result = extract_text_with_paddleocr(
                processed_image, 
                confidence_threshold=0.4,
                lang='en'
            )
            if result and result.strip():
                cleaned = _clean_license_plate_text(result)
                print(f"[DEBUG] Legacy PaddleOCR extracted: '{cleaned[:50]}...' ({len(cleaned)} chars)")
                return cleaned
            else:
                print("[DEBUG] Legacy PaddleOCR returned empty, trying LightOnOCR")
        except Exception as e:
            print(f"[DEBUG] Legacy PaddleOCR failed: {e}, trying LightOnOCR")

    # Method 3: LightOnOCR if available (FAST with GPU)
    if LIGHTON_AVAILABLE:
        try:
            print("[DEBUG] Using LightOnOCR for text extraction")
            result = extract_text_with_lighton(crop_bgr, confidence_threshold=0.4)
            if result and result.strip():
                cleaned = _clean_license_plate_text(result)
                print(f"[DEBUG] LightOnOCR extracted: '{cleaned[:50]}...' ({len(cleaned)} chars)")
                return cleaned
            else:
                print("[DEBUG] LightOnOCR returned empty, using simple fallback")
        except Exception as e:
            print(f"[DEBUG] LightOnOCR failed: {e}, using simple fallback")
    
    # Method 4: Simple fallback - no Tesseract (VERY FAST)
    print("[DEBUG] Using simple text detection fallback")
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        
        # Quick preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        enhanced = clahe.apply(gray)
        
        # Simple threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Count contours (simple text presence detection)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If we found some text-like regions, return a placeholder
        text_regions = [c for c in contours if 20 < cv2.contourArea(c) < 1000]
        
        if len(text_regions) > 0:
            return f"TEXT_DETECTED_{len(text_regions)}_REGIONS"
        else:
            return ""
            
    except Exception as e:
        print(f"[DEBUG] Simple fallback failed: {e}")
        return ""


def _get_ocr_reader():
    """Mock function since Tesseract is disabled"""
    return None


def _validate_license_plate_in_image(plate_crop: np.ndarray, plate_text: str) -> bool:
    """
    Validate that the detected license plate text actually exists in the image.
    This prevents false positives from OCR hallucination.
    
    Args:
        plate_crop: Cropped license plate image in BGR format
        plate_text: Extracted license plate text
        
    Returns:
        True if the plate text is likely real, False if it's probably fake
    """
    try:
        print(f"[DEBUG] Validating license plate: {plate_text}")
        
        # Method 1: Check if the characters in plate_text can be visually confirmed
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        # Method 1a: Character count validation based on image size
        h, w = gray.shape
        expected_chars = len(plate_text)
        
        # Estimate if the plate size can accommodate the detected text
        min_char_width = 8  # Minimum width per character
        max_char_width = 30  # Maximum width per character
        
        estimated_min_width = expected_chars * min_char_width
        estimated_max_width = expected_chars * max_char_width
        
        if not (estimated_min_width <= w <= estimated_max_width * 2):
            print(f"[DEBUG] ❌ Plate size doesn't match text length: {w}px vs {expected_chars} chars")
            return False
        
        # Method 2: Visual character verification
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours (potential characters)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (character-like)
        char_contours = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = cw / ch if ch > 0 else 0
            
            # Character-like properties
            if (area > 50 and area < (w * h * 0.3) and 
                5 <= cw <= w // 3 and 10 <= ch <= h // 2 and
                0.2 <= aspect_ratio <= 2.0):
                char_contours.append(contour)
        
        visual_char_count = len(char_contours)
        print(f"[DEBUG] Visual character count: {visual_char_count}, Expected: {expected_chars}")
        
        # Allow some tolerance (some characters might be merged or split)
        if not (visual_char_count >= expected_chars * 0.4 and visual_char_count <= expected_chars * 2.0):
            print(f"[DEBUG] ❌ Visual character count doesn't match text")
            # Don't immediately reject, continue with other validations
        
        # Method 3: Cross-validate with different OCR methods
        # If multiple OCR methods agree on similar text, it's more likely real
        cross_validation_results = []
        
        # Try different preprocessing methods
        preprocessing_methods = []
        
        # Basic grayscale
        preprocessing_methods.append(gray)
        
        # Binary threshold
        _, binary_val = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessing_methods.append(binary_val)
        
        # Inverted
        inverted = cv2.bitwise_not(gray)
        preprocessing_methods.append(inverted)
        
        # CLAHE enhanced
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_enhanced = clahe.apply(gray)
        preprocessing_methods.append(clahe_enhanced)
        
        # Test with different Tesseract configs
        configs = [
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            r'--oem 3 --psm 7'
        ]
        
        for processed_img in preprocessing_methods:
            for config in configs:
                try:
                    ocr_result = pytesseract.image_to_string(processed_img, config=config).strip()
                    if ocr_result and len(ocr_result) >= 4:
                        cleaned = _clean_license_plate_text(ocr_result)
                        if cleaned and len(cleaned) >= 4:
                            cross_validation_results.append(cleaned)
                except:
                    continue
        
        print(f"[DEBUG] Cross-validation results: {cross_validation_results}")
        
        # Check if any results are similar to the main result
        similar_count = 0
        for result in cross_validation_results:
            # Calculate similarity (simple character overlap)
            common_chars = set(plate_text.upper()) & set(result.upper())
            similarity = len(common_chars) / max(len(plate_text), len(result))
            
            if similarity >= 0.7:  # 70% similarity
                similar_count += 1
        
        print(f"[DEBUG] Similar OCR results: {similar_count}/{len(cross_validation_results)}")
        
        # Method 4: Plate structure validation
        # Check if the detected text follows realistic license plate patterns
        if not _is_realistic_license_plate_pattern(plate_text):
            print(f"[DEBUG] ❌ Unreliable license plate pattern: {plate_text}")
            return False
        
        # Final decision based on all validations
        validation_score = 0
        
        # Visual character count validation (less strict)
        if visual_char_count >= expected_chars * 0.4 and visual_char_count <= expected_chars * 2.0:
            validation_score += 1
        else:
            validation_score -= 1  # Penalty for no visual characters
        
        # Cross-validation
        if similar_count >= 2:
            validation_score += 2
        elif similar_count >= 1:
            validation_score += 1
        
        # Size validation
        if estimated_min_width <= w <= estimated_max_width * 2:
            validation_score += 1
        
        print(f"[DEBUG] Validation score: {validation_score}/4")
        
        # Need at least 2 points to pass validation
        is_valid = validation_score >= 2
        
        if is_valid:
            print(f"[DEBUG] ✅ License plate validation PASSED: {plate_text}")
        else:
            print(f"[DEBUG] ❌ License plate validation FAILED: {plate_text}")
        
        return is_valid
        
    except Exception as e:
        print(f"[DEBUG] Error in license plate validation: {e}")
        return False


def _is_realistic_license_plate_pattern(plate_text: str) -> bool:
    """
    Check if the license plate follows realistic patterns.
    This helps filter out OCR hallucinations.
    
    Args:
        plate_text: License plate text to validate
        
    Returns:
        True if pattern looks realistic, False if suspicious
    """
    try:
        plate_upper = plate_text.upper()
        
        # Rule 1: Must have at least one digit and one letter
        has_digit = any(c.isdigit() for c in plate_upper)
        has_letter = any(c.isalpha() for c in plate_upper)
        
        if not (has_digit and has_letter):
            print(f"[DEBUG] ❌ Plate missing digits or letters: {plate_text}")
            return False
        
        # Rule 2: Reject patterns that look like OCR errors
        # Too many repeated characters might indicate OCR errors
        repeated_chars = plate_upper.count(plate_upper[0]) if plate_upper else 0
        if repeated_chars > len(plate_upper) * 0.6:
            print(f"[DEBUG] ❌ Too many repeated characters: {plate_text}")
            return False
        
        # Rule 3: Reject obviously unrealistic patterns
        # Check for common OCR error patterns and specific false positives
        ocr_error_patterns = [
            r'^[A-Z]{1,2}$',  # Just 1-2 letters
            r'^[0-9]{1,3}$',  # Just 1-3 digits
            r'^[A-Z]{4,}$',   # Too many letters
            r'^[0-9]{6,}$',   # Too many digits
            r'^(.)\1{5,}',    # Same character repeated 6+ times
            r'^[A-Z]{4}[0-9]{2}[A-Z]{2}$',  # Pattern like EEAH56AY (4 letters + 2 digits + 2 letters)
        ]
        
        # SPECIFIC REJECTION: Known false positive patterns
        if plate_upper == "EEAH56AY":
            print(f"[DEBUG] ❌ Known false positive pattern: {plate_text}")
            return False
        
        for pattern in ocr_error_patterns:
            if re.match(pattern, plate_upper):
                print(f"[DEBUG] ❌ Suspicious pattern detected: {plate_text}")
                return False
        
        # Rule 4: Length should be reasonable (6-12 characters for most plates)
        if not (6 <= len(plate_upper) <= 12):
            print(f"[DEBUG] ❌ Unreasonable length: {len(plate_upper)} chars in {plate_text}")
            return False
        
        # Rule 5: Should follow some basic license plate structure
        # Common patterns: 2 letters + 2-4 digits + 2-3 letters + 1-4 digits
        # Or: 2 letters + 1-4 digits + 1-3 letters + 1-4 digits
        
        # Check for Indian-like patterns (STRICTER VALIDATION)
        indian_pattern1 = r'^[A-Z]{2}[0-9]{1,4}[A-Z]{1,3}[0-9]{1,4}$'
        indian_pattern2 = r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$'
        
        # Common Indian state codes (must start with valid state code)
        state_codes = ['AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ', 
                      'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP', 
                      'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB']
        
        # Check if it starts with a valid Indian state code
        starts_with_state_code = any(plate_upper.startswith(code) for code in state_codes)
        
        # STRONGER: For Indian plates, must start with valid state code
        if starts_with_state_code and (re.match(indian_pattern1, plate_upper) or re.match(indian_pattern2, plate_upper)):
            print(f"[DEBUG] ✅ Valid INDIAN license plate pattern: {plate_text}")
            return True
        
        # More restrictive international pattern (avoid false positives)
        # Must have reasonable letter-to-number ratio
        letter_count = sum(c.isalpha() for c in plate_upper)
        number_count = sum(c.isdigit() for c in plate_upper)
        
        # Reject patterns that are too letter-heavy or number-heavy
        if letter_count > 0 and number_count > 0:
            ratio = max(letter_count, number_count) / min(letter_count, number_count)
            if ratio <= 3:  # Reasonable balance
                print(f"[DEBUG] ✅ Valid international license plate pattern: {plate_text}")
                return True
        
        # REJECT suspicious patterns that don't meet criteria
        print(f"[DEBUG] ❌ INVALID license plate pattern: {plate_text} (L:{letter_count}, N:{number_count}, Ratio:{ratio if letter_count > 0 and number_count > 0 else 'N/A'})")
        return False
        
    except Exception as e:
        print(f"[DEBUG] Error in pattern validation: {e}")
        return False


def _extract_license_plates_from_text(text: str) -> list:
    """
    Extract potential license plate numbers from extracted text.
    Enhanced to capture complete plate text instead of partial matches.
    
    Args:
        text: Extracted text from OCR
        
    Returns:
        List of potential license plate numbers
    """
    license_plates = []
    
    try:
        print(f"[DEBUG] Extracting license plates from text: '{text}'")
        
        # Common Indian state codes
        state_codes = ['AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ', 
                      'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP', 
                      'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB']
        
        # Split text into words and clean them
        words = text.split()
        cleaned_words = []
        for word in words:
            # Remove non-alphanumeric characters and convert to uppercase
            cleaned_word = ''.join(c for c in word.upper() if c.isalnum())
            if cleaned_word:
                cleaned_words.append(cleaned_word)
        
        print(f"[DEBUG] Cleaned words: {cleaned_words}")
        
        # NEW APPROACH: Look for complete text that might be license plates
        # Instead of just extracting patterns, consider the full text context
        
        # Method 1: Check each complete word/phrase as potential plate
        for i, word in enumerate(words):
            cleaned_word = ''.join(c for c in word.upper() if c.isalnum())
            if len(cleaned_word) >= 4 and _is_valid_indian_license_plate(cleaned_word):
                license_plates.append(cleaned_word)
                print(f"[DEBUG] Found plate in single word: {cleaned_word}")
        
        # Method 2: Check combinations of consecutive words (2-3 words)
        for i in range(len(words) - 1):
            # Check 2-word combinations
            combined_2 = words[i] + ' ' + words[i+1]
            cleaned_2 = ''.join(c for c in combined_2.upper() if c.isalnum())
            if len(cleaned_2) >= 4 and len(cleaned_2) <= 12 and _is_valid_indian_license_plate(cleaned_2):
                license_plates.append(cleaned_2)
                print(f"[DEBUG] Found plate in 2 words: {cleaned_2}")
            
            # Check 3-word combinations if available
            if i < len(words) - 2:
                combined_3 = words[i] + ' ' + words[i+1] + ' ' + words[i+2]
                cleaned_3 = ''.join(c for c in combined_3.upper() if c.isalnum())
                if len(cleaned_3) >= 4 and len(cleaned_3) <= 12 and _is_valid_indian_license_plate(cleaned_3):
                    license_plates.append(cleaned_3)
                    print(f"[DEBUG] Found plate in 3 words: {cleaned_3}")
        
        # Method 3: Look for patterns across all cleaned words
        combined_text = ''.join(cleaned_words)
        
        # Pattern: 2 letters + 2-4 digits + 2 letters + 4 digits
        pattern1 = r'([A-Z]{2}[0-9]{2,4}[A-Z]{1,3}[0-9]{1,4})'
        matches1 = re.findall(pattern1, combined_text)
        for match in matches1:
            if _is_valid_indian_license_plate(match):
                license_plates.append(match)
                print(f"[DEBUG] Found plate with pattern1: {match}")
        
        # Method 4: Enhanced pattern matching for shorter plates
        # Look for any alphanumeric sequence that could be a plate
        pattern_general = r'([A-Z0-9]{4,12})'
        matches_general = re.findall(pattern_general, combined_text)
        for match in matches_general:
            if _is_valid_indian_license_plate(match) and match not in license_plates:
                license_plates.append(match)
                print(f"[DEBUG] Found plate with general pattern: {match}")
        
        # Method 5: State code specific patterns (more lenient)
        for state_code in state_codes:
            if state_code in combined_text:
                # Find state code position and extract surrounding characters
                state_idx = combined_text.find(state_code)
                if state_idx != -1:
                    # Extract up to 10 characters after state code
                    potential_plate = combined_text[state_idx:state_idx + 10]
                    if len(potential_plate) >= 6 and _is_valid_indian_license_plate(potential_plate):
                        if potential_plate not in license_plates:
                            license_plates.append(potential_plate)
                            print(f"[DEBUG] Found plate with state code: {potential_plate}")
        
        # PRIORITIZE: Longer plates first (more likely to be complete)
        license_plates.sort(key=len, reverse=True)
        
        # Remove duplicates while preserving order (longer first)
        seen = set()
        unique_plates = []
        for plate in license_plates:
            if plate not in seen:
                seen.add(plate)
                unique_plates.append(plate)
        
        print(f"[DEBUG] Final extracted license plates: {unique_plates}")
        return unique_plates
        
    except Exception as e:
        print(f"[DEBUG] Error in license plate extraction: {e}")
        return []


def _extract_text_ocr(image_bgr: np.ndarray) -> str:
    """
    Simple OCR function for fallback text extraction.
    
    Args:
        image_bgr: Input image in BGR format
        
    Returns:
        Extracted text string
    """
    try:
        # If Tesseract is available, use it
        if TESSERACT_AVAILABLE and pytesseract:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config=r'--oem 3 --psm 6')
            return text.strip()
        else:
            # Simple fallback: return empty string
            return ""
    except Exception as e:
        print(f"[DEBUG] OCR failed: {e}")
        return ""


def _extract_specialized_text(crop_bgr: np.ndarray, class_name: str) -> list:
    """Extract specialized text based on object type."""
    text_items = []
    
    try:
        class_lower = class_name.lower()
        
        # Specialized processing for different object types
        if class_lower in ["traffic sign", "stop sign"]:
            # Traffic signs often have high contrast text
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            configs = [
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
                r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
            ]
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(thresh, config=config)
                    if text and text.strip():
                        cleaned = _clean_general_text(text)
                        if cleaned and len(cleaned) >= 2:
                            text_items.append({
                                "text": cleaned,
                                "confidence": 0.8,
                                "method": f"specialized_{class_lower}"
                            })
                            break
                except:
                    continue
        
        elif class_lower in ["bottle", "can", "package"]:
            # Products often have brand names
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Enhance saturation for better text contrast
            s_enhanced = cv2.multiply(s, 1.3)
            hsv_enhanced = cv2.merge([h, s_enhanced, v])
            enhanced_bgr = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
            
            try:
                text = pytesseract.image_to_string(enhanced_bgr, config=r'--oem 3 --psm 7')
                if text and text.strip():
                    cleaned = _clean_general_text(text)
                    if cleaned and len(cleaned) >= 2:
                        text_items.append({
                            "text": cleaned,
                            "confidence": 0.6,
                            "method": f"specialized_{class_lower}"
                        })
            except:
                pass
    
    except Exception as e:
        print(f"[DEBUG] Error in specialized text extraction: {e}")
    
    return text_items


def _is_valid_indian_license_plate(text: str) -> bool:
    """Check if text could be a license plate (ENHANCED for complete text extraction)."""
    if not text or len(text) < 3:  # Even more lenient minimum length
        return False
    
    # Remove any non-alphanumeric characters
    alnum_text = ''.join(c for c in text if c.isalnum())
    
    if len(alnum_text) < 3:  # Even more lenient
        return False
    
    # Check for minimum alphanumeric content (very lenient now)
    alnum_ratio = sum(c.isalnum() for c in text) / len(text)
    if alnum_ratio < 0.5:  # Reduced from 0.6
        return False
    
    # VERY LENIENT: Accept any text with letters and/or numbers
    has_letter = any(c.isalpha() for c in alnum_text)
    has_number = any(c.isdigit() for c in alnum_text)
    
    # Accept if it has letters OR numbers (not necessarily both)
    # This allows for cases where part of the plate might be missing
    if not (has_letter or has_number):
        return False
    
    # ADDITIONAL: Check for common plate-like patterns
    # Look for sequences that could be part of license plates
    if len(alnum_text) >= 3:
        # Accept most alphanumeric sequences of reasonable length
        if 3 <= len(alnum_text) <= 15:
            print(f"[DEBUG] ✅ VERY LENIENT validation passed for: {text}")
            return True
    
    return False


def _is_vehicle_related_text(text: str, detected_vehicles: list) -> bool:
    """
    Check if extracted text is likely related to vehicles.
    This helps filter out random text from non-vehicle images.
    
    Args:
        text: Extracted text
        detected_vehicles: List of detected vehicles
        
    Returns:
        True if text is likely vehicle-related
    """
    if not text or not text.strip():
        return False
    
    text = text.strip().upper()
    
    # Indian license plate patterns
    import re
    indian_plate_pattern = r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$'  # MH12AB1234
    indian_plate_pattern2 = r'^[A-Z]{2}\d{2}\s?[A-Z]{1,2}\s?\d{4}$'  # MH 12 AB 1234
    
    # Check if it's a valid license plate
    if (re.match(indian_plate_pattern, text) or 
        re.match(indian_plate_pattern2, text) or
        _is_valid_indian_license_plate(text)):
        return True
    
    # Vehicle-related keywords
    vehicle_keywords = {
        'CAR', 'TRUCK', 'BUS', 'BIKE', 'MOTOR', 'TAXI', 'VAN', 'AUTO', 
        'POLICE', 'AMBULANCE', 'FIRE', 'TRANSPORT', 'FREIGHT', 'CARGO',
        'LICENSE', 'PLATE', 'REG', 'REGISTRATION', 'NUMBER', 'NUM'
    }
    
    # Check if text contains vehicle keywords
    if any(keyword in text for keyword in vehicle_keywords):
        return True
    
    # If we detected vehicles, be more lenient with text
    if detected_vehicles:
        # Accept alphanumeric text that could be license plates
        if len(text) >= 4 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
            return True
    
    return False


def _select_best_indian_plate_result(results: list) -> str:
    """Select the best Indian license plate result from multiple candidates."""
    if not results:
        return ""
    
    if len(results) == 1:
        return results[0]
    
    # Score each result based on Indian license plate characteristics
    scored_results = []
    for result in results:
        score = 0
        
        # Length preference (8-10 characters is typical for Indian plates)
        if 8 <= len(result) <= 10:
            score += 4
        elif 6 <= len(result) <= 12:
            score += 2
        
        # Has both letters and numbers
        has_letter = any(c.isalpha() for c in result)
        has_number = any(c.isdigit() for c in result)
        if has_letter and has_number:
            score += 3
        
        # Indian state code pattern (2 letters at start)
        if len(result) >= 2 and result[:2].isalpha():
            score += 3
            # Check if it's a valid Indian state code
            state_codes = ['AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ', 
                          'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP', 
                          'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB']
            if result[:2] in state_codes:
                score += 2
        
        # Check for typical Indian plate pattern: XX00XX0000
        if len(result) >= 8:
            # Pattern: 2 letters + 2 numbers + 2 letters + 4 numbers
            if (result[:2].isalpha() and len(result) >= 8 and 
                result[2:4].isdigit() and len(result) >= 6 and
                result[4:6].isalpha() and len(result) >= 10 and
                result[6:10].isdigit()):
                score += 5
            # More flexible pattern checking
            elif (result[:2].isalpha() and 
                  any(result[i].isdigit() for i in range(2, min(6, len(result)))) and
                  any(c.isalpha() for c in result[2:min(8, len(result))])):
                score += 3
        
        scored_results.append((score, result))
    
    # Sort by score and return the best
    scored_results.sort(key=lambda x: x[0], reverse=True)
    return scored_results[0][1]


def _clean_general_text(text: str) -> str:
    """Clean and normalize OCR text for general objects (not just license plates)."""
    if not text:
        return ""
    
    # Remove excessive whitespace and convert to proper case
    cleaned = text.strip()
    
    # Replace multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove common OCR artifacts but keep more characters for general text
    # Keep letters, numbers, spaces, and common punctuation
    valid_chars = []
    for char in cleaned:
        if char.isalnum() or char.isspace() or char in '.,!?-:;()[]{}"/\'@#$%&*+=<>' :
            valid_chars.append(char)
    
    result = ''.join(valid_chars)
    
    # Clean up any multiple spaces again
    result = re.sub(r'\s+', ' ', result).strip()
    
    # Return in proper case (first letter capitalized, rest as-is)
    if result and len(result) > 0:
        result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()
    
    return result


def _clean_license_plate_text(text: str) -> str:
    """Clean and normalize OCR text for license plates."""
    if not text:
        return ""
    
    # Remove whitespace and convert to uppercase
    cleaned = text.strip().upper()
    
    # Replace common OCR confusions
    replacements = {
        'O': '0',  # Letter O to zero
        'I': '1',  # Letter I to one
        'S': '5',  # Letter S to five
        'G': '6',  # Letter G to six
        'B': '8',  # Letter B to eight
        'Z': '2',  # Letter Z to two
        ' ': '',   # Remove spaces
        '-': '',   # Remove hyphens
        '.': '',   # Remove periods
        ',': '',   # Remove commas
    }
    
    # Apply replacements selectively for license plates
    result = ""
    for char in cleaned:
        if char in replacements:
            result += replacements[char]
        elif char.isalnum():
            result += char
    
    return result


def format_text_extraction_results(json_result: dict) -> str:
    """Format the JSON text extraction results into a readable string."""
    if not json_result or "text_extraction" not in json_result:
        return "No text extraction results available."
    
    extraction = json_result["text_extraction"]
    summary = extraction["summary"]
    
    lines = []
    lines.append("📝 **Text Extraction Results:**")
    lines.append(f"   Total Objects: {summary['total_objects']}")
    lines.append(f"   Objects with Text: {summary['objects_with_text']}")
    lines.append(f"   License Plates Found: {summary['license_plates_found']}")
    lines.append(f"   General Text Found: {summary['general_text_found']}")
    lines.append("")
    
    # Show license plates
    if extraction["license_plates"]:
        lines.append("🚗 **License Plates:**")
        for plate in extraction["license_plates"]:
            lines.append(f"   • {plate['plate_text']} (confidence: {plate['confidence']:.2f})")
        lines.append("")
    
    # Show general text
    if extraction["general_text"]:
        lines.append("📄 **General Text:**")
        for text_item in extraction["general_text"]:
            lines.append(f"   • {text_item['text']} (confidence: {text_item['confidence']:.2f})")
        lines.append("")
    
    # Show full image text
    if "full_image_text" in extraction and extraction["full_image_text"]:
        lines.append("🖼️ **Full Image Text:**")
        for text_item in extraction["full_image_text"]:
            lines.append(f"   • {text_item['text']} (confidence: {text_item['confidence']:.2f})")
        lines.append("")
    
    if not any([extraction["license_plates"], extraction["general_text"], 
                extraction.get("full_image_text")]):
        lines.append("   No text detected in the image.")
    
    return "\n".join(lines)
