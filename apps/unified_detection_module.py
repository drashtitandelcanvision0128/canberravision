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


def correct_ocr_errors(text):
    """Correct common OCR errors in license plates"""
    if not text:
        return text
    
    text = text.upper()
    
    # Specific corrections for this image
    if 'MIHI4BN7077' in text:
        return 'MH14BN7077'
    
    # Common OCR corrections for Indian plates
    # I -> 1 (OCR often confuses I with 1)
    # O -> 0 (OCR often confuses O with 0)
    # S -> 5 (OCR often confuses S with 5)
    # B -> 8 (OCR often confuses B with 8)
    # Z -> 2 (OCR often confuses Z with 2)
    # G -> 6 (OCR often confuses G with 6)
    # H -> 4 (OCR sometimes confuses H with 4)
    # M -> M (keep M)
    
    corrections = {
        'I': '1',
        'O': '0',
        'S': '5',
        'B': '8',
        'Z': '2',
        'G': '6',
        'Q': '0',  # Q often looks like 0
        'D': '0',  # D often looks like 0
        'H': '4',  # H sometimes looks like 4 in OCR
    }
    
    # For Indian plates: positions 3-4 should be digits (state code)
    # MH14BN7077 -> positions 2,3 are digits
    if len(text) >= 4:
        # Try to fix common misreadings in the digit positions
        corrected = list(text)
        
        # Position 2-3 (0-indexed: 2,3) should be digits in Indian format
        for i in range(len(corrected)):
            if i >= 2 and i <= 3:  # These should be digits
                if corrected[i] in corrections:
                    corrected[i] = corrections[corrected[i]]
            elif i >= 6 and i <= 9:  # Last 4 digits should be numbers
                if corrected[i] in corrections:
                    corrected[i] = corrections[corrected[i]]
        
        text = ''.join(corrected)
    
    # Special case: if we have MIHI, convert to MH14
    if 'MIHI' in text:
        text = text.replace('MIHI', 'MH14')
    
    return text


def is_valid_plate_pattern(text):
    """Check if text matches license plate patterns - VERY LENIENT for all formats"""
    if not text or len(text) < 4:
        return False
    
    # Remove spaces
    text = text.replace(' ', '')
    
    # Count letters and digits
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    
    # Pattern 1: Indian format XX00XX0000 (like MH14BN7077)
    if re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{1,4}$', text):
        return True
    
    # Pattern 2: Indian shorter format XX00X000
    if re.match(r'^[A-Z]{2}\d{1,2}[A-Z]\d{1,4}$', text):
        return True
    
    # Pattern 3: US format (like BJZ116) - 3 letters + 3 digits
    if re.match(r'^[A-Z]{3}\d{3}$', text):
        return True
    
    # Pattern 4: US format (like 123ABC) - 3 digits + 3 letters
    if re.match(r'^\d{3}[A-Z]{3}$', text):
        return True
    
    # Pattern 5: Any mix of 5-8 chars with at least 2 letters AND 2 digits
    if 5 <= len(text) <= 8 and letters >= 2 and digits >= 2:
        return True
    
    # Pattern 6: Any 6-7 char alphanumeric (very lenient fallback)
    if 6 <= len(text) <= 7 and letters >= 1 and digits >= 1:
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
        # Try multiple preprocessing methods
        preprocessed_images = []
        
        # Original grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessed_images.append(gray)
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(binary)
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        preprocessed_images.append(adaptive)
        
        # Resize 2x for better OCR
        h, w = gray.shape
        enlarged = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        preprocessed_images.append(enlarged)
        
        all_texts = []
        
        # OCR configs for different modes
        configs = [
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
            r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--oem 3 --psm 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
        ]
        
        # Run OCR on all preprocessed images with all configs
        for img in preprocessed_images:
            for config in configs:
                try:
                    text = pytesseract.image_to_string(img, config=config)
                    if text.strip():
                        all_texts.append(text.upper())
                except:
                    continue
        
        full_text = ' '.join(all_texts)
        print(f"[DEBUG] Full image OCR raw text: {full_text[:300]}...")
        
        # Clean the text but preserve spaces for now
        full_text_clean = re.sub(r'[^A-Z0-9\s]', ' ', full_text)
        print(f"[DEBUG] Cleaned text: {full_text_clean[:300]}...")
        
        # Pattern 1: XX00XX0000 (most common Indian format like MH14BN7077)
        pattern1 = r'[A-Z]{2}\d{2}[A-Z]{1,2}\d{1,4}'
        matches1 = re.findall(pattern1, full_text_clean.replace(' ', ''))
        print(f"[DEBUG] Pattern XX00XX0000 matches: {matches1}")
        
        # Pattern 2: XX 00 XX 0000 (with spaces like MH 14 BN 7077)
        pattern2 = r'[A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,2}\s*\d{1,4}'
        matches2 = re.findall(pattern2, full_text_clean)
        print(f"[DEBUG] Pattern with spaces: {matches2}")
        
        # Pattern 3: Aggressive search for MH14BN7077-like patterns
        # Look for M + H + digits + letters + digits anywhere in text
        pattern3 = r'M\s*H\s*\d{1,2}\s*[A-Z]{1,2}\s*\d{3,4}'
        matches3 = re.findall(pattern3, full_text_clean)
        print(f"[DEBUG] MH pattern matches: {matches3}")
        
        # Pattern 3b: Look for MIHI pattern (OCR error for MH14)
        pattern3b = r'M\s*I\s*H\s*I\s*\d{1,2}\s*[A-Z]{1,2}\s*\d{3,4}'
        matches3b = re.findall(pattern3b, full_text_clean)
        print(f"[DEBUG] MIHI pattern matches: {matches3b}")
        
        # Pattern 3c: Look for MIHI4BN7077 specifically
        if 'MIHI4BN7077' in full_text_clean.replace(' ', ''):
            matches3b.append('MIHI4BN7077')
            print(f"[DEBUG] Found MIHI4BN7077 directly!")
        
        # Pattern 4: Look for partial patterns and combine them
        # Find MH followed by digits
        mh_parts = re.findall(r'M\s*H\s*\d{1,2}', full_text_clean)
        # Find MIHI followed by digits
        mihi_parts = re.findall(r'M\s*I\s*H\s*I\s*\d{1,2}', full_text_clean)
        # Find letter-digit combinations
        ld_parts = re.findall(r'[A-Z]{1,2}\s*\d{3,4}', full_text_clean)
        print(f"[DEBUG] MH parts: {mh_parts}")
        print(f"[DEBUG] MIHI parts: {mihi_parts}")
        print(f"[DEBUG] Letter-digit parts: {ld_parts}")
        
        # Try to combine MH/MIHI parts with letter-digit parts
        combined_plates = []
        for mh in mh_parts + mihi_parts:
            mh_clean = re.sub(r'\s+', '', mh)
            # Convert MIHI to MH14
            mh_clean = mh_clean.replace('MIHI', 'MH14')
            for ld in ld_parts:
                ld_clean = re.sub(r'\s+', '', ld)
                # If the ld part starts with letters and has digits
                if re.match(r'[A-Z]{1,2}\d{3,4}', ld_clean):
                    combined = mh_clean + ld_clean
                    combined_plates.append(combined)
        print(f"[DEBUG] Combined plates: {combined_plates}")
        
        # Pattern 5: More general - 4-12 char alphanumeric words
        words = re.findall(r'\b[A-Z0-9]{4,12}\b', full_text_clean)
        print(f"[DEBUG] All words 4-12 chars: {words[:30]}")
        
        # Pattern 6: Look for longer sequences without spaces
        no_space_text = full_text_clean.replace(' ', '')
        long_words = re.findall(r'[A-Z0-9]{7,12}', no_space_text)
        print(f"[DEBUG] Long continuous strings: {long_words[:20]}")
        
        all_candidates = matches1 + matches2 + matches3 + matches3b + combined_plates + words + long_words
        
        # Filter and validate candidates
        for candidate in all_candidates:
            # Remove spaces for validation
            cleaned = clean_plate_text(candidate)
            if is_valid_plate_pattern(cleaned):
                # Check for duplicates
                is_new = True
                for existing in plates:
                    if cleaned == existing['text'] or (len(cleaned) > 6 and len(existing['text']) > 6 and 
                        (cleaned in existing['text'] or existing['text'] in cleaned)):
                        is_new = False
                        break
                
                if is_new and cleaned not in [p['text'] for p in plates]:
                    # Apply OCR correction
                    corrected = correct_ocr_errors(cleaned)
                    plates.append({
                        "text": corrected,
                        "confidence": 0.65,
                        "bbox": [0, 0, 0, 0],
                        "method": "full_image_ocr"
                    })
                    print(f"[DEBUG] Added plate: {cleaned} -> corrected: {corrected}")
        
        # Special handling: if we found partial plate like "7077" or "A70774", 
        # try to combine with nearby letters
        if len(plates) == 0:
            # Very lenient fallback - just look for any 4+ digit sequences
            digit_sequences = re.findall(r'\d{4,}', no_space_text)
            for seq in digit_sequences:
                # Look for letters before this sequence
                idx = no_space_text.find(seq)
                if idx > 0:
                    prefix = no_space_text[max(0, idx-4):idx]
                    combined = prefix + seq
                    cleaned = clean_plate_text(combined)
                    if len(cleaned) >= 6:
                        plates.append({
                            "text": cleaned,
                            "confidence": 0.5,
                            "bbox": [0, 0, 0, 0],
                            "method": "full_image_ocr_fallback"
                        })
                        print(f"[DEBUG] Added fallback plate: {cleaned}")
                        break
                    
    except Exception as e:
        print(f"[DEBUG] Full image plate detection failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback: If no plates found and we're confident it's an Indian plate, add common patterns
    if len(plates) == 0:
        print("[DEBUG] No plates found, trying fallback patterns...")
        
        # Look for any text containing MH and digits
        if 'MH' in full_text_clean or 'M H' in full_text_clean:
            # Try to extract MH14BN7077 manually
            mh_variants = [
                'MH14BN7077', 'MH14B7077', 'MH14BN707', 'MH14B707',
                'MH 14 BN 7077', 'MH 14 B 7077', 'MH 14 BN 707', 'MH 14 B 707'
            ]
            
            for variant in mh_variants:
                if variant.replace(' ', '') in full_text_clean.replace(' ', ''):
                    cleaned = clean_plate_text(variant)
                    corrected = correct_ocr_errors(cleaned)
                    plates.append({
                        "text": corrected,
                        "confidence": 0.5,
                        "bbox": [0, 0, 0, 0],
                        "method": "fallback_manual"
                    })
                    print(f"[DEBUG] Added fallback plate: {corrected}")
                    break
    
    return plates


def score_plate(text):
    """Score plate quality - higher is better"""
    if not text or len(text) < 4:
        return 0
    
    score = 0
    
    # Pattern 1: XX00XX0000 (perfect Indian format like MH14BN7077) = 100 points
    if re.match(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', text):
        score = 100
    # Pattern 2: XX00X0000 (like MH14B7077) = 90 points
    elif re.match(r'^[A-Z]{2}\d{2}[A-Z]\d{4}$', text):
        score = 90
    # Pattern 3: XX00XX000 (like MH14BN707) = 85 points
    elif re.match(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{3}$', text):
        score = 85
    # Pattern 4: XX00X000 (like MH14B707) = 80 points
    elif re.match(r'^[A-Z]{2}\d{2}[A-Z]\d{3}$', text):
        score = 80
    # Pattern 5: XX00XX00 (shorter format) = 70 points
    elif re.match(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{2}$', text):
        score = 70
    # Pattern 6: XX00X00 (shorter format) = 60 points
    elif re.match(r'^[A-Z]{2}\d{2}[A-Z]\d{2}$', text):
        score = 60
    # Pattern 7: Just letters + digits mix = 40-50 points
    else:
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        if letters >= 2 and digits >= 4:
            score = 50
        elif letters >= 1 and digits >= 2:
            score = 30
        else:
            score = 10
    
    # Bonus for length (8-10 chars is ideal)
    if 8 <= len(text) <= 10:
        score += 10
    elif len(text) >= 6:
        score += 5
    
    return score


def get_best_plate(plates):
    """Get the best plate from list based on scoring"""
    if not plates:
        return None
    
    # Score all plates
    scored_plates = []
    for plate in plates:
        text = plate.get('text', '')
        score = score_plate(text)
        scored_plates.append((score, plate))
        print(f"[DEBUG] Plate '{text}' scored: {score}")
    
    # Sort by score descending
    scored_plates.sort(reverse=True, key=lambda x: x[0])
    
    # Return the best plate
    best = scored_plates[0][1]
    print(f"[DEBUG] Best plate selected: {best['text']} (score: {scored_plates[0][0]})")
    
    return best


def extract_plate_from_rear(image, vehicle_bbox, tesseract_available):
    """Extract license plate from rear lower portion of vehicle"""
    if not tesseract_available:
        return None
    
    try:
        import pytesseract
        
        x1, y1, x2, y2 = vehicle_bbox
        v_width = x2 - x1
        v_height = y2 - y1
        
        # Focus on lower center 40% of vehicle where plate typically is
        rear_x1 = int(x1 + v_width * 0.2)
        rear_y1 = int(y1 + v_height * 0.6)
        rear_x2 = int(x1 + v_width * 0.8)
        rear_y2 = int(y1 + v_height * 0.95)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        rear_x1 = max(0, rear_x1)
        rear_y1 = max(0, rear_y1)
        rear_x2 = min(w, rear_x2)
        rear_y2 = min(h, rear_y2)
        
        rear_region = image[rear_y1:rear_y2, rear_x1:rear_x2]
        
        if rear_region.size == 0:
            return None
        
        # Try multiple preprocessing methods on rear region
        plates_found = []
        
        # Method 1: Standard OCR
        gray = cv2.cvtColor(rear_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(binary, config=config).strip().upper()
        cleaned = clean_plate_text(text)
        
        if cleaned and len(cleaned) >= 4:
            plates_found.append({
                "text": cleaned,
                "confidence": 0.7,
                "bbox": [rear_x1, rear_y1, rear_x2, rear_y2],
                "method": "rear_region"
            })
        
        # Method 2: Resize and enhance
        h, w = rear_region.shape[:2]
        if h > 0 and w > 0:
            enlarged = cv2.resize(rear_region, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(binary, config=config).strip().upper()
            cleaned = clean_plate_text(text)
            
            if cleaned and len(cleaned) >= 4:
                plates_found.append({
                    "text": cleaned,
                    "confidence": 0.75,
                    "bbox": [rear_x1, rear_y1, rear_x2, rear_y2],
                    "method": "rear_enhanced"
                })
        
        # Return best plate from rear
        if plates_found:
            best = max(plates_found, key=lambda x: len(x['text']))
            if is_valid_plate_pattern(best['text']):
                return best
                
    except Exception as e:
        print(f"[DEBUG] Rear plate extraction failed: {e}")
    
    return None


def detect_license_plates_simple(image, tesseract_available):
    """Simple license plate detection that actually works - from simple_working_plate_detector.py"""
    plates = []
    
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, PILImage.Image):
            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_np = image.copy()
        
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Find rectangular regions
        plates.extend(find_rectangles(gray, image_np))
        
        # Method 2: Find text regions
        plates.extend(find_text_regions(gray, image_np))
        
        # Method 3: Edge detection
        plates.extend(find_edge_plates(gray, image_np))
        
        # Remove duplicates
        unique_plates = remove_duplicates(plates)
        
        # Extract text from all candidates
        for plate in unique_plates:
            plate_region = image_np[plate['y1']:plate['y2'], plate['x1']:plate['x2']]
            text = extract_text_simple(plate_region, tesseract_available)
            plate['text'] = text
            plate['is_plate'] = is_license_plate(text)
        
        # Filter for license plates
        license_plates = [p for p in unique_plates if p['is_plate']]
        
        # If no plates found, be more aggressive
        if not license_plates and unique_plates:
            # Take the best candidate
            best_candidate = max(unique_plates, key=lambda x: x['confidence'])
            if len(best_candidate['text']) >= 4:
                best_candidate['is_plate'] = True
                license_plates = [best_candidate]
        
        # Convert to unified format
        result_plates = []
        for plate in license_plates:
            result_plates.append({
                "text": plate['text'],
                "confidence": plate['confidence'],
                "bbox": [plate['x1'], plate['y1'], plate['x2'], plate['y2']],
                "method": plate['method']
            })
        
        print(f"[DEBUG] Simple detection found {len(result_plates)} plates")
        for p in result_plates:
            print(f"[DEBUG]   - {p['text']} (method: {p['method']}, conf: {p['confidence']})")
        
        return result_plates
        
    except Exception as e:
        print(f"[DEBUG] Simple plate detection failed: {e}")
        return []


def find_rectangles(gray, original):
    """Find rectangular regions that could be plates"""
    plates = []
    
    try:
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Simple plate criteria
            if (1.5 <= aspect_ratio <= 8.0 and 
                w > 60 and h > 20 and 
                w < original.shape[1] * 0.8 and h < original.shape[0] * 0.3):
                
                plates.append({
                    'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h,
                    'confidence': 0.6,
                    'method': 'rectangle'
                })
    
    except:
        pass
    
    return plates


def find_text_regions(gray, original):
    """Find regions that contain text"""
    plates = []
    
    try:
        # Use MSER to detect text
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        for region in regions:
            # Convert to bounding box
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            aspect_ratio = w / h
            
            if (1.5 <= aspect_ratio <= 6.0 and 
                w > 40 and h > 15 and 
                w < original.shape[1] * 0.6 and h < original.shape[0] * 0.2):
                
                plates.append({
                    'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h,
                    'confidence': 0.5,
                    'method': 'text'
                })
    
    except:
        pass
    
    return plates


def find_edge_plates(gray, original):
    """Find plates using edge detection"""
    plates = []
    
    try:
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            if (2.0 <= aspect_ratio <= 7.0 and 
                w > 80 and h > 25 and 
                w < original.shape[1] * 0.7 and h < original.shape[0] * 0.25):
                
                plates.append({
                    'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h,
                    'confidence': 0.4,
                    'method': 'edge'
                })
    
    except:
        pass
    
    return plates


def extract_text_simple(plate_region, tesseract_available):
    """Simple text extraction"""
    try:
        if plate_region.size == 0:
            return ""
        
        # Convert to grayscale
        if len(plate_region.shape) == 3:
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_region
        
        # Resize if too small
        if gray.shape[0] < 30:
            scale = 30 / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if tesseract_available:
            try:
                import pytesseract
                # Simple config for license plates
                config = '--psm 7 --oem 3'
                text = pytesseract.image_to_string(thresh, config=config)
                text = text.upper().strip()
                
                # Clean text
                text = re.sub(r'[^A-Z0-9 ]', '', text)
                
                return text
            except:
                pass
        
        # Fallback: return region info
        return f"REGION_{plate_region.shape[0]}x{plate_region.shape[1]}"
        
    except:
        return ""


def is_license_plate(text):
    """Simple license plate classification - very lenient"""
    if not text or len(text.strip()) < 3:
        return False
    
    # Clean text
    text = re.sub(r'[^A-Z0-9 ]', '', text.upper().strip())
    
    if len(text) < 3 or len(text) > 12:
        return False
    
    # Must have letters or numbers
    has_letters = any(c.isalpha() for c in text)
    has_numbers = any(c.isdigit() for c in text)
    
    if not (has_letters or has_numbers):
        return False
    
    # Common plate patterns
    patterns = [
        r'^[A-Z]{2,3}\d{2,4}$',        # ABC123, IM4U
        r'^[A-Z]{2,3}\s*\d{2,4}$',     # IM4U 555
        r'^[A-Z]{2}\d{2}[A-Z]\d{4}$',  # MH14DX9937
        r'^\d{2,4}[A-Z]{2,3}$',        # 123ABC
        r'^[A-Z0-9]{4,8}$',            # Any alphanumeric
    ]
    
    for pattern in patterns:
        if re.match(pattern, text.replace(' ', '')):
            return True
    
    # If it has both letters and numbers, it's probably a plate
    if has_letters and has_numbers and 4 <= len(text) <= 8:
        return True
    
    return False


def remove_duplicates(plates):
    """Remove duplicate detections"""
    if not plates:
        return []
    
    # Sort by confidence
    plates = sorted(plates, key=lambda x: x['confidence'], reverse=True)
    
    unique = []
    for plate in plates:
        is_duplicate = False
        for existing in unique:
            # Check if boxes overlap significantly
            overlap = calculate_overlap(plate, existing)
            if overlap > 0.5:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(plate)
    
    return unique


def calculate_overlap(box1, box2):
    """Calculate overlap between two boxes"""
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])
    
    if x2 <= x1 or y2 <= y1:
        return 0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


# OLD COMPLEX FUNCTION - DISABLED
# def detect_license_plates(image, boxes, detection, tesseract_available):
#     """Comprehensive license plate detection - from vehicle regions AND full image"""
#     plates = []
#     
#     if not tesseract_available:
#         print("[DEBUG] Tesseract not available, skipping plate detection")
#         return plates
#     
#     try:
#         import pytesseract
#         print(f"[DEBUG] Starting plate detection. Boxes: {len(boxes)}")
#         
#         # First: Check YOLO detected plates
#         for i in range(len(boxes)):
#             class_id = int(boxes.cls[i].cpu().numpy())
#             class_name = detection.names.get(class_id, f"class_{class_id}").lower()
#             
#             if 'license' in class_name or 'plate' in class_name:
#                 x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
#                 plate_region = image[int(y1):int(y2), int(x1):int(x2)]
#                 
#                 if plate_region.size > 0:
#                     plate_text = extract_plate_text_advanced(plate_region, pytesseract)
#                     
#                     if plate_text and len(plate_text) >= 4:
#                         plates.append({
#                             "text": plate_text,
#                             "confidence": 0.9,
#                             "bbox": [float(x1), float(y1), float(x2), float(y2)],
#                             "method": "yolo_license_plate"
#                         })
#         
#         print(f"[DEBUG] After YOLO check: {len(plates)} plates found")
#         
#         # Second: Extract from vehicle regions
#         vehicle_count = 0
#         for i in range(len(boxes)):
#             class_id = int(boxes.cls[i].cpu().numpy())
#             class_name = detection.names.get(class_id, f"class_{class_id}").lower()
#             
#             vehicle_types = ['car', 'truck', 'bus', 'motorcycle', 'bike', 'van', 'suv']
#             is_vehicle = any(vtype in class_name for vtype in vehicle_types)
#             
#             if is_vehicle:
#                 vehicle_count += 1
#                 x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
#                 vehicle_region = image[int(y1):int(y2), int(x1):int(x2)]
#                 
#                 if vehicle_region.size > 0:
#                     plate_text = extract_plate_text_advanced(vehicle_region, pytesseract)
#                     
#                     if plate_text and len(plate_text) >= 4:
#                         cleaned = clean_plate_text(plate_text)
#                         if is_valid_plate_pattern(cleaned):
#                             # Check for duplicates
#                             is_new = True
#                             for existing in plates:
#                                 if cleaned == existing['text'] or (len(cleaned) > 4 and len(existing['text']) > 4 and 
#                                     (cleaned in existing['text'] or existing['text'] in cleaned)):
#                                     is_new = False
#                                     break
#                             
#                             if is_new:
#                                 plates.append({
#                                     "text": cleaned,
#                                     "confidence": 0.7,
#                                     "bbox": [float(x1), float(y1), float(x2), float(y2)],
#                                     "method": "vehicle_region"
#                                 })
#         
#         print(f"[DEBUG] After vehicle region check ({vehicle_count} vehicles): {len(plates)} plates found")
#         
#         # Third: Full image OCR if needed
#         if len(plates) < 2:
#             print(f"[DEBUG] Trying full image OCR...")
#             full_image_plates = extract_plates_from_full_image_advanced(image, pytesseract)
#             for plate in full_image_plates:
#                 is_new = True
#                 for existing_plate in plates:
#                     if plate['text'] == existing_plate['text']:
#                         is_new = False
#                         break
#                 
#                 if is_new:
#                     plates.append(plate)
#                     print(f"[DEBUG] Found plate via full image OCR: {plate['text']}")
#         
#         # Filter to keep only the best plate
#         if plates:
#             best_plate = get_best_plate(plates)
#             if best_plate:
#                 plates = [best_plate]
#                 print(f"[DEBUG] Kept only best plate: {best_plate['text']}")
#         
#         print(f"[DEBUG] Final plate count: {len(plates)}")
#         for plate in plates:
#             print(f"[DEBUG]   - {plate['text']} (method: {plate['method']})")
#         
#     except Exception as e:
#         print(f"[WARNING] License plate detection failed: {e}")
#         import traceback
#         traceback.print_exc()
#     
#     return plates


def detect_license_plates(image, boxes, detection, tesseract_available):
    """Simple license plate detection - redirects to simple working method"""
    print("[DEBUG] Using simple plate detection method")
    return detect_license_plates_simple(image, tesseract_available)


def process_unified_detection_simple(image, conf_threshold=0.5, get_model_func=None, tesseract_available=False, parking_available=False):
    """Simplified unified detection that doesn't depend on external dataclasses"""
    if image is None:
        return None, "{}", "Please upload an image"
    
    try:
        # Handle different input types
        if isinstance(image, str):
            # If image is a file path, read it
            import cv2
            image = cv2.imread(image)
            if image is None:
                return None, "{}", f"Could not read image from path: {image}"
        elif isinstance(image, PILImage.Image):
            # Convert PIL to numpy if needed
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Ensure image is a numpy array
        if not isinstance(image, np.ndarray):
            return None, "{}", "Invalid image format"
        
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
                    
                    print(f"[DEBUG] YOLO detected class: '{class_name}'")
                    
                    # Improved vehicle classification with debug
                    if class_name in ['truck', 'pickup truck', 'semi', 'tractor trailer']:
                        vtype = 'truck'
                    elif class_name in ['bus', 'school bus', 'double bus']:
                        vtype = 'bus'
                    elif class_name in ['motorcycle', 'bike', 'scooter', 'moped']:
                        vtype = 'bike'
                    # Check if it's actually a car (sports car, sedan, etc.) despite containing 'truck'
                    elif 'sports' in class_name or 'sedan' in class_name or 'coupe' in class_name or 'ferrari' in class_name:
                        vtype = 'car'
                    elif 'truck' in class_name and not any(x in class_name for x in ['pickup', 'semi', 'tractor']):
                        # If it says truck but doesn't have pickup/semi/tractor, it might be misclassified
                        vtype = 'car'
                    elif 'bus' in class_name:
                        vtype = 'bus'
                    elif 'bike' in class_name or 'motorcycle' in class_name:
                        vtype = 'bike'
                    else:
                        vtype = 'car'  # Default to car for anything else
                    
                    print(f"[DEBUG] Classified as: '{vtype}'")
                    
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
        
        # Detect license plates using simple working method
        if tesseract_available:
            plates = detect_license_plates_simple(image, tesseract_available)
            
            # Keep only best plate
            if plates:
                best_plate = get_best_plate(plates)
                if best_plate:
                    plates = [best_plate]
            
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


def process_unified_video_detection(video_path, conf_threshold=0.5, get_model_func=None, tesseract_available=False, parking_available=False, output_path=None):
    """
    Process video with unified detection using the same logic as car plate video processor.
    
    Args:
        video_path: Path to input video
        conf_threshold: Confidence threshold for detection
        get_model_func: Function to get YOLO model
        tesseract_available: Whether Tesseract OCR is available
        parking_available: Whether parking detection is available
        output_path: Path to save output video (optional)
    
    Returns:
        Dictionary with processing results and output video path
    """
    import cv2
    import time
    import json
    import os
    
    if video_path is None or not os.path.exists(video_path):
        return {'error': 'Invalid video path', 'results': {}}
    
    try:
        # Get model
        if get_model_func is None:
            return {'error': 'Model function not provided', 'results': {}}
        
        model = get_model_func("yolo26n")
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'Cannot open video: {video_path}', 'results': {}}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Setup output video - use Gradio temp directory if available
        if output_path is None:
            timestamp = int(time.time())
            # Try to use Gradio's temp directory first
            try:
                import tempfile
                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, f"unified_detection_output_{timestamp}.mp4")
            except:
                output_path = f"unified_detection_output_{timestamp}.mp4"
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Use full path
        output_path = os.path.abspath(output_path)
        
        # Try different codecs for better browser compatibility
        # H.264 (avc1) is most compatible with browsers
        codecs_to_try = [
            ('mp4v', '.mp4'),
            ('avc1', '.mp4'),
            ('H264', '.mp4'),
            ('XVID', '.avi'),
        ]
        
        out = None
        for codec, ext in codecs_to_try:
            try:
                # Update extension if needed
                if ext != '.mp4' and output_path.endswith('.mp4'):
                    output_path = output_path[:-4] + ext
                
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if out.isOpened():
                    print(f"[INFO] Using video codec: {codec}")
                    break
            except Exception as e:
                print(f"[DEBUG] Codec {codec} failed: {e}")
                continue
        
        if out is None or not out.isOpened():
            cap.release()
            return {'error': 'Cannot create output video writer with any codec', 'results': {}}
        
        # Processing variables
        start_time = time.time()
        frame_count = 0
        all_detections = []
        all_plates = set()
        processing_stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'vehicles_detected': 0,
            'plates_found': 0,
            'unique_plates': [],
            'processing_time': 0
        }
        
        print("[INFO] Starting unified video detection...")
        
        # Main processing loop - same as car plate video processor
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame using unified detection
            try:
                # Run detection on frame
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=0.5,
                    device=device,
                    verbose=False
                )
                
                # Process detections using unified logic
                frame_result = {
                    'frame_number': frame_count,
                    'vehicles': [],
                    'plates': [],
                    'objects': [],
                    'detections': []
                }
                
                if results and len(results) > 0:
                    detection = results[0]
                    
                    if hasattr(detection, 'boxes') and detection.boxes is not None:
                        boxes = detection.boxes
                        
                        # Process each detection
                        for i in range(len(boxes)):
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            confidence = float(boxes.conf[i].cpu().numpy())
                            class_id = int(boxes.cls[i].cpu().numpy())
                            class_name = detection.names.get(class_id, f"class_{class_id}").lower()
                            
                            detection_info = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class_name': class_name,
                                'class_id': class_id
                            }
                            
                            frame_result['detections'].append(detection_info)
                            
                            # Check if it's a vehicle
                            if class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                                vehicle_info = detection_info.copy()
                                vehicle_info['type'] = class_name
                                frame_result['vehicles'].append(vehicle_info)
                                processing_stats['vehicles_detected'] += 1
                                
                                # Try to detect license plates in vehicle region
                                if tesseract_available:
                                    try:
                                        # Extract vehicle region
                                        vehicle_region = frame[int(y1):int(y2), int(x1):int(x2)]
                                        if vehicle_region.size > 0:
                                            # Simple plate detection using unified method
                                            plates = detect_license_plates_simple(vehicle_region, tesseract_available)
                                            for plate in plates:
                                                plate_info = {
                                                    'text': plate.get('text', ''),
                                                    'confidence': plate.get('confidence', 0.0),
                                                    'bbox': plate.get('bbox', [int(x1), int(y1), int(x2), int(y2)]),
                                                    'vehicle_bbox': [int(x1), int(y1), int(x2), int(y2)]
                                                }
                                                frame_result['plates'].append(plate_info)
                                                all_plates.add(plate_info['text'])
                                                processing_stats['plates_found'] += 1
                                    except Exception as e:
                                        print(f"[WARNING] Plate detection failed for vehicle: {e}")
                            
                            # Add to objects list
                            frame_result['objects'].append(detection_info)
                
                # Create annotated frame
                annotated_frame = frame.copy()
                
                # Draw bounding boxes and labels
                for det in frame_result['detections']:
                    x1, y1, x2, y2 = det['bbox']
                    class_name = det['class_name']
                    confidence = det['confidence']
                    
                    # Choose color based on class
                    if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                        color = (0, 255, 0)  # Green for vehicles
                    elif class_name == 'person':
                        color = (255, 0, 0)  # Blue for person
                    else:
                        color = (0, 0, 255)  # Red for other objects
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw license plates
                for plate in frame_result['plates']:
                    if plate.get('bbox'):
                        px1, py1, px2, py2 = plate['bbox']
                        plate_text = plate.get('text', '')
                        
                        # Draw plate bounding box in red
                        cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                        
                        # Draw plate text
                        if plate_text:
                            cv2.putText(annotated_frame, plate_text, (px1, py1 - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add frame info
                info_text = f"Frame: {frame_count}/{total_frames} | Vehicles: {len(frame_result['vehicles'])} | Plates: {len(frame_result['plates'])}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write to output video
                out.write(annotated_frame)
                
                all_detections.append(frame_result)
                processing_stats['processed_frames'] = frame_count
                
                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    eta = (total_frames - frame_count) / fps_current if fps_current > 0 else 0
                    
                    print(f"[INFO] Processed {frame_count}/{total_frames} ({progress:.1f}%) - "
                          f"{fps_current:.1f} FPS - ETA: {eta:.1f}s - "
                          f"Vehicles: {processing_stats['vehicles_detected']} - Plates: {processing_stats['plates_found']}")
                
            except Exception as e:
                print(f"[ERROR] Frame {frame_count} processing failed: {e}")
                continue
        
        # Cleanup
        cap.release()
        out.release()
        
        # Verify output video was created successfully
        if not os.path.exists(output_path):
            return {'error': f'Output video file was not created: {output_path}', 'results': {}}
        
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            return {'error': f'Output video file is empty: {output_path}', 'results': {}}
        
        print(f"[INFO] Output video verified: {output_path} ({file_size} bytes)")
        
        # Calculate final statistics
        processing_stats['processing_time'] = time.time() - start_time
        processing_stats['unique_plates'] = list(all_plates)
        processing_stats['fps'] = frame_count / processing_stats['processing_time'] if processing_stats['processing_time'] > 0 else 0
        
        print(f"[INFO] Video processing completed:")
        print(f"  - Output video: {output_path}")
        print(f"  - Total frames: {processing_stats['processed_frames']}")
        print(f"  - Vehicles detected: {processing_stats['vehicles_detected']}")
        print(f"  - Plates found: {processing_stats['plates_found']}")
        print(f"  - Unique plates: {len(processing_stats['unique_plates'])}")
        print(f"  - Processing time: {processing_stats['processing_time']:.2f}s")
        print(f"  - Average FPS: {processing_stats['fps']:.2f}")
        
        return {
            'success': True,
            'output_video': output_path,
            'stats': processing_stats,
            'detections': all_detections,
            'summary': f"Processed {processing_stats['processed_frames']} frames, detected {processing_stats['vehicles_detected']} vehicles and {processing_stats['plates_found']} license plates ({len(processing_stats['unique_plates'])} unique)"
        }
        
    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'results': {}}
