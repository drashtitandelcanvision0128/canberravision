"""
PRODUCTION-GRADE LICENSE PLATE OCR
Implements strict ROI cropping, contour-based text isolation, and format-aware filtering.
"""

import cv2
import numpy as np
import re
from typing import Tuple, List, Optional, Dict

# Debug mode - save intermediate images
DEBUG_MODE = True
DEBUG_DIR = "debug_plates"

if DEBUG_MODE:
    import os
    os.makedirs(DEBUG_DIR, exist_ok=True)


def extract_license_plate_text_strict(plate_crop: np.ndarray, plate_id: str = "unknown") -> Dict:
    """
    PRODUCTION-READY: Extract license plate text with strict validation.
    
    Args:
        plate_crop: The license plate region from YOLO detection
        plate_id: Identifier for debugging
        
    Returns:
        Dict with 'plate_text', 'confidence', 'country', 'debug_images'
    """
    result = {
        'plate_text': '',
        'confidence': 0.0,
        'country': '',
        'debug_images': {}
    }
    
    if plate_crop is None or plate_crop.size == 0:
        print(f"[ERROR] Empty plate crop for {plate_id}")
        return result
    
    print(f"\n{'='*60}")
    print(f"[STRICT-OCR] Processing plate: {plate_id}")
    print(f"[STRICT-OCR] Input size: {plate_crop.shape}")
    
    # STEP 1: Strict ROI Cropping with minimal padding
    cropped = _strict_crop_plate(plate_crop, padding=3)
    result['debug_images']['01_strict_crop'] = cropped.copy()
    print(f"[STRICT-OCR] Strict crop size: {cropped.shape}")
    
    # STEP 2: Grayscale and Noise Reduction
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    result['debug_images']['02_denoised'] = denoised.copy()
    
    # STEP 3: High-contrast thresholding for text isolation
    text_mask = _isolate_text_regions(denoised)
    result['debug_images']['03_text_mask'] = text_mask.copy()
    
    # STEP 4: Contour-based character filtering
    char_regions = _extract_character_regions(text_mask, denoised)
    if char_regions is None:
        print(f"[ERROR] No character regions found for {plate_id}")
        return result
    
    result['debug_images']['04_char_regions'] = char_regions.copy()
    
    # STEP 5: Perspective correction if needed
    straightened = _straighten_plate(char_regions, text_mask)
    result['debug_images']['05_straightened'] = straightened.copy()
    
    # STEP 6: Multi-pass OCR on cleaned plate
    ocr_results = _multi_pass_ocr(straightened)
    print(f"[STRICT-OCR] OCR candidates: {ocr_results}")
    
    # STEP 7: Format-aware validation and selection
    best_result = _select_by_format(ocr_results)
    
    if best_result:
        result['plate_text'] = best_result['text']
        result['confidence'] = best_result['confidence']
        result['country'] = best_result['country']
        
        print(f"[STRICT-OCR] ✅ FINAL: '{best_result['text']}' ({best_result['country']}, conf: {best_result['confidence']:.2f})")
    else:
        print(f"[STRICT-OCR] ❌ No valid plate text found")
    
    # Save debug images
    if DEBUG_MODE:
        _save_debug_images(result['debug_images'], plate_id)
    
    print(f"{'='*60}\n")
    return result


def _strict_crop_plate(plate_crop: np.ndarray, padding: int = 3) -> np.ndarray:
    """
    Strict ROI cropping - remove excess background, keep only plate area.
    Uses edge detection to find the actual plate boundaries.
    """
    h, w = plate_crop.shape[:2]
    
    # Convert to grayscale
    if len(plate_crop.shape) == 3:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_crop
    
    # Edge detection to find plate boundaries
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return plate_crop
    
    # Find the largest rectangular contour (the plate)
    max_area = 0
    best_rect = None
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        aspect = cw / max(ch, 1)
        
        # Plate aspect ratio: typically 3:1 to 6:1
        if 2.0 < aspect < 8.0 and area > max_area and area > (h * w * 0.1):
            max_area = area
            best_rect = (x, y, cw, ch)
    
    if best_rect:
        x, y, cw, ch = best_rect
        # Add small padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + cw + padding)
        y2 = min(h, y + ch + padding)
        
        cropped = plate_crop[y1:y2, x1:x2]
        return cropped
    
    return plate_crop


def _isolate_text_regions(gray: np.ndarray) -> np.ndarray:
    """
    Isolate text regions using adaptive thresholding and morphological ops.
    Returns binary mask of text regions.
    """
    # Resize for better processing
    h, w = gray.shape
    scale = 300 / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Adaptive threshold - inverted (text white on black)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 5)
    
    # Remove small noise
    kernel_small = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Connect character parts horizontally
    kernel_wide = np.ones((3, 15), np.uint8)
    text_connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_wide, iterations=1)
    
    return text_connected


def _extract_character_regions(text_mask: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract only character regions based on contour filtering.
    Filters by aspect ratio, size, and position.
    """
    # Find contours
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    h, w = text_mask.shape
    char_contours = []
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        area = cw * ch
        
        # Character filters:
        # - Aspect ratio: 0.2 to 1.2 (tall to wide characters)
        # - Height: 15% to 70% of plate height
        # - Width: at least 5 pixels
        if (0.15 < aspect < 1.5 and 
            h * 0.15 < ch < h * 0.8 and
            cw >= 5 and
            area > 50):
            char_contours.append((x, y, cw, ch, area))
    
    if not char_contours:
        # Fallback: return original
        return gray
    
    # Sort by x position (left to right)
    char_contours.sort(key=lambda c: c[0])
    
    # Find bounding box of all characters
    min_x = min(c[0] for c in char_contours)
    min_y = min(c[1] for c in char_contours)
    max_x = max(c[0] + c[2] for c in char_contours)
    max_y = max(c[1] + c[3] for c in char_contours)
    
    # Add padding
    pad = 5
    min_x = max(0, min_x - pad)
    min_y = max(0, min_y - pad)
    max_x = min(w, max_x + pad)
    max_y = min(h, max_y + pad)
    
    # Extract character region
    char_region = gray[min_y:max_y, min_x:max_x]
    
    # Scale up for OCR
    if char_region.shape[1] < 200:
        scale = 300 / char_region.shape[1]
        new_w = int(char_region.shape[1] * scale)
        new_h = int(char_region.shape[0] * scale)
        char_region = cv2.resize(char_region, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Binarize for OCR
    _, binary = cv2.threshold(char_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def _straighten_plate(char_regions: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    """
    Apply perspective correction if plate is tilted.
    Uses contour analysis to detect and correct rotation.
    """
    # Check if we need perspective correction
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 3:
        return char_regions
    
    # Find main text line
    points = []
    for cnt in contours[:10]:  # Top 10 contours
        x, y, w, h = cv2.boundingRect(cnt)
        points.append([x, y])
        points.append([x + w, y + h])
    
    if len(points) < 4:
        return char_regions
    
    # Check for skew using minAreaRect
    points = np.array(points)
    rect = cv2.minAreaRect(points)
    angle = rect[2]
    
    # If angle is significant, rotate
    if abs(angle) > 5 and abs(angle) < 85:
        h, w = char_regions.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(char_regions, M, (w, h), 
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=255)
        return rotated
    
    return char_regions


def _multi_pass_ocr(cleaned_plate: np.ndarray) -> List[Dict]:
    """
    Run OCR multiple times with different preprocessing and return all results.
    """
    results = []
    
    # Ensure we have the right format for Tesseract
    if len(cleaned_plate.shape) == 3:
        gray = cv2.cvtColor(cleaned_plate, cv2.COLOR_BGR2GRAY)
    else:
        gray = cleaned_plate
    
    # Preprocess variants
    variants = [
        ("original", gray),
        ("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)),
        ("inverted", cv2.bitwise_not(gray)),
    ]
    
    # Tesseract configs
    configs = [
        '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    ]
    
    import pytesseract
    
    for variant_name, img in variants:
        for config in configs:
            try:
                text = pytesseract.image_to_string(img, config=config).strip()
                cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                
                if cleaned_text and len(cleaned_text) >= 5:
                    results.append({
                        'text': cleaned_text,
                        'raw': text,
                        'variant': variant_name,
                        'config': config[:20]
                    })
            except:
                continue
    
    # Also try EasyOCR if available
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        easy_results = reader.readtext(rgb, detail=0, paragraph=False)
        
        for er in easy_results:
            cleaned = re.sub(r'[^A-Z0-9]', '', er.upper())
            if cleaned and len(cleaned) >= 5:
                results.append({
                    'text': cleaned,
                    'raw': er,
                    'variant': 'easyocr',
                    'config': 'easyocr'
                })
    except:
        pass
    
    return results


def _select_by_format(ocr_results: List[Dict]) -> Optional[Dict]:
    """
    Select best OCR result based on format matching for different countries.
    """
    if not ocr_results:
        return None
    
    scored_results = []
    
    for result in ocr_results:
        text = result['text']
        score = 0.0
        country = ''
        
        # UK format: AB12 CDE or AB12CDE (2 letters, 2 digits, space, 3 letters)
        if re.match(r'^[A-Z]{2}\d{2}\s?[A-Z]{3}$', text):
            score = 100.0
            country = 'UK'
        # Indian format: MH12AB1234
        elif re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$', text):
            score = 95.0
            country = 'India'
        # US format: ABC1234 or 123ABC
        elif re.match(r'^[A-Z]{1,3}\d{1,4}$', text) or re.match(r'^\d{1,4}[A-Z]{1,3}$', text):
            score = 80.0
            country = 'US'
        # EU format variations
        elif re.match(r'^[A-Z]{1,3}[\-]?\d{1,4}[\-]?[A-Z]{0,3}$', text):
            score = 75.0
            country = 'EU'
        # Generic alphanumeric
        elif re.match(r'^[A-Z0-9]{5,10}$', text):
            score = 50.0
            country = 'Unknown'
        else:
            score = 20.0
            country = 'Unknown'
        
        # Penalize unrealistic patterns
        if re.match(r'^[0O]+[1IL]+$|^[1IL]+[0O]+$', text):  # Only 0s and 1s
            score -= 40
        
        # Boost for realistic character distribution
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        if letters >= 2 and digits >= 2:
            score += 10
        
        scored_results.append({
            **result,
            'score': score,
            'confidence': min(0.99, score / 100),
            'country': country
        })
    
    # Sort by score
    scored_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Return best if score is good enough
    if scored_results and scored_results[0]['score'] >= 50:
        return scored_results[0]
    
    return None


def _save_debug_images(images: Dict, plate_id: str):
    """Save debug images for analysis."""
    for name, img in images.items():
        if img is not None:
            filename = f"{DEBUG_DIR}/{plate_id}_{name}.png"
            cv2.imwrite(filename, img)
    print(f"[STRICT-OCR] Debug images saved to {DEBUG_DIR}/")


# Legacy function wrapper for compatibility
def _extract_text_from_license_plate_crop(plate_crop: np.ndarray) -> str:
    """Wrapper that returns just the text for backward compatibility."""
    result = extract_license_plate_text_strict(plate_crop)
    return result.get('plate_text', '')
