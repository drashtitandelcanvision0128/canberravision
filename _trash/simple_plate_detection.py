"""
Simple Universal License Plate Detection
Koi bhi vehicle ho - car, motorcycle, bike
Bas uske aage jo number plate hai, uska text nikalo
Simple logic for all countries
"""

import cv2
import numpy as np
import re
from typing import List, Tuple, Optional

class SimpleLicensePlateDetector:
    """
    Simple universal license plate detector
    Kaam ki baat - koi bhi vehicle, bas number plate text nikalo
    """
    
    def __init__(self):
        # Simple regex for any license plate (sabhi countries ke liye)
        self.plate_patterns = [
            # 5-12 characters, letters aur numbers (90% plates)
            r'[A-Z0-9]{5,12}',
            
            # Letter-number combinations (common format)
            r'[A-Z]{2,4}[0-9]{2,6}',
            r'[0-9]{2,6}[A-Z]{2,4}',
            
            # With spaces/dashes
            r'[A-Z]{2,4}[- ]?[0-9]{2,6}[- ]?[A-Z]{0,3}',
            r'[0-9]{2,6}[- ]?[A-Z]{2,4}[- ]?[0-9]{0,3}',
        ]
    
    def find_license_plate_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Image mein license plate regions dhundo
        Simple logic - rectangle shapes jo text jaise dikhte hain
        """
        regions = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Edge detection (license plates have straight edges)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours (license plate ki shape)
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Too small ya too large skip karo
                if area < 1000 or area > (image.shape[0] * image.shape[1] * 0.1):
                    continue
                
                # Bounding box nikalo
                x, y, w, h = cv2.boundingRect(contour)
                
                # License plate ka aspect ratio (usually 2:1 to 6:1)
                aspect_ratio = w / h
                
                if 1.5 <= aspect_ratio <= 7.0:
                    # Thoda border add karo
                    x1 = max(0, x - 5)
                    y1 = max(0, y - 5)
                    x2 = min(image.shape[1], x + w + 5)
                    y2 = min(image.shape[0], y + h + 5)
                    
                    regions.append((x1, y1, x2, y2))
            
            # Method 2: White/light color detection (most plates are white)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # White/light color range
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 30, 255])
            
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find white regions
            white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in white_contours:
                area = cv2.contourArea(contour)
                if area < 800 or area > (image.shape[0] * image.shape[1] * 0.05):
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 2.0 <= aspect_ratio <= 6.0:
                    # Check contrast (text should have good contrast)
                    roi = image[y:y+h, x:x+w]
                    if roi.size > 0:
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        contrast = np.std(roi_gray)
                        
                        if contrast > 25:  # Good contrast for text
                            x1 = max(0, x - 3)
                            y1 = max(0, y - 3)
                            x2 = min(image.shape[1], x + w + 3)
                            y2 = min(image.shape[0], y + h + 3)
                            
                            # Duplicate check
                            is_duplicate = False
                            for existing in regions:
                                ex1, ey1, ex2, ey2 = existing
                                overlap = max(0, min(x2, ex2) - max(x1, ex1)) * max(0, min(y2, ey2) - max(y1, ey1))
                                if overlap > (w * h * 0.3):
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                regions.append((x1, y1, x2, y2))
            
        except Exception as e:
            print(f"[ERROR] Plate region detection failed: {e}")
        
        return regions
    
    def extract_text_from_regions(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[str]:
        """
        Detected regions se text extract karo
        Simple OCR with multiple attempts
        """
        extracted_texts = []
        
        try:
            import pytesseract
        except ImportError:
            print("[ERROR] pytesseract not available")
            return extracted_texts
        
        for i, (x1, y1, x2, y2) in enumerate(regions):
            try:
                # Region crop karo
                plate_region = image[y1:y2, x1:x2]
                
                if plate_region.size == 0:
                    continue
                
                print(f"[DEBUG] Processing region {i+1}: ({x1}, {y1}, {x2}, {y2})")
                
                # Multiple preprocessing attempts
                texts = []
                
                # Attempt 1: Direct OCR
                try:
                    text = pytesseract.image_to_string(plate_region, config='--psm 7 --oem 3')
                    if text.strip():
                        texts.append(text.strip())
                except:
                    pass
                
                # Attempt 2: Grayscale
                try:
                    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(gray, config='--psm 7 --oem 3')
                    if text.strip():
                        texts.append(text.strip())
                except:
                    pass
                
                # Attempt 3: Enhanced contrast
                try:
                    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    text = pytesseract.image_to_string(enhanced, config='--psm 7 --oem 3')
                    if text.strip():
                        texts.append(text.strip())
                except:
                    pass
                
                # Attempt 4: Binary threshold
                try:
                    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    text = pytesseract.image_to_string(binary, config='--psm 7 --oem 3')
                    if text.strip():
                        texts.append(text.strip())
                except:
                    pass
                
                # Best text select karo
                if texts:
                    best_text = self.select_best_plate_text(texts)
                    if best_text:
                        extracted_texts.append(best_text)
                        print(f"[DEBUG] ✅ Extracted: {best_text}")
                
            except Exception as e:
                print(f"[ERROR] Text extraction from region {i+1} failed: {e}")
                continue
        
        return extracted_texts
    
    def select_best_plate_text(self, texts: List[str]) -> Optional[str]:
        """
        Multiple texts se best choose karo
        Simple logic - license plate jaisa text
        """
        best_text = None
        best_score = 0
        
        for text in texts:
            # Clean text
            cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            if len(cleaned) < 5 or len(cleaned) > 15:
                continue
            
            # Score calculate karo
            score = 0
            
            # Length score (5-10 characters optimal)
            if 5 <= len(cleaned) <= 10:
                score += 3
            elif 11 <= len(cleaned) <= 15:
                score += 1
            
            # Character mix score (good plates have mix of letters and numbers)
            has_letters = bool(re.search(r'[A-Z]', cleaned))
            has_numbers = bool(re.search(r'[0-9]', cleaned))
            
            if has_letters and has_numbers:
                score += 2
            elif has_letters or has_numbers:
                score += 1
            
            # Pattern matching score
            for pattern in self.plate_patterns:
                if re.match(pattern, cleaned):
                    score += 2
                    break
            
            if score > best_score:
                best_score = score
                best_text = cleaned
        
        return best_text if best_score >= 2 else None
    
    def detect_and_extract_plates(self, image: np.ndarray) -> List[str]:
        """
        Complete process - regions dhundo, text extract karo
        Simple one-line function
        """
        print("[DEBUG] Starting simple license plate detection...")
        
        # Step 1: Plate regions dhundo
        regions = self.find_license_plate_regions(image)
        print(f"[DEBUG] Found {len(regions)} potential plate regions")
        
        if not regions:
            print("[DEBUG] No plate regions found")
            return []
        
        # Step 2: Text extract karo
        texts = self.extract_text_from_regions(image, regions)
        print(f"[DEBUG] Extracted {len(texts)} plate texts")
        
        # Step 3: Remove duplicates
        unique_texts = []
        seen = set()
        
        for text in texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)
        
        print(f"[DEBUG] Final unique plates: {unique_texts}")
        return unique_texts


# Simple integration function
def extract_license_plates_simple(image: np.ndarray) -> List[str]:
    """
    Most simple function - bas image do, plates le lo
    Koi bhi vehicle, koi bhi country - sabpe kaam karega
    """
    detector = SimpleLicensePlateDetector()
    return detector.detect_and_extract_plates(image)


def validate_plate_simple(plate_text: str) -> bool:
    """
    Simple validation - koi bhi text hai aur plate jaisa hai
    """
    if not plate_text or len(plate_text.strip()) < 5:
        return False
    
    cleaned = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    
    # Basic checks
    if len(cleaned) < 5 or len(cleaned) > 15:
        return False
    
    # Should have at least one letter or number
    if not re.search(r'[A-Z0-9]', cleaned):
        return False
    
    return True


# Integration with existing app.py
def simple_plate_integration(image_bgr: np.ndarray, existing_results: dict = None) -> dict:
    """
    Existing YOLO26 results ke saath simple plate detection add karo
    """
    # Simple plate detection
    simple_plates = extract_license_plates_simple(image_bgr)
    
    # Results format
    simple_results = {
        'simple_plate_detection': {
            'enabled': True,
            'plates_found': simple_plates,
            'total_plates': len(simple_plates),
            'method': 'simple_universal'
        }
    }
    
    # Agar existing results hain to unmein add karo
    if existing_results:
        if 'text_extraction' not in existing_results:
            existing_results['text_extraction'] = {}
        
        if 'license_plates' not in existing_results['text_extraction']:
            existing_results['text_extraction']['license_plates'] = []
        
        # Simple plates ko existing format mein add karo
        for i, plate_text in enumerate(simple_plates):
            plate_info = {
                'object_id': f'simple_plate_{i}',
                'plate_text': plate_text,
                'confidence': 0.8,  # Fixed confidence for simple method
                'method': 'simple_universal_detection'
            }
            existing_results['text_extraction']['license_plates'].append(plate_info)
        
        # Simple results bhi add karo
        existing_results.update(simple_results)
    
    return simple_results


if __name__ == "__main__":
    print("🚗 Simple Universal License Plate Detection")
    print("=" * 50)
    print("Logic:")
    print("1. Koi bhi vehicle ho - car, motorcycle, bike")
    print("2. Uske aage jo number plate hai")
    print("3. Bas ussi ka text nikalo")
    print("4. Sabhi countries mein same logic")
    print()
    
    # Test with sample
    detector = SimpleLicensePlateDetector()
    
    # Sample plates for testing
    test_plates = [
        "ABC123",
        "AB12CDE", 
        "123-ABC",
        "XYZ 9999",
        "MH20EE7602"
    ]
    
    print("🧪 Testing plate validation:")
    for plate in test_plates:
        is_valid = validate_plate_simple(plate)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  {plate:<12} → {status}")
    
    print("\n📖 Usage:")
    print("  plates = extract_license_plates_simple(image)")
    print("  is_valid = validate_plate_simple('ABC123')")
    print("  results = simple_plate_integration(image, existing_results)")
    
    print("\n✅ Simple logic ready - sab countries mein kaam karega!")
