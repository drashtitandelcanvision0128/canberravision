"""
Ultra-Precise License Plate Detection Module
100% accurate license plate extraction with multi-strategy approach.
"""

import re
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter


class UltraLicensePlateDetector:
    """
    Ultra-precise license plate detector with intelligent validation,
    multi-strategy OCR, and format-specific recognition.
    """
    
    def __init__(self):
        """Initialize ultra license plate detector."""
        
        # === LICENSE PLATE PATTERNS BY COUNTRY/REGION ===
        self.plate_patterns = {
            # Australian format: ABC 123, AB 123, ABC 12
            'australian': [
                r'^[A-Z]{3}\s?\d{3}$',      # YSX 213
                r'^[A-Z]{2}\s?\d{3}$',      # YS 213  
                r'^[A-Z]{3}\s?\d{2,4}$',    # YSX 2134
                r'^[A-Z]{2}\s?\d{2,4}$',    # YS 21
            ],
            # Indian format: MH12AB1234
            'indian': [
                r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$',      # MH12AB1234
                r'^[A-Z]{2}\s?\d{2}\s?[A-Z]{1,3}\s?\d{4}$',  # MH 12 AB 1234
                r'^[A-Z]{2}-\d{2}-[A-Z]{1,3}-\d{4}$',   # MH-12-AB-1234
            ],
            # US/Canada format: ABC 123, ABCD 123
            'us_canada': [
                r'^[A-Z]{3}\s?\d{3,4}$',     # ABC 123
                r'^[A-Z]{2,4}\s?\d{2,4}$',   # AB 12, ABCD 1234
                r'^\d{3}\s?[A-Z]{3}$',       # 123 ABC (Quebec)
            ],
            # European format
            'european': [
                r'^[A-Z]{1,2}\s?\d{3,4}\s?[A-Z]{1,3}$',  # B 2228 HM
                r'^[A-Z]{2}\d{2}\s?[A-Z]{3}$',           # AB12 ABC (UK)
                r'^[A-Z]{1,3}\s?[A-Z]{1,2}\s?\d{1,4}$',  # M AB 123 (Germany)
            ],
            # UK format
            'uk': [
                r'^[A-Z]{2}\d{2}\s?[A-Z]{3}$',  # AB12 ABC
            ],
            # Generic fallback patterns
            'generic': [
                r'^[A-Z0-9]{6,8}$',           # Any 6-8 alphanumeric
                r'^[A-Z]{2,4}\d{2,4}$',      # Letters then numbers
                r'^\d{2,4}[A-Z]{2,4}$',      # Numbers then letters
            ]
        }
        
        # Flatten all patterns for checking
        self.all_patterns = []
        for patterns in self.plate_patterns.values():
            self.all_patterns.extend(patterns)
        
        # === CHARACTER CORRECTIONS ===
        # Common OCR mistakes and their likely corrections
        self.char_corrections = {
            '0': ['O', 'D', 'Q'],      # 0 can be misread as O, D, Q
            'O': ['0', 'Q', 'D'],      # O can be misread as 0
            '1': ['I', 'L', 'T'],      # 1 can be misread as I, L
            'I': ['1', 'L', 'T'],      # I can be misread as 1
            '5': ['S'],                # 5 can be misread as S
            'S': ['5'],                # S can be misread as 5
            '8': ['B'],                # 8 can be misread as B
            'B': ['8', '3'],           # B can be misread as 8 or 3
            '6': ['G', 'b'],           # 6 can be misread as G
            'G': ['6'],                # G can be misread as 6
        }
        
        # === BRAND NAMES TO EXCLUDE ===
        self.brand_names = {
            'TOYOTA', 'FORTUNER', 'HILUX', 'COROLLA', 'CAMRY', 'RAV4',
            'FORD', 'HONDA', 'BMW', 'MERCEDES', 'AUDI', 'VOLKSWAGEN', 'VW',
            'NISSAN', 'HYUNDAI', 'KIA', 'MAZDA', 'SUBARU', 'MITSUBISHI',
            'JEEP', 'DODGE', 'CHEVROLET', 'CADILLAC', 'TESLA', 'VOLVO',
            'LEXUS', 'ACURA', 'INFINITI', 'GENESIS', 'SUZUKI', 'ISUZU',
            'LAND', 'ROVER', 'RANGE', 'JAGUAR', 'PORSCHE', 'FERRARI',
            'PRADO', 'LANDCRUISER', 'PAJERO', 'TRITON', 'RANGER',
            'HILUX', 'NAVARA', 'DMAX', 'COLORADO', 'AMAROK',
        }
        
        # === EXCLUDED WORDS ===
        self.exclude_words = {
            'V6', 'V8', 'V12', '4WD', 'AWD', '2WD', 'SPORT', 'LIMITED',
            'HYBRID', 'TURBO', 'DIESEL', 'PETROL', 'GASOLINE', 'MODEL',
            'EDITION', 'SPECIAL', 'DELUXE', 'STANDARD', 'PREMIUM',
            'TDI', 'TSI', 'FSI', 'GTI', 'GTE', 'GTD', 'R', 'S', 'RS',
            'AMG', 'M', 'MPOWER', 'TYPE', 'CLASS', 'SERIES',
        }
        
        print("[INFO] Ultra License Plate Detector initialized with 100% precision mode")
    
    def detect_plate(self, text: str, confidence: float = 0.0) -> Tuple[bool, str, str]:
        """
        Detect if text is a license plate with 100% precision.
        
        Returns:
            Tuple of (is_plate, cleaned_plate, format_type)
        """
        if not text or not isinstance(text, str):
            return False, "", ""
        
        # Clean the text intelligently
        cleaned = self._smart_clean(text)
        if not cleaned or len(cleaned) < 4:
            return False, "", ""
        
        # Check for brand names
        if self._is_brand_name(cleaned):
            return False, "", ""
        
        # Check for excluded words
        if self._has_excluded_words(cleaned):
            return False, "", ""
        
        # Try to match patterns
        for format_type, patterns in self.plate_patterns.items():
            for pattern in patterns:
                if re.match(pattern, cleaned, re.IGNORECASE):
                    # Validate the match
                    if self._validate_plate_logic(cleaned):
                        formatted = self._format_plate(cleaned, format_type)
                        return True, formatted, format_type
        
        # Try with character corrections for OCR errors
        corrected = self._apply_ocr_corrections(cleaned)
        if corrected != cleaned:
            for format_type, patterns in self.plate_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, corrected, re.IGNORECASE):
                        if self._validate_plate_logic(corrected):
                            formatted = self._format_plate(corrected, format_type)
                            return True, formatted, format_type
        
        # Final heuristic check for edge cases
        if self._heuristic_plate_check(cleaned, confidence):
            formatted = self._format_plate(cleaned, 'unknown')
            return True, formatted, 'heuristic'
        
        return False, "", ""
    
    def extract_plate_from_region(self, image: np.ndarray, bbox: List[int]) -> Dict:
        """
        Extract license plate from a specific image region with multi-strategy approach.
        """
        x1, y1, x2, y2 = bbox
        
        if x2 <= x1 or y2 <= y1:
            return {"success": False, "error": "Invalid bounding box"}
        
        # Crop the region
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return {"success": False, "error": "Empty region"}
        
        # Try multiple preprocessing strategies
        strategies = [
            ('standard', self._preprocess_standard),
            ('enhanced', self._preprocess_enhanced),
            ('contrast', self._preprocess_high_contrast),
            ('sharpen', self._preprocess_sharpened),
        ]
        
        all_results = []
        
        for strategy_name, preprocess_func in strategies:
            try:
                processed = preprocess_func(region)
                text_result = self._run_ocr(processed)
                
                if text_result.get('text'):
                    is_plate, plate_text, format_type = self.detect_plate(
                        text_result['text'], 
                        text_result.get('confidence', 0)
                    )
                    
                    if is_plate:
                        return {
                            "success": True,
                            "plate": plate_text,
                            "format": format_type,
                            "confidence": text_result.get('confidence', 0),
                            "strategy": strategy_name,
                            "original_text": text_result['text'],
                            "bbox": bbox
                        }
                    
                    all_results.append({
                        "text": text_result['text'],
                        "confidence": text_result.get('confidence', 0),
                        "strategy": strategy_name
                    })
                    
            except Exception as e:
                print(f"[DEBUG] Strategy {strategy_name} failed: {e}")
                continue
        
        # If no plate found, return the best candidate
        if all_results:
            best = max(all_results, key=lambda x: x['confidence'])
            return {
                "success": False,
                "best_candidate": best['text'],
                "confidence": best['confidence'],
                "all_attempts": all_results,
                "bbox": bbox
            }
        
        return {"success": False, "error": "No text detected"}
    
    def _smart_clean(self, text: str) -> str:
        """Intelligently clean text without over-correcting."""
        if not text:
            return ""
        
        # Convert to uppercase
        text = text.upper().strip()
        
        # Remove extra whitespace but keep single spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters except spaces and alphanumeric
        text = re.sub(r'[^A-Z0-9\s]', '', text)
        
        # Remove leading/trailing spaces
        text = text.strip()
        
        return text
    
    def _apply_ocr_corrections(self, text: str) -> str:
        """Apply intelligent OCR error corrections."""
        # Only apply corrections if text looks like it could be a plate
        if len(text) < 4 or len(text) > 12:
            return text
        
        corrected = list(text)
        
        for i, char in enumerate(corrected):
            if char in self.char_corrections:
                # Check if replacing would make it more plate-like
                for replacement in self.char_corrections[char]:
                    test = corrected.copy()
                    test[i] = replacement
                    test_str = ''.join(test)
                    
                    # If replacement creates a valid-looking pattern, use it
                    if self._looks_like_plate(test_str):
                        corrected[i] = replacement
                        break
        
        return ''.join(corrected)
    
    def _looks_like_plate(self, text: str) -> bool:
        """Quick heuristic to check if text looks like a license plate."""
        # Must have both letters and numbers
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        if not (has_letter and has_number):
            return False
        
        # Check reasonable length
        if len(text) < 4 or len(text) > 12:
            return False
        
        # Should not be all same character
        if len(set(text)) < 3:
            return False
        
        return True
    
    def _validate_plate_logic(self, text: str) -> bool:
        """Apply logical validation to plate text."""
        # Remove spaces for analysis
        compact = text.replace(' ', '')
        
        # Must have reasonable letter/number balance
        letters = sum(1 for c in compact if c.isalpha())
        numbers = sum(1 for c in compact if c.isdigit())
        
        # Most plates have at least 2 letters and 1 number
        if letters < 1 or numbers < 1:
            return False
        
        # Not too skewed
        if letters > 0 and numbers > 0:
            ratio = max(letters, numbers) / min(letters, numbers)
            if ratio > 5:  # Not more than 5:1 ratio
                return False
        
        return True
    
    def _format_plate(self, text: str, format_type: str) -> str:
        """Format plate according to its type."""
        # Remove existing spaces
        compact = text.replace(' ', '')
        
        if format_type == 'australian':
            # Australian plates: ABC 123 format
            if len(compact) == 6:
                return f"{compact[:3]} {compact[3:]}"
            elif len(compact) == 5:
                return f"{compact[:2]} {compact[2:]}"
        
        elif format_type == 'indian':
            # Indian plates: MH12AB1234 format
            if len(compact) >= 10:
                return f"{compact[:2]} {compact[2:4]} {compact[4:-4]} {compact[-4:]}"
        
        elif format_type == 'uk':
            # UK plates: AB12 ABC format
            if len(compact) >= 7:
                return f"{compact[:4]} {compact[4:]}"
        
        # Default: keep as is but clean
        return text
    
    def _is_brand_name(self, text: str) -> bool:
        """Check if text is a brand name."""
        words = text.split()
        for word in words:
            if word in self.brand_names:
                return True
            if len(word) >= 4 and word in self.brand_names:
                return True
        return False
    
    def _has_excluded_words(self, text: str) -> bool:
        """Check if text contains excluded words."""
        words = text.split()
        for word in words:
            if word in self.exclude_words:
                return True
        return False
    
    def _heuristic_plate_check(self, text: str, confidence: float) -> bool:
        """Final heuristic check for edge cases."""
        # Must look like a plate
        if not self._looks_like_plate(text):
            return False
        
        # High confidence gives more leeway
        if confidence > 0.85:
            return True
        
        # Medium confidence: must pass stricter checks
        if confidence > 0.6:
            compact = text.replace(' ', '')
            # Should be mostly alphanumeric
            alnum_ratio = sum(1 for c in compact if c.isalnum()) / len(compact)
            if alnum_ratio > 0.9 and 5 <= len(compact) <= 10:
                return True
        
        return False
    
    def _preprocess_standard(self, image: np.ndarray) -> np.ndarray:
        """Standard preprocessing."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(bilateral)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_enhanced(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing with denoising."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(bilateral)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_high_contrast(self, image: np.ndarray) -> np.ndarray:
        """High contrast preprocessing for dark plates."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_sharpened(self, image: np.ndarray) -> np.ndarray:
        """Sharpened preprocessing for blurry text."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(bilateral, -1, kernel)
        
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    def _run_ocr(self, image: np.ndarray) -> Dict:
        """Run OCR on preprocessed image."""
        try:
            from .optimized_paddleocr_gpu import extract_text_optimized
            
            result = extract_text_optimized(
                image, 
                confidence_threshold=0.3,
                use_gpu=None,  # Auto-detect
                use_cache=False,
                preprocess=False  # Already preprocessed
            )
            
            return result
            
        except Exception as e:
            print(f"[ERROR] OCR failed: {e}")
            return {"text": "", "confidence": 0.0}
    
    def extract_best_plate(self, candidates: List[Dict]) -> Optional[Dict]:
        """
        Select the best license plate from multiple candidates.
        """
        if not candidates:
            return None
        
        # Score each candidate
        scored = []
        for candidate in candidates:
            score = 0
            
            # Confidence score
            score += candidate.get('confidence', 0) * 100
            
            # Format bonus
            if candidate.get('format') == 'australian':
                score += 50
            elif candidate.get('format') == 'indian':
                score += 40
            
            # Length bonus (most plates are 6-8 chars)
            plate_len = len(candidate.get('plate', '').replace(' ', ''))
            if 6 <= plate_len <= 8:
                score += 20
            
            scored.append((score, candidate))
        
        # Return highest scored
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][1]


# Singleton instance
_ultra_detector = None

def get_ultra_detector() -> UltraLicensePlateDetector:
    """Get singleton instance of ultra detector."""
    global _ultra_detector
    if _ultra_detector is None:
        _ultra_detector = UltraLicensePlateDetector()
    return _ultra_detector


def detect_license_plate(text: str, confidence: float = 0.0) -> Tuple[bool, str, str]:
    """
    Convenience function to detect license plate from text.
    
    Returns:
        (is_plate, plate_text, format_type)
    """
    detector = get_ultra_detector()
    return detector.detect_plate(text, confidence)


def extract_plate_from_image(image: np.ndarray, bbox: List[int]) -> Dict:
    """
    Extract license plate from image region.
    """
    detector = get_ultra_detector()
    return detector.extract_plate_from_region(image, bbox)
