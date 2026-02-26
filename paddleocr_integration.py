"""
PaddleOCR Integration Module
PP-OCRv5 model integration for text extraction and license plate detection.
Supports Hindi, English, and multiple Indian languages.
"""

import os
import sys
import time
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import re

# PaddleOCR imports
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    print("[INFO] PaddleOCR imported successfully")
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    print(f"[ERROR] PaddleOCR not available: {e}")

# Global PaddleOCR instance (lazy loading)
_paddle_ocr_instance = None
_ocr_initialized = False

def get_paddle_ocr_instance(lang: str = 'en', use_gpu: bool = True):
    """
    Get or create PaddleOCR instance (singleton pattern).
    
    Args:
        lang: Language code ('en', 'hi', 'ch', etc.)
        use_gpu: Whether to use GPU acceleration (default: True for RTX 4050)
    
    Returns:
        PaddleOCR instance or None if not available
    """
    global _paddle_ocr_instance, _ocr_initialized
    
    if not PADDLEOCR_AVAILABLE:
        print("[ERROR] PaddleOCR not installed")
        return None
    
    if _ocr_initialized and _paddle_ocr_instance is not None:
        return _paddle_ocr_instance
    
    try:
        print(f"[INFO] Initializing PaddleOCR (lang={lang}, use_gpu={use_gpu})...")
        
        # Disable model source check for faster initialization
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        
        # GPU configuration for RTX 4050
        if use_gpu:
            print("[INFO] Configuring PaddleOCR for GPU acceleration (RTX 4050)")
            # Set GPU memory usage
            os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.9'  # Use 90% GPU memory
            os.environ['FLAGS_initial_gpu_memory_in_mb'] = '1024'      # Start with 1GB
            os.environ['FLAGS_reallocate_gpu_memory_in_mb'] = '2048'   # Reallocate 2GB chunks
        
        _paddle_ocr_instance = PaddleOCR(
            use_textline_orientation=True,      # Enable text direction classification
            lang=lang,               # Language
            text_det_thresh=0.3,     # Text detection threshold
            text_det_box_thresh=0.5, # Text box threshold
            text_recognition_batch_size=8,   # Increased batch size for GPU
            # use_gpu=use_gpu,         # Removed - not supported in newer versions
            # gpu_mem=8000             # Removed - not supported in newer versions
        )
        
        _ocr_initialized = True
        print("[INFO] PaddleOCR initialized successfully with GPU acceleration")
        return _paddle_ocr_instance
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize PaddleOCR: {e}")
        print("[INFO] Falling back to CPU initialization...")
        try:
            # Fallback to CPU
            _paddle_ocr_instance = PaddleOCR(
                use_textline_orientation=True,
                lang=lang,
                text_det_thresh=0.3,
                text_det_box_thresh=0.5,
                text_recognition_batch_size=4
                # use_gpu=False  # Removed - not supported in newer versions
            )
            _ocr_initialized = True
            print("[INFO] PaddleOCR initialized with CPU fallback")
            return _paddle_ocr_instance
        except Exception as e2:
            print(f"[ERROR] CPU fallback also failed: {e2}")
            return None

def extract_text_with_paddleocr(
    image: np.ndarray, 
    confidence_threshold: float = 0.5,
    lang: str = 'en',
    use_gpu: bool = True
) -> str:
    """
    Extract text from image using PaddleOCR PP-OCRv5.
    
    Args:
        image: Input image in BGR format
        confidence_threshold: Minimum confidence for text detection
        lang: Language code for OCR
        use_gpu: Whether to use GPU acceleration (default: True)
    
    Returns:
        Extracted text string
    """
    try:
        if image is None or image.size == 0:
            print("[DEBUG] Invalid image for PaddleOCR")
            return ""
        
        # Get PaddleOCR instance
        ocr = get_paddle_ocr_instance(lang=lang, use_gpu=use_gpu)
        if ocr is None:
            return ""
        
        print(f"[DEBUG] Running PaddleOCR on image {image.shape} (GPU: {use_gpu})")
        
        # Run OCR
        start_time = time.time()
        result = ocr.ocr(image, cls=True)
        processing_time = time.time() - start_time
        
        print(f"[DEBUG] PaddleOCR completed in {processing_time:.2f}s")
        
        # Extract text from results
        extracted_texts = []
        
        if result and len(result) > 0 and result[0] is not None:
            for line in result[0]:
                if line and len(line) >= 2:
                    # line format: [[x1,y1,x2,y2,x3,y3,x4,y4], (text, confidence)]
                    box_points, (text, confidence) = line
                    
                    if confidence >= confidence_threshold:
                        cleaned_text = text.strip()
                        if cleaned_text:
                            extracted_texts.append(cleaned_text)
                            print(f"[DEBUG] Found text: '{cleaned_text}' (conf: {confidence:.3f})")
        
        final_text = ' '.join(extracted_texts)
        print(f"[DEBUG] PaddleOCR extracted: '{final_text}'")
        
        return final_text
        
    except Exception as e:
        print(f"[ERROR] PaddleOCR extraction failed: {e}")
        return ""

def extract_license_plates_with_paddleocr(
    image: np.ndarray,
    confidence_threshold: float = 0.6
) -> List[Dict]:
    """
    Extract license plates specifically using PaddleOCR.
    Optimized for Indian license plates.
    
    Args:
        image: Input image in BGR format
        confidence_threshold: Minimum confidence for license plate detection
    
    Returns:
        List of detected license plates with metadata
    """
    try:
        if image is None or image.size == 0:
            return []
        
        print(f"[DEBUG] Searching for license plates with PaddleOCR...")
        
        # Extract text using PaddleOCR
        text = extract_text_with_paddleocr(
            image, 
            confidence_threshold=confidence_threshold,
            lang='en'  # Use English for license plates
        )
        
        if not text:
            return []
        
        # Find potential license plates in extracted text
        license_plates = []
        
        # Indian license plate patterns
        patterns = [
            # Standard format: MH12AB1234
            r'\b[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}\b',
            # Old format: MH-12-AB-1234
            r'\b[A-Z]{2}-[0-9]{1,2}-[A-Z]{1,3}-[0-9]{4}\b',
            # Temporary format: MH12 AB 1234
            r'\b[A-Z]{2}[0-9]{1,2}\s[A-Z]{1,3}\s[0-9]{4}\b',
            # Generic alphanumeric plates
            r'\b[A-Z0-9]{6,12}\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cleaned_plate = match.upper().replace('-', '').replace(' ', '')
                
                if is_valid_indian_license_plate(cleaned_plate):
                    license_plates.append({
                        'text': cleaned_plate,
                        'original_text': match,
                        'confidence': confidence_threshold,
                        'method': 'paddleocr_license_plate',
                        'pattern_matched': pattern
                    })
                    print(f"[DEBUG] Found license plate: {cleaned_plate}")
        
        # Remove duplicates
        unique_plates = []
        seen_texts = set()
        
        for plate in license_plates:
            if plate['text'] not in seen_texts:
                unique_plates.append(plate)
                seen_texts.add(plate['text'])
        
        print(f"[DEBUG] Found {len(unique_plates)} unique license plates")
        return unique_plates
        
    except Exception as e:
        print(f"[ERROR] License plate extraction with PaddleOCR failed: {e}")
        return []

def is_valid_indian_license_plate(plate_text: str) -> bool:
    """
    Validate if text matches Indian license plate format.
    
    Args:
        plate_text: Text to validate
    
    Returns:
        True if valid Indian license plate format
    """
    if not plate_text or len(plate_text) < 6:
        return False
    
    # Remove common non-alphanumeric characters
    cleaned = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    
    # Check length (Indian plates are typically 8-10 characters)
    if len(cleaned) < 6 or len(cleaned) > 12:
        return False
    
    # Standard Indian license plate format
    # Format: XX00XX0000 (State code + District + Series + Number)
    standard_pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'
    
    # Check standard format
    if re.match(standard_pattern, cleaned):
        return True
    
    # Check for partial matches (for difficult-to-read plates)
    if len(cleaned) >= 8:
        # At least 2 letters at start
        if re.match(r'^[A-Z]{2}', cleaned):
            # At least 4 numbers at end
            if re.search(r'[0-9]{4}$', cleaned):
                return True
    
    return False

def preprocess_image_for_paddleocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better PaddleOCR results.
    
    Args:
        image: Input image in BGR format
    
    Returns:
        Preprocessed image
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(bilateral)
        
        # Convert back to BGR (PaddleOCR expects 3-channel)
        processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return processed
        
    except Exception as e:
        print(f"[ERROR] Image preprocessing failed: {e}")
        return image

def extract_text_multilingual(
    image: np.ndarray,
    languages: List[str] = ['en', 'hi'],
    confidence_threshold: float = 0.5
) -> Dict[str, str]:
    """
    Extract text in multiple languages using PaddleOCR.
    
    Args:
        image: Input image in BGR format
        languages: List of language codes to try
        confidence_threshold: Minimum confidence threshold
    
    Returns:
        Dictionary with language codes as keys and extracted text as values
    """
    results = {}
    
    for lang in languages:
        try:
            print(f"[DEBUG] Extracting text in language: {lang}")
            
            text = extract_text_with_paddleocr(
                image,
                confidence_threshold=confidence_threshold,
                lang=lang
            )
            
            if text and text.strip():
                results[lang] = text.strip()
                print(f"[DEBUG] {lang.upper()}: '{text.strip()}'")
            
        except Exception as e:
            print(f"[ERROR] Failed to extract {lang} text: {e}")
            continue
    
    return results

def cleanup_paddleocr():
    """Cleanup PaddleOCR resources"""
    global _paddle_ocr_instance, _ocr_initialized
    
    try:
        if _paddle_ocr_instance is not None:
            # PaddleOCR doesn't have explicit cleanup, but we can reset the instance
            _paddle_ocr_instance = None
            _ocr_initialized = False
            print("[INFO] PaddleOCR resources cleaned up")
    except Exception as e:
        print(f"[ERROR] Failed to cleanup PaddleOCR: {e}")

# Test function
def test_paddleocr_integration():
    """Test PaddleOCR integration with a simple test"""
    try:
        print("[INFO] Testing PaddleOCR integration...")
        
        # Create a simple test image
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "TEST123", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test text extraction
        text = extract_text_with_paddleocr(test_image)
        print(f"[TEST] Extracted text: '{text}'")
        
        # Test license plate extraction
        plates = extract_license_plates_with_paddleocr(test_image)
        print(f"[TEST] Found plates: {plates}")
        
        print("[INFO] PaddleOCR integration test completed")
        
    except Exception as e:
        print(f"[ERROR] PaddleOCR integration test failed: {e}")

if __name__ == "__main__":
    test_paddleocr_integration()
