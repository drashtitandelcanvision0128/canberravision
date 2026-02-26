"""
GPU-Optimized Text Extraction using EasyOCR
Fallback solution for Python 3.14 with RTX 4050 GPU support
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import re
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Try EasyOCR as GPU alternative
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("[INFO] 🚀 EasyOCR loaded with GPU support")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[WARNING] EasyOCR not available")

# Global variables for optimized GPU usage
_easyocr_reader = None
_gpu_available = False
_device_info = {}
_cache_lock = threading.Lock()
_text_cache = {}

def initialize_gpu_environment():
    """Initialize GPU environment and detect CUDA availability"""
    global _gpu_available, _device_info
    
    try:
        # Check PyTorch CUDA availability
        _gpu_available = torch.cuda.is_available()
        
        if _gpu_available:
            _device_info = {
                'pytorch_cuda': True,
                'device_count': torch.cuda.device_count(),
                'device_name': torch.cuda.get_device_name(0),
                'device_memory': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'current_device': torch.cuda.current_device()
            }
            print(f"[INFO] 🚀 CUDA GPU Detected: {_device_info['device_name']}")
            print(f"[INFO] GPU Memory: {_device_info['device_memory']:.1f} GB")
        else:
            _device_info = {'pytorch_cuda': False}
            print("[INFO] ⚠️ CUDA not available, using CPU mode")
            
        return _gpu_available
        
    except Exception as e:
        print(f"[ERROR] GPU initialization failed: {e}")
        _gpu_available = False
        return False

def get_easyocr_reader(gpu: bool = None):
    """Get or create EasyOCR reader with GPU support"""
    global _easyocr_reader, _gpu_available
    
    if not EASYOCR_AVAILABLE:
        print("[ERROR] EasyOCR not installed")
        return None
    
    if _easyocr_reader is not None:
        return _easyocr_reader
    
    try:
        # Auto-detect GPU if not specified
        if gpu is None:
            gpu = _gpu_available
        
        print(f"[INFO] Initializing EasyOCR (GPU: {gpu})")
        
        # Create EasyOCR reader
        _easyocr_reader = easyocr.Reader(
            ['en', 'hi'],  # English and Hindi
            gpu=gpu,
            verbose=False,
            detector=True,
            recognizer=True
        )
        
        device_type = "🚀 GPU" if gpu else "💻 CPU"
        print(f"[INFO] EasyOCR initialized successfully on {device_type}")
        return _easyocr_reader
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize EasyOCR: {e}")
        
        # Try CPU fallback
        if gpu:
            print("[INFO] Attempting CPU fallback...")
            try:
                _easyocr_reader = easyocr.Reader(
                    ['en', 'hi'],
                    gpu=False,
                    verbose=False
                )
                print("[INFO] EasyOCR initialized with CPU fallback")
                return _easyocr_reader
            except Exception as e2:
                print(f"[ERROR] CPU fallback failed: {e2}")
        
        return None

def get_image_hash(image: np.ndarray) -> str:
    """Generate hash for image caching"""
    try:
        resized = cv2.resize(image, (64, 64))
        hash_bytes = hashlib.md5(resized.tobytes()).hexdigest()
        return hash_bytes
    except:
        return str(time.time())

def extract_text_optimized(
    image: np.ndarray, 
    confidence_threshold: float = 0.5,
    use_gpu: Optional[bool] = None,
    use_cache: bool = True,
    preprocess: bool = True
) -> Dict:
    """
    Extract text from image using EasyOCR with GPU acceleration.
    
    Args:
        image: Input image in BGR format
        confidence_threshold: Minimum confidence for text detection
        use_gpu: Whether to use GPU (auto-detect if None)
        use_cache: Use result caching for faster repeated processing
        preprocess: Apply preprocessing for better results
    
    Returns:
        Dictionary containing extracted text and metadata
    """
    start_time = time.time()
    
    try:
        if image is None or image.size == 0:
            return {"text": "", "confidence": 0.0, "processing_time": 0.0, "method": "none"}
        
        # Check cache first
        if use_cache:
            image_hash = get_image_hash(image)
            cache_key = f"{image_hash}_{confidence_threshold}"
            
            with _cache_lock:
                if cache_key in _text_cache:
                    cached_result = _text_cache[cache_key].copy()
                    cached_result["cached"] = True
                    print(f"[DEBUG] Using cached OCR result")
                    return cached_result
        
        # Preprocess image for better OCR results
        processed_image = image
        if preprocess:
            processed_image = preprocess_image_for_ocr(image)
        
        # Get EasyOCR reader
        reader = get_easyocr_reader(gpu=use_gpu)
        if reader is None:
            return {"text": "", "confidence": 0.0, "processing_time": 0.0, "method": "error"}
        
        print(f"[DEBUG] Running EasyOCR on {image.shape} (GPU: {use_gpu})")
        
        # Run OCR
        ocr_start = time.time()
        results = reader.readtext(processed_image)
        ocr_time = time.time() - ocr_start
        
        # Extract and process results
        extracted_texts = []
        total_confidence = 0.0
        text_regions = []
        
        for (bbox, text, confidence) in results:
            if confidence >= confidence_threshold:
                cleaned_text = text.strip()
                if cleaned_text:
                    extracted_texts.append(cleaned_text)
                    total_confidence += confidence
                    
                    # Add text region information
                    text_regions.append({
                        "text": cleaned_text,
                        "confidence": confidence,
                        "bbox": bbox,
                        "area": calculate_bbox_area(bbox)
                    })
        
        # Calculate metrics
        final_text = ' '.join(extracted_texts)
        avg_confidence = total_confidence / len(extracted_texts) if extracted_texts else 0.0
        processing_time = time.time() - start_time
        
        result_dict = {
            "text": final_text,
            "confidence": avg_confidence,
            "processing_time": processing_time,
            "ocr_time": ocr_time,
            "method": "easyocr_gpu" if use_gpu else "easyocr_cpu",
            "text_count": len(extracted_texts),
            "text_regions": text_regions,
            "cached": False,
            "image_shape": image.shape,
            "device": "GPU" if use_gpu else "CPU"
        }
        
        # Cache result for future use
        if use_cache and final_text:
            with _cache_lock:
                _text_cache[cache_key] = result_dict.copy()
                
                # Limit cache size
                if len(_text_cache) > 100:
                    oldest_keys = list(_text_cache.keys())[:-50]
                    for key in oldest_keys:
                        del _text_cache[key]
        
        print(f"[DEBUG] EasyOCR extracted '{final_text}' in {processing_time:.3f}s (OCR: {ocr_time:.3f}s)")
        return result_dict
        
    except Exception as e:
        print(f"[ERROR] EasyOCR extraction failed: {e}")
        return {"text": "", "confidence": 0.0, "processing_time": time.time() - start_time, "method": "error"}

def extract_license_plates_optimized(
    image: np.ndarray,
    confidence_threshold: float = 0.6,
    use_gpu: Optional[bool] = None
) -> List[Dict]:
    """
    Extract license plates with EasyOCR.
    
    Args:
        image: Input image in BGR format
        confidence_threshold: Minimum confidence for license plate detection
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        List of detected license plates with metadata
    """
    try:
        if image is None or image.size == 0:
            return []
        
        print(f"[DEBUG] Optimized license plate search...")
        
        # Extract text using EasyOCR
        text_result = extract_text_optimized(
            image, 
            confidence_threshold=confidence_threshold,
            use_gpu=use_gpu,
            preprocess=True
        )
        
        if not text_result["text"]:
            return []
        
        # Find license plates in extracted text
        license_plates = []
        extracted_text = text_result["text"]
        
        # Indian license plate patterns
        patterns = [
            r'\b[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}\b',  # MH12AB1234
            r'\b[A-Z]{2}-[0-9]{1,2}-[A-Z]{1,3}-[0-9]{4}\b',  # MH-12-AB-1234
            r'\b[A-Z]{2}[0-9]{1,2}\s[A-Z]{1,3}\s[0-9]{4}\b',  # MH12 AB 1234
            r'\b[A-Z0-9]{6,12}\b'  # Generic alphanumeric
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, extracted_text, re.IGNORECASE)
            for match in matches:
                cleaned_plate = match.upper().replace('-', '').replace(' ', '')
                
                if is_valid_indian_license_plate(cleaned_plate):
                    # Find corresponding text region for confidence
                    plate_confidence = confidence_threshold
                    for region in text_result["text_regions"]:
                        if cleaned_plate in region["text"]:
                            plate_confidence = region["confidence"]
                            break
                    
                    license_plates.append({
                        'text': cleaned_plate,
                        'original_text': match,
                        'confidence': plate_confidence,
                        'method': 'easyocr_optimized',
                        'pattern_matched': pattern,
                        'processing_time': text_result["processing_time"],
                        'device': text_result["device"]
                    })
                    print(f"[DEBUG] ✅ Found license plate: {cleaned_plate} (conf: {plate_confidence:.3f})")
        
        # Remove duplicates and return best results
        unique_plates = []
        seen_texts = set()
        
        for plate in license_plates:
            if plate['text'] not in seen_texts:
                unique_plates.append(plate)
                seen_texts.add(plate['text'])
        
        print(f"[DEBUG] Found {len(unique_plates)} unique license plates")
        return unique_plates
        
    except Exception as e:
        print(f"[ERROR] Optimized license plate extraction failed: {e}")
        return []

def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Optimized image preprocessing for better OCR results.
    
    Args:
        image: Input image in BGR format
    
    Returns:
        Preprocessed image in BGR format
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for noise reduction while preserving edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(bilateral)
        
        # Optional: Apply sharpening for better text clarity
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Convert back to BGR (EasyOCR expects RGB, but we'll handle conversion)
        processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        return processed
        
    except Exception as e:
        print(f"[ERROR] Image preprocessing failed: {e}")
        return image

def calculate_bbox_area(bbox_points: List) -> float:
    """Calculate area of bounding box from corner points"""
    try:
        if len(bbox_points) >= 4:
            # EasyOCR bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            points = np.array(bbox_points)
            # Calculate polygon area
            return cv2.contourArea(points)
        return 0.0
    except:
        return 0.0

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
    
    # Remove non-alphanumeric characters
    cleaned = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    
    # Check length (Indian plates are typically 8-10 characters)
    if len(cleaned) < 6 or len(cleaned) > 12:
        return False
    
    # Standard Indian license plate format
    standard_pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'
    
    # Check standard format
    if re.match(standard_pattern, cleaned):
        return True
    
    # Check for partial matches
    if len(cleaned) >= 8:
        if re.match(r'^[A-Z]{2}', cleaned):  # At least 2 letters at start
            if re.search(r'[0-9]{4}$', cleaned):  # At least 4 numbers at end
                return True
    
    return False

def batch_extract_text(
    images: List[np.ndarray],
    confidence_threshold: float = 0.5,
    use_gpu: Optional[bool] = None,
    max_workers: int = 4
) -> List[Dict]:
    """
    Extract text from multiple images in parallel for maximum performance.
    
    Args:
        images: List of input images in BGR format
        confidence_threshold: Minimum confidence for text detection
        use_gpu: Whether to use GPU acceleration
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of dictionaries containing extracted text for each image
    """
    if not images:
        return []
    
    print(f"[INFO] Processing {len(images)} images in parallel...")
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for image in images:
            future = executor.submit(
                extract_text_optimized,
                image,
                confidence_threshold=confidence_threshold,
                use_gpu=use_gpu,
                use_cache=True,
                preprocess=True
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)  # 30 second timeout per image
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Batch processing failed: {e}")
                results.append({"text": "", "confidence": 0.0, "processing_time": 0.0, "method": "error"})
    
    total_time = time.time() - start_time
    print(f"[INFO] Batch processing completed in {total_time:.2f}s ({total_time/len(images):.3f}s per image)")
    
    return results

def cleanup_ocr_cache():
    """Clean up OCR cache to free memory"""
    global _text_cache
    with _cache_lock:
        _text_cache.clear()
    print("[INFO] OCR cache cleared")

def get_gpu_info() -> Dict:
    """Get GPU information"""
    return _device_info.copy()

# Initialize GPU environment on module import
initialize_gpu_environment()

# Test function
def test_easyocr_gpu():
    """Test EasyOCR GPU integration"""
    try:
        print("[INFO] Testing EasyOCR GPU integration...")
        
        # Create test image
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "MH12AB1234", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test optimized text extraction
        result = extract_text_optimized(test_image)
        print(f"[TEST] EasyOCR extraction: '{result['text']}' (conf: {result['confidence']:.3f})")
        
        # Test license plate extraction
        plates = extract_license_plates_optimized(test_image)
        print(f"[TEST] License plates found: {len(plates)}")
        
        # Test batch processing
        batch_results = batch_extract_text([test_image, test_image])
        print(f"[TEST] Batch processing: {len(batch_results)} results")
        
        print("[INFO] EasyOCR GPU test completed")
        
    except Exception as e:
        print(f"[ERROR] EasyOCR GPU test failed: {e}")

if __name__ == "__main__":
    test_easyocr_gpu()
