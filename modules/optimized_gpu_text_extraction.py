"""
Optimized GPU Text Extraction for YOLO26
Fast GPU-accelerated text extraction using PyTorch CUDA + OpenCV
Compatible with RTX 4050 and modern GPUs
"""

import os
import sys
import time
import threading
import hashlib
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

# PyTorch for GPU acceleration
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    print("[INFO] PyTorch available for GPU acceleration")
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Global GPU optimization variables
_gpu_available = False
_device_info = {}
_cache_lock = threading.Lock()
_text_cache = {}

def initialize_gpu_environment():
    """Initialize GPU environment and detect CUDA availability"""
    global _gpu_available, _device_info
    
    try:
        if TORCH_AVAILABLE:
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
                
                # Set GPU optimization flags
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Enable mixed precision for faster processing
                torch.set_float32_matmul_precision('high')
                
            else:
                _device_info = {'pytorch_cuda': False}
                print("[INFO] ⚠️ CUDA not available, using CPU mode")
        else:
            _gpu_available = False
            print("[WARNING] PyTorch not available for GPU acceleration")
            
        return _gpu_available
        
    except Exception as e:
        print(f"[ERROR] GPU initialization failed: {e}")
        _gpu_available = False
        return False

def get_image_hash(image: np.ndarray) -> str:
    """Generate hash for image caching"""
    try:
        # Use a smaller sample for hashing to improve speed
        sample = image[::4, ::4] if image.shape[0] > 100 else image
        return hashlib.md5(sample.tobytes()).hexdigest()[:16]
    except:
        return str(time.time())

def preprocess_image_for_ocr_gpu(image: np.ndarray) -> np.ndarray:
    """GPU-optimized image preprocessing for OCR"""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # GPU-accelerated preprocessing if available
        if _gpu_available and TORCH_AVAILABLE:
            # Convert to tensor and move to GPU
            gray_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).cuda() / 255.0
            
            # Apply GPU filters
            # Contrast enhancement
            gray_tensor = torch.clamp(gray_tensor * 1.2, 0, 1)
            
            # Move back to CPU
            processed = (gray_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        else:
            processed = gray
        
        # Apply OpenCV preprocessing
        processed = cv2.equalizeHist(processed)
        
        # Noise reduction
        processed = cv2.bilateralFilter(processed, 9, 75, 75)
        
        # Thresholding
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return processed
        
    except Exception as e:
        print(f"[DEBUG] GPU preprocessing failed: {e}")
        return image

def detect_text_regions_opencv(image: np.ndarray, min_confidence: float = 0.5) -> List[Dict]:
    """Fast text region detection using OpenCV"""
    try:
        # Use MSER algorithm for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(image)
        
        text_regions = []
        h, w = image.shape[:2]
        
        for region in regions:
            if len(region) < 10:  # Skip very small regions
                continue
                
            # Get bounding box
            x, y, w_region, h_region = cv2.boundingRect(region)
            
            # Filter regions by aspect ratio and size
            aspect_ratio = w_region / h_region if h_region > 0 else 0
            area = w_region * h_region
            
            # Text-like characteristics
            if (aspect_ratio > 0.5 and aspect_ratio < 10 and 
                area > 100 and area < (h * w) * 0.5 and
                w_region > 20 and h_region > 10):
                
                confidence = min(1.0, area / 1000)  # Simple confidence scoring
                
                if confidence >= min_confidence:
                    text_regions.append({
                        'bbox': [x, y, x + w_region, y + h_region],
                        'confidence': confidence,
                        'area': area
                    })
        
        # Sort by confidence
        text_regions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return text_regions[:10]  # Return top 10 regions
        
    except Exception as e:
        print(f"[DEBUG] Text region detection failed: {e}")
        return []

def extract_text_from_region_tesseract(image: np.ndarray, region: Dict) -> Dict:
    """Extract text from a specific region using Tesseract"""
    try:
        import pytesseract
        
        x1, y1, x2, y2 = region['bbox']
        region_img = image[y1:y2, x1:x2]
        
        if region_img.size == 0:
            return {"text": "", "confidence": 0.0}
        
        # Preprocess region
        processed_region = preprocess_image_for_ocr_gpu(region_img)
        
        # Configure Tesseract for license plates
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        # Extract text
        text = pytesseract.image_to_string(processed_region, config=custom_config)
        confidence = 0.0
        
        try:
            data = pytesseract.image_to_data(processed_region, config=custom_config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            confidence = max(confidences) / 100.0 if confidences else 0.0
        except:
            pass
        
        return {
            "text": text.strip().upper(),
            "confidence": confidence,
            "bbox": region['bbox']
        }
        
    except ImportError:
        print("[WARNING] Tesseract not available")
        return {"text": "", "confidence": 0.0}
    except Exception as e:
        print(f"[DEBUG] Region OCR failed: {e}")
        return {"text": "", "confidence": 0.0}

def extract_text_optimized(
    image: np.ndarray, 
    confidence_threshold: float = 0.5,
    use_gpu: Optional[bool] = None,
    use_cache: bool = True,
    preprocess: bool = True
) -> Dict:
    """
    Extract text from image using optimized GPU processing.
    
    Args:
        image: Input image in BGR format
        confidence_threshold: Minimum confidence for text detection
        use_gpu: Whether to use GPU (auto-detect if None)
        use_cache: Use result caching for faster repeated processing
        preprocess: Apply preprocessing for better results
    
    Returns:
        Dictionary with extracted text and metadata
    """
    try:
        if image is None or image.size == 0:
            return {"text": "", "confidence": 0.0, "processing_time": 0.0, "method": "none"}
        
        # Auto-detect GPU if not specified
        if use_gpu is None:
            use_gpu = _gpu_available
        
        # Check cache first
        if use_cache:
            image_hash = get_image_hash(image)
            with _cache_lock:
                if image_hash in _text_cache:
                    cached_result = _text_cache[image_hash]
                    print(f"[DEBUG] Using cached OCR result")
                    return cached_result
        
        print(f"[DEBUG] Running optimized text extraction on {image.shape} (GPU: {use_gpu})")
        start_time = time.time()
        
        # Preprocess image for better OCR results
        processed_image = image
        if preprocess:
            processed_image = preprocess_image_for_ocr_gpu(image)
        
        # Detect text regions
        text_regions = detect_text_regions_opencv(processed_image, confidence_threshold)
        
        # Extract text from regions
        all_text_results = []
        
        for region in text_regions:
            result = extract_text_from_region_tesseract(processed_image, region)
            if result["text"] and result["confidence"] > 0.3:
                all_text_results.append(result)
        
        # Combine results
        if all_text_results:
            # Sort by confidence and take the best
            all_text_results.sort(key=lambda x: x["confidence"], reverse=True)
            best_result = all_text_results[0]
            final_text = best_result["text"]
            avg_confidence = best_result["confidence"]
        else:
            final_text = ""
            avg_confidence = 0.0
        
        processing_time = time.time() - start_time
        
        result = {
            "text": final_text,
            "confidence": avg_confidence,
            "processing_time": processing_time,
            "method": "opencv_gpu" if use_gpu else "opencv_cpu",
            "text_count": len(all_text_results),
            "text_regions": text_regions,
            "cached": False,
            "image_shape": image.shape,
            "device": "GPU" if use_gpu else "CPU"
        }
        
        # Cache result for future use
        if use_cache and image_hash:
            with _cache_lock:
                _text_cache[image_hash] = result
        
        print(f"[DEBUG] Text extraction completed: '{final_text}' (conf: {avg_confidence:.2f}, time: {processing_time:.3f}s)")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Optimized text extraction failed: {e}")
        return {
            "text": "",
            "confidence": 0.0,
            "processing_time": 0.0,
            "method": "error",
            "error": str(e)
        }

def extract_license_plates_optimized(
    image: np.ndarray,
    confidence_threshold: float = 0.6,
    use_gpu: Optional[bool] = None
) -> List[Dict]:
    """
    Extract license plates using optimized methods.
    
    Args:
        image: Input image in BGR format
        confidence_threshold: Minimum confidence for license plate detection
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        List of detected license plates with metadata
    """
    try:
        print(f"[DEBUG] Running optimized license plate extraction on {image.shape}")
        
        # Use general text extraction with license plate specific settings
        text_result = extract_text_optimized(
            image, 
            confidence_threshold=confidence_threshold,
            use_gpu=use_gpu,
            preprocess=True
        )
        
        if text_result["text"]:
            # Validate license plate format
            plate_text = text_result["text"].upper()
            
            # Basic validation for license plates
            if (len(plate_text) >= 4 and len(plate_text) <= 10 and
                any(c.isdigit() for c in plate_text) and
                any(c.isalpha() for c in plate_text)):
                
                return [{
                    "text": plate_text,
                    "confidence": text_result["confidence"],
                    "bbox": text_result.get("text_regions", [{}])[0].get("bbox", [0, 0, 0, 0]),
                    "method": text_result["method"],
                    "processing_time": text_result["processing_time"]
                }]
        
        return []
        
    except Exception as e:
        print(f"[ERROR] License plate extraction failed: {e}")
        return []

def get_gpu_info() -> Dict:
    """Get GPU information"""
    return _device_info.copy()

def clear_cache():
    """Clear text extraction cache"""
    global _text_cache
    with _cache_lock:
        _text_cache.clear()
    print("[INFO] OCR cache cleared")

# Initialize GPU environment on module import
initialize_gpu_environment()

# Test function
def test_optimized_text_extraction():
    """Test the optimized text extraction"""
    print("[TEST] Testing optimized text extraction...")
    
    # Create a test image
    test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    
    # Test extraction
    result = extract_text_optimized(test_image, use_cache=False)
    
    print(f"[TEST] Result: {result}")
    print(f"[TEST] GPU Available: {_gpu_available}")
    print(f"[TEST] Device Info: {_device_info}")

if __name__ == "__main__":
    test_optimized_text_extraction()
