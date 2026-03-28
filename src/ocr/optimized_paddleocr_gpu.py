"""
Optimized PaddleOCR GPU Integration Module
High-performance text extraction with GPU acceleration and smart caching.
Designed for fast object detection + accurate text extraction workflow.
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

# PaddleOCR imports
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    print("[INFO] PaddleOCR imported successfully")
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    print(f"[ERROR] PaddleOCR not available: {e}")

# Global variables for optimized GPU usage
_paddle_ocr_instances = {}
_ocr_initialized = False
_gpu_available = False
_device_info = {}
_cache_lock = threading.Lock()
_text_cache = {}
_ocr_init_failed = False

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
            
        # Set PaddleOCR environment variables for optimal performance
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        os.environ['FLAGS_use_cuda_malloc_async'] = 'True'  # Async memory allocation
        os.environ['FLAGS_memory_fraction_of_eager_deletion'] = '0.5'  # Memory optimization
        
        if _gpu_available:
            # GPU-specific optimizations
            os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.8'  # Use 80% GPU memory
            os.environ['FLAGS_initial_gpu_memory_in_mb'] = '1024'      # Start with 1GB
            os.environ['FLAGS_reallocate_gpu_memory_in_mb'] = '2048'   # Reallocate 2GB chunks
            os.environ['FLAGS_cuda_graph_mode'] = '1'  # Enable CUDA graphs for performance
            
        return _gpu_available
        
    except Exception as e:
        print(f"[ERROR] GPU initialization failed: {e}")
        _gpu_available = False
        return False

def get_paddle_ocr_instance(
    lang: str = 'en', 
    use_gpu: Optional[bool] = None,
    batch_size: int = 8,
    precision: str = 'fp32'
) -> Optional["PaddleOCR"]:
    """
    Get or create optimized PaddleOCR instance with GPU support.
    
    Args:
        lang: Language code ('en', 'hi', 'ch', etc.)
        use_gpu: Whether to use GPU (auto-detect if None)
        batch_size: Batch size for processing (higher for GPU)
        precision: Model precision ('fp32', 'fp16')
    
    Returns:
        Optimized PaddleOCR instance or None if not available
    """
    global _paddle_ocr_instances, _ocr_initialized, _gpu_available, _ocr_init_failed
    
    if not PADDLEOCR_AVAILABLE:
        print("[ERROR] PaddleOCR not installed")
        return None

    # If initialization already failed in this process, don't keep retrying every frame
    if _ocr_init_failed:
        return None
    
    # Auto-detect GPU if not specified
    if use_gpu is None:
        use_gpu = _gpu_available
    
    # Create instance key
    instance_key = f"{lang}_{use_gpu}_{batch_size}_{precision}"
    
    # Return existing instance if available
    if instance_key in _paddle_ocr_instances:
        return _paddle_ocr_instances[instance_key]

    def _create_ocr_with_retry(initial_kwargs: Dict) -> Optional[PaddleOCR]:
        """Create PaddleOCR while handling version-specific unsupported kwargs."""
        kwargs = dict(initial_kwargs)
        last_err = None
        for attempt in range(10):
            try:
                print(f"[DEBUG] OCR attempt {attempt+1}: kwargs={list(kwargs.keys())}")
                return PaddleOCR(**kwargs)
            except Exception as e:
                last_err = e
                msg = str(e)
                print(f"[DEBUG] OCR attempt {attempt+1} failed: {msg}")
                # PaddleOCR often raises: "Unknown argument: <name>"
                if "Unknown argument:" in msg:
                    bad = msg.split("Unknown argument:", 1)[1].strip()
                    bad = bad.split()[0].strip().strip(',').strip()  # defensive
                    if bad in kwargs:
                        print(f"[DEBUG] Removing unsupported kwarg: {bad}")
                        kwargs.pop(bad, None)
                        continue
                break
        if last_err is not None:
            print(f"[ERROR] All OCR attempts failed. Last error: {last_err}")
            raise last_err
        return None
    
    try:
        print(f"[INFO] Initializing optimized PaddleOCR (lang={lang}, gpu={use_gpu}, batch={batch_size})")

        # NOTE: PaddleOCR argument support varies significantly by version.
        # We retry automatically if a kwarg is not supported by the installed paddleocr version.
        base_kwargs = {
            'lang': lang,
            'use_textline_orientation': True,
            'text_det_thresh': 0.3,
            'text_det_box_thresh': 0.5,
            'text_recognition_batch_size': batch_size,
            'show_log': False,
            'drop_score': 0.3,
            'use_space_char': True,
        }
        ocr_instance = _create_ocr_with_retry(base_kwargs)
        
        # Cache the instance
        _paddle_ocr_instances[instance_key] = ocr_instance
        _ocr_initialized = True
        
        device_type = "🚀 GPU" if use_gpu else "💻 CPU"
        print(f"[INFO] PaddleOCR initialized successfully on {device_type}")
        return ocr_instance
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize PaddleOCR: {e}")

        # Single CPU fallback attempt with minimal args
        print("[INFO] Attempting CPU fallback...")
        try:
            cpu_kwargs = {
                'lang': lang,
                'use_textline_orientation': True,
                'text_det_thresh': 0.3,
                'text_det_box_thresh': 0.5,
                'text_recognition_batch_size': 4,
                'show_log': False,
                'drop_score': 0.3,
                'use_space_char': True,
            }
            cpu_instance = _create_ocr_with_retry(cpu_kwargs)
            _paddle_ocr_instances[instance_key] = cpu_instance
            print("[INFO] PaddleOCR initialized with CPU fallback")
            return cpu_instance
        except Exception as e2:
            print(f"[ERROR] CPU fallback failed: {e2}")
            _ocr_init_failed = True
            
            # Last-ditch: try with only 'lang' kwarg
            print("[INFO] Trying minimal OCR init (lang only)...")
            try:
                minimal_instance = PaddleOCR(lang='en')
                _paddle_ocr_instances[instance_key] = minimal_instance
                _ocr_init_failed = False  # reset flag since we succeeded
                print("[INFO] OCR initialized with minimal args (lang only)")
                return minimal_instance
            except Exception as e3:
                print(f"[ERROR] Minimal OCR init also failed: {e3}")
                _ocr_init_failed = True
                return None

def get_image_hash(image: np.ndarray) -> str:
    """Generate hash for image caching"""
    try:
        # Resize image to standard size for consistent hashing
        resized = cv2.resize(image, (64, 64))
        hash_bytes = hashlib.md5(resized.tobytes()).hexdigest()
        return hash_bytes
    except:
        return str(time.time())

def extract_text_optimized(
    image: np.ndarray, 
    confidence_threshold: float = 0.5,
    lang: str = 'en',
    use_gpu: Optional[bool] = None,
    use_cache: bool = True,
    preprocess: bool = True
) -> Dict:
    """
    Extract text from image using optimized PaddleOCR with GPU acceleration.
    
    Args:
        image: Input image in BGR format
        confidence_threshold: Minimum confidence for text detection
        lang: Language code for OCR
        use_gpu: Whether to use GPU (auto-detect if None)
        use_cache: Use result caching for faster repeated processing
        preprocess: Apply preprocessing for better results
    
    Returns:
        Dictionary containing extracted text and metadata
    """
    start_time = time.time()
    
    try:
        print(f"[DEBUG] 🚀 extract_text_optimized called with: image_shape={image.shape if image is not None else 'None'}, threshold={confidence_threshold}")
        
        if image is None or image.size == 0:
            print(f"[DEBUG] ❌ Invalid image provided")
            return {"text": "", "confidence": 0.0, "processing_time": 0.0, "method": "none"}
        
        # Check cache first
        if use_cache:
            image_hash = get_image_hash(image)
            cache_key = f"{image_hash}_{lang}_{confidence_threshold}"
            
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
        
        # Get optimized PaddleOCR instance
        print(f"[DEBUG] Getting PaddleOCR instance...")
        ocr = get_paddle_ocr_instance(lang=lang, use_gpu=use_gpu)
        if ocr is None:
            print(f"[DEBUG] ❌ Failed to get PaddleOCR instance")
            return {"text": "", "confidence": 0.0, "processing_time": 0.0, "method": "error"}
        
        print(f"[DEBUG] ✅ PaddleOCR instance obtained, running OCR on {image.shape} (GPU: {use_gpu})")
        
        # Run OCR with optimized parameters
        ocr_start = time.time()
        result = ocr.ocr(processed_image, cls=True)
        ocr_time = time.time() - ocr_start
        
        print(f"[DEBUG] OCR completed in {ocr_time:.3f}s, result: {result}")
        
        # Extract and process results
        extracted_texts = []
        total_confidence = 0.0
        text_regions = []
        
        if result and len(result) > 0 and result[0] is not None:
            print(f"[DEBUG] Processing {len(result[0])} detected text lines...")
            for i, line in enumerate(result[0]):
                if line and len(line) >= 2:
                    # line format: [[x1,y1,x2,y2,x3,y3,x4,y4], (text, confidence)]
                    box_points, (text, confidence) = line
                    
                    print(f"[DEBUG] Line {i+1}: text='{text}', confidence={confidence:.3f}")
                    
                    if confidence >= confidence_threshold:
                        cleaned_text = text.strip()
                        if cleaned_text:
                            extracted_texts.append(cleaned_text)
                            total_confidence += confidence
                            
                            # Add text region information
                            text_regions.append({
                                "text": cleaned_text,
                                "confidence": confidence,
                                "bbox": box_points,
                                "area": calculate_bbox_area(box_points)
                            })
                            print(f"[DEBUG] ✅ Accepted: '{cleaned_text}' (conf: {confidence:.3f})")
                        else:
                            print(f"[DEBUG] ❌ Rejected empty text")
                    else:
                        print(f"[DEBUG] ❌ Rejected low confidence: '{text}' (conf: {confidence:.3f} < {confidence_threshold})")
                else:
                    print(f"[DEBUG] ❌ Invalid line format: {line}")
        else:
            print(f"[DEBUG] ❌ No text detected - result: {result}")
        
        # Calculate metrics
        final_text = ' '.join(extracted_texts)
        avg_confidence = total_confidence / len(extracted_texts) if extracted_texts else 0.0
        processing_time = time.time() - start_time
        
        print(f"[DEBUG] Final result: '{final_text}' (avg_conf: {avg_confidence:.3f}, texts_found: {len(extracted_texts)})")
        
        result_dict = {
            "text": final_text,
            "confidence": avg_confidence,
            "processing_time": processing_time,
            "ocr_time": ocr_time,
            "method": "paddleocr_gpu" if use_gpu else "paddleocr_cpu",
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
        
        print(f"[DEBUG] PaddleOCR extracted '{final_text}' in {processing_time:.3f}s (OCR: {ocr_time:.3f}s)")
        return result_dict
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"[ERROR] ❌ Optimized PaddleOCR extraction failed: {e}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return {"text": "", "confidence": 0.0, "processing_time": processing_time, "method": "error", "error": str(e)}

def extract_license_plates_optimized(
    image: np.ndarray,
    confidence_threshold: float = 0.6,
    use_gpu: Optional[bool] = None
) -> List[Dict]:
    """
    Extract license plates with optimized PaddleOCR.
    
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
        
        # Extract text using optimized PaddleOCR
        text_result = extract_text_optimized(
            image, 
            confidence_threshold=confidence_threshold,
            lang='en',  # English for license plates
            use_gpu=use_gpu,
            preprocess=True
        )
        
        if not text_result["text"]:
            return []
        
        # Find license plates in extracted text
        license_plates = []
        extracted_text = text_result["text"]
        
        # Indian license plate patterns (optimized)
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
                        'method': 'paddleocr_optimized',
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

def preprocess_image_for_ocr(image: np.ndarray, enhance_for_video: bool = True) -> np.ndarray:
    """
    Advanced image preprocessing for better OCR results with angle correction.
    Enhanced for video frames with rotation and perspective handling.
    
    Args:
        image: Input image in BGR format
        enhance_for_video: Apply video-specific enhancements (angle correction, etc.)
    
    Returns:
        Preprocessed image in BGR format
    """
    try:
        if enhance_for_video:
            # Use advanced preprocessing for videos
            return preprocess_image_for_video_ocr(image)
        else:
            # Use standard preprocessing for images
            return _standard_preprocess(image)
        
    except Exception as e:
        print(f"[ERROR] Image preprocessing failed: {e}")
        return image

def _standard_preprocess(image: np.ndarray) -> np.ndarray:
    """Standard preprocessing for regular images"""
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
        
        # Convert back to BGR (PaddleOCR expects 3-channel)
        processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        return processed
        
    except Exception as e:
        print(f"[ERROR] Standard preprocessing failed: {e}")
        return image

def preprocess_image_for_video_ocr(image: np.ndarray) -> np.ndarray:
    """
    Advanced preprocessing specifically for video frames with angle correction.
    Handles rotated text, perspective issues, and video compression artifacts.
    
    Args:
        image: Input video frame in BGR format
    
    Returns:
        Preprocessed image in BGR format with corrected angles
    """
    try:
        print(f"[DEBUG] Applying advanced video preprocessing with angle correction...")
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Denoise for video compression artifacts
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Step 3: Apply bilateral filter for edge preservation
        bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        # Step 4: Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
        enhanced = clahe.apply(bilateral)
        
        # Step 5: Detect and correct text angle (deskewing)
        deskewed = _correct_text_angle(enhanced)
        
        # Step 6: Apply adaptive threshold for better text separation
        adaptive_thresh = cv2.adaptiveThreshold(
            deskewed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Step 7: Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Step 8: Advanced sharpening for text clarity
        sharpened = _advanced_sharpen(morph)
        
        # Step 9: Convert back to 3-channel for PaddleOCR
        processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        print(f"[DEBUG] Video preprocessing completed with angle correction")
        return processed
        
    except Exception as e:
        print(f"[ERROR] Video preprocessing failed: {e}")
        return image

def _correct_text_angle(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct text angle in the image.
    Handles rotated text in video frames.
    
    Args:
        image: Grayscale image
    
    Returns:
        Angle-corrected image
    """
    try:
        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Find lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:50]:  # Process first 50 lines
                angle = theta * 180 / np.pi
                # Convert to angle relative to horizontal
                if angle > 90:
                    angle = angle - 180
                elif angle > 45:
                    angle = angle - 90
                elif angle < -45:
                    angle = angle + 90
                angles.append(angle)
            
            if angles:
                # Find the most common angle
                median_angle = np.median(angles)
                
                # Only rotate if angle is significant (> 2 degrees)
                if abs(median_angle) > 2:
                    print(f"[DEBUG] Correcting text angle: {median_angle:.2f} degrees")
                    
                    # Rotate the image
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(image, M, (w, h), 
                                           flags=cv2.INTER_CUBIC, 
                                           borderMode=cv2.BORDER_REPLICATE)
                    return rotated
        
        return image
        
    except Exception as e:
        print(f"[DEBUG] Angle correction failed: {e}")
        return image

def _advanced_sharpen(image: np.ndarray) -> np.ndarray:
    """
    Advanced sharpening specifically for text in video frames.
    
    Args:
        image: Grayscale image
    
    Returns:
        Sharpened image
    """
    try:
        # Multiple sharpening kernels for different text types
        kernels = [
            np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]),  # Standard sharpen
            np.array([[0,-1,0], [-1,5,-1], [0,-1,0]]),      # Light sharpen
            np.array([[-2,-1,-2], [-1,12,-1], [-2,-1,-2]])  # Heavy sharpen
        ]
        
        # Apply standard sharpening
        sharpened = cv2.filter2D(image, -1, kernels[0])
        
        # Apply unsharp masking for fine details
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Combine results
        result = cv2.addWeighted(sharpened, 0.7, unsharp_mask, 0.3, 0)
        
        return result
        
    except Exception as e:
        print(f"[DEBUG] Advanced sharpening failed: {e}")
        return image

def extract_text_with_multiple_angles(image: np.ndarray, confidence_threshold: float = 0.3, lang: str = 'en', use_gpu: Optional[bool] = None) -> Dict:
    """
    Extract text using multiple angle approaches for best results.
    Especially useful for video frames with rotated text.
    
    Args:
        image: Input image in BGR format
        confidence_threshold: Minimum confidence for text detection
        lang: Language code for OCR
        use_gpu: Whether to use GPU (auto-detect if None)
    
    Returns:
        Dictionary containing extracted text and metadata
    """
    try:
        print(f"[DEBUG] Running multi-angle text extraction...")
        
        # Method 1: Standard preprocessing
        standard_result = extract_text_optimized(
            image, confidence_threshold, lang, use_gpu, use_cache=False, preprocess=False
        )
        
        # Apply standard preprocessing
        standard_processed = preprocess_image_for_ocr(image, enhance_for_video=False)
        standard_preprocessed_result = extract_text_optimized(
            standard_processed, confidence_threshold, lang, use_gpu, use_cache=False, preprocess=False
        )
        
        # Method 2: Video preprocessing with angle correction
        video_processed = preprocess_image_for_ocr(image, enhance_for_video=True)
        video_result = extract_text_optimized(
            video_processed, confidence_threshold, lang, use_gpu, use_cache=False, preprocess=False
        )
        
        # Method 3: Try different rotation angles if needed
        rotation_results = []
        if video_result["text"] and len(video_result["text"]) < 3:  # If very little text found
            for angle in [-15, -10, -5, 5, 10, 15]:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), 
                                       flags=cv2.INTER_CUBIC, 
                                       borderMode=cv2.BORDER_REPLICATE)
                
                rotated_result = extract_text_optimized(
                    rotated, confidence_threshold * 0.8, lang, use_gpu, use_cache=False, preprocess=False
                )
                
                if rotated_result["text"] and len(rotated_result["text"]) > len(video_result["text"]):
                    rotation_results.append(rotated_result)
        
        # Select the best result
        all_results = [standard_result, standard_preprocessed_result, video_result] + rotation_results
        
        # Filter valid results and sort by text length and confidence
        valid_results = [r for r in all_results if r["text"] and r["text"].strip()]
        
        if valid_results:
            # Score each result (text length + confidence)
            def score_result(result):
                text_length = len(result["text"].strip())
                confidence = result["confidence"]
                return text_length * confidence
            
            best_result = max(valid_results, key=score_result)
            
            # Add method information
            best_result["extraction_method"] = "multi_angle_enhanced"
            best_result["angle_corrected"] = True
            
            print(f"[DEBUG] Multi-angle extraction best result: '{best_result['text']}' (conf: {best_result['confidence']:.3f})")
            return best_result
        
        # Fallback to video result
        video_result["extraction_method"] = "video_enhanced"
        video_result["angle_corrected"] = True
        return video_result
        
    except Exception as e:
        print(f"[ERROR] Multi-angle extraction failed: {e}")
        # Fallback to standard method
        return extract_text_optimized(image, confidence_threshold, lang, use_gpu, use_cache=True, preprocess=True)

def calculate_bbox_area(bbox_points: List) -> float:
    """Calculate area of bounding box from corner points"""
    try:
        if len(bbox_points) >= 4:
            # Flatten points and reshape
            points = np.array(bbox_points).reshape(-1, 2)
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
    lang: str = 'en',
    use_gpu: Optional[bool] = None,
    max_workers: int = 4
) -> List[Dict]:
    """
    Extract text from multiple images in parallel for maximum performance.
    
    Args:
        images: List of input images in BGR format
        confidence_threshold: Minimum confidence for text detection
        lang: Language code for OCR
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
                lang=lang,
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
def test_optimized_paddleocr():
    """Test optimized PaddleOCR integration"""
    try:
        print("[INFO] Testing optimized PaddleOCR integration...")
        
        # Create test image
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "TEST123", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test optimized text extraction
        result = extract_text_optimized(test_image)
        print(f"[TEST] Optimized extraction: '{result['text']}' (conf: {result['confidence']:.3f})")
        
        # Test license plate extraction
        plates = extract_license_plates_optimized(test_image)
        print(f"[TEST] License plates found: {len(plates)}")
        
        # Test batch processing
        batch_results = batch_extract_text([test_image, test_image])
        print(f"[TEST] Batch processing: {len(batch_results)} results")
        
        print("[INFO] Optimized PaddleOCR test completed")
        
    except Exception as e:
        print(f"[ERROR] Optimized PaddleOCR test failed: {e}")

if __name__ == "__main__":
    test_optimized_paddleocr()
