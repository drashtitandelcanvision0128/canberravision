"""
Webcam Processing Module for YOLO26
Handle all webcam/live camera operations
Real-time processing with performance optimization
"""

import cv2
import numpy as np
import time
import threading
import os
import sys
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from queue import Queue
import json

# YOLOv8 (Ultralytics) import
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

from collections import deque

# Tesseract fallback (optional)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    if sys.platform.startswith("win"):
        try:
            _default_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(_default_tesseract):
                pytesseract.pytesseract.tesseract_cmd = _default_tesseract
        except Exception:
            pass
except Exception:
    pytesseract = None
    TESSERACT_AVAILABLE = False

# Import torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# OCR imports (optional)
try:
    from optimized_paddleocr_gpu import extract_text_optimized
    OPTIMIZED_PADDLEOCR_AVAILABLE = True
except ImportError:
    OPTIMIZED_PADDLEOCR_AVAILABLE = False
    extract_text_optimized = None

try:
    from paddleocr_integration import extract_text_with_paddleocr
    LEGACY_PADDLEOCR_AVAILABLE = True
except ImportError:
    LEGACY_PADDLEOCR_AVAILABLE = False
    extract_text_with_paddleocr = None

# Import our modules
try:
    from image_processor import ImageProcessor, image_processor
    from simple_plate_detection import extract_license_plates_simple
    from advanced_color_detection import get_advanced_color_detector, detect_object_color_advanced
    from kmeans_color_detector import enhanced_detector, detect_colors_enhanced, categorize_object_enhanced
    from fallback_color_detector import fallback_detector, detect_colors_fallback, get_fallback_status
    IMAGE_PROCESSOR_AVAILABLE = True
    ADVANCED_COLOR_AVAILABLE = True
    ENHANCED_COLOR_AVAILABLE = True
    FALLBACK_COLOR_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Some modules not available: {e}")
    IMAGE_PROCESSOR_AVAILABLE = False
    ADVANCED_COLOR_AVAILABLE = False
    ENHANCED_COLOR_AVAILABLE = False
    FALLBACK_COLOR_AVAILABLE = False

class WebcamProcessor:
    """
    Handle all webcam/live camera processing operations
    """
    
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.processing_thread = None
        self.frame_queue = Queue(maxsize=30)  # Buffer for 30 frames
        self.result_queue = Queue(maxsize=30)
        self.current_frame = None
        self.latest_result = None
        
        # Enhanced detection features
        self.color_detector = None
        self.enhanced_detector = None
        self.fallback_detector = None
        self.enable_advanced_colors = True
        self.enable_enhanced_colors = True
        self.enable_fallback_colors = True
        self.enable_general_objects = True
        self.color_calibration_frame = None
        self.calibration_applied = False

        # OCR settings (kept conservative for performance)
        self.enable_ocr = True
        self.ocr_every_n = 10
        self._frame_idx = 0
        self._ocr_cache = {}
        self._tesseract_checked = False
        self._tesseract_ok = False
        self._last_ocr_debug_ts = 0.0

        # OCR class filtering
        self._ocr_skip_classes = {
            'person',
            'bird',
            'cat',
            'dog',
            'horse',
            'sheep',
            'cow',
            'elephant',
            'bear',
            'zebra',
            'giraffe',
        }
        self._ocr_allow_classes = {
            'cup',
            'bottle',
            'book',
            'laptop',
            'cell phone',
            'tv',
            'keyboard',
            'remote',
            'backpack',
            'handbag',
            'suitcase',
            'tie',
            'umbrella',
            'license plate',
        }
        self._ocr_vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}

        # License plate de-dup buffer
        self._plate_history = deque(maxlen=50)
        self._plate_seen_ttl_s = 8.0

        # YOLOv8s model cache
        self._yolo_v8_model = None
        self._yolo_v8_names = None
        
        # Initialize advanced color detector
        if ADVANCED_COLOR_AVAILABLE:
            try:
                self.color_detector = get_advanced_color_detector()
                print("[INFO] Advanced color detector initialized")
            except Exception as e:
                print(f"[WARNING] Failed to initialize advanced color detector: {e}")
                self.color_detector = None
        
        # Initialize Enhanced K-means + ResNet-18 color detector
        if ENHANCED_COLOR_AVAILABLE:
            try:
                self.enhanced_detector = enhanced_detector
                print("[INFO] Enhanced K-means + ResNet-18 color detector initialized")
            except Exception as e:
                print(f"[WARNING] Failed to initialize enhanced color detector: {e}")
                self.enhanced_detector = None
        
        # Initialize Fallback color detector
        if FALLBACK_COLOR_AVAILABLE:
            try:
                self.fallback_detector = fallback_detector
                fallback_status = get_fallback_status()
                print(f"[INFO] Fallback color detector initialized (Level {fallback_status['current_level']})")
                print(f"[INFO] Fallback recommendation: {fallback_status['recommended_usage']}")
            except Exception as e:
                print(f"[WARNING] Failed to initialize fallback color detector: {e}")
                self.fallback_detector = None
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'start_time': None,
            'fps': 0,
            'total_detections': 0,
            'unique_plates': set(),
            'processing_times': [],
            'objects_detected': 0,
            'colors_detected': 0,
            'unique_objects': set(),
            'unique_colors': set(),
            'unique_texts': set()
        }
        
        print("[INFO] Webcam Processor initialized")

    def _get_yolov8s_model(self):
        if self._yolo_v8_model is not None:
            return self._yolo_v8_model
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics is not available. Please install ultralytics.")

        # Load YOLOv8s for general object detection
        self._yolo_v8_model = YOLO("yolov8s.pt")
        try:
            self._yolo_v8_names = self._yolo_v8_model.names
        except Exception:
            self._yolo_v8_names = None
        return self._yolo_v8_model
    
    def list_available_cameras(self) -> List[Dict]:
        """List all available cameras"""
        cameras = []
        
        # Try camera indices 0-9
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                cameras.append({
                    'index': i,
                    'name': f"Camera {i}",
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'available': True
                })
                cap.release()
            else:
                cameras.append({
                    'index': i,
                    'name': f"Camera {i}",
                    'available': False
                })
        
        return cameras
    
    def start_camera(self, camera_index: int = 0, resolution: Tuple[int, int] = (640, 480), 
                    fps: int = 30) -> Dict:
        """
        Start webcam capture
        
        Args:
            camera_index: Camera device index
            resolution: Camera resolution (width, height)
            fps: Target FPS
            
        Returns:
            Status dictionary
        """
        try:
            if self.is_running:
                return {'error': 'Camera already running'}
            
            # Open camera
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                return {'error': f'Failed to open camera {camera_index}'}
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            
            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            print(f"[INFO] Camera started: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Initialize statistics
            self.stats['start_time'] = time.time()
            self.stats['frames_processed'] = 0
            self.stats['unique_plates'] = set()
            self.stats['processing_times'] = []
            
            # Start processing thread
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            return {
                'success': True,
                'camera_index': camera_index,
                'resolution': f"{actual_width}x{actual_height}",
                'fps': actual_fps,
                'message': 'Camera started successfully'
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to start camera: {e}")
            return {'error': str(e)}
    
    def stop_camera(self) -> Dict:
        """Stop webcam capture"""
        try:
            if not self.is_running:
                return {'error': 'Camera not running'}
            
            # Stop processing
            self.is_running = False
            
            # Wait for thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2)
            
            # Release camera
            if self.camera:
                self.camera.release()
                self.camera = None
            
            # Clear queues
            while not self.frame_queue.empty():
                self.frame_queue.get()
            
            while not self.result_queue.empty():
                self.result_queue.get()
            
            print("[INFO] Camera stopped")
            
            return {
                'success': True,
                'message': 'Camera stopped successfully',
                'final_stats': self.get_current_stats()
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to stop camera: {e}")
            return {'error': str(e)}
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from camera"""
        try:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    self.current_frame = frame
                    return frame
            return None
        except Exception as e:
            print(f"[ERROR] Failed to get frame: {e}")
            return None
    
    def get_processed_frame(self) -> Optional[np.ndarray]:
        """Get latest processed frame with annotations"""
        try:
            if not self.result_queue.empty():
                result = self.result_queue.get()
                self.latest_result = result
                return result.get('annotated_frame')
            return None
        except Exception as e:
            print(f"[ERROR] Failed to get processed frame: {e}")
            return None
    
    def get_latest_result(self) -> Optional[Dict]:
        """Get latest processing result"""
        return self.latest_result
    
    def _processing_loop(self):
        """Maximum speed GPU-only processing loop for real-time performance"""
        last_fps_time = time.time()
        frame_count = 0
        
        # Prefer GPU but allow CPU fallback
        using_gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                using_gpu = True
                print(f"[INFO] 🚀 MAXIMUM MODE (GPU): {torch.cuda.get_device_name(0)}")
                torch.cuda.empty_cache()
            else:
                print("[WARNING] CUDA not available - running webcam in CPU fallback mode")
        except Exception as e:
            print(f"[WARNING] GPU check failed - running webcam in CPU fallback mode: {e}")
            using_gpu = False
        
        while self.is_running:
            try:
                # Get frame immediately
                frame = self.get_frame()
                if frame is None:
                    time.sleep(0.001)  # Minimal sleep
                    continue
                
                # Process EVERY frame for real-time - no skipping
                start_time = time.time()
                result = self._process_frame_realtime_maximum_speed(frame)
                self._frame_idx += 1
                processing_time = time.time() - start_time
                
                # Very strict timeout on GPU; more lenient on CPU fallback
                if using_gpu and processing_time > 0.033:  # 30 FPS = 33ms per frame
                    continue
                
                # Minimal statistics update
                self.stats['frames_processed'] += 1
                self.stats['processing_times'].append(processing_time)
                
                # Keep only last 20 processing times
                if len(self.stats['processing_times']) > 20:
                    self.stats['processing_times'].pop(0)
                
                # Ultra-fast annotation
                annotated_frame = self._create_annotated_frame_maximum_speed(frame, result)
                result['annotated_frame'] = annotated_frame
                
                # Update FPS every second
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self.stats['fps'] = frame_count / (current_time - last_fps_time)
                    frame_count = 0
                    last_fps_time = current_time
                
                # Add to queue (drop old if full)
                if self.result_queue.full():
                    self.result_queue.get()
                
                self.result_queue.put(result)
                
            except Exception as e:
                print(f"[ERROR] Max speed processing error: {e}")
                time.sleep(0.001)
    
    def _process_frame_realtime_maximum_speed(self, frame: np.ndarray) -> Dict:
        """Maximum speed frame processing - GPU only with Enhanced K-means + ResNet-18 + Fallback color detection"""
        try:
            result = {
                'timestamp': datetime.now().isoformat(),
                'processing_mode': 'maximum_gpu_speed',
                'license_plates': [],
                'objects': [],
                'ocr_results': [],
                'colors': [],
                'enhanced_colors': [],
                'fallback_colors': [],
                'scene_analysis': {},
                'frame_info': {
                    'shape': frame.shape,
                    'gpu_only': True
                }
            }
            
            # Object detection
            if self.enable_general_objects:
                objects = self._detect_all_objects(frame)
                result['objects'] = objects
                
                # Quick stats update
                for obj in objects:
                    self.stats['unique_objects'].add(obj['class_name'])
                self.stats['objects_detected'] += len(objects)
                
                # Multi-level color detection with fallback
                if objects:
                    try:
                        # Process each object with fallback color detection
                        for obj in objects:
                            if 'bounding_box' in obj:
                                x1, y1, x2, y2 = obj['bounding_box']
                                
                                # Try Enhanced detection first
                                color_result = None
                                processing_method = 'unknown'
                                fallback_level = 0
                                
                                if self.enable_enhanced_colors and self.enhanced_detector:
                                    try:
                                        color_result = detect_colors_enhanced(frame, (x1, y1, x2, y2))
                                        if color_result.get('success'):
                                            processing_method = 'enhanced_resnet_kmeans'
                                            fallback_level = 1
                                            
                                            # Extract ResNet-18 features for this object
                                            object_features = None
                                            if self.enhanced_detector.enable_resnet:
                                                object_features = self.enhanced_detector.extract_resnet_features(frame[y1:y2, x1:x2])
                                            
                                            # Enhanced categorization
                                            categorized_obj = categorize_object_enhanced(
                                                obj['class_name'], color_result, obj['confidence'], object_features
                                            )
                                            
                                            obj.update({
                                                'enhanced_color_info': color_result,
                                                'enhanced_category': categorized_obj,
                                                'color_family': color_result.get('primary_color', {}).get('family', 'Unknown'),
                                                'color_shade': color_result.get('primary_color', {}).get('shade', 'Unknown'),
                                                'resnet_enhanced': color_result.get('resnet_features', {}).get('enhanced', False),
                                                'processing_method': processing_method,
                                                'fallback_level': fallback_level
                                            })
                                    except Exception as e:
                                        print(f"[DEBUG] Enhanced detection failed for {obj['class_name']}: {e}")
                                        color_result = None
                                
                                # Fallback to fallback detector if enhanced failed
                                if color_result is None and self.enable_fallback_colors and self.fallback_detector:
                                    try:
                                        color_result = detect_colors_fallback(frame, (x1, y1, x2, y2))
                                        if color_result.get('success'):
                                            processing_method = color_result.get('fallback_method', 'fallback')
                                            fallback_level = color_result.get('fallback_level', 4)
                                            
                                            obj.update({
                                                'fallback_color_info': color_result,
                                                'color_family': color_result.get('primary_color', {}).get('family', 'Unknown'),
                                                'color_shade': color_result.get('primary_color', {}).get('shade', 'Unknown'),
                                                'processing_method': processing_method,
                                                'fallback_level': fallback_level,
                                                'resnet_enhanced': False
                                            })
                                    except Exception as e:
                                        print(f"[DEBUG] Fallback detection failed for {obj['class_name']}: {e}")
                                        color_result = None
                                
                                # If all else failed, use simple color detection
                                if color_result is None:
                                    try:
                                        color_result = self._simple_color_detection(frame[y1:y2, x1:x2])
                                        processing_method = 'simple_hsv'
                                        fallback_level = 5
                                        
                                        obj.update({
                                            'simple_color_info': color_result,
                                            'color_family': color_result.get('family', 'Unknown'),
                                            'color_shade': color_result.get('shade', 'Unknown'),
                                            'processing_method': processing_method,
                                            'fallback_level': fallback_level,
                                            'resnet_enhanced': False
                                        })
                                    except Exception as e:
                                        print(f"[DEBUG] Simple detection failed for {obj['class_name']}: {e}")
                        
                        # Collect color results
                        enhanced_colors = [obj.get('enhanced_color_info', {}) for obj in objects if obj.get('enhanced_color_info')]
                        fallback_colors = [obj.get('fallback_color_info', {}) for obj in objects if obj.get('fallback_color_info')]
                        simple_colors = [obj.get('simple_color_info', {}) for obj in objects if obj.get('simple_color_info')]
                        
                        result['enhanced_colors'] = enhanced_colors
                        result['fallback_colors'] = fallback_colors
                        result['colors'] = enhanced_colors + fallback_colors + simple_colors
                        
                        # Update stats
                        self.stats['colors_detected'] += len([c for c in result['colors'] if c.get('success')])
                        
                        # Add processing summary
                        result['color_processing_summary'] = {
                            'total_objects': len(objects),
                            'enhanced_processed': len(enhanced_colors),
                            'fallback_processed': len(fallback_colors),
                            'simple_processed': len(simple_colors),
                            'success_rate': len([c for c in result['colors'] if c.get('success')]) / len(objects) if objects else 0
                        }
                        
                    except Exception as e:
                        print(f"[ERROR] Color processing failed: {e}")

            # OCR on all objects now (not just priority list), with specialized crops for some classes
            should_run_ocr = self.enable_ocr and objects
            if should_run_ocr:
                # Filter objects for OCR (skip people/animals/birds)
                objects_for_ocr = []
                for o in objects:
                    cn = str(o.get('class_name', '')).strip().lower()
                    if not cn:
                        continue
                    if cn in self._ocr_skip_classes:
                        continue
                    if (cn in self._ocr_allow_classes) or (cn in self._ocr_vehicle_classes):
                        objects_for_ocr.append(o)

                if not objects_for_ocr:
                    result['ocr_results'] = []
                    objects_for_ocr = []

                # Always run OCR for priority text objects regardless of frame skip
                has_priority = any(
                    str(obj.get('class_name', '')).strip().lower() in ('cup', 'bottle', 'book', 'license plate', 'cell phone', 'laptop')
                    for obj in objects_for_ocr[:4]
                )
                run_now = has_priority or (self._frame_idx % int(max(1, self.ocr_every_n)) == 0)
                if run_now:
                    try:
                        # Webcam previews are often mirrored; flip crops before OCR to read text correctly.
                        result['ocr_results'] = self._extract_text_for_objects(frame, objects_for_ocr, mirrored=True)
                        for item in result['ocr_results']:
                            text = (item.get('text') or '').strip()
                            if text:
                                self.stats['unique_texts'].add(text)
                    except Exception as e:
                        print(f"[DEBUG] OCR extraction failed: {e}")
                        result['ocr_results'] = []
                
                # Vehicle -> license plate detection + OCR (runs even if general OCR is throttled)
                try:
                    vehicle_classes = {"car", "truck", "bus", "motorcycle"}
                    vehicles = [o for o in objects if str(o.get('class_name', '')).strip().lower() in vehicle_classes]
                    if vehicles:
                        plates = self._detect_and_read_license_plates(frame, vehicles)
                        result['license_plates'] = plates
                        for p in plates:
                            t = (p.get('text') or '').strip()
                            if t:
                                self.stats['unique_plates'].add(t)
                except Exception as e:
                    print(f"[DEBUG] Vehicle plate pipeline failed: {e}")
                    result['license_plates'] = []
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Max speed frame processing failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'license_plates': [],
                'objects': [],
                'ocr_results': [],
                'colors': [],
                'enhanced_colors': [],
                'fallback_colors': []
            }

    def _extract_text_for_objects(self, frame: np.ndarray, objects: List[Dict], mirrored: bool = False) -> List[Dict]:
        try:
            h, w = frame.shape[:2]
            results = []

            def _clean_text(s: str) -> str:
                try:
                    s = (s or "")
                    s = s.upper()
                    s = re.sub(r"[^A-Z0-9]+", "", s)
                    return s
                except Exception:
                    return ""

            def _split_alpha_num(s: str) -> tuple[str, str]:
                try:
                    letters = re.sub(r"[^A-Z]+", "", s)
                    digits = re.sub(r"[^0-9]+", "", s)
                    return letters, digits
                except Exception:
                    return "", ""

            def _filter_noise_digits(cleaned: str, ocr_conf: float, class_name: str, is_plate_like: bool) -> str:
                """Prevent digit hallucination for brand-like text on non-plate objects."""
                try:
                    if not cleaned:
                        return ""
                    letters, digits = _split_alpha_num(cleaned)
                    has_letters = bool(letters)
                    has_digits = bool(digits)

                    # Always keep alphanumeric for plates/vehicles
                    if is_plate_like:
                        return cleaned

                    # If only letters => OK
                    if has_letters and (not has_digits):
                        return letters

                    # If only digits on non-plate objects: usually noise; suppress unless very confident
                    if has_digits and (not has_letters):
                        return cleaned if float(ocr_conf or 0.0) >= 0.80 else ""

                    # Mixed letters+digits: if confidence is low, drop digits and keep letters only
                    if has_letters and has_digits:
                        if float(ocr_conf or 0.0) < 0.65:
                            return letters
                        return cleaned

                    return ""
                except Exception:
                    return cleaned

            # OCR all provided objects (caller already bounds this list for real-time performance)
            for obj in objects:
                if 'bounding_box' not in obj:
                    continue
                x1, y1, x2, y2 = obj['bounding_box']
                x1 = int(max(0, min(w - 1, x1)))
                y1 = int(max(0, min(h - 1, y1)))
                x2 = int(max(0, min(w - 1, x2)))
                y2 = int(max(0, min(h - 1, y2)))
                if x2 <= x1 or y2 <= y1:
                    continue

                # Skip tiny crops (usually unreadable) - lowered threshold for cups/plates/person/vehicles
                class_name = str(obj.get('class_name', '')).strip().lower()
                is_priority = class_name in ('cup', 'bottle', 'book', 'license plate', 'person', 'car', 'truck', 'bus', 'motorcycle')

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                if mirrored:
                    try:
                        crop = cv2.flip(crop, 1)
                    except Exception:
                        pass

                # Expand crop slightly for text-heavy objects so printed labels aren't clipped
                if is_priority and class_name in ('cup', 'bottle', 'book', 'person'):
                    try:
                        pad_x = int(max(2, (x2 - x1) * 0.08))
                        pad_y = int(max(2, (y2 - y1) * 0.08))
                        ex1 = int(max(0, x1 - pad_x))
                        ey1 = int(max(0, y1 - pad_y))
                        ex2 = int(min(w, x2 + pad_x))
                        ey2 = int(min(h, y2 + pad_y))
                        if ex2 > ex1 and ey2 > ey1:
                            crop = frame[ey1:ey2, ex1:ex2]
                            x1, y1, x2, y2 = ex1, ey1, ex2, ey2
                            if mirrored:
                                try:
                                    crop = cv2.flip(crop, 1)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                min_crop_w = 30 if is_priority else 80
                min_crop_h = 20 if is_priority else 60
                conf_thresh = 0.2 if is_priority else 0.4
                cache_ttl = 0.5 if is_priority else 5.0
                
                if is_priority:
                    print(f"[DEBUG] Priority OCR for {class_name}: bbox={obj['bounding_box']}, size={x2-x1}x{y2-y1}")
                
                if (x2 - x1) < min_crop_w or (y2 - y1) < min_crop_h:
                    continue

                # Cache key is coarse to avoid flicker
                key = f"{class_name}:{x1//20}:{y1//20}:{x2//20}:{y2//20}"
                cached = self._ocr_cache.get(key)
                if cached and isinstance(cached, dict):
                    cached_ttl = float(cached.get('ttl', cache_ttl))
                    cached_text = ((cached.get('data') or {}).get('text') or '').strip()
                    # Don't hold onto empty text for long; allow quick retries
                    if not cached_text:
                        cached_ttl = min(cached_ttl, 0.25 if is_priority else 0.75)
                    if (time.time() - cached.get('ts', 0)) < cached_ttl:
                        results.append(cached['data'])
                        continue

                text = ''
                text_conf = 0.0

                # If car, try quick plate region extraction first (if available)
                if class_name == 'car':
                    try:
                        plates = extract_license_plates_simple(crop)
                        if plates:
                            # plates could be list of crops or regions; handle both
                            plate_crop = None
                            first = plates[0]
                            if isinstance(first, np.ndarray):
                                plate_crop = first
                            elif isinstance(first, (tuple, list)) and len(first) == 4:
                                px1, py1, px2, py2 = first
                                plate_crop = crop[int(py1):int(py2), int(px1):int(px2)]
                            if plate_crop is not None and plate_crop.size > 0:
                                t, c = self._run_ocr_on_crop_with_conf(plate_crop, confidence_threshold=0.3, is_plate=True)
                                text = t
                                text_conf = c
                    except Exception:
                        pass

                # General text for any object with lower threshold for priority objects
                if not text:
                    # More permissive threshold for priority objects to avoid missing faint/blurred print
                    thresh = 0.12 if class_name in ('cup', 'bottle', 'book', 'license plate', 'person', 'car', 'truck', 'bus', 'motorcycle') else 0.4

                    best_text = ''
                    best_len = 0

                    # For cups/bottles, focus OCR on likely label region as well as full crop
                    # For person, try torso region (likely T-shirt/shirt)
                    # For vehicles, try lower third (likely license plate)
                    crops_to_try = [crop]
                    if class_name in ('cup', 'bottle'):
                        ch, cw = crop.shape[:2]
                        y_start = int(max(0, ch * 0.18))
                        y_end = int(min(ch, ch * 0.88))
                        x_start = int(max(0, cw * 0.08))
                        x_end = int(min(cw, cw * 0.92))
                        if y_end > y_start and x_end > x_start:
                            label_crop = crop[y_start:y_end, x_start:x_end]
                            if label_crop.size:
                                crops_to_try.insert(0, label_crop)
                    elif class_name == 'person':
                        # Try torso region (roughly upper 30% to 70% of height)
                        ch, cw = crop.shape[:2]
                        y_start = int(max(0, ch * 0.30))
                        y_end = int(min(ch, ch * 0.75))
                        x_start = int(max(0, cw * 0.10))
                        x_end = int(min(cw, cw * 0.90))
                        if y_end > y_start and x_end > x_start:
                            torso_crop = crop[y_start:y_end, x_start:x_end]
                            if torso_crop.size:
                                crops_to_try.insert(0, torso_crop)
                    elif class_name in ('car', 'truck', 'bus', 'motorcycle'):
                        # Try lower third region (likely license plate)
                        ch, cw = crop.shape[:2]
                        y_start = int(max(0, ch * 0.65))  # lower 35%
                        y_end = ch
                        x_start = int(max(0, cw * 0.05))
                        x_end = int(min(cw, cw * 0.95))
                        if y_end > y_start and x_end > x_start:
                            plate_crop = crop[y_start:y_end, x_start:x_end]
                            if plate_crop.size:
                                crops_to_try.insert(0, plate_crop)

                    for c in crops_to_try:
                        t, cconf = self._run_ocr_on_crop_with_conf(
                            c,
                            confidence_threshold=thresh,
                            is_plate=(class_name in ('license plate', 'car', 'truck', 'bus', 'motorcycle')),
                        )
                        t = (t or '').strip()
                        if t and len(t) > best_len:
                            best_text = t
                            best_len = len(t)
                            text_conf = float(cconf or 0.0)
                            if best_len >= 10:
                                break

                    # If still empty for priority objects, do a more permissive pass
                    if (not best_text) and class_name in ('cup', 'bottle', 'book', 'license plate', 'person', 'car', 'truck', 'bus', 'motorcycle'):
                        for c in crops_to_try:
                            t, cconf = self._run_ocr_on_crop_with_conf(
                                c,
                                confidence_threshold=0.15,
                                is_plate=(class_name in ('license plate', 'car', 'truck', 'bus', 'motorcycle')),
                            )
                            t = (t or '').strip()
                            if t and len(t) > best_len:
                                best_text = t
                                best_len = len(t)
                                text_conf = float(cconf or 0.0)
                                if best_len >= 8:
                                    break

                    text = best_text

                cleaned = _clean_text(text)
                plate_like = (class_name in ('license plate', 'car', 'truck', 'bus', 'motorcycle'))
                cleaned = _filter_noise_digits(cleaned, float(text_conf or 0.0), class_name, plate_like)
                # Suppress meaningless very short results
                if cleaned and len(cleaned) < 3:
                    letters, digits = _split_alpha_num(cleaned)
                    if len(letters) < 3 and (not plate_like):
                        cleaned = ""
                text = cleaned

                data = {
                    'object_id': obj.get('object_id'),
                    'class_name': obj.get('class_name'),
                    'bounding_box': (x1, y1, x2, y2),
                    'text': text,
                    'confidence': float(text_conf or 0.0),
                }
                # Cache, but empty text should expire quickly so we keep retrying as camera moves
                entry_ttl = cache_ttl
                if not (text or '').strip():
                    entry_ttl = 0.25 if is_priority else 0.75
                self._ocr_cache[key] = {'ts': time.time(), 'ttl': float(entry_ttl), 'data': data}
                results.append(data)

            # Limit cache growth
            if len(self._ocr_cache) > 200:
                for k in list(self._ocr_cache.keys())[:100]:
                    self._ocr_cache.pop(k, None)

            return results
        except Exception as e:
            print(f"[DEBUG] _extract_text_for_objects failed: {e}")
            return []

    def _extract_colors_for_objects(self, frame: np.ndarray, objects: List[Dict]) -> List[Dict]:
        """Extract color information for objects and return in JSON-friendly format"""
        try:
            print(f"[DEBUG] Color extraction called for {len(objects)} objects")
            colors = []
            h, w = frame.shape[:2]
            
            for obj in objects:
                print(f"[DEBUG] Processing object: {obj}")
                if 'bounding_box' not in obj:
                    print(f"[DEBUG] No bounding box for object {obj}")
                    continue
                    
                x1, y1, x2, y2 = obj['bounding_box']
                print(f"[DEBUG] Bounding box: {x1}, {y1}, {x2}, {y2}")
                x1 = int(max(0, min(w - 1, x1)))
                y1 = int(max(0, min(h - 1, y1)))
                x2 = int(max(0, min(w - 1, x2)))
                y2 = int(max(0, min(h - 1, y2)))
                
                if x2 <= x1 or y2 <= y1:
                    print(f"[DEBUG] Invalid bounding box after clipping: {x1}, {y1}, {x2}, {y2}")
                    continue
                
                color_result = None
                processing_method = 'unknown'
                fallback_level = 0
                
                # Try Enhanced detection first
                if self.enable_enhanced_colors and self.enhanced_detector:
                    try:
                        print(f"[DEBUG] Trying enhanced color detection for {obj.get('class_name')}")
                        color_result = detect_colors_enhanced(frame, (x1, y1, x2, y2))
                        print(f"[DEBUG] Enhanced result: {color_result}")
                        if color_result.get('success'):
                            processing_method = 'enhanced_resnet_kmeans'
                            fallback_level = 1
                    except Exception as e:
                        print(f"[DEBUG] Enhanced color extraction failed: {e}")
                        color_result = None
                
                # Fallback to fallback detector if enhanced failed
                if color_result is None and self.enable_fallback_colors and self.fallback_detector:
                    try:
                        print(f"[DEBUG] Trying fallback color detection for {obj.get('class_name')}")
                        color_result = detect_colors_fallback(frame, (x1, y1, x2, y2))
                        print(f"[DEBUG] Fallback result: {color_result}")
                        if color_result.get('success'):
                            processing_method = color_result.get('fallback_method', 'fallback')
                            fallback_level = color_result.get('fallback_level', 4)
                    except Exception as e:
                        print(f"[DEBUG] Fallback color extraction failed: {e}")
                        color_result = None
                
                # If all else failed, use simple color detection
                if color_result is None:
                    try:
                        print(f"[DEBUG] Trying simple color detection for {obj.get('class_name')}")
                        color_result = self._simple_color_detection(frame[y1:y2, x1:x2])
                        print(f"[DEBUG] Simple result: {color_result}")
                        processing_method = 'simple_hsv'
                        fallback_level = 5
                    except Exception as e:
                        print(f"[DEBUG] Simple color extraction failed: {e}")
                        color_result = None
                
                # Create color entry for JSON
                if color_result:
                    # Handle different color result formats
                    color_entry = None
                    
                    if color_result.get('success'):
                        # Enhanced/fallback format
                        color_entry = {
                            'object_id': obj.get('object_id'),
                            'class_name': obj.get('class_name'),
                            'bounding_box': (x1, y1, x2, y2),
                            'color_family': color_result.get('primary_color', {}).get('family', 'Unknown'),
                            'color_shade': color_result.get('primary_color', {}).get('shade', 'Unknown'),
                            'processing_method': processing_method,
                            'fallback_level': fallback_level,
                            'confidence': color_result.get('primary_color', {}).get('confidence', 0.0)
                        }
                    elif 'final_color' in color_result:
                        # Simple detection format
                        color_entry = {
                            'object_id': obj.get('object_id'),
                            'class_name': obj.get('class_name'),
                            'bounding_box': (x1, y1, x2, y2),
                            'color_family': color_result.get('final_color', 'Unknown'),
                            'color_shade': 'medium',  # Simple detection doesn't provide shade
                            'processing_method': processing_method,
                            'fallback_level': fallback_level,
                            'confidence': color_result.get('confidence', 0.0)
                        }
                    
                    if color_entry:
                        colors.append(color_entry)
                        print(f"[DEBUG] Color found for {obj.get('class_name')}: {color_entry['color_family']} {color_entry['color_shade']} (conf: {color_entry['confidence']:.2f})")
            
            print(f"[DEBUG] Color extraction completed: {len(colors)} colors extracted")
            return colors
            
        except Exception as e:
            print(f"[ERROR] Color extraction failed: {e}")
            return []

    def _run_ocr_on_crop_with_conf(self, crop: np.ndarray, confidence_threshold: float = 0.4, is_plate: bool = False) -> Tuple[str, float]:
        try:
            if crop is None or crop.size == 0:
                return '', 0.0

            # Throttled debug info (avoid spamming console)
            try:
                now = time.time()
                if (now - float(self._last_ocr_debug_ts or 0.0)) > 2.0:
                    self._last_ocr_debug_ts = now
                    if isinstance(crop, np.ndarray):
                        print(f"[DEBUG] OCR input shape={getattr(crop, 'shape', None)} dtype={getattr(crop, 'dtype', None)}")
            except Exception:
                pass

            # One-time check: pytesseract may be installed but the Tesseract engine might be missing.
            if (not self._tesseract_checked) and TESSERACT_AVAILABLE and pytesseract is not None:
                self._tesseract_checked = True
                try:
                    _ = pytesseract.get_tesseract_version()
                    self._tesseract_ok = True
                except Exception:
                    self._tesseract_ok = False
                    print("[WARN] Tesseract engine not found/configured. Install Tesseract-OCR and add it to PATH for pytesseract fallback.")

            # Preprocess for better OCR: resize if too small, enhance contrast, try multiple orientations
            processed_crops = self._preprocess_for_ocr(crop, is_plate=is_plate)

            best_text = ''
            best_conf = 0.0

            for idx, (proc_crop, angle) in enumerate(processed_crops):
                # Prefer optimized PaddleOCR GPU
                if OPTIMIZED_PADDLEOCR_AVAILABLE and extract_text_optimized is not None:
                    try:
                        res = extract_text_optimized(
                            proc_crop,
                            confidence_threshold=float(confidence_threshold),
                            lang='en',
                            use_gpu=None,
                            use_cache=False,  # disable cache for multi-angle attempts
                            preprocess=False, # we already preprocessed
                        )
                        text = (res.get('text') or '').strip()
                        conf = res.get('confidence', 0.0)
                        if text:
                            print(f"[DEBUG] OCR variant {idx} (angle={angle}): '{text}' conf={conf:.2f}")
                        if text and conf > best_conf:
                            best_text = text
                            best_conf = conf
                            print(f"[DEBUG] New best: '{best_text}' conf={best_conf:.2f} from angle={angle}")
                            if conf > 0.85:  # early exit if very confident
                                break
                    except Exception as e:
                        print(f"[DEBUG] Optimized PaddleOCR failed: {e}")
                
                # Fallback to EasyOCR if PaddleOCR fails
                if not best_text and idx == 0:  # only try on first variant to avoid spam
                    try:
                        import easyocr
                        reader = easyocr.Reader(['en'], gpu=False)
                        results = reader.readtext(proc_crop)
                        for (bbox, txt, conf) in results:
                            txt = txt.strip()
                            if txt and conf > confidence_threshold:
                                text = txt
                                conf_score = float(conf)
                                print(f"[DEBUG] EasyOCR found: '{text}' conf={conf_score:.2f}")
                                if text and conf_score > best_conf:
                                    best_text = text
                                    best_conf = conf_score
                                    print(f"[DEBUG] EasyOCR new best: '{best_text}' conf={best_conf:.2f}")
                    except Exception as e:
                        print(f"[DEBUG] EasyOCR fallback failed: {e}")
                else:
                    # Legacy PaddleOCR fallback
                    if LEGACY_PADDLEOCR_AVAILABLE and extract_text_with_paddleocr is not None:
                        text = extract_text_with_paddleocr(
                            proc_crop,
                            confidence_threshold=float(confidence_threshold),
                            lang='en',
                            use_gpu=True,
                        )
                        if text and text.strip():
                            text = text.strip()
                            # Estimate confidence by length and format (simple heuristic)
                            est_conf = 0.7
                            if is_plate and len(text) >= 6 and any(c.isdigit() for c in text):
                                est_conf = 0.8
                            if est_conf > best_conf:
                                best_text = text
                                best_conf = est_conf

            # Last resort fallback: pytesseract (works even if PaddleOCR is broken)
            if (not best_text) and self._tesseract_ok and TESSERACT_AVAILABLE and pytesseract is not None and processed_crops:
                try:
                    img = processed_crops[0][0]
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.GaussianBlur(gray, (3, 3), 0)
                    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    psm = "7" if is_plate else "6"
                    config = f"--oem 3 --psm {psm}"
                    t = pytesseract.image_to_string(th, config=config)
                    t = (t or "").strip()
                    t = " ".join(t.split())
                    if t:
                        best_text = t
                except Exception:
                    pass

            return best_text, float(best_conf or 0.0)
        except Exception as e:
            print(f"[DEBUG] _run_ocr_on_crop failed: {e}")
            return '', 0.0

    def _run_ocr_on_crop(self, crop: np.ndarray, confidence_threshold: float = 0.4, is_plate: bool = False) -> str:
        """Backward-compatible wrapper returning only text."""
        t, _c = self._run_ocr_on_crop_with_conf(crop, confidence_threshold=confidence_threshold, is_plate=is_plate)
        return t

    def _preprocess_for_ocr(self, crop: np.ndarray, is_plate: bool = False) -> List[Tuple[np.ndarray, float]]:
        """Generate multiple preprocessed versions of the crop for OCR, with rotation attempts."""
        processed = []
        try:
            # Ensure BGR 3-channel input
            if crop is None or (not isinstance(crop, np.ndarray)) or crop.size == 0:
                return [(crop, 0.0)]
            if len(crop.shape) == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            elif len(crop.shape) == 3 and crop.shape[2] == 4:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)

            h, w = crop.shape[:2]

            # 1) Basic resize if too small (target ~300-400 px width for OCR)
            target_w = 350 if not is_plate else 400
            if w < target_w:
                scale = target_w / w
                new_h = int(h * scale)
                crop = cv2.resize(crop, (target_w, new_h), interpolation=cv2.INTER_CUBIC)

            # Always include original crop first (safe fallback)
            processed.append((crop, 0.0))

            # 2) Blur detection + conditional enhancement (real-time safe)
            gray0 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            try:
                blur_score = float(cv2.Laplacian(gray0, cv2.CV_64F).var())
            except Exception:
                blur_score = 0.0

            # Threshold tuned for resized crops (~350-400px wide). Lower => more blurry.
            blur_thresh = 70.0 if not is_plate else 90.0
            is_blurry = blur_score > 0.0 and blur_score < blur_thresh

            base_bgr = crop
            base_gray = gray0

            if is_blurry:
                # Contrast enhancement (CLAHE)
                clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
                g = clahe.apply(gray0)

                # Denoise (fast)
                try:
                    g = cv2.fastNlMeansDenoising(g, None, h=10, templateWindowSize=7, searchWindowSize=21)
                except Exception:
                    pass

                # Unsharp mask (gentle sharpening for blur)
                try:
                    blur = cv2.GaussianBlur(g, (0, 0), 1.0)
                    sharp = cv2.addWeighted(g, 1.35, blur, -0.35, 0)
                    g = sharp
                except Exception:
                    pass

                base_gray = g
                base_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)

                # Add enhanced version as an additional candidate
                processed.append((base_bgr, 0.0))

                # Throttled blur debug log
                try:
                    now = time.time()
                    if (now - float(self._last_ocr_debug_ts or 0.0)) > 2.0:
                        print(f"[DEBUG] Blur detected score={blur_score:.1f} thresh={blur_thresh:.1f} crop={crop.shape}")
                except Exception:
                    pass

            # Use enhanced (if created) for augmentations; otherwise original
            aug_base = base_bgr if is_blurry else crop

            # 3) Try flips and rotations for text on cups or skewed plates
            # Horizontal flip for mirrored text (common in webcam/camera)
            flipped_h = cv2.flip(aug_base, 1)
            processed.append((flipped_h, 0.0))

            # Vertical flip
            flipped_v = cv2.flip(aug_base, 0)
            processed.append((flipped_v, 0.0))

            # Both flips (equivalent to 180 rotation)
            flipped_hv = cv2.flip(aug_base, -1)
            processed.append((flipped_hv, 0.0))

            # Rotations
            angles = [90, 180, 270] if not is_plate else [90, 270]
            for angle in angles:
                rotated = cv2.rotate(aug_base, cv2.ROTATE_90_CLOCKWISE if angle == 90 else (cv2.ROTATE_180 if angle == 180 else cv2.ROTATE_90_COUNTERCLOCKWISE))
                processed.append((rotated, float(angle)))

            # Also try flips of rotated versions for cups
            if not is_plate:
                for angle in [90, 270]:
                    rotated = cv2.rotate(aug_base, cv2.ROTATE_90_CLOCKWISE if angle == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
                    flipped_rot = cv2.flip(rotated, 1)  # horizontal flip of rotated
                    processed.append((flipped_rot, float(angle + 1000)))  # use angle+1000 to distinguish

            # 4) Enhanced license plate preprocessing
            if is_plate:
                # Multiple preprocessing approaches for license plates
                
                # 4a) Adaptive threshold with different parameters
                adaptive_thresh1 = cv2.adaptiveThreshold(base_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                adaptive_thresh1_bgr = cv2.cvtColor(adaptive_thresh1, cv2.COLOR_GRAY2BGR)
                processed.append((adaptive_thresh1_bgr, 0.0))
                
                adaptive_thresh2 = cv2.adaptiveThreshold(base_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
                adaptive_thresh2_bgr = cv2.cvtColor(adaptive_thresh2, cv2.COLOR_GRAY2BGR)
                processed.append((adaptive_thresh2_bgr, 0.0))
                
                # 4b) Otsu thresholding with different preprocessing
                blur_otsu = cv2.GaussianBlur(base_gray, (5, 5), 0)
                _, otsu1 = cv2.threshold(blur_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                otsu1_bgr = cv2.cvtColor(otsu1, cv2.COLOR_GRAY2BGR)
                processed.append((otsu1_bgr, 0.0))
                
                # 4c) Morphological operations for text enhancement
                kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
                
                # Remove small noise
                cleaned1 = cv2.morphologyEx(base_gray, cv2.MORPH_OPEN, kernel_small)
                cleaned1_bgr = cv2.cvtColor(cleaned1, cv2.COLOR_GRAY2BGR)
                processed.append((cleaned1_bgr, 0.0))
                
                # Connect text components
                cleaned2 = cv2.morphologyEx(base_gray, cv2.MORPH_CLOSE, kernel_medium)
                cleaned2_bgr = cv2.cvtColor(cleaned2, cv2.COLOR_GRAY2BGR)
                processed.append((cleaned2_bgr, 0.0))
                
                # 4d) Contrast stretching
                contrast_stretched = cv2.normalize(base_gray, None, 0, 255, cv2.NORM_MINMAX)
                contrast_stretched_bgr = cv2.cvtColor(contrast_stretched, cv2.COLOR_GRAY2BGR)
                processed.append((contrast_stretched_bgr, 0.0))
                
                # 4e) Local contrast enhancement using CLAHE with different parameters
                clahe_aggressive = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
                clahe_result = clahe_aggressive.apply(base_gray)
                clahe_result_bgr = cv2.cvtColor(clahe_result, cv2.COLOR_GRAY2BGR)
                processed.append((clahe_result_bgr, 0.0))
                
                # 4f) Edge enhancement for better text boundaries
                edges = cv2.Canny(base_gray, 100, 200)
                edges_dilated = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
                edges_enhanced = cv2.addWeighted(base_gray, 0.8, edges_dilated, 0.2, 0)
                edges_enhanced_bgr = cv2.cvtColor(edges_enhanced, cv2.COLOR_GRAY2BGR)
                processed.append((edges_enhanced_bgr, 0.0))
                
                # 4g) Gamma correction for different lighting conditions
                gamma_corrected = np.array(255 * (base_gray / 255) ** 0.7, dtype='uint8')
                gamma_corrected_bgr = cv2.cvtColor(gamma_corrected, cv2.COLOR_GRAY2BGR)
                processed.append((gamma_corrected_bgr, 0.0))
                
                # 4h) Original binarize method (kept for compatibility)
                _, binary = cv2.threshold(base_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                deskewed_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
                processed.append((deskewed_bgr, 0.0))

        except Exception as e:
            print(f"[DEBUG] _preprocess_for_ocr failed: {e}")
            # Fallback: return original crop
            processed = [(crop, 0.0)] if crop is not None else [(crop, 0.0)]

        return processed
    
    def _detect_all_objects(self, frame: np.ndarray) -> List[Dict]:
        """YOLOv8s object detection for webcam (conf > 0.5)"""
        try:
            model = self._get_yolov8s_model()

            height, width = frame.shape[:2]
            device = 0 if (TORCH_AVAILABLE and torch is not None and torch.cuda.is_available()) else "cpu"

            # Mild downsampling for smoother performance while keeping accuracy reasonable
            max_size = 640
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                small_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            else:
                small_frame = frame
                new_width, new_height = width, height

            import torch
            with torch.no_grad():
                detection_results = model.predict(
                    source=small_frame,
                    conf=0.5,
                    iou=0.5,
                    max_det=50,
                    device=device,
                    verbose=False,
                    half=True if device != "cpu" else False,
                )
            
            objects = []
            
            if detection_results and len(detection_results) > 0:
                detection = detection_results[0]
                
                if hasattr(detection, 'boxes') and detection.boxes is not None:
                    boxes = detection.boxes
                    # Keep everything on GPU as long as possible
                    xyxy = boxes.xyxy
                    conf = boxes.conf
                    cls = boxes.cls
                    names = detection.names
                    
                    # Move to CPU only at the end
                    xyxy = xyxy.cpu().numpy()
                    conf = conf.cpu().numpy()
                    cls = cls.cpu().numpy()
                    
                    # Scale coordinates back to original frame size
                    scale_x = width / new_width
                    scale_y = height / new_height
                    
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        confidence = float(conf[i])
                        class_id = int(cls[i])
                        class_name = names.get(class_id, f"class_{class_id}")
                        
                        # Scale coordinates back
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        if confidence >= 0.5:
                            cname = str(class_name).strip().lower()
                            category = "Vehicle" if cname in ("car", "truck", "bus", "motorcycle", "bicycle") else "Object"
                            if cname == "person":
                                category = "Person"
                            box_color = (0, 255, 0) if category == "Vehicle" else (255, 255, 0)

                            objects.append({
                                'object_id': f"{class_name}_{i}",
                                'class_name': class_name,
                                'display_name': str(class_name),
                                'category': category,
                                'confidence': confidence,
                                'bounding_box': (x1, y1, x2, y2),
                                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                                'size': (x2 - x1, y2 - y1),
                                'color': box_color
                            })
            
            return objects
            
        except Exception as e:
            print(f"[ERROR] Ultra-fast GPU detection failed: {e}")
            return []

    def _detect_license_plate_regions(self, vehicle_crop: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            if vehicle_crop is None or vehicle_crop.size == 0:
                return []

            h, w = vehicle_crop.shape[:2]
            
            # Try multiple detection strategies
            all_candidates = []
            
            # Strategy 1: Enhanced edge-based detection with position weighting
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)  # Increased filter size
            
            # Try multiple Canny thresholds for better edge detection
            edges_list = []
            for low_thresh in [30, 50, 70]:
                edges = cv2.Canny(gray, low_thresh, low_thresh * 3)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
                edges = cv2.dilate(edges, kernel, iterations=1)
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
                edges_list.append(edges)
            
            # Combine all edge detections
            combined_edges = np.zeros_like(edges_list[0])
            for edges in edges_list:
                combined_edges = cv2.bitwise_or(combined_edges, edges)
            
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                if cw <= 0 or ch <= 0:
                    continue
                    
                area = cw * ch
                min_area = (w * h) * 0.002  # Reduced minimum area requirement
                if area < min_area:
                    continue
                
                # Relaxed aspect ratio constraints (1.5 to 8.0)
                ar = cw / float(ch)
                if ar < 1.5 or ar > 8.0:
                    continue
                
                # IMPROVED: Position-based scoring to prefer lower regions (license plates are usually low)
                position_score = 1.0
                y_center = y + ch / 2
                y_ratio = y_center / h
                
                # Strongly prefer lower 60% of the vehicle
                if y_ratio < 0.4:  # Top 40% - unlikely for license plates
                    position_score = 0.3
                elif y_ratio < 0.6:  # 40-60% - less likely
                    position_score = 0.6
                elif y_ratio >= 0.6:  # Bottom 40% - most likely
                    position_score = 1.2
                
                # Penalize regions that are too high up (likely grills/emblems)
                if y < int(0.25 * h):  # Top quarter - very unlikely
                    continue
                
                # Add padding around the detected region
                padding_x = int(cw * 0.1)
                padding_y = int(ch * 0.1)
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(w, x + cw + padding_x)
                y2 = min(h, y + ch + padding_y)
                
                # Apply position score to area for ranking
                weighted_area = area * position_score
                all_candidates.append((x1, y1, x2, y2, weighted_area))
            
            # Strategy 2: MSER detection for text-like regions with position filtering
            try:
                mser = cv2.MSER_create()
                mser.set_delta(5)
                mser.set_min_area(int((w * h) * 0.001))
                mser.set_max_area(int((w * h) * 0.3))
                
                regions, _ = mser.detectRegions(gray)
                for region in regions:
                    x, y, cw, ch = cv2.boundingRect(region)
                    
                    # Skip regions in top half (more likely to be grills/emblems)
                    y_center = y + ch / 2
                    if y_center / h < 0.5:
                        continue
                    
                    area = cw * ch
                    if area < (w * h) * 0.001:
                        continue
                    
                    ar = cw / float(ch)
                    if ar < 1.0 or ar > 10.0:
                        continue
                    
                    # Add padding
                    padding_x = int(cw * 0.15)
                    padding_y = int(ch * 0.15)
                    x1 = max(0, x - padding_x)
                    y1 = max(0, y - padding_y)
                    x2 = min(w, x + cw + padding_x)
                    y2 = min(h, y + ch + padding_y)
                    
                    # Higher score for lower regions
                    position_score = 1.2 if (y_center / h) >= 0.6 else 0.8
                    weighted_area = area * position_score
                    all_candidates.append((x1, y1, x2, y2, weighted_area))
            except Exception:
                pass  # MSER not available, continue with edge detection
            
            # Strategy 3: Focused lower region search (license plates are typically in lower half)
            lower_region_candidates = []
            
            # Focus on the lower 50% of the vehicle where license plates are most common
            for y_start_ratio in [0.5, 0.6, 0.7, 0.8]:
                y_start = int(h * y_start_ratio)
                y_end = h
                
                # Split lower region into horizontal strips
                strip_height = int((y_end - y_start) / 3)
                for i in range(3):
                    strip_y1 = y_start + (i * strip_height)
                    strip_y2 = min(y_end, strip_y1 + strip_height)
                    
                    if strip_y2 > strip_y1:
                        # Lower strips get higher scores
                        strip_score = 1.0 + (i * 0.2)  # Lower strips = higher score
                        weighted_area = (w * (strip_y2 - strip_y1)) * 0.5 * strip_score
                        lower_region_candidates.append((0, strip_y1, w, strip_y2, weighted_area))
            
            all_candidates.extend(lower_region_candidates)
            
            # Remove duplicates and sort by weighted area
            unique_candidates = []
            seen = set()
            
            for x1, y1, x2, y2, area in all_candidates:
                # Create a normalized key for deduplication
                key = (x1//10, y1//10, x2//10, y2//10)
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append((x1, y1, x2, y2, area))
            
            # Sort by weighted area (largest first) and return top candidates
            unique_candidates.sort(key=lambda t: t[4], reverse=True)
            return [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in unique_candidates[:4]]  # Return up to 4 candidates

        except Exception as e:
            print(f"[DEBUG] License plate region detection failed: {e}")
            return []

    def _normalize_plate_text(self, text: str) -> str:
        """Enhanced license plate text normalization that preserves spaces and improves accuracy."""
        if not text:
            return ""
        
        # Convert to uppercase and strip whitespace
        text = text.upper().strip()
        
        # Remove common OCR artifacts but preserve spaces
        # Remove special characters except spaces
        cleaned = re.sub(r'[^\w\s]', '', text)
        
        # Fix common OCR confusions for license plates
        replacements = {
            '0': 'O',  # Replace 0 with O (common in plates like CZ17)
            '1': 'I',  # Replace 1 with I
            '8': 'B',  # Replace 8 with B
            '5': 'S',  # Replace 5 with S
            '2': 'Z',  # Replace 2 with Z
        }
        
        # Only replace if it makes sense in context (for letters)
        for old, new in replacements.items():
            # Be more conservative - only replace if it's likely a letter position
            if old in cleaned:
                # Check if this character is likely in a letter position
                # This is a heuristic - in practice, we might need more sophisticated logic
                pass  # Skip automatic replacement for now
        
        # Normalize multiple spaces to single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove leading/trailing spaces again
        cleaned = cleaned.strip()
        
        # Ensure we have a reasonable length for license plates
        if len(cleaned) < 4:
            return ""
        
        return cleaned

    def _is_recent_plate(self, plate_text: str) -> bool:
        now = time.time()
        for t, ts in list(self._plate_history):
            if t == plate_text and (now - ts) < self._plate_seen_ttl_s:
                return True
        return False

    def _mark_plate_seen(self, plate_text: str):
        self._plate_history.append((plate_text, time.time()))

    def _detect_and_read_license_plates(self, frame: np.ndarray, vehicles: List[Dict]) -> List[Dict]:
        try:
            h, w = frame.shape[:2]
            plates_out: List[Dict] = []

            for v in vehicles[:3]:
                if 'bounding_box' not in v:
                    continue
                x1, y1, x2, y2 = v['bounding_box']
                x1 = int(max(0, min(w - 1, x1)))
                y1 = int(max(0, min(h - 1, y1)))
                x2 = int(max(0, min(w - 1, x2)))
                y2 = int(max(0, min(h - 1, y2)))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                regions = self._detect_license_plate_regions(crop)
                
                # Try each detected region with different OCR approaches
                best_plate_text = ""
                best_confidence = 0.0
                best_bbox = None
                
                for (px1, py1, px2, py2) in regions:
                    plate_crop = crop[int(py1):int(py2), int(px1):int(px2)]
                    if plate_crop is None or plate_crop.size == 0:
                        continue

                    # Try multiple OCR approaches with different confidence thresholds
                    ocr_attempts = [
                        (0.15, "low"),    # Very permissive
                        (0.25, "medium"), # Original threshold
                        (0.35, "high"),   # Strict
                    ]
                    
                    for conf_thresh, desc in ocr_attempts:
                        raw = self._run_ocr_on_crop(plate_crop, confidence_threshold=conf_thresh, is_plate=True)
                        norm = self._normalize_plate_text(raw)
                        
                        # More lenient length check - allow 4+ characters
                        if len(norm.replace(" ", "")) < 4:
                            continue
                        
                        # Validate that this looks like a license plate
                        if self._is_valid_license_plate_text(norm):
                            # Use a simple confidence estimation based on text characteristics
                            estimated_conf = self._estimate_plate_confidence(norm, raw)
                            
                            if estimated_conf > best_confidence and len(norm) > len(best_plate_text):
                                best_plate_text = norm
                                best_confidence = estimated_conf
                                best_bbox = (x1 + int(px1), y1 + int(py1), x1 + int(px2), y1 + int(py2))
                                
                                print(f"[DEBUG] Better plate found: '{norm}' (conf: {estimated_conf:.2f}, method: {desc})")

                # If we found a good plate, add it to results
                if best_plate_text and not self._is_recent_plate(best_plate_text):
                    self._mark_plate_seen(best_plate_text)
                    plates_out.append({
                        'text': best_plate_text,
                        'confidence': best_confidence,
                        'vehicle_object_id': v.get('object_id'),
                        'vehicle_class': v.get('class_name'),
                        'bounding_box': best_bbox,
                    })

            return plates_out
        except Exception as e:
            print(f"[DEBUG] License plate detection failed: {e}")
            return []
    
    def _is_valid_license_plate_text(self, text: str) -> bool:
        """Check if text looks like a valid license plate with stricter validation."""
        if not text or len(text.strip()) < 4:
            return False
        
        # Remove spaces for validation but keep them for display
        clean_text = text.replace(" ", "").upper()
        
        # Must contain at least one letter and one number
        has_letters = any(c.isalpha() for c in clean_text)
        has_numbers = any(c.isdigit() for c in clean_text)
        
        if not (has_letters and has_numbers):
            return False
        
        # Reasonable length for license plates (4-10 characters)
        if len(clean_text) < 4 or len(clean_text) > 10:
            return False
        
        # Should be mostly alphanumeric
        alnum_ratio = sum(c.isalnum() for c in clean_text) / len(clean_text)
        if alnum_ratio < 0.8:
            return False
        
        # NEW: Reject common grill/emblem text patterns that look like plates
        # These are typical patterns found in car grills that get misdetected
        grill_patterns = [
            r'^[A-Z]+\d+[A-Z]+\d+$',  # Pattern like JA5E555 (alternating letters and numbers)
        ]
        
        # Only reject if it's a very short alternating pattern (likely a model badge)
        # Valid plates like IM4U555 should pass
        is_alternating = False
        if len(clean_text) <= 6:
            alternating_count = 0
            for i in range(len(clean_text) - 1):
                if (clean_text[i].isalpha() and clean_text[i+1].isdigit()) or \
                   (clean_text[i].isdigit() and clean_text[i+1].isalpha()):
                    alternating_count += 1
            # If most characters alternate and it's short, likely a grill badge
            if alternating_count >= len(clean_text) * 0.7:
                is_alternating = True
        
        if is_alternating:
            print(f"[DEBUG] Rejected grill-like alternating pattern: {clean_text}")
            return False
        
        # NEW: Prefer more realistic license plate formats
        # Valid formats typically have better letter-number distribution
        if not self._has_realistic_plate_format(clean_text):
            print(f"[DEBUG] Rejected unrealistic plate format: {clean_text}")
            return False
        
        return True
    
    def _has_grill_like_pattern(self, text: str) -> bool:
        """Check if text has a pattern typical of grill/emblem text."""
        # Pattern 1: Alternating letters and numbers (like JA5E555)
        if len(text) >= 6:
            alternating_count = 0
            for i in range(len(text) - 1):
                if (text[i].isalpha() and text[i+1].isdigit()) or (text[i].isdigit() and text[i+1].isalpha()):
                    alternating_count += 1
            
            # If most transitions are alternating, it's likely grill text
            if alternating_count >= len(text) * 0.6:
                return True
        
        # Pattern 2: Repeated characters (common in emblems)
        if any(text.count(char) >= 2 for char in set(text)):
            # Check if repeated characters are in suspicious positions
            for char in set(text):
                if text.count(char) >= 2:
                    indices = [i for i, c in enumerate(text) if c == char]
                    # If same character appears in alternating pattern
                    if len(indices) >= 2 and indices[1] - indices[0] == 2:
                        return True
        
        return False
    
    def _has_realistic_plate_format(self, text: str) -> bool:
        """Check if text follows realistic license plate format patterns."""
        # Pattern 1: Letters (2-4) + Numbers (1-4) - most common
        if re.match(r'^[A-Z]{2,4}\d{1,4}$', text):
            return True
        
        # Pattern 2: Numbers (1-4) + Letters (2-4)
        if re.match(r'^\d{1,4}[A-Z]{2,4}$', text):
            return True
        
        # Pattern 3: Letters (2) + Numbers (2) + Letters (2) + Numbers (4) - Indian format
        if re.match(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', text):
            return True
        
        # Pattern 4: Letters (2-3) + Space + Numbers (1-4)
        if re.match(r'^[A-Z]{2,3}\s\d{1,4}$', text):
            return True
        
        # Pattern 5: Numbers (1-2) + Space + Letters (2-3) + Numbers (1-4)
        if re.match(r'^\d{1,2}\s[A-Z]{2,3}\d{1,4}$', text):
            return True
        
        # Pattern 6: Letters (2-4) + Numbers (2-4) + Letters (2-3) - European format
        if re.match(r'^[A-Z]{2,4}\d{2,4}[A-Z]{2,3}$', text):
            return True
        
        # Pattern 7: Numbers (2-3) + Letters (2-3) + Numbers (2-3) - Some European formats
        if re.match(r'^\d{2,3}[A-Z]{2,3}\d{2,3}$', text):
            return True
        
        # Pattern 8: Letters (3-6) + Numbers (1-4) - Many international formats
        if re.match(r'^[A-Z]{3,6}\d{1,4}$', text):
            return True
        
        # Pattern 9: Numbers (1-4) + Letters (3-6) - Some international formats
        if re.match(r'^\d{1,4}[A-Z]{3,6}$', text):
            return True
        
        # NEW Pattern 10: Mixed alphanumeric (4-8 chars) like IM4U555, B2228HM
        # Allows letters and numbers mixed together (common in many countries)
        if re.match(r'^[A-Z0-9]{4,8}$', text):
            # Must have at least 2 letters and 2 numbers
            letters = sum(c.isalpha() for c in text)
            numbers = sum(c.isdigit() for c in text)
            if letters >= 2 and numbers >= 2:
                return True
        
        # NEW Pattern 11: Letters-Numbers-Letters-Numbers format like IM4U555
        # Pattern: 2-4 letters, 1-4 numbers, 0-3 letters, 0-4 numbers
        if re.match(r'^[A-Z]{2,4}\d{1,4}[A-Z]{0,3}\d{0,4}$', text):
            # Must have at least one number section and reasonable total length
            if len(text) >= 5 and len(text) <= 10:
                return True
        
        # NEW Pattern 12: Bulgarian format - 1-2 letters + 4 numbers + 2 letters like B2228HM
        if re.match(r'^[A-Z]{1,2}\d{4}[A-Z]{2}$', text):
            return True
        
        # NEW Pattern 13: Malaysian/Singapore format like IM4U555 (letters with embedded numbers)
        if re.match(r'^[A-Z]{1,3}\d{1,2}[A-Z]{1,3}\d{1,4}$', text):
            return True
        
        # If none of the realistic patterns match, be more suspicious
        return False
    
    def _estimate_plate_confidence(self, normalized_text: str, raw_text: str) -> float:
        """Estimate confidence in the detected license plate text with improved validation."""
        if not normalized_text:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Bonus for reasonable length
        clean_len = len(normalized_text.replace(" ", ""))
        if 6 <= clean_len <= 8:
            confidence += 0.1
        
        # Bonus for having both letters and numbers
        has_letters = any(c.isalpha() for c in normalized_text)
        has_numbers = any(c.isdigit() for c in normalized_text)
        if has_letters and has_numbers:
            confidence += 0.1
        
        # Bonus for space in reasonable position (like "CZ17 KOD")
        if " " in normalized_text:
            parts = normalized_text.split()
            if len(parts) == 2:
                # Check if it follows pattern like "AB12 CDE" or "CZ17 KOD"
                if (len(parts[0]) >= 2 and len(parts[1]) >= 2):
                    confidence += 0.15
        
        # NEW: Higher confidence for realistic plate formats
        if self._has_realistic_plate_format(normalized_text.replace(" ", "")):
            confidence += 0.2
        
        # NEW: Penalty for grill-like patterns
        if self._has_grill_like_pattern(normalized_text.replace(" ", "")):
            confidence -= 0.3
        
        # NEW: Bonus for proper letter-number balance
        letter_count = sum(c.isalpha() for c in normalized_text.replace(" ", ""))
        number_count = sum(c.isdigit() for c in normalized_text.replace(" ", ""))
        
        if letter_count > 0 and number_count > 0:
            ratio = max(letter_count, number_count) / min(letter_count, number_count)
            # Prefer balanced letter-number ratios (not too skewed)
            if ratio <= 3:
                confidence += 0.1
            elif ratio > 6:
                confidence -= 0.2  # Penalty for very skewed ratios
        
        # NEW: Penalty for patterns that look like brand models
        if self._looks_like_brand_model(normalized_text):
            confidence -= 0.4
        
        # Ensure confidence stays within bounds
        return max(0.1, min(0.95, confidence))
    
    def _looks_like_brand_model(self, text: str) -> bool:
        """Check if text looks like a car brand/model rather than a license plate."""
        text = text.upper().replace(" ", "")
        
        # Common brand model patterns that get confused with plates
        brand_model_patterns = [
            r'^[A-Z]{2,4}\d{2,4}$',  # Like JA55, BMW320, etc.
            r'^\d{3,4}[A-Z]{1,3}$',  # Like 328I, 750LI
        ]
        
        # Check if it matches brand model patterns
        for pattern in brand_model_patterns:
            if re.match(pattern, text):
                # Additional heuristics to distinguish from actual plates
                if self._has_grill_like_pattern(text):
                    return True
                
                # If it's short and looks like a model code
                if len(text) <= 5:
                    return True
        
        return False
    
    def _detect_object_colors(self, frame: np.ndarray, objects: List[Dict]) -> List[Dict]:
        """Detect colors for detected objects using advanced color detection"""
        colors = []
        
        try:
            for obj in objects:
                if 'bounding_box' in obj:
                    x1, y1, x2, y2 = obj['bounding_box']
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        # Use advanced color detection
                        if self.color_detector:
                            color_result = detect_object_color_advanced(crop, use_hybrid=True)
                        else:
                            # Fallback to simple HSV detection
                            color_result = self._simple_color_detection(crop)
                        
                        colors.append({
                            'object_id': obj['object_id'],
                            'object_class': obj['class_name'],
                            'color_info': color_result,
                            'bounding_box': obj['bounding_box']
                        })
        except Exception as e:
            print(f"[ERROR] Advanced color detection failed: {e}")
        
        return colors
    
    def _simple_color_detection(self, crop: np.ndarray) -> Dict:
        """Simple HSV-based color detection as fallback"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Define color ranges
            color_ranges = {
                'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
                'orange': [(10, 50, 50), (25, 255, 255)],
                'yellow': [(25, 50, 50), (35, 255, 255)],
                'green': [(35, 50, 50), (85, 255, 255)],
                'blue': [(100, 50, 50), (130, 255, 255)],
                'purple': [(130, 50, 50), (170, 255, 255)],
                'black': [(0, 0, 0), (180, 255, 50)],
                'white': [(0, 0, 200), (180, 30, 255)],
                'gray': [(0, 0, 50), (180, 30, 200)]
            }
            
            max_pixels = 0
            detected_color = 'unknown'
            
            for color, ranges in color_ranges.items():
                pixel_count = 0
                for i in range(0, len(ranges), 2):
                    lower = np.array(ranges[i])
                    upper = np.array(ranges[i+1])
                    mask = cv2.inRange(hsv, lower, upper)
                    pixel_count += cv2.countNonZero(mask)
                
                if pixel_count > max_pixels:
                    max_pixels = pixel_count
                    detected_color = color
            
            return {
                'final_color': detected_color,
                'confidence': 0.6,
                'method': 'simple_hsv_fallback'
            }
            
        except Exception as e:
            print(f"[ERROR] Simple color detection failed: {e}")
            return {
                'final_color': 'unknown',
                'confidence': 0.0,
                'method': 'error'
            }
    
    def _apply_color_calibration(self, frame: np.ndarray) -> np.ndarray:
        """Apply color calibration to improve color detection accuracy"""
        try:
            if not self.calibration_applied:
                # Basic white balance adjustment
                balanced = self._apply_white_balance(frame)
                
                # Contrast and brightness enhancement
                enhanced = cv2.convertScaleAbs(balanced, alpha=1.2, beta=10)
                
                return enhanced
            else:
                return frame
        except Exception as e:
            print(f"[ERROR] Color calibration failed: {e}")
            return frame
    
    def _apply_white_balance(self, frame: np.ndarray) -> np.ndarray:
        """Apply simple white balance correction"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels back
            lab = cv2.merge([l, a, b])
            
            # Convert back to BGR
            balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return balanced
        except Exception as e:
            print(f"[ERROR] White balance failed: {e}")
            return frame
    
    def _create_annotated_frame_maximum_speed(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Maximum speed annotation for real-time performance with Enhanced K-means + ResNet-18 + Fallback colors"""
        try:
            annotated = frame.copy()
            
            # Ultra-simple header
            timestamp = result.get('timestamp', datetime.now().isoformat())
            time_str = timestamp.split('T')[1][:8]
            
            cv2.putText(annotated, f"⚡ MAX SPEED + FALLBACK - {time_str}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # FPS counter
            fps_text = f"🔥 {self.stats['fps']:.0f} FPS"
            cv2.putText(annotated, fps_text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw license plates (if found) - IMPROVED: Better formatting like image processing
            for plate in (result.get('license_plates', []) or [])[:5]:
                bbox = plate.get('bounding_box')
                if not bbox:
                    continue
                x1, y1, x2, y2 = bbox
                
                # Validate plate text
                plate_text = (plate.get('text') or '').strip()
                if not plate_text or len(plate_text) < 4:
                    continue
                
                # Draw yellow box for license plates (matching image processing style)
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
                
                # Create enhanced label with "Plate: " prefix
                label = f"Plate: {plate_text[:16]}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Position label above the plate
                ly = int(y1) - 10
                if ly - text_height - baseline < 0:
                    ly = int(y2) + text_height + 10
                
                # Draw yellow background for label
                cv2.rectangle(annotated, 
                             (int(x1), ly - text_height - baseline), 
                             (int(x1) + text_width + 8, ly + 4), 
                             (0, 255, 255), -1)
                
                # Draw label text in black
                cv2.putText(annotated, label, (int(x1) + 4, ly), 
                           font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # Draw objects with multi-level fallback color information
            objects = result.get('objects', [])
            ocr_results = result.get('ocr_results', []) or []
            ocr_by_object = {}
            for item in ocr_results:
                oid = item.get('object_id')
                if oid is not None:
                    ocr_by_object[oid] = item
            
            # Limit to 6 objects for maximum speed
            for i, obj in enumerate(objects[:6]):
                if 'bounding_box' in obj:
                    x1, y1, x2, y2 = obj['bounding_box']
                    
                    # Get color information from any available source
                    color_family = obj.get('color_family', 'Unknown')
                    color_shade = obj.get('color_shade', 'Unknown')
                    fallback_level = obj.get('fallback_level', 0)
                    processing_method = obj.get('processing_method', 'unknown')
                    resnet_enhanced = obj.get('resnet_enhanced', False)
                    
                    # Determine display name and method indicator
                    if 'enhanced_category' in obj:
                        display_name = obj['enhanced_category'].get('enhanced_label', obj['display_name'])
                    else:
                        display_name = obj.get('display_name', obj['class_name'])
                    
                    # Method indicators
                    if fallback_level == 1:
                        method_indicator = "🧠"  # ResNet-18 + K-means
                    elif fallback_level == 2:
                        method_indicator = "🎯"  # Basic K-means
                    elif fallback_level == 3:
                        method_indicator = "🎨"  # HSV Detection
                    elif fallback_level == 4:
                        method_indicator = "🔧"  # Basic Classification
                    else:
                        method_indicator = "❓"  # Unknown
                    
                    box_color = obj.get('color', (255, 255, 255))
                    
                    # Fastest possible rectangle
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Enhanced label with fallback information
                    if color_family != 'Unknown':
                        label = f"{display_name[:10]} {method_indicator}{color_shade[:6]}"
                    else:
                        label = f"{display_name[:15]} {method_indicator}"
                    
                    # Fast label background and text
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0]
                    cv2.rectangle(annotated, (x1, y1 - 20), (x1 + label_size[0], y1), box_color, -1)
                    cv2.putText(annotated, label, 
                               (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)

                    # OCR text (if available)
                    ocr_item = ocr_by_object.get(obj.get('object_id'))
                    if ocr_item:
                        ocr_text = (ocr_item.get('text') or '').strip()
                        if ocr_text:
                            ocr_text = ocr_text[:28]
                            ocr_size = cv2.getTextSize(ocr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                            oy = y2 + 18
                            if oy + 8 > annotated.shape[0]:
                                oy = max(20, y1 - 26)
                            cv2.rectangle(annotated, (x1, oy - 16), (x1 + ocr_size[0] + 6, oy + 4), (0, 0, 0), -1)
                            cv2.putText(annotated, ocr_text, (x1 + 3, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Enhanced stats with fallback information
            obj_text = f"🎯 Objects: {len(objects)}"
            cv2.putText(annotated, obj_text, 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            lp_count = len(result.get('license_plates', []) or [])
            if lp_count:
                lp_text = f"🚗 Plates: {lp_count} (unique: {len(self.stats['unique_plates'])})"
                cv2.putText(annotated, lp_text, (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if self.enable_ocr:
                ocr_text = f"🔤 OCR: {len(ocr_results)} (unique: {len(self.stats['unique_texts'])})"
                cv2.putText(annotated, ocr_text,
                           (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Add color detection stats with fallback breakdown
            color_summary = result.get('color_processing_summary', {})
            if color_summary:
                total_processed = color_summary.get('total_objects', 0)
                enhanced_count = color_summary.get('enhanced_processed', 0)
                fallback_count = color_summary.get('fallback_processed', 0)
                simple_count = color_summary.get('simple_processed', 0)
                success_rate = color_summary.get('success_rate', 0)
                
                color_text = f"🛡️ Colors: {total_processed} (✨{enhanced_count} 🔄{fallback_count} 🔧{simple_count})"
                cv2.putText(annotated, color_text, 
                           (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                # Success rate
                success_text = f"📊 Success: {success_rate:.1%}"
                cv2.putText(annotated, success_text, 
                           (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Fallback system status
            if self.fallback_detector:
                fallback_status = get_fallback_status()
                current_level = fallback_status.get('current_level', 0)
                if current_level >= 1:
                    method_text = "🚀 GPU + ResNet-18 + K-means + Fallback | ROBUST"
                else:
                    method_text = "🚀 GPU + K-means + Fallback | ROBUST"
            else:
                method_text = "🚀 GPU + Basic Detection | FALLBACK"
            
            cv2.putText(annotated, method_text, 
                       (10, annotated.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            print(f"[ERROR] Max speed annotation failed: {e}")
            return frame
        """Ultra-fast annotated frame for maximum performance"""
        try:
            annotated = frame.copy()
            
            # Add performance header
            timestamp = result.get('timestamp', datetime.now().isoformat())
            time_str = timestamp.split('T')[1][:8]
            mode = result.get('processing_mode', 'fast')
            gpu_enabled = result.get('frame_info', {}).get('gpu_enabled', False)
            gpu_status = "🚀 GPU" if gpu_enabled else "💻 CPU"
            
            cv2.putText(annotated, f"⚡ YOLO26 ULTRA FAST - {time_str}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add FPS and GPU status
            fps_text = f"🔥 {self.stats['fps']:.1f} FPS | {gpu_status}"
            cv2.putText(annotated, fps_text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw objects with minimal processing
            objects = result.get('objects', [])
            
            # Limit to 10 objects for maximum speed
            for i, obj in enumerate(objects[:10]):
                if 'bounding_box' in obj:
                    x1, y1, x2, y2 = obj['bounding_box']
                    display_name = obj.get('display_name', obj['class_name'])
                    confidence = obj['confidence']
                    box_color = obj.get('color', (255, 255, 255))
                    
                    # Fast rectangle drawing
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Simple label (no confidence for speed)
                    label = display_name
                    
                    # Fast label drawing
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated, (x1, y1 - 22), (x1 + label_size[0], y1), box_color, -1)
                    cv2.putText(annotated, label, 
                               (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Minimal stats
            stats_y = 90
            obj_text = f"🎯 Objects: {len(objects)}"
            cv2.putText(annotated, obj_text, 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Performance indicator
            perf_text = f"⚡ {mode.upper()} | Smooth Mode"
            cv2.putText(annotated, perf_text, 
                       (10, annotated.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            print(f"[ERROR] Ultra-fast annotation failed: {e}")
            return frame
        """Fast annotated frame creation for smooth real-time display"""
        try:
            annotated = frame.copy()
            
            # Add timestamp and FPS
            timestamp = result.get('timestamp', datetime.now().isoformat())
            time_str = timestamp.split('T')[1][:8]  # HH:MM:SS
            cv2.putText(annotated, f"🚀 YOLO26 FAST Live - {time_str}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add FPS
            fps_text = f"FPS: {self.stats['fps']:.1f} ⚡"
            cv2.putText(annotated, fps_text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw detected objects quickly
            objects = result.get('objects', [])
            
            for i, obj in enumerate(objects[:15]):  # Limit to 15 objects for speed
                if 'bounding_box' in obj:
                    x1, y1, x2, y2 = obj['bounding_box']
                    class_name = obj['class_name']
                    confidence = obj['confidence']
                    display_name = obj.get('display_name', class_name)
                    category = obj.get('category', 'Unknown')
                    box_color = obj.get('color', (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Create simple label
                    label = f"{display_name} {confidence:.2f}"
                    if category != 'Unknown' and category != class_name:
                        label = f"{display_name}"
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated, (x1, y1 - 25), (x1 + label_size[0], y1), box_color, -1)
                    
                    # Draw label text
                    cv2.putText(annotated, label, 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw license plates (only if any) - IMPROVED: Better formatting with yellow box
            plates = result.get('license_plates', [])
            if plates and len(plates) > 0:
                for i, plate in enumerate(plates[:3]):  # Max 3 plates
                    if 'bounding_box' in plate and plate['bounding_box']:
                        x1, y1, x2, y2 = plate['bounding_box']
                        
                        # Get plate text with validation
                        plate_text = plate.get('plate_text', plate.get('text', 'Unknown')).strip()
                        if not plate_text or len(plate_text) < 4:
                            continue
                        
                        # Yellow box for license plates (matching image style)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
                        
                        # Add plate text with "Plate: " prefix
                        label = f"Plate: {plate_text[:16]}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        
                        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # Position label above the plate
                        ly = y1 - 10
                        if ly - text_height - baseline < 0:
                            ly = y2 + text_height + 10
                        
                        # Draw yellow background for label
                        cv2.rectangle(annotated, 
                                     (x1, ly - text_height - baseline), 
                                     (x1 + text_width + 8, ly + 4), 
                                     (0, 255, 255), -1)
                        
                        # Draw label text in black
                        cv2.putText(annotated, label, (x1 + 4, ly), 
                                   font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            
            # Simple stats display
            stats_y = 90
            detection_text = f"🎯 Objects: {len(objects)}"
            cv2.putText(annotated, detection_text, 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if plates:
                stats_y += 30
                plate_text = f"📗 Plates: {len(plates)}"
                cv2.putText(annotated, plate_text, 
                           (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add performance indicator
            mode = result.get('processing_mode', 'fast')
            perf_text = f"⚡ {mode.upper()} | Smooth Mode"
            cv2.putText(annotated, perf_text, 
                       (10, annotated.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            print(f"[ERROR] Failed to create fast annotated frame: {e}")
            return frame
        """Create annotated frame for real-time display with enhanced visualization"""
        try:
            annotated = frame.copy()
            
            # Add timestamp
            timestamp = result.get('timestamp', datetime.now().isoformat())
            time_str = timestamp.split('T')[1][:8]  # HH:MM:SS
            cv2.putText(annotated, f"YOLO26 Enhanced Live - {time_str}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add FPS
            fps_text = f"FPS: {self.stats['fps']:.1f}"
            cv2.putText(annotated, fps_text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw all detected objects with colors
            objects = result.get('objects', [])
            colors = result.get('colors', [])
            
            # Create color map for objects
            color_map = {color['object_id']: color['color_info'] for color in colors}
            
            # Define colors for drawing boxes
            box_colors = {
                'person': (0, 255, 255),     # Yellow
                'car': (255, 0, 0),           # Blue
                'truck': (0, 0, 255),         # Red
                'bicycle': (0, 255, 0),       # Green
                'motorcycle': (255, 255, 0),   # Cyan
                'bus': (255, 0, 255),         # Magenta
                'dog': (128, 0, 128),         # Purple
                'cat': (255, 165, 0),         # Orange
                'default': (255, 255, 255)    # White
            }
            
            for i, obj in enumerate(objects):
                if 'bounding_box' in obj:
                    x1, y1, x2, y2 = obj['bounding_box']
                    class_name = obj['class_name']
                    confidence = obj['confidence']
                    
                    # Use enhanced display name and color from object detection
                    display_name = obj.get('display_name', class_name)
                    category = obj.get('category', 'Unknown')
                    box_color = obj.get('color', (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Get color information
                    color_info = color_map.get(obj['object_id'], {})
                    detected_color = color_info.get('final_color', 'unknown')
                    color_confidence = color_info.get('confidence', 0.0)
                    
                    # Create enhanced label text with gender and category
                    label = f"{display_name} {confidence:.2f}"
                    if category != 'Unknown' and category != class_name:
                        label += f" [{category}]"
                    if detected_color != 'unknown' and color_confidence > 0.5:
                        label += f" [{detected_color}]"
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated, (x1, y1 - 25), (x1 + label_size[0], y1), box_color, -1)
                    
                    # Draw label text
                    cv2.putText(annotated, label, 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw license plates - IMPROVED: Better formatting with yellow box
            plates = result.get('license_plates', [])
            for i, plate in enumerate(plates):
                if 'bounding_box' in plate and plate['bounding_box']:
                    x1, y1, x2, y2 = plate['bounding_box']
                    
                    # Get plate text with validation
                    plate_text = plate.get('plate_text', plate.get('text', 'Unknown')).strip()
                    if not plate_text or len(plate_text) < 4:
                        continue
                    
                    # Yellow box for license plates (matching image style)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    
                    # Add plate text with "Plate: " prefix
                    label = f"Plate: {plate_text[:20]}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Position label above the plate
                    ly = y1 - 10
                    if ly - text_height - baseline < 0:
                        ly = y2 + text_height + 10
                    
                    # Draw yellow background for label
                    cv2.rectangle(annotated, 
                                 (x1, ly - text_height - baseline), 
                                 (x1 + text_width + 8, ly + 4), 
                                 (0, 255, 255), -1)
                    
                    # Draw label text in black
                    cv2.putText(annotated, label, (x1 + 4, ly), 
                               font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            
            # Add detection statistics
            stats_y = 90
            detection_text = f"Objects: {len(objects)}"
            cv2.putText(annotated, detection_text, 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            stats_y += 30
            plate_text = f"Plates: {len(plates)}"
            cv2.putText(annotated, plate_text, 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            stats_y += 30
            unique_objects = len(self.stats['unique_objects'])
            obj_text = f"Unique Objects: {unique_objects}"
            cv2.putText(annotated, obj_text, 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            stats_y += 30
            unique_colors = len(self.stats['unique_colors'])
            color_text = f"Colors Found: {unique_colors}"
            cv2.putText(annotated, color_text, 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add processing mode and calibration info
            mode = result.get('processing_mode', 'unknown')
            frame_info = result.get('frame_info', {})
            calibration_status = "Calibrated" if frame_info.get('calibration_applied', False) else "Raw"
            
            mode_text = f"Mode: {mode} | {calibration_status}"
            cv2.putText(annotated, mode_text, 
                       (10, annotated.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            print(f"[ERROR] Failed to create annotated frame: {e}")
            return frame
    
    def capture_snapshot(self, save_path: Optional[str] = None) -> Dict:
        """Capture current frame snapshot"""
        try:
            if self.current_frame is None:
                return {'error': 'No frame available'}
            
            # Generate filename if not provided
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"outputs/webcam_snapshot_{timestamp}.jpg"
            
            # Save frame
            success = cv2.imwrite(save_path, self.current_frame)
            
            if success:
                return {
                    'success': True,
                    'path': save_path,
                    'timestamp': datetime.now().isoformat(),
                    'message': 'Snapshot saved successfully'
                }
            else:
                return {'error': 'Failed to save snapshot'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def enable_enhanced_features(self, enable_colors: bool = True, enable_objects: bool = True, enable_calibration: bool = True):
        """Enable or disable enhanced detection features"""
        self.enable_advanced_colors = enable_colors
        self.enable_general_objects = enable_objects
        self.calibration_applied = enable_calibration  # Reset calibration when enabling
        
        print(f"[INFO] Enhanced features updated:")
        print(f"  - Advanced Colors: {'Enabled' if enable_colors else 'Disabled'}")
        print(f"  - General Objects: {'Enabled' if enable_objects else 'Disabled'}")
        print(f"  - Color Calibration: {'Enabled' if enable_calibration else 'Disabled'}")
    
    def calibrate_colors(self, frame: np.ndarray) -> Dict:
        """Set calibration frame for color correction"""
        try:
            self.color_calibration_frame = frame.copy()
            self.calibration_applied = True
            
            # Analyze the calibration frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mean_hsv = cv2.mean(hsv)
            
            calibration_info = {
                'timestamp': datetime.now().isoformat(),
                'mean_hsv': mean_hsv[:3],
                'frame_shape': frame.shape
            }
            
            print(f"[INFO] Color calibration applied")
            print(f"  - Mean HSV: {mean_hsv[:3]}")
            
            return {
                'success': True,
                'calibration_info': calibration_info,
                'message': 'Color calibration applied successfully'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def reset_calibration(self):
        """Reset color calibration"""
        self.color_calibration_frame = None
        self.calibration_applied = False
        print("[INFO] Color calibration reset")
    
    def get_detection_summary(self) -> Dict:
        """Get comprehensive detection summary"""
        try:
            runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
            
            # Calculate average processing time
            if self.stats['processing_times']:
                avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            else:
                avg_processing_time = 0
            
            return {
                'session_info': {
                    'is_running': self.is_running,
                    'runtime_seconds': runtime,
                    'frames_processed': self.stats['frames_processed'],
                    'current_fps': self.stats['fps'],
                    'avg_processing_time_ms': avg_processing_time * 1000
                },
                'detection_stats': {
                    'total_detections': self.stats['total_detections'],
                    'objects_detected': self.stats['objects_detected'],
                    'colors_detected': self.stats['colors_detected'],
                    'unique_plates_found': list(self.stats['unique_plates']),
                    'unique_objects_found': list(self.stats['unique_objects']),
                    'unique_colors_found': list(self.stats['unique_colors'])
                },
                'feature_status': {
                    'advanced_colors_enabled': self.enable_advanced_colors,
                    'general_objects_enabled': self.enable_general_objects,
                    'color_calibration_applied': self.calibration_applied,
                    'color_detector_available': self.color_detector is not None
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_current_stats(self) -> Dict:
        """Get current processing statistics"""
        try:
            runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
            
            # Calculate average processing time
            if self.stats['processing_times']:
                avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            else:
                avg_processing_time = 0
            
            return {
                'is_running': self.is_running,
                'runtime_seconds': runtime,
                'frames_processed': self.stats['frames_processed'],
                'current_fps': self.stats['fps'],
                'avg_processing_time_ms': avg_processing_time * 1000,
                'total_detections': self.stats['total_detections'],
                'unique_plates_found': list(self.stats['unique_plates']),
                'unique_plates_count': len(self.stats['unique_plates']),
                'processing_queue_size': self.result_queue.qsize()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def export_session_data(self, filename: Optional[str] = None) -> Dict:
        """Export session data to JSON file"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"outputs/webcam_session_{timestamp}.json"
            
            session_data = {
                'session_info': {
                    'start_time': self.stats['start_time'],
                    'export_time': datetime.now().isoformat(),
                    'total_runtime': time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
                },
                'statistics': self.get_current_stats(),
                'detected_plates': list(self.stats['unique_plates']),
                'performance_metrics': {
                    'avg_processing_times': self.stats['processing_times'],
                    'fps_history': []  # Would need to implement FPS history
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            return {
                'success': True,
                'filename': filename,
                'message': 'Session data exported successfully'
            }
            
        except Exception as e:
            return {'error': str(e)}


# Global instance for easy access
webcam_processor = WebcamProcessor()

# Convenience functions
def start_webcam(camera_index: int = 0, **kwargs) -> Dict:
    """Start webcam with default settings"""
    return webcam_processor.start_camera(camera_index, **kwargs)

def stop_webcam() -> Dict:
    """Stop webcam"""
    return webcam_processor.stop_camera()

def get_webcam_frame() -> Optional[np.ndarray]:
    """Get current webcam frame"""
    return webcam_processor.get_processed_frame()

def get_webcam_stats() -> Dict:
    """Get webcam statistics"""
    return webcam_processor.get_current_stats()


if __name__ == "__main__":
    print("📷 Webcam Processor Module")
    print("=" * 30)
    
    # Test camera listing
    print("🔍 Scanning for cameras...")
    cameras = webcam_processor.list_available_cameras()
    
    available_cameras = [c for c in cameras if c.get('available', False)]
    print(f"Found {len(available_cameras)} available cameras")
    
    for camera in available_cameras:
        print(f"  - {camera['name']}: {camera['resolution']} @ {camera['fps']}fps")
    
    print("\n📖 Usage:")
    print("   from webcam_processor import start_webcam, stop_webcam, get_webcam_frame")
    print("   start_webcam(0)")
    print("   while True:")
    print("       frame = get_webcam_frame()")
    print("       if frame is not None:")
    print("           cv2.imshow('YOLO26 Live', frame)")
    print("           if cv2.waitKey(1) & 0xFF == ord('q'):")
    print("               break")
    print("   stop_webcam()")
    
    print("\n✅ Webcam processor ready!")
    print("   Features:")
    print("   - Multi-threaded processing")
    print("   - Real-time performance optimization")
    print("   - Live statistics")
    print("   - Session data export")
    print("   - Snapshot capture")
