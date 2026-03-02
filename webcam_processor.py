"""
Webcam Processing Module for YOLO26
Handle all webcam/live camera operations
Real-time processing with performance optimization
"""

import cv2
import numpy as np
import time
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from queue import Queue
import json

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
                # Always run OCR for cups/plates regardless of frame skip
                has_priority = any(str(obj.get('class_name', '')).strip().lower() in ('cup', 'bottle', 'book', 'license plate', 'cell phone') for obj in objects[:4])
                run_now = has_priority or (self._frame_idx % int(max(1, self.ocr_every_n)) == 0)
                if run_now:
                    try:
                        result['ocr_results'] = self._extract_text_for_objects(frame, objects)
                        for item in result['ocr_results']:
                            text = (item.get('text') or '').strip()
                            if text:
                                self.stats['unique_texts'].add(text)
                    except Exception as e:
                        print(f"[DEBUG] OCR processing failed: {e}")
            
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

    def _extract_text_for_objects(self, frame: np.ndarray, objects: List[Dict]) -> List[Dict]:
        try:
            h, w = frame.shape[:2]
            results = []

            # Keep OCR bounded for speed; now process all objects up to limit
            for obj in objects[:6]:
                if 'bounding_box' not in obj:
                    continue
                x1, y1, x2, y2 = obj['bounding_box']
                x1 = int(max(0, min(w - 1, x1)))
                y1 = int(max(0, min(h - 1, y1)))
                x2 = int(max(0, min(w - 1, x2)))
                y2 = int(max(0, min(h - 1, y2)))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Skip tiny crops (usually unreadable) - lowered threshold for cups/plates/person/vehicles
                class_name = str(obj.get('class_name', '')).strip().lower()
                is_priority = class_name in ('cup', 'bottle', 'book', 'license plate', 'person', 'car', 'truck', 'bus', 'motorcycle')
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
                if cached and isinstance(cached, dict) and (time.time() - cached.get('ts', 0)) < cache_ttl:
                    results.append(cached['data'])
                    continue

                text = ''

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
                                text = self._run_ocr_on_crop(plate_crop, confidence_threshold=0.3, is_plate=True)
                    except Exception:
                        pass

                # General text for any object with lower threshold for priority objects
                if not text:
                    thresh = 0.2 if class_name in ('cup', 'bottle', 'book', 'license plate', 'person', 'car', 'truck', 'bus', 'motorcycle') else 0.4

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
                        t = self._run_ocr_on_crop(
                            c,
                            confidence_threshold=thresh,
                            is_plate=(class_name in ('license plate', 'car', 'truck', 'bus', 'motorcycle')),
                        )
                        t = (t or '').strip()
                        if t and len(t) > best_len:
                            best_text = t
                            best_len = len(t)
                            if best_len >= 10:
                                break

                    # If still empty for priority objects, do a more permissive pass
                    if (not best_text) and class_name in ('cup', 'bottle', 'book', 'license plate', 'person', 'car', 'truck', 'bus', 'motorcycle'):
                        for c in crops_to_try:
                            t = self._run_ocr_on_crop(
                                c,
                                confidence_threshold=0.15,
                                is_plate=(class_name in ('license plate', 'car', 'truck', 'bus', 'motorcycle')),
                            )
                            t = (t or '').strip()
                            if t and len(t) > best_len:
                                best_text = t
                                best_len = len(t)
                                if best_len >= 8:
                                    break

                    text = best_text

                data = {
                    'object_id': obj.get('object_id'),
                    'class_name': obj.get('class_name'),
                    'bounding_box': (x1, y1, x2, y2),
                    'text': text
                }
                self._ocr_cache[key] = {'ts': time.time(), 'data': data}
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

    def _run_ocr_on_crop(self, crop: np.ndarray, confidence_threshold: float = 0.4, is_plate: bool = False) -> str:
        try:
            if crop is None or crop.size == 0:
                return ''

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

            return best_text
        except Exception as e:
            print(f"[DEBUG] _run_ocr_on_crop failed: {e}")
            return ''

    def _preprocess_for_ocr(self, crop: np.ndarray, is_plate: bool = False) -> List[Tuple[np.ndarray, float]]:
        """Generate multiple preprocessed versions of the crop for OCR, with rotation attempts."""
        processed = []
        try:
            h, w = crop.shape[:2]

            # 1) Basic resize if too small (target ~300-400 px width for OCR)
            target_w = 350 if not is_plate else 400
            if w < target_w:
                scale = target_w / w
                new_h = int(h * scale)
                crop = cv2.resize(crop, (target_w, new_h), interpolation=cv2.INTER_CUBIC)

            # 2) Enhance contrast and sharpen
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)

            # Convert back to BGR for PaddleOCR
            enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            processed.append((enhanced_bgr, 0.0))

            # 3) Try flips and rotations for text on cups or skewed plates
            # Horizontal flip for mirrored text (common in webcam/camera)
            flipped_h = cv2.flip(enhanced_bgr, 1)
            processed.append((flipped_h, 0.0))

            # Vertical flip
            flipped_v = cv2.flip(enhanced_bgr, 0)
            processed.append((flipped_v, 0.0))

            # Both flips (equivalent to 180 rotation)
            flipped_hv = cv2.flip(enhanced_bgr, -1)
            processed.append((flipped_hv, 0.0))

            # Rotations
            angles = [90, 180, 270] if not is_plate else [90, 270]
            for angle in angles:
                rotated = cv2.rotate(enhanced_bgr, cv2.ROTATE_90_CLOCKWISE if angle == 90 else (cv2.ROTATE_180 if angle == 180 else cv2.ROTATE_90_COUNTERCLOCKWISE))
                processed.append((rotated, float(angle)))

            # Also try flips of rotated versions for cups
            if not is_plate:
                for angle in [90, 270]:
                    rotated = cv2.rotate(enhanced_bgr, cv2.ROTATE_90_CLOCKWISE if angle == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
                    flipped_rot = cv2.flip(rotated, 1)  # horizontal flip of rotated
                    processed.append((flipped_rot, float(angle + 1000)))  # use angle+1000 to distinguish

            # 4) For plates, also try a slight deskew using morphological operations
            if is_plate:
                # Binarize
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # Morph open to reduce noise
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                deskewed_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
                processed.append((deskewed_bgr, 0.0))

        except Exception as e:
            print(f"[DEBUG] _preprocess_for_ocr failed: {e}")
            # Fallback: return original crop
            processed.append((crop, 0.0))

        return processed
    
    def _detect_all_objects(self, frame: np.ndarray) -> List[Dict]:
        """Maximum speed GPU-only object detection"""
        try:
            # Import YOLO from your main app
            from app import get_model, _get_device
            
            # Use the nano model for maximum speed
            model = get_model("yolo26n.pt")
            
            device = _get_device()
            
            # Ultra aggressive downsampling for maximum speed
            height, width = frame.shape[:2]
            max_size = 320  # Even smaller for maximum speed
            
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                # Use fastest interpolation
                small_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            else:
                small_frame = frame
                new_width, new_height = width, height
            
            # Maximum speed detection settings
            import torch
            try:
                with torch.no_grad():  # Disable gradient calculation
                    detection_results = model.predict(
                        source=small_frame,
                        conf=0.15,      # Very low confidence for more detections
                        iou=0.35,       # Lower IoU for faster NMS
                        max_det=6,      # Even fewer detections for speed
                        device=device,
                        verbose=False,
                        half=True if device != "cpu" else False,
                    )
            except Exception as e:
                if device != "cpu":
                    print(f"[WARNING] GPU detection failed, falling back to CPU: {e}")
                    device = "cpu"
                    with torch.no_grad():
                        detection_results = model.predict(
                            source=small_frame,
                            conf=0.15,
                            iou=0.35,
                            max_det=6,
                            device=device,
                            verbose=False,
                            half=False,
                        )
                else:
                    raise
            
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
                        
                        # Very low confidence threshold
                        if confidence > 0.15:
                            # Ultra-fast classification with emojis
                            ultra_fast_categories = {
                                'person': ('👤 Person', 'Person', (0, 255, 255)),
                                'bird': ('🐦 Bird', 'Bird', (255, 105, 180)),
                                'cat': ('🐱 Cat', 'Animal', (255, 0, 255)),
                                'dog': ('🐕 Dog', 'Animal', (255, 0, 255)),
                                'cup': ('☕ Cup', 'Drinkware', (139, 69, 19)),
                                'bottle': ('🍶 Bottle', 'Drinkware', (139, 69, 19)),
                                'cell phone': ('📱 Phone', 'Electronics', (0, 191, 255)),
                                'laptop': ('💻 Laptop', 'Electronics', (0, 191, 255)),
                                'car': ('🚗 Car', 'Vehicle', (255, 165, 0)),
                                'bicycle': ('🚴 Bicycle', 'Vehicle', (0, 255, 255)),
                                'motorcycle': ('🏍️ Moto', 'Vehicle', (255, 165, 0)),
                                'bus': ('🚌 Bus', 'Vehicle', (255, 165, 0)),
                                'truck': ('🚚 Truck', 'Vehicle', (255, 165, 0)),
                                'chair': ('🪑 Chair', 'Furniture', (139, 69, 19)),
                                'book': ('📚 Book', 'Object', (128, 128, 128)),
                                'clock': ('🕐 Clock', 'Object', (128, 128, 128)),
                                'handbag': ('👜 Bag', 'Object', (128, 128, 128)),
                                'backpack': ('🎒 Pack', 'Object', (128, 128, 128)),
                            }
                            
                            display_name, category, color = ultra_fast_categories.get(class_name.lower(), 
                                (class_name.title(), 'Object', (255, 255, 255)))
                            
                            objects.append({
                                'object_id': f"{class_name}_{i}",
                                'class_name': class_name,
                                'display_name': display_name,
                                'category': category,
                                'confidence': confidence,
                                'bounding_box': (x1, y1, x2, y2),
                                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                                'size': (x2 - x1, y2 - y1),
                                'color': color
                            })
            
            return objects
            
        except Exception as e:
            print(f"[ERROR] Ultra-fast GPU detection failed: {e}")
            return []
    
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
            
            # Draw license plates (only if any)
            plates = result.get('license_plates', [])
            if plates and len(plates) > 0:
                for i, plate in enumerate(plates[:3]):  # Max 3 plates
                    if 'bounding_box' in plate and plate['bounding_box']:
                        x1, y1, x2, y2 = plate['bounding_box']
                        
                        # Green box for license plates
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Add plate text
                        plate_text = plate.get('plate_text', 'Unknown')
                        cv2.putText(annotated, f"📗 {plate_text}", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
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
            
            # Draw license plates
            plates = result.get('license_plates', [])
            for i, plate in enumerate(plates):
                if 'bounding_box' in plate and plate['bounding_box']:
                    x1, y1, x2, y2 = plate['bounding_box']
                    
                    # Green box for license plates
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Add plate text
                    plate_text = plate.get('plate_text', 'Unknown')
                    cv2.putText(annotated, f"Plate: {plate_text}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
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
