"""
Enhanced Car and License Plate Video Processor
Detects cars in video and extracts number plates with real-time display
"""

import cv2
import numpy as np
import time
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Generator
from pathlib import Path

# Import YOLO and OCR modules
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] YOLO not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not available")

# Import OCR modules
try:
    from optimized_paddleocr_gpu import extract_text_optimized
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("[WARNING] PaddleOCR not available")

try:
    from international_license_plates import InternationalLicensePlateRecognizer
    INTERNATIONAL_PLATES_AVAILABLE = True
except ImportError:
    INTERNATIONAL_PLATES_AVAILABLE = False
    print("[WARNING] International plates not available")

class CarPlateVideoProcessor:
    """
    Enhanced video processor for car detection and license plate recognition
    """
    
    def __init__(self, model_path: str = "yolo26n.pt", use_gpu: bool = True):
        """
        Initialize the processor
        
        Args:
            model_path: Path to YOLO model
            use_gpu: Whether to use GPU acceleration
        """
        self.model_path = model_path
        self.use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.model = None
        self.plate_recognizer = None
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'cars_detected': 0,
            'plates_found': 0,
            'unique_plates': set(),
            'processing_time': 0
        }
        
        # Plate tracking for stable detection across frames
        self.plate_trackers = {}
        self.tracked_plates = {}
        self.next_plate_id = 1
        self.tracking_timeout = 30  # frames to keep plate without seeing it
        
        # Initialize components
        self._initialize_model()
        self._initialize_plate_recognizer()
        
        print(f"[INFO] Car Plate Video Processor initialized")
        print(f"[INFO] Model: {model_path}")
        print(f"[INFO] GPU: {'Enabled' if self.use_gpu else 'Disabled'}")
        print(f"[INFO] OCR: {'Available' if PADDLEOCR_AVAILABLE else 'Not Available'}")
    
    def _initialize_model(self):
        """Initialize YOLO model"""
        if not YOLO_AVAILABLE:
            print("[ERROR] YOLO not available")
            return
        
        try:
            self.model = YOLO(self.model_path)
            
            if self.use_gpu:
                device = "cuda:0"
                self.model.to(device)
                print(f"[INFO] Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("[INFO] Model loaded on CPU")
                
            self.device = device
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model = None
    
    def _initialize_plate_recognizer(self):
        """Initialize license plate recognizer"""
        if INTERNATIONAL_PLATES_AVAILABLE:
            try:
                self.plate_recognizer = InternationalLicensePlateRecognizer()
                print("[INFO] International license plate recognizer initialized")
            except Exception as e:
                print(f"[WARNING] Failed to initialize plate recognizer: {e}")
                self.plate_recognizer = None
    
    def process_video(self, video_path: str, output_path: str = None, 
                     show_realtime: bool = True, save_frames: bool = True) -> Dict:
        """
        Process video for car detection and license plate recognition
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            show_realtime: Whether to show real-time processing
            save_frames: Whether to save frames with detections
            
        Returns:
            Processing results with all detected plates
        """
        if not self.model:
            return {'error': 'Model not loaded'}
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'Cannot open video: {video_path}'}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Setup output video
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"car_plate_output_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return {'error': 'Cannot create output video writer'}
        
        # Processing variables
        start_time = time.time()
        frame_count = 0
        all_detections = []
        saved_frames = []
        
        print("[INFO] Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            frame_result = self._process_frame(frame, frame_count)
            all_detections.append(frame_result)
            
            # Create annotated frame
            annotated_frame = self._create_annotated_frame(frame, frame_result)
            
            # Write to output
            out.write(annotated_frame)
            
            # Save frames with detections
            if save_frames and frame_result['cars_detected'] > 0:
                frame_path = self._save_frame(annotated_frame, frame_count, frame_result)
                if frame_path:
                    saved_frames.append(frame_path)
            
            # Show real-time display
            if show_realtime:
                cv2.imshow('Car & Plate Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Processing stopped by user")
                    break
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                eta = (total_frames - frame_count) / fps_current if fps_current > 0 else 0
                
                print(f"[INFO] Processed {frame_count}/{total_frames} ({progress:.1f}%) - "
                      f"{fps_current:.1f} FPS - ETA: {eta:.1f}s")
        
        # Cleanup
        cap.release()
        out.release()
        if show_realtime:
            cv2.destroyAllWindows()
        
        # Calculate final statistics
        processing_time = time.time() - start_time
        self.stats['processing_time'] = processing_time
        self.stats['total_frames'] = frame_count
        
        # Create results summary
        results = self._create_results_summary(all_detections, saved_frames, output_path, processing_time)
        
        print(f"[INFO] Processing completed in {processing_time:.1f}s")
        print(f"[INFO] Cars detected: {self.stats['cars_detected']}")
        print(f"[INFO] Plates found: {self.stats['plates_found']}")
        print(f"[INFO] Unique plates: {len(self.stats['unique_plates'])}")
        print(f"[INFO] Output saved: {output_path}")
        
        return results
    
    def _process_frame(self, frame: np.ndarray, frame_number: int) -> Dict:
        """Process single frame for car detection and plate recognition"""
        result = {
            'frame_number': frame_number,
            'cars_detected': 0,
            'cars': [],
            'plates_found': 0,
            'plates': [],
            'processing_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Detect objects
            detections = self.model.predict(
                source=frame,
                conf=0.5,
                iou=0.5,
                device=self.device,
                verbose=False
            )
            
            if detections and len(detections) > 0:
                detection = detections[0]
                
                if hasattr(detection, 'boxes') and detection.boxes is not None:
                    boxes = detection.boxes
                    
                    print(f"[DEBUG] Frame {frame_number}: Found {len(boxes)} detections")
                    
                    # Process each detection
                    for i in range(len(boxes)):
                        # Get bounding box
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = detection.names.get(class_id, f"class_{class_id}")
                        
                        # Check if it's a vehicle
                        if self._is_vehicle(class_name):
                            print(f"[DEBUG] Processing vehicle: {class_name} at ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
                            
                            car_info = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class_name': class_name,
                                'plates': []
                            }
                            
                            # Extract license plates from car region
                            plates = self._extract_plates_from_region(frame, [int(x1), int(y1), int(x2), int(y2)])
                            
                            print(f"[DEBUG] Found {len(plates)} plates for this vehicle")
                            
                            # Track plates for stable detection
                            stable_plates = self._track_plates(plates, [int(x1), int(y1), int(x2), int(y2)], frame_number)
                            car_info['plates'] = stable_plates
                            
                            result['cars'].append(car_info)
                            result['cars_detected'] += 1
                            result['plates_found'] += len(plates)
                            
                            # Add to global stats
                            self.stats['cars_detected'] += 1
                            self.stats['plates_found'] += len(plates)
                            
                            for plate in plates:
                                self.stats['unique_plates'].add(plate['text'])
                                print(f"[DEBUG] Added plate to stats: '{plate['text']}'")
                        else:
                            if i < 5:  # Only print first 5 non-vehicle detections
                                print(f"[DEBUG] Skipping non-vehicle: {class_name}")
                else:
                    print(f"[DEBUG] Frame {frame_number}: No boxes found in detection")
            else:
                print(f"[DEBUG] Frame {frame_number}: No detections")
        
        except Exception as e:
            print(f"[ERROR] Frame {frame_number} processing failed: {e}")
            import traceback
            traceback.print_exc()
        
        result['processing_time'] = time.time() - start_time
        
        if result['plates_found'] > 0:
            print(f"[DEBUG] Frame {frame_number} result: {result['cars_detected']} cars, {result['plates_found']} plates")
        
        return result
    
    def _is_vehicle(self, class_name: str) -> bool:
        """Check if detected class is a vehicle"""
        vehicle_classes = {
            'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'van',
            'taxi', 'ambulance', 'police', 'fire truck', 'tractor',
            'scooter', 'bike', 'auto', 'rickshaw', 'lorry', 'suv'
        }
        return class_name.lower() in vehicle_classes
    
    def _extract_plates_from_region(self, frame: np.ndarray, bbox: List[int]) -> List[Dict]:
        """
        Extract license plates from a specific region with multi-angle support.
        Handles plates at various angles by trying multiple rotations.
        """
        plates = []
        
        if not PADDLEOCR_AVAILABLE:
            return plates
        
        x1, y1, x2, y2 = bbox
        
        # Extract region
        region = frame[y1:y2, x1:x2]
        
        if region.size == 0:
            return plates
        
        try:
            # First attempt: Try OCR on original region
            plate = self._try_ocr_on_region(region, bbox)
            if plate:
                plates.append(plate)
                return plates
            
            # Second attempt: Try multiple angles for rotated plates
            angles = [-45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, -60]
            print(f"[DEBUG] Trying {len(angles)} rotation angles...")
            for angle in angles:
                rotated_region = self._rotate_image(region, angle)
                if rotated_region is None or rotated_region.size == 0:
                    continue
                
                plate = self._try_ocr_on_region(rotated_region, bbox, angle_hint=angle)
                if plate:
                    plate['angle'] = angle
                    plates.append(plate)
                    print(f"[DEBUG] ✅ Found plate at angle {angle}: '{plate['text']}'")
                    return plates
            
            # Third attempt: Try perspective correction if plate appears skewed
            # Look for rectangular contours that might be plates
            warped_regions = self._find_plate_candidates(region)
            for i, warped in enumerate(warped_regions):
                if warped is None or warped.size == 0:
                    continue
                plate = self._try_ocr_on_region(warped, bbox, method='perspective')
                if plate:
                    plate['method'] = 'perspective_correction'
                    plates.append(plate)
                    return plates
            
            # Fourth attempt: Try different preprocessing on original
            preprocessed_variants = self._create_preprocessing_variants(region)
            for variant_name, variant_img in preprocessed_variants.items():
                if variant_img is None or variant_img.size == 0:
                    continue
                plate = self._try_ocr_on_region(variant_img, bbox, method=variant_name)
                if plate:
                    plate['method'] = f'preprocessing_{variant_name}'
                    plates.append(plate)
                    return plates
                    
        except Exception as e:
            print(f"[DEBUG] Multi-angle plate extraction failed: {e}")
        
        return plates
    
    def _try_ocr_on_region(self, region: np.ndarray, bbox: List[int], 
                          angle_hint: int = 0, method: str = 'direct') -> Optional[Dict]:
        """Try OCR on a region and validate if it's a license plate."""
        try:
            # Try multiple OCR methods
            text = None
            confidence = 0.0
            device = 'CPU'
            
            # Method 1: Try Tesseract first (more reliable for angled plates)
            try:
                import pytesseract
                
                # Preprocess for Tesseract
                if len(region.shape) == 3:
                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                else:
                    gray = region
                
                # Enhance contrast
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                
                # Apply threshold
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Try multiple Tesseract configs
                configs = [
                    r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    r'--oem 3 --psm 7',
                    r'--oem 3 --psm 8',
                    r'--oem 3 --psm 6',
                ]
                
                for config in configs:
                    tess_text = pytesseract.image_to_string(binary, config=config)
                    tess_clean = tess_text.strip().upper()
                    
                    if tess_clean and len(tess_clean) >= 3:
                        # Validate it looks like a plate
                        has_letters = sum(c.isalpha() for c in tess_clean) >= 1
                        has_numbers = sum(c.isdigit() for c in tess_clean) >= 1
                        
                        if has_letters and has_numbers:
                            text = tess_clean
                            confidence = 0.85
                            device = 'Tesseract'
                            print(f"[DEBUG] ✅ Tesseract found plate: '{text}' (config: {config})")
                            break
                
                if text:
                    plate_info = {
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox,
                        'method': f'tesseract_{method}',
                        'device': device,
                        'angle_hint': angle_hint
                    }
                    
                    # Try to identify country
                    if self.plate_recognizer:
                        country = self._identify_plate_country(text)
                        if country:
                            plate_info['country'] = country
                    
                    print(f"[DEBUG] ✅ License plate detected: '{text}' (method: {method}, angle: {angle_hint})")
                    return plate_info
                    
            except Exception as e:
                print(f"[DEBUG] Tesseract failed: {e}")
            
            # Method 2: Use PaddleOCR as fallback
            if PADDLEOCR_AVAILABLE and text is None:
                ocr_result = extract_text_optimized(
                    region,
                    confidence_threshold=0.15,  # Very low threshold for difficult plates
                    lang='en',
                    use_gpu=self.use_gpu,
                    use_cache=False,
                    preprocess=True
                )
                
                if ocr_result and ocr_result.get('text'):
                    paddle_text = ocr_result['text'].strip()
                    paddle_conf = ocr_result.get('confidence', 0.0)
                    
                    # Check if it looks like a license plate (relaxed validation)
                    if self._is_license_plate_relaxed(paddle_text):
                        plate_info = {
                            'text': paddle_text,
                            'confidence': paddle_conf,
                            'bbox': bbox,
                            'method': f'paddleocr_{method}',
                            'device': 'GPU' if self.use_gpu else 'CPU',
                            'angle_hint': angle_hint
                        }
                        
                        # Try to identify country
                        if self.plate_recognizer:
                            country = self._identify_plate_country(paddle_text)
                            if country:
                                plate_info['country'] = country
                        
                        print(f"[DEBUG] ✅ License plate detected: '{paddle_text}' (method: {method}, angle: {angle_hint})")
                        return plate_info
                    else:
                        print(f"[DEBUG] ❌ PaddleOCR text rejected: '{paddle_text}' - not a valid plate format")
                        
        except Exception as e:
            print(f"[DEBUG] OCR attempt failed: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def _is_license_plate_relaxed(self, text: str) -> bool:
        """
        Relaxed license plate validation for angled/partial detections.
        Very lenient to catch more plates.
        """
        if not text or not isinstance(text, str):
            return False
        
        # Clean text
        clean_text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
        
        print(f"[DEBUG] Validating plate text: '{clean_text}' (original: '{text}')")
        
        # Relaxed length check (2-12 chars)
        if len(clean_text) < 2 or len(clean_text) > 12:
            print(f"[DEBUG] ❌ Text length {len(clean_text)} not in range [2, 12]")
            return False
        
        # Must have at least 1 letter OR 1 number (very relaxed)
        has_letters = sum(c.isalpha() for c in clean_text) >= 1
        has_numbers = sum(c.isdigit() for c in clean_text) >= 1
        
        print(f"[DEBUG] Has letters: {has_letters}, Has numbers: {has_numbers}")
        
        # Accept if has at least one letter AND one number
        if has_letters and has_numbers:
            print(f"[DEBUG] ✅ Accepted as plate: '{clean_text}'")
            return True
        
        # Also accept pure alphanumeric that looks like a plate (e.g., "ABC123")
        if clean_text.isalnum() and len(clean_text) >= 4:
            print(f"[DEBUG] ✅ Accepted alphanumeric as plate: '{clean_text}'")
            return True
        
        print(f"[DEBUG] ❌ Rejected: '{clean_text}'")
        return False
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle.
        """
        if image is None or image.size == 0:
            return image
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix for new center
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Rotate with black background
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
        
        return rotated
    
    def _find_plate_candidates(self, region: np.ndarray) -> List[np.ndarray]:
        """
        Find potential license plate regions by looking for rectangular contours.
        Returns list of warped (perspective-corrected) regions.
        """
        warped_regions = []
        
        try:
            # Convert to grayscale
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that look like plates
            for contour in contours:
                # Approximate contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Look for rectangles (4 corners)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Aspect ratio check (plates are typically 2-5 times wider than tall)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if 2.0 <= aspect_ratio <= 6.0 and w > 50 and h > 15:
                        # Try to warp perspective
                        try:
                            pts = approx.reshape(4, 2)
                            rect = np.zeros((4, 2), dtype="float32")
                            
                            # Order points: top-left, top-right, bottom-right, bottom-left
                            s = pts.sum(axis=1)
                            rect[0] = pts[np.argmin(s)]
                            rect[2] = pts[np.argmax(s)]
                            
                            diff = np.diff(pts, axis=1)
                            rect[1] = pts[np.argmin(diff)]
                            rect[3] = pts[np.argmax(diff)]
                            
                            # Compute width and height
                            widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
                            widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
                            maxWidth = max(int(widthA), int(widthB))
                            
                            heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
                            heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
                            maxHeight = max(int(heightA), int(heightB))
                            
                            if maxWidth > 50 and maxHeight > 15:
                                # Destination points
                                dst = np.array([
                                    [0, 0],
                                    [maxWidth - 1, 0],
                                    [maxWidth - 1, maxHeight - 1],
                                    [0, maxHeight - 1]], dtype="float32")
                                
                                # Perspective transform
                                M = cv2.getPerspectiveTransform(rect, dst)
                                warped = cv2.warpPerspective(region, M, (maxWidth, maxHeight))
                                
                                if warped.size > 0:
                                    warped_regions.append(warped)
                        except Exception as e:
                            continue
            
            # Also add the original region scaled to different sizes
            for scale in [0.8, 1.2, 1.5]:
                scaled = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                warped_regions.append(scaled)
                
        except Exception as e:
            print(f"[DEBUG] Plate candidate finding failed: {e}")
        
        return warped_regions
    
    def _create_preprocessing_variants(self, region: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create different preprocessing variants of the region for better OCR.
        """
        variants = {}
        
        try:
            # Convert to grayscale if needed
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            
            # Variant 1: High contrast
            _, high_contrast = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants['high_contrast'] = cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2BGR)
            
            # Variant 2: Adaptive threshold
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            variants['adaptive'] = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
            
            # Variant 3: Denoised
            denoised = cv2.fastNlMeansDenoisingColored(region, None, 10, 10, 7, 21)
            variants['denoised'] = denoised
            
            # Variant 4: Sharpened
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(region, -1, kernel)
            variants['sharpened'] = sharpened
            
            # Variant 5: Upscaled
            upscaled = cv2.resize(region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            variants['upscaled'] = upscaled
            
            # Variant 6: Grayscale
            variants['grayscale'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
        except Exception as e:
            print(f"[DEBUG] Preprocessing variants failed: {e}")
        
        return variants
    
    def _is_license_plate(self, text: str) -> bool:
        """
        Check if text looks like a license plate.
        Enhanced to handle various formats including short plates like NH320.
        """
        if not text or not isinstance(text, str):
            return False
        
        # Clean text - remove spaces and special characters
        clean_text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
        
        # Relaxed length check (3-10 chars for various plate formats)
        if len(clean_text) < 3 or len(clean_text) > 10:
            return False
        
        # Must have at least 1 letter and 1 number (even for short plates)
        has_letters = sum(c.isalpha() for c in clean_text) >= 1
        has_numbers = sum(c.isdigit() for c in clean_text) >= 1
        
        # Special case: Short plates like NH320 (2 letters + 3 numbers)
        # This matches Indian, European, and many international formats
        if len(clean_text) <= 6:
            # At least one letter and one number
            return has_letters and has_numbers
        
        # For longer plates, require at least 2 of each
        if len(clean_text) > 6:
            has_2_letters = sum(c.isalpha() for c in clean_text) >= 2
            has_2_numbers = sum(c.isdigit() for c in clean_text) >= 2
            return has_2_letters and has_2_numbers
        
        return has_letters and has_numbers
    
    def _identify_plate_country(self, text: str) -> Optional[str]:
        """Identify the country of a license plate"""
        if not self.plate_recognizer:
            return None
        
        try:
            # This would use the international plate recognizer
            # For now, return a simple guess based on patterns
            clean_text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
            
            # Simple pattern matching (can be enhanced)
            if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$', clean_text):
                return 'UK'
            elif re.match(r'^[A-Z]{3}[0-9]{3}$', clean_text):
                return 'Canada'
            elif re.match(r'^[A-Z]{3}[ -]?[0-9]{1,4}$', clean_text):
                return 'USA'
            
        except Exception as e:
            print(f"[DEBUG] Country identification failed: {e}")
        
        return None
    
    def _track_plates(self, plates: List[Dict], car_bbox: List[int], frame_number: int) -> List[Dict]:
        """Track plates for stable detection"""
        tracked_plates = []
        
        for plate in plates:
            plate_text = plate['text']
            plate_bbox = plate['bbox']
            plate_confidence = plate['confidence']
            
            # Check if plate is already tracked
            if plate_text in self.plate_trackers:
                tracker = self.plate_trackers[plate_text]
                
                # Update tracker with new detection
                tracker['detections'].append({
                    'frame_number': frame_number,
                    'bbox': plate_bbox,
                    'confidence': plate_confidence
                })
                
                # Calculate stable plate text and confidence
                stable_text = tracker['text']
                stable_confidence = sum(d['confidence'] for d in tracker['detections']) / len(tracker['detections'])
                
                # Create stable plate info
                stable_plate = {
                    'stable_text': stable_text,
                    'confidence': stable_confidence,
                    'bbox': plate_bbox,
                    'method': 'tracked',
                    'device': 'GPU' if self.use_gpu else 'CPU'
                }
                
                # Try to identify country if international recognizer is available
                if self.plate_recognizer:
                    country = self._identify_plate_country(stable_text)
                    if country:
                        stable_plate['country'] = country
                
                tracked_plates.append(stable_plate)
            else:
                # Create new tracker for plate
                self.plate_trackers[plate_text] = {
                    'text': plate_text,
                    'detections': [{
                        'frame_number': frame_number,
                        'bbox': plate_bbox,
                        'confidence': plate_confidence
                    }]
                }
                
                # Add plate to tracked plates
                tracked_plates.append(plate)
        
        return tracked_plates
    
    def _create_annotated_frame(self, frame: np.ndarray, frame_result: Dict) -> np.ndarray:
        """Create annotated frame with detections"""
        annotated = frame.copy()
        
        print(f"[DEBUG] Annotating frame {frame_result['frame_number']}: {frame_result['cars_detected']} cars, {frame_result['plates_found']} plates")
        
        # Draw cars and their plates
        for car_idx, car in enumerate(frame_result['cars']):
            bbox = car['bbox']
            x1, y1, x2, y2 = bbox
            
            print(f"[DEBUG] Car {car_idx}: {car['class_name']} with {len(car['plates'])} plates")
            
            # Draw car bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add car label with BIGGER font
            car_label = f"{car['class_name']} ({car['confidence']:.2f})"
            cv2.putText(annotated, car_label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            # Draw license plates - IMPROVED: Better formatting with yellow box and BIGGER text
            for i, plate in enumerate(car['plates']):
                plate_text = plate.get('stable_text', plate.get('text', ''))
                confidence = plate.get('confidence', 0)
                country = plate.get('country', '')
                
                print(f"[DEBUG] Drawing plate {i}: '{plate_text}' (conf: {confidence:.2f})")
                
                # Get plate bbox if available
                plate_bbox = plate.get('bbox')
                if plate_bbox:
                    px1, py1, px2, py2 = plate_bbox
                    # Draw yellow box around license plate region
                    cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 255), 4)
                
                # Position for plate text label
                plate_y = y1 - 50 - (i * 40)
                
                # Create plate label with "Plate: " prefix
                if country:
                    plate_label = f"Plate: {plate_text} ({confidence:.2f}) [{country}]"
                else:
                    plate_label = f"Plate: {plate_text} ({confidence:.2f})"
                
                # Draw background rectangle for plate text - BIGGER
                (text_width, text_height), _ = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
                cv2.rectangle(annotated, (x1, plate_y - text_height - 5), 
                             (x1 + text_width + 10, plate_y + 10), (0, 255, 255), -1)
                cv2.putText(annotated, plate_label, (x1 + 5, plate_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
        
        # Add frame info with BIGGER font
        frame_text = f"Frame: {frame_result['frame_number']} | Cars: {frame_result['cars_detected']} | Plates: {frame_result['plates_found']}"
        cv2.putText(annotated, frame_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Add processing time with BIGGER font
        time_text = f"Time: {frame_result['processing_time']:.3f}s"
        cv2.putText(annotated, time_text, (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Add unique plates count with BIGGER font
        unique_text = f"Unique Plates: {len(self.stats['unique_plates'])}"
        cv2.putText(annotated, unique_text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        return annotated
    
    def _save_frame(self, frame: np.ndarray, frame_number: int, frame_result: Dict) -> Optional[str]:
        """Save frame with detections"""
        try:
            # Create output directory
            output_dir = "detected_frames"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save frame
            filename = f"frame_{frame_number:06d}_cars_{frame_result['cars_detected']}_plates_{frame_result['plates_found']}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            
            return filepath
        except Exception as e:
            print(f"[ERROR] Failed to save frame {frame_number}: {e}")
            return None
    
    def _create_results_summary(self, all_detections: List[Dict], saved_frames: List[str], 
                              output_path: str, processing_time: float) -> Dict:
        """Create comprehensive results summary"""
        
        # Collect all plates
        all_plates = []
        for detection in all_detections:
            for car in detection['cars']:
                for plate in car['plates']:
                    plate_copy = plate.copy()
                    plate_copy['frame_number'] = detection['frame_number']
                    all_plates.append(plate_copy)
        
        # Create summary
        results = {
            'video_info': {
                'output_path': output_path,
                'processing_time': processing_time,
                'total_frames': len(all_detections),
                'fps_processed': len(all_detections) / processing_time if processing_time > 0 else 0
            },
            'detection_summary': {
                'total_cars_detected': self.stats['cars_detected'],
                'total_plates_found': self.stats['plates_found'],
                'unique_plates': list(self.stats['unique_plates']),
                'unique_plates_count': len(self.stats['unique_plates']),
                'frames_with_detections': len([d for d in all_detections if d['cars_detected'] > 0])
            },
            'all_plates': all_plates,
            'saved_frames': saved_frames,
            'plates_by_frame': {},
            'most_common_plates': []
        }
        
        # Group plates by frame
        for plate in all_plates:
            frame_num = plate['frame_number']
            if frame_num not in results['plates_by_frame']:
                results['plates_by_frame'][frame_num] = []
            results['plates_by_frame'][frame_num].append(plate)
        
        # Find most common plates
        plate_counts = {}
        for plate in all_plates:
            text = plate['text']
            plate_counts[text] = plate_counts.get(text, 0) + 1
        
        results['most_common_plates'] = sorted(
            plate_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]  # Top 10 most common plates
        
        # Save results to JSON
        json_path = output_path.replace('.mp4', '_results.json')
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Results saved to: {json_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save results JSON: {e}")
        
        return results

# Convenience function for easy usage
def process_video_for_cars_and_plates(video_path: str, model_path: str = "yolo26n.pt", 
                                    output_path: str = None, show_realtime: bool = True) -> Dict:
    """
    Process video for car detection and license plate recognition
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLO model
        output_path: Path to save output video
        show_realtime: Whether to show real-time processing
        
    Returns:
        Processing results with all detected plates
    """
    processor = CarPlateVideoProcessor(model_path=model_path, use_gpu=True)
    return processor.process_video(video_path, output_path, show_realtime)

if __name__ == "__main__":
    print("🚗 Car & License Plate Video Processor")
    print("=" * 50)
    
    # Example usage
    video_path = "input_video.mp4"  # Replace with actual video path
    
    print("📖 Usage:")
    print("   from car_plate_video_processor import process_video_for_cars_and_plates")
    print("   results = process_video_for_cars_and_plates('video.mp4')")
    print("   print(f'Found {len(results[\"unique_plates\"])} unique plates')")
    
    print("\n✅ Processor ready!")
    print("   Features:")
    print("   - Real-time car detection")
    print("   - License plate recognition")
    print("   - International plate support")
    print("   - GPU acceleration")
    print("   - Frame saving with detections")
    print("   - Comprehensive results export")
