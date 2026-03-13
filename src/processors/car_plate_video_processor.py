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
                    
                    # Process each detection
                    for i in range(len(boxes)):
                        # Get bounding box
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = detection.names.get(class_id, f"class_{class_id}")
                        
                        # Check if it's a vehicle
                        if self._is_vehicle(class_name):
                            car_info = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class_name': class_name,
                                'plates': []
                            }
                            
                            # Extract license plates from car region
                            plates = self._extract_plates_from_region(frame, [int(x1), int(y1), int(x2), int(y2)])
                            car_info['plates'] = plates
                            
                            result['cars'].append(car_info)
                            result['cars_detected'] += 1
                            result['plates_found'] += len(plates)
                            
                            # Add to global stats
                            self.stats['cars_detected'] += 1
                            self.stats['plates_found'] += len(plates)
                            
                            for plate in plates:
                                self.stats['unique_plates'].add(plate['text'])
        
        except Exception as e:
            print(f"[ERROR] Frame {frame_number} processing failed: {e}")
        
        result['processing_time'] = time.time() - start_time
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
        """Extract license plates from a specific region"""
        plates = []
        
        if not PADDLEOCR_AVAILABLE:
            return plates
        
        x1, y1, x2, y2 = bbox
        
        # Extract region
        region = frame[y1:y2, x1:x2]
        
        if region.size == 0:
            return plates
        
        try:
            # Use PaddleOCR to extract text
            ocr_result = extract_text_optimized(
                region,
                confidence_threshold=0.3,
                lang='en',
                use_gpu=self.use_gpu,
                use_cache=False,
                preprocess=True
            )
            
            if ocr_result and ocr_result.get('text'):
                text = ocr_result['text'].strip()
                confidence = ocr_result.get('confidence', 0.0)
                
                # Check if it looks like a license plate
                if self._is_license_plate(text):
                    plate_info = {
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox,
                        'method': 'paddleocr',
                        'device': 'GPU' if self.use_gpu else 'CPU'
                    }
                    
                    # Try to identify country if international recognizer is available
                    if self.plate_recognizer:
                        country = self._identify_plate_country(text)
                        if country:
                            plate_info['country'] = country
                    
                    plates.append(plate_info)
        
        except Exception as e:
            print(f"[DEBUG] OCR extraction failed: {e}")
        
        return plates
    
    def _is_license_plate(self, text: str) -> bool:
        """Check if text looks like a license plate"""
        # Remove spaces and special characters
        clean_text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
        
        # License plates typically have 5-10 characters
        if len(clean_text) < 4 or len(clean_text) > 12:
            return False
        
        # Check if it has both letters and numbers (common for plates)
        has_letters = bool(re.search(r'[A-Z]', clean_text))
        has_numbers = bool(re.search(r'[0-9]', clean_text))
        
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
    
    def _create_annotated_frame(self, frame: np.ndarray, frame_result: Dict) -> np.ndarray:
        """Create annotated frame with detections"""
        annotated = frame.copy()
        
        # Draw cars and their plates
        for car in frame_result['cars']:
            bbox = car['bbox']
            x1, y1, x2, y2 = bbox
            
            # Draw car bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add car label
            car_label = f"{car['class_name']} ({car['confidence']:.2f})"
            cv2.putText(annotated, car_label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw license plates
            for i, plate in enumerate(car['plates']):
                plate_text = plate['text']
                confidence = plate['confidence']
                country = plate.get('country', '')
                
                # Position for plate text
                plate_y = y1 - 30 - (i * 25)
                
                # Create plate label
                if country:
                    plate_label = f"🚗 {plate_text} ({confidence:.2f}) [{country}]"
                else:
                    plate_label = f"🚗 {plate_text} ({confidence:.2f})"
                
                # Draw background rectangle for plate text
                (text_width, text_height), _ = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated, (x1, plate_y - text_height - 5), 
                             (x1 + text_width + 5, plate_y + 5), (0, 0, 255), -1)
                cv2.putText(annotated, plate_label, (x1, plate_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add frame info
        frame_text = f"Frame: {frame_result['frame_number']} | Cars: {frame_result['cars_detected']} | Plates: {frame_result['plates_found']}"
        cv2.putText(annotated, frame_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add processing time
        time_text = f"Time: {frame_result['processing_time']:.3f}s"
        cv2.putText(annotated, time_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add unique plates count
        unique_text = f"Unique Plates: {len(self.stats['unique_plates'])}"
        cv2.putText(annotated, unique_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
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
