"""
Enhanced License Plate Detector for International Plates
Supports Japanese, English, and multi-language license plates
"""

import cv2
import numpy as np
import os
import time
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Try to import OCR libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    # Set Tesseract path for Windows
    if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

class EnhancedPlateDetector:
    """Enhanced license plate detector with multi-language support"""
    
    def __init__(self):
        self.ocr_engines = []
        self._initialize_ocr_engines()
        
    def _initialize_ocr_engines(self):
        """Initialize available OCR engines"""
        
        # Initialize PaddleOCR
        if PADDLEOCR_AVAILABLE:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
                self.ocr_engines.append('paddleocr')
                print("✅ PaddleOCR initialized")
            except Exception as e:
                print(f"❌ PaddleOCR failed: {e}")
        
        # Initialize Tesseract
        if TESSERACT_AVAILABLE:
            try:
                # Test Tesseract
                test_img = np.zeros((100, 200, 3), dtype=np.uint8)
                cv2.putText(test_img, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                pytesseract.image_to_string(test_img)
                self.ocr_engines.append('tesseract')
                print("✅ Tesseract initialized")
            except Exception as e:
                print(f"❌ Tesseract failed: {e}")
        
        print(f"🔧 Available OCR engines: {self.ocr_engines}")
    
    def extract_text_from_image(self, image: np.ndarray) -> List[Dict]:
        """Extract text using all available OCR engines"""
        results = []
        
        for engine in self.ocr_engines:
            try:
                if engine == 'paddleocr':
                    text_results = self._extract_with_paddleocr(image)
                elif engine == 'tesseract':
                    text_results = self._extract_with_tesseract(image)
                else:
                    continue
                
                results.extend(text_results)
                
            except Exception as e:
                print(f"❌ {engine} failed: {e}")
        
        return results
    
    def _extract_with_paddleocr(self, image: np.ndarray) -> List[Dict]:
        """Extract text using PaddleOCR"""
        try:
            results = self.paddle_ocr.ocr(image, cls=True)
            
            extracted_texts = []
            if results and results[0]:
                for line in results[0]:
                    if line:
                        bbox, (text, confidence) = line
                        if confidence > 0.3:  # Lower threshold for better detection
                            extracted_texts.append({
                                'text': text.strip(),
                                'confidence': confidence,
                                'engine': 'paddleocr',
                                'bbox': bbox
                            })
            
            return extracted_texts
            
        except Exception as e:
            print(f"❌ PaddleOCR extraction failed: {e}")
            return []
    
    def _extract_with_tesseract(self, image: np.ndarray) -> List[Dict]:
        """Extract text using Tesseract with multiple languages"""
        try:
            extracted_texts = []
            
            # Try different language configurations
            languages = ['eng', 'jpn', 'eng+jpn']  # English, Japanese, both
            
            for lang in languages:
                try:
                    # Configure Tesseract for better plate detection
                    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789日本'
                    
                    text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
                    confidence = pytesseract.image_to_data(image, lang=lang, config=custom_config, output_type=pytesseract.Output.DICT)
                    
                    if text.strip():
                        # Get average confidence
                        confidences = [conf for conf in confidence['conf'] if conf > 0]
                        avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.5
                        
                        extracted_texts.append({
                            'text': text.strip(),
                            'confidence': avg_confidence,
                            'engine': f'tesseract_{lang}',
                            'bbox': None
                        })
                
                except Exception as e:
                    continue  # Try next language
            
            return extracted_texts
            
        except Exception as e:
            print(f"❌ Tesseract extraction failed: {e}")
            return []
    
    def is_license_plate(self, text: str) -> bool:
        """Enhanced license plate validation for international formats"""
        if not text or len(text.strip()) < 3:
            return False
        
        text = text.strip().upper()
        
        # Japanese plate patterns
        japanese_patterns = [
            r'^[A-Z0-9]{2,6}$',  # Standard Japanese
            r'^日本\s*[0-9]{1,4}\s*[A-Z0-9]{2,4}$',  # Japanese with characters
            r'^[ひらがなカタカナ]{1,4}\s*[0-9]{1,4}\s*[A-Z0-9]{2,4}$',  # Hiragana/Katakana
        ]
        
        # International patterns
        international_patterns = [
            r'^[A-Z]{2,3}[ -]?[0-9]{2,4}[ -]?[A-Z]{0,3}$',  # Most countries
            r'^[0-9]{2,4}[ -]?[A-Z]{2,3}[ -]?[0-9]{0,2}$',  # Some formats
            r'^[A-Z0-9]{4,8}$',  # Simple alphanumeric
        ]
        
        all_patterns = japanese_patterns + international_patterns
        
        for pattern in all_patterns:
            if re.match(pattern, text):
                return True
        
        # Additional check: contains both letters and numbers (common for plates)
        has_letters = bool(re.search(r'[A-Z]', text))
        has_numbers = bool(re.search(r'[0-9]', text))
        has_japanese = bool(re.search(r'[日本ひらがなカタカナ]', text))
        
        return (has_letters and has_numbers) or has_japanese
    
    def preprocess_plate_region(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Denoise
            denoised = cv2.medianBlur(binary, 3)
            
            # Enhance contrast
            enhanced = cv2.convertScaleAbs(denoised, alpha=1.5, beta=10)
            
            return enhanced
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return image
    
    def detect_license_plates_in_frame(self, frame: np.ndarray, car_bboxes: List[List[int]]) -> List[Dict]:
        """Detect license plates in car regions"""
        plates = []
        
        for bbox in car_bboxes:
            x1, y1, x2, y2 = bbox
            
            # Extract car region
            car_region = frame[y1:y2, x1:x2]
            
            if car_region.size == 0:
                continue
            
            # Try to find license plate within car region
            plate_texts = self._find_plates_in_region(car_region, bbox)
            plates.extend(plate_texts)
        
        return plates
    
    def _find_plates_in_region(self, region: np.ndarray, car_bbox: List[int]) -> List[Dict]:
        """Find license plates in a specific region"""
        plates = []
        
        # Method 1: Try to detect rectangular regions (potential plates)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours for plate-like shapes
        plate_candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # License plates typically have aspect ratio between 2:1 and 5:1
            if 1.5 < aspect_ratio < 6.0 and w > 60 and h > 20:
                plate_candidates.append((x, y, w, h))
        
        # Extract text from candidates
        for x, y, w, h in plate_candidates:
            plate_region = region[y:y+h, x:x+w]
            
            # Preprocess for better OCR
            processed = self.preprocess_plate_region(plate_region)
            
            # Extract text
            text_results = self.extract_text_from_image(processed)
            
            for result in text_results:
                text = result['text']
                if self.is_license_plate(text):
                    # Adjust bbox to frame coordinates
                    frame_x = car_bbox[0] + x
                    frame_y = car_bbox[1] + y
                    
                    plates.append({
                        'text': text,
                        'confidence': result['confidence'],
                        'bbox': [frame_x, frame_y, frame_x + w, frame_y + h],
                        'engine': result['engine'],
                        'method': 'contour_detection'
                    })
        
        # Method 2: If no plates found, try OCR on entire car region
        if not plates:
            processed_region = self.preprocess_plate_region(region)
            text_results = self.extract_text_from_image(processed_region)
            
            for result in text_results:
                text = result['text']
                if self.is_license_plate(text):
                    plates.append({
                        'text': text,
                        'confidence': result['confidence'],
                        'bbox': car_bbox,
                        'engine': result['engine'],
                        'method': 'full_region_ocr'
                    })
        
        return plates

def process_video_with_enhanced_detection(video_path: str, output_path: str = None, show_realtime: bool = True):
    """Process video with enhanced license plate detection"""
    
    print("🚗 Starting Enhanced Car & License Plate Detection")
    print(f"📁 Input video: {video_path}")
    
    if not os.path.exists(video_path):
        return {'error': f'Video file not found: {video_path}'}
    
    if output_path is None:
        timestamp = int(time.time())
        output_path = f"enhanced_plate_detection_{timestamp}.mp4"
    
    try:
        # Initialize enhanced detector
        detector = EnhancedPlateDetector()
        
        # Initialize YOLO for car detection
        from ultralytics import YOLO
        model = YOLO("yolo26n.pt")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Cannot open video'}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing variables
        frame_count = 0
        all_plates = []
        unique_plates = set()
        cars_detected = 0
        
        print("🔄 Starting enhanced processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect cars
            results = model.predict(frame, conf=0.5, verbose=False)
            
            car_bboxes = []
            annotated_frame = frame.copy()
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = result.names.get(class_id, f"class_{class_id}")
                        
                        # Check if it's a vehicle
                        if class_name.lower() in ['car', 'truck', 'bus', 'motorcycle', 'van']:
                            car_bboxes = [int(x1), int(y1), int(x2), int(y2)]
                            cars_detected += 1
                            
                            # Draw car box
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"{class_name} ({confidence:.2f})", 
                                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # Detect license plates in this car
                            plates = detector.detect_license_plates_in_frame(frame, [car_bboxes])
                            
                            for plate in plates:
                                plate_text = plate['text']
                                plate_confidence = plate['confidence']
                                
                                # Add to results
                                all_plates.append({
                                    'text': plate_text,
                                    'confidence': plate_confidence,
                                    'frame_number': frame_count,
                                    'bbox': plate['bbox'],
                                    'engine': plate['engine']
                                })
                                unique_plates.add(plate_text)
                                
                                # Draw plate box and text
                                px1, py1, px2, py2 = plate['bbox']
                                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                                
                                # Create plate label
                                plate_label = f"📋 {plate_text} ({plate_confidence:.2f})"
                                cv2.putText(annotated_frame, plate_label, 
                                           (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add frame info
            cv2.putText(annotated_frame, f"Frame: {frame_count} | Cars: {cars_detected} | Plates: {len(unique_plates)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(annotated_frame)
            
            # Show real-time
            if show_realtime:
                cv2.imshow('Enhanced Plate Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processed {frame_count}/{total_frames} ({progress:.1f}%) - Found {len(unique_plates)} plates")
        
        # Cleanup
        cap.release()
        out.release()
        if show_realtime:
            cv2.destroyAllWindows()
        
        # Create results
        results = {
            'cars_detected': cars_detected,
            'plates_found': len(all_plates),
            'unique_plates': list(unique_plates),
            'all_plates': all_plates,
            'video_info': {
                'input_path': video_path,
                'output_path': output_path,
                'frames_processed': frame_count
            }
        }
        
        # Save detailed results
        json_path = output_path.replace('.mp4', '_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Enhanced processing completed!")
        print(f"📁 Output video: {output_path}")
        print(f"📁 Results JSON: {json_path}")
        print(f"🚗 Cars detected: {cars_detected}")
        print(f"📋 Plates found: {len(all_plates)}")
        print(f"🎯 Unique plates: {len(unique_plates)}")
        
        if unique_plates:
            print("\n📋 Detected License Plates:")
            for i, plate in enumerate(unique_plates, 1):
                print(f"   {i}. {plate}")
        
        return results
        
    except Exception as e:
        print(f"❌ Enhanced processing failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

if __name__ == "__main__":
    # Test with available video
    video_files = [f for f in os.listdir('.') if f.lower().endswith('.mp4')]
    if video_files:
        print(f"Testing with: {video_files[0]}")
        results = process_video_with_enhanced_detection(video_files[0])
        print(f"Results: {results}")
    else:
        print("No video files found for testing")
