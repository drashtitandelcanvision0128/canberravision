"""
Image Processing Module for YOLO26
Handle all image-related operations
Clean, modular, and easy to maintain
"""

import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import our modules
try:
    from simple_plate_detection import extract_license_plates_simple, validate_plate_simple
    from enhanced_detection import enhanced_license_plate_detection
    from international_license_plates import extract_international_license_plates
    from video_output_handler import save_detection_frame
    SIMPLE_PLATE_AVAILABLE = True
    ENHANCED_AVAILABLE = True
    INTERNATIONAL_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Some modules not available: {e}")
    SIMPLE_PLATE_AVAILABLE = False
    ENHANCED_AVAILABLE = False
    INTERNATIONAL_AVAILABLE = False

class ImageProcessor:
    """
    Handle all image processing operations
    """
    
    def __init__(self):
        self.processed_count = 0
        self.start_time = time.time()
        print("[INFO] Image Processor initialized")
    
    def process_image(self, image: np.ndarray, use_enhanced: bool = True, 
                     use_international: bool = True) -> Dict:
        """
        Main image processing function
        
        Args:
            image: Input image in BGR format
            use_enhanced: Use enhanced detection for challenging images
            use_international: Use international plate recognition
            
        Returns:
            Complete processing results
        """
        try:
            start_time = time.time()
            self.processed_count += 1
            
            print(f"[INFO] Processing image #{self.processed_count}")
            
            # Initialize results
            results = {
                'image_info': {
                    'shape': image.shape,
                    'size_mb': image.nbytes / (1024 * 1024),
                    'timestamp': datetime.now().isoformat()
                },
                'detections': {
                    'objects': [],
                    'license_plates': [],
                    'colors': [],
                    'text_found': []
                },
                'processing_info': {
                    'method_used': [],
                    'processing_time': 0,
                    'enhanced_used': False,
                    'international_used': False
                },
                'output_files': {
                    'saved_frames': [],
                    'detection_image': None
                }
            }
            
            # Step 1: Basic YOLO detection
            yolo_results = self._detect_objects_yolo(image)
            results['detections']['objects'] = yolo_results['objects']
            results['processing_info']['method_used'].append('yolo_detection')
            
            # Step 2: License plate detection
            plate_results = self._detect_license_plates(image, use_enhanced)
            results['detections']['license_plates'] = plate_results['plates']
            results['processing_info']['enhanced_used'] = plate_results['enhanced_used']
            
            if plate_results['enhanced_used']:
                results['processing_info']['method_used'].append('enhanced_detection')
            
            # Step 3: Color detection
            color_results = self._detect_colors(image, results['detections']['objects'])
            results['detections']['colors'] = color_results
            results['processing_info']['method_used'].append('color_detection')
            
            # Step 4: Text extraction and recognition
            text_results = self._extract_text_from_plates(image, results['detections']['license_plates'])
            results['detections']['text_found'] = text_results['texts']
            
            if use_international and INTERNATIONAL_AVAILABLE:
                international_results = self._process_international_plates(image, text_results['raw_texts'])
                results['international_plates'] = international_results
                results['processing_info']['international_used'] = True
                results['processing_info']['method_used'].append('international_recognition')
            
            # Step 5: Create annotated image
            annotated_image = self._create_annotated_image(image, results)
            results['output_files']['detection_image'] = annotated_image
            
            # Step 6: Save detection snapshots
            if results['detections']['license_plates']:
                saved_frame = save_detection_frame(
                    annotated_image, 
                    self.processed_count, 
                    results['detections']['license_plates']
                )
                if saved_frame:
                    results['output_files']['saved_frames'].append(saved_frame)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            results['processing_info']['processing_time'] = processing_time
            
            print(f"[INFO] Image processing completed in {processing_time:.2f}s")
            print(f"[INFO] Found {len(results['detections']['license_plates'])} license plates")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Image processing failed: {e}")
            return {'error': str(e)}
    
    def _detect_objects_yolo(self, image: np.ndarray) -> Dict:
        """Detect objects using YOLO"""
        try:
            # Import YOLO from your main app
            from app import get_model
            
            model = get_model("yolo26n.pt")
            detection_results = model(image)
            
            objects = []
            
            if detection_results and len(detection_results) > 0:
                detection = detection_results[0]
                
                if hasattr(detection, 'boxes') and detection.boxes is not None:
                    boxes = detection.boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy()
                    names = detection.names
                    
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = map(int, xyxy[i])
                        confidence = float(conf[i])
                        class_id = int(cls[i])
                        class_name = names.get(class_id, f"class_{class_id}")
                        
                        objects.append({
                            'object_id': f"{class_name}_{i}",
                            'class_name': class_name,
                            'confidence': confidence,
                            'bounding_box': (x1, y1, x2, y2)
                        })
            
            return {'objects': objects}
            
        except Exception as e:
            print(f"[ERROR] YOLO detection failed: {e}")
            return {'objects': []}
    
    def _detect_license_plates(self, image: np.ndarray, use_enhanced: bool = True) -> Dict:
        """Detect license plates with fallback to enhanced detection"""
        plates = []
        enhanced_used = False
        
        try:
            # Method 1: Simple detection
            if SIMPLE_PLATE_AVAILABLE:
                simple_plates = extract_license_plates_simple(image)
                for i, plate_text in enumerate(simple_plates):
                    if validate_plate_simple(plate_text):
                        plates.append({
                            'plate_id': f"simple_plate_{i}",
                            'plate_text': plate_text,
                            'confidence': 0.8,
                            'method': 'simple_detection',
                            'bounding_box': None  # Simple method doesn't provide bbox
                        })
            
            # Method 2: Enhanced detection (if no plates found or enabled)
            if use_enhanced and ENHANCED_AVAILABLE and (not plates or True):
                try:
                    enhanced_result = enhanced_license_plate_detection(image)
                    if enhanced_result.get('plate_detected'):
                        enhanced_used = True
                        
                        # Add enhanced result
                        if enhanced_result.get('plate_text'):
                            plates.append({
                                'plate_id': 'enhanced_plate',
                                'plate_text': enhanced_result['plate_text'],
                                'confidence': 0.9,
                                'method': 'enhanced_detection',
                                'bounding_box': enhanced_result.get('plate_bbox'),
                                'color': enhanced_result.get('color', 'unknown')
                            })
                            
                except Exception as e:
                    print(f"[WARNING] Enhanced detection failed: {e}")
            
        except Exception as e:
            print(f"[ERROR] License plate detection failed: {e}")
        
        return {
            'plates': plates,
            'enhanced_used': enhanced_used
        }
    
    def _detect_colors(self, image: np.ndarray, objects: List[Dict]) -> List[Dict]:
        """Detect colors for objects"""
        colors = []
        
        try:
            for obj in objects:
                if 'bounding_box' in obj:
                    x1, y1, x2, y2 = obj['bounding_box']
                    crop = image[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        color = self._classify_color(crop)
                        colors.append({
                            'object_id': obj['object_id'],
                            'color': color,
                            'method': 'hsv_classification'
                        })
        except Exception as e:
            print(f"[ERROR] Color detection failed: {e}")
        
        return colors
    
    def _classify_color(self, crop: np.ndarray) -> str:
        """Simple color classification"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Define color ranges
            color_ranges = {
                'white': ([0, 0, 200], [180, 30, 255]),
                'black': ([0, 0, 0], [180, 255, 50]),
                'gray': ([0, 0, 50], [180, 30, 150]),
                'red': ([0, 50, 50], [10, 255, 255]),
                'blue': ([100, 50, 50], [130, 255, 255]),
                'green': ([40, 50, 50], [80, 255, 255]),
                'yellow': ([20, 50, 50], [40, 255, 255]),
                'brown': ([8, 50, 50], [20, 255, 255]),
            }
            
            # Count pixels for each color
            color_counts = {}
            total_pixels = hsv.shape[0] * hsv.shape[1]
            
            for color_name, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                count = cv2.countNonZero(mask)
                percentage = (count / total_pixels) * 100
                color_counts[color_name] = percentage
            
            # Find dominant color
            if color_counts:
                dominant_color = max(color_counts, key=color_counts.get)
                if color_counts[dominant_color] > 10:  # Minimum 10%
                    return dominant_color
            
            return "unknown"
            
        except Exception as e:
            print(f"[ERROR] Color classification failed: {e}")
            return "unknown"
    
    def _extract_text_from_plates(self, image: np.ndarray, plates: List[Dict]) -> Dict:
        """Extract text from detected license plates"""
        texts = []
        raw_texts = []
        
        try:
            for plate in plates:
                plate_text = plate.get('plate_text', '')
                if plate_text:
                    texts.append({
                        'plate_id': plate['plate_id'],
                        'text': plate_text,
                        'confidence': plate.get('confidence', 0.8),
                        'method': plate.get('method', 'unknown')
                    })
                    raw_texts.append(plate_text)
        
        except Exception as e:
            print(f"[ERROR] Text extraction failed: {e}")
        
        return {
            'texts': texts,
            'raw_texts': raw_texts
        }
    
    def _process_international_plates(self, image: np.ndarray, raw_texts: List[str]) -> Dict:
        """Process plates through international recognition"""
        try:
            if not INTERNATIONAL_AVAILABLE or not raw_texts:
                return {'enabled': False, 'plates': []}
            
            international_results = extract_international_license_plates(image, raw_texts)
            
            return {
                'enabled': True,
                'plates': international_results['international_plates'],
                'statistics': international_results['statistics'],
                'supported_countries': international_results['supported_countries']
            }
            
        except Exception as e:
            print(f"[ERROR] International processing failed: {e}")
            return {'enabled': False, 'error': str(e)}
    
    def _create_annotated_image(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """Create annotated image with detections"""
        try:
            annotated = image.copy()
            
            # Draw license plates
            for plate in results['detections']['license_plates']:
                if 'bounding_box' in plate and plate['bounding_box']:
                    x1, y1, x2, y2 = plate['bounding_box']
                    
                    # Green box for license plates
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add plate text
                    plate_text = plate.get('plate_text', 'Unknown')
                    cv2.putText(annotated, f"Plate: {plate_text}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw objects
            for obj in results['detections']['objects']:
                if 'bounding_box' in obj:
                    x1, y1, x2, y2 = obj['bounding_box']
                    
                    # Blue box for objects
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Add object info
                    class_name = obj.get('class_name', 'Object')
                    confidence = obj.get('confidence', 0)
                    
                    text = f"{class_name}: {confidence:.2f}"
                    cv2.putText(annotated, text, 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Add info text
            info_text = f"Processed: {results['image_info']['timestamp'].split('T')[1][:8]}"
            cv2.putText(annotated, info_text, 
                       (10, annotated.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            print(f"[ERROR] Failed to create annotated image: {e}")
            return image
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        runtime = time.time() - self.start_time
        
        return {
            'total_processed': self.processed_count,
            'runtime_seconds': runtime,
            'avg_time_per_image': runtime / max(1, self.processed_count),
            'images_per_minute': (self.processed_count / runtime) * 60 if runtime > 0 else 0
        }


# Global instance for easy access
image_processor = ImageProcessor()

# Convenience functions
def process_single_image(image: np.ndarray, **kwargs) -> Dict:
    """Process single image with default settings"""
    return image_processor.process_image(image, **kwargs)

def get_image_processing_stats() -> Dict:
    """Get image processing statistics"""
    return image_processor.get_processing_stats()


if __name__ == "__main__":
    print("🖼️  Image Processor Module")
    print("=" * 30)
    
    # Test with dummy image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("🧪 Testing image processing...")
    result = process_single_image(test_image)
    
    if 'error' not in result:
        print("✅ Image processing test passed!")
        print(f"   Objects found: {len(result['detections']['objects'])}")
        print(f"   License plates: {len(result['detections']['license_plates'])}")
        print(f"   Processing time: {result['processing_info']['processing_time']:.2f}s")
    else:
        print(f"❌ Test failed: {result['error']}")
    
    print("\n📖 Usage:")
    print("   from image_processor import process_single_image")
    print("   result = process_single_image(image)")
    print("   print(result['detections']['license_plates'])")
    
    print("\n✅ Image processor ready!")
