"""
Fast Detection + Text Flow Module
Optimized YOLO detection + PaddleOCR text extraction with GPU acceleration.
Designed for maximum speed and accuracy in object detection + text extraction workflow.
"""

import time
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
import json

# Import optimized PaddleOCR
try:
    from optimized_paddleocr_gpu import (
        extract_text_optimized, 
        extract_license_plates_optimized,
        batch_extract_text,
        get_gpu_info,
        initialize_gpu_environment
    )
    OPTIMIZED_PADDLEOCR_AVAILABLE = True
    print("[INFO] 🚀 Optimized PaddleOCR GPU available for fast detection flow")
except ImportError:
    OPTIMIZED_PADDLEOCR_AVAILABLE = False
    print("[WARNING] Optimized PaddleOCR GPU not available")

# Import YOLO utilities
try:
    from modules.utils import get_model, _get_device, _classify_color_bgr
    from modules.text_extraction import extract_text_from_image_json
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("[WARNING] Utils modules not available")

@dataclass
class DetectionResult:
    """Data class for detection results"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    color: str
    text_found: List[Dict] = None
    license_plate: Optional[Dict] = None
    processing_time: float = 0.0

@dataclass
class FastDetectionResult:
    """Data class for fast detection results"""
    image_shape: Tuple[int, int, int]
    total_detections: int
    detections: List[DetectionResult]
    text_summary: Dict
    processing_time: Dict
    device_info: Dict
    gpu_accelerated: bool

class FastDetectionFlow:
    """
    Fast Detection + Text Flow with GPU optimization.
    Combines YOLO object detection with PaddleOCR text extraction.
    """
    
    def __init__(self, model_name: str = "yolo26n", use_gpu: Optional[bool] = None):
        """
        Initialize fast detection flow.
        
        Args:
            model_name: YOLO model name to use
            use_gpu: Whether to use GPU (auto-detect if None)
        """
        self.model_name = model_name
        self.use_gpu = use_gpu if use_gpu is not None else torch.cuda.is_available()
        self.device = _get_device() if UTILS_AVAILABLE else "cpu"
        
        # Initialize GPU environment
        if OPTIMIZED_PADDLEOCR_AVAILABLE:
            initialize_gpu_environment()
        
        # Load YOLO model
        self.model = None
        if UTILS_AVAILABLE:
            try:
                self.model = get_model(model_name)
                print(f"[INFO] YOLO model '{model_name}' loaded on {self.device}")
            except Exception as e:
                print(f"[ERROR] Failed to load YOLO model: {e}")
        
        # Performance metrics
        self.processing_times = {
            'detection': [],
            'text_extraction': [],
            'total': []
        }
        
        print(f"[INFO] Fast Detection Flow initialized (GPU: {self.use_gpu}, Device: {self.device})")
    
    def detect_objects_fast(
        self, 
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        imgsz: int = 640
    ) -> List[Dict]:
        """
        Fast object detection using YOLO with GPU optimization.
        
        Args:
            image: Input image in BGR format
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            imgsz: Image size for inference
            
        Returns:
            List of detected objects with metadata
        """
        start_time = time.time()
        
        try:
            if self.model is None:
                print("[ERROR] YOLO model not loaded")
                return []
            
            # Run YOLO detection with optimizations
            results = self.model.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
                half=True if self.device != "cpu" else False,  # FP16 on CUDA
                max_det=50  # Limit detections for speed
            )
            
            if not results:
                return []
            
            detection = results[0]
            detections = []
            
            if hasattr(detection, 'boxes') and detection.boxes is not None:
                boxes = detection.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                names = detection.names
                
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Get class information
                    class_id = int(cls[i]) if i < len(cls) else -1
                    class_name = names.get(class_id, f"class_{class_id}")
                    confidence = float(conf[i]) if i < len(conf) else 0.0
                    
                    detections.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'class_id': class_id
                    })
            
            processing_time = time.time() - start_time
            self.processing_times['detection'].append(processing_time)
            
            print(f"[DEBUG] Fast detection: {len(detections)} objects in {processing_time:.3f}s")
            return detections
            
        except Exception as e:
            print(f"[ERROR] Fast detection failed: {e}")
            return []
    
    def extract_text_from_detections(
        self, 
        image: np.ndarray,
        detections: List[Dict],
        extract_license_plates: bool = True,
        extract_general_text: bool = True,
        confidence_threshold: float = 0.5
    ) -> List[DetectionResult]:
        """
        Extract text from detected objects using optimized PaddleOCR.
        
        Args:
            image: Original image in BGR format
            detections: List of detected objects
            extract_license_plates: Whether to extract license plates
            extract_general_text: Whether to extract general text
            confidence_threshold: Text confidence threshold
            
        Returns:
            List of DetectionResult objects with text information
        """
        start_time = time.time()
        
        if not OPTIMIZED_PADDLEOCR_AVAILABLE:
            print("[WARNING] Optimized PaddleOCR not available, skipping text extraction")
            return []
        
        detection_results = []
        
        # Process detections in parallel for maximum speed
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # Crop object region
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Submit text extraction task
                future = executor.submit(
                    self._extract_text_from_crop,
                    crop,
                    detection,
                    extract_license_plates,
                    extract_general_text,
                    confidence_threshold
                )
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=10)  # 10 second timeout
                    if result:
                        detection_results.append(result)
                except Exception as e:
                    print(f"[ERROR] Text extraction failed: {e}")
        
        processing_time = time.time() - start_time
        self.processing_times['text_extraction'].append(processing_time)
        
        print(f"[DEBUG] Text extraction: {len(detection_results)} objects processed in {processing_time:.3f}s")
        return detection_results
    
    def _extract_text_from_crop(
        self,
        crop: np.ndarray,
        detection: Dict,
        extract_license_plates: bool,
        extract_general_text: bool,
        confidence_threshold: float
    ) -> Optional[DetectionResult]:
        """Extract text from a single crop"""
        try:
            # Get color information
            color = "unknown"
            if UTILS_AVAILABLE:
                color = _classify_color_bgr(crop)
            
            # Initialize detection result
            detection_result = DetectionResult(
                class_name=detection['class_name'],
                confidence=detection['confidence'],
                bbox=detection['bbox'],
                color=color
            )
            
            text_start = time.time()
            
            # Extract license plates
            if extract_license_plates:
                try:
                    license_plates = extract_license_plates_optimized(
                        crop,
                        confidence_threshold=confidence_threshold,
                        use_gpu=self.use_gpu
                    )
                    
                    if license_plates:
                        # Take the best license plate
                        best_plate = max(license_plates, key=lambda x: x['confidence'])
                        detection_result.license_plate = best_plate
                        detection_result.text_found = [{
                            "text": best_plate['text'],
                            "type": "license_plate",
                            "confidence": best_plate['confidence'],
                            "method": best_plate['method']
                        }]
                        print(f"[DEBUG] License plate found: {best_plate['text']}")
                except Exception as e:
                    print(f"[DEBUG] License plate extraction failed: {e}")
            
            # Extract general text
            if extract_general_text and not detection_result.text_found:
                try:
                    text_result = extract_text_optimized(
                        crop,
                        confidence_threshold=confidence_threshold,
                        lang='en',
                        use_gpu=self.use_gpu,
                        use_cache=True,
                        preprocess=True
                    )
                    
                    if text_result["text"] and text_result["text"].strip():
                        detection_result.text_found = [{
                            "text": text_result["text"],
                            "type": "general_text",
                            "confidence": text_result["confidence"],
                            "method": text_result["method"]
                        }]
                        print(f"[DEBUG] General text found: {text_result['text'][:50]}...")
                except Exception as e:
                    print(f"[DEBUG] General text extraction failed: {e}")
            
            detection_result.processing_time = time.time() - text_start
            return detection_result
            
        except Exception as e:
            print(f"[ERROR] Crop text extraction failed: {e}")
            return None
    
    def process_image_fast(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        imgsz: int = 640,
        extract_text: bool = True,
        confidence_threshold: float = 0.5
    ) -> FastDetectionResult:
        """
        Process image with fast detection + text extraction flow.
        
        Args:
            image: Input image in BGR format
            conf_threshold: Detection confidence threshold
            iou_threshold: Detection IoU threshold
            imgsz: Detection image size
            extract_text: Whether to extract text
            confidence_threshold: Text confidence threshold
            
        Returns:
            FastDetectionResult with all detections and text
        """
        total_start = time.time()
        
        print(f"[INFO] 🚀 Starting fast detection flow on {image.shape}")
        
        # Step 1: Fast object detection
        detections = self.detect_objects_fast(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz
        )
        
        # Step 2: Extract text from detections
        detection_results = []
        text_summary = {
            'total_objects_with_text': 0,
            'license_plates_found': 0,
            'general_text_found': 0,
            'text_details': []
        }
        
        if extract_text and detections:
            detection_results = self.extract_text_from_detections(
                image,
                detections,
                extract_license_plates=True,
                extract_general_text=True,
                confidence_threshold=confidence_threshold
            )
            
            # Generate text summary
            for result in detection_results:
                if result.text_found:
                    text_summary['total_objects_with_text'] += 1
                    
                    for text_item in result.text_found:
                        if text_item['type'] == 'license_plate':
                            text_summary['license_plates_found'] += 1
                        else:
                            text_summary['general_text_found'] += 1
                        
                        text_summary['text_details'].append({
                            'object_class': result.class_name,
                            'text': text_item['text'],
                            'type': text_item['type'],
                            'confidence': text_item['confidence'],
                            'method': text_item['method']
                        })
        
        # Calculate processing times
        total_time = time.time() - total_start
        processing_times = {
            'detection': np.mean(self.processing_times['detection']) if self.processing_times['detection'] else 0,
            'text_extraction': np.mean(self.processing_times['text_extraction']) if self.processing_times['text_extraction'] else 0,
            'total': total_time
        }
        
        # Get device info
        device_info = get_gpu_info() if OPTIMIZED_PADDLEOCR_AVAILABLE else {'pytorch_cuda': False}
        
        # Create result
        result = FastDetectionResult(
            image_shape=image.shape,
            total_detections=len(detections),
            detections=detection_results,
            text_summary=text_summary,
            processing_time=processing_times,
            device_info=device_info,
            gpu_accelerated=self.use_gpu
        )
        
        print(f"[INFO] ✅ Fast detection flow completed in {total_time:.3f}s")
        print(f"[INFO]   Detections: {len(detections)}, Objects with text: {text_summary['total_objects_with_text']}")
        print(f"[INFO]   License plates: {text_summary['license_plates_found']}, General text: {text_summary['general_text_found']}")
        
        return result
    
    def annotate_image(
        self, 
        image: np.ndarray, 
        result: FastDetectionResult,
        show_labels: bool = True,
        show_confidence: bool = True,
        show_text: bool = True
    ) -> np.ndarray:
        """
        Annotate image with detection results and extracted text.
        
        Args:
            image: Original image in BGR format
            result: FastDetectionResult from process_image_fast
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores
            show_text: Whether to show extracted text
            
        Returns:
            Annotated image in BGR format
        """
        annotated = image.copy()
        
        for detection in result.detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            color = (0, 255, 0) if detection.license_plate else (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_parts = []
            
            if show_labels:
                label_parts.append(detection.class_name)
            
            if show_confidence:
                label_parts.append(f"{detection.confidence:.2f}")
            
            # Add text information
            if show_text and detection.text_found:
                text_items = [item["text"] for item in detection.text_found[:2]]  # Limit to 2 text items
                label_parts.extend(text_items)
            
            # Draw label
            if label_parts:
                label = " ".join(label_parts)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Background rectangle for label
                cv2.rectangle(
                    annotated,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )
                
                # Label text
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
        
        return annotated
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {
            'avg_detection_time': np.mean(self.processing_times['detection']) if self.processing_times['detection'] else 0,
            'avg_text_extraction_time': np.mean(self.processing_times['text_extraction']) if self.processing_times['text_extraction'] else 0,
            'total_processed': len(self.processing_times['total']),
            'gpu_accelerated': self.use_gpu
        }
        
        if OPTIMIZED_PADDLEOCR_AVAILABLE:
            stats['gpu_info'] = get_gpu_info()
        
        return stats

def create_fast_detection_flow(model_name: str = "yolo26n", use_gpu: Optional[bool] = None) -> FastDetectionFlow:
    """
    Create and initialize fast detection flow.
    
    Args:
        model_name: YOLO model name
        use_gpu: Whether to use GPU (auto-detect if None)
    
    Returns:
        Initialized FastDetectionFlow instance
    """
    return FastDetectionFlow(model_name=model_name, use_gpu=use_gpu)

# Test function
def test_fast_detection_flow():
    """Test the fast detection flow"""
    try:
        print("[INFO] Testing fast detection flow...")
        
        # Create test image
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "TEST CAR", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.rectangle(test_image, (50, 150), (300, 250), (0, 0, 255), 2)  # Simulate car detection
        
        # Initialize fast detection flow
        flow = create_fast_detection_flow(use_gpu=None)
        
        # Process image
        result = flow.process_image_fast(
            test_image,
            conf_threshold=0.25,
            extract_text=True
        )
        
        # Annotate image
        annotated = flow.annotate_image(test_image, result)
        
        print(f"[TEST] Detection completed: {result.total_detections} objects")
        print(f"[TEST] Text summary: {result.text_summary}")
        print(f"[TEST] Processing time: {result.processing_time['total']:.3f}s")
        
        # Get performance stats
        stats = flow.get_performance_stats()
        print(f"[TEST] Performance stats: {stats}")
        
        print("[INFO] Fast detection flow test completed")
        
    except Exception as e:
        print(f"[ERROR] Fast detection flow test failed: {e}")

if __name__ == "__main__":
    test_fast_detection_flow()
