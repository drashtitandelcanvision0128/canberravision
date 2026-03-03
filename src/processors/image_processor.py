"""
Image Processor Module
Handles image processing with object detection, OCR, and color analysis.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
import time
import os

from ..core.processor import ImageProcessor as BaseImageProcessor
from ..core.detector import YOLODetector
from ..ocr.text_extractor import TextExtractor
from ..utils.color_detector import ColorDetector
from ..config.settings import get_config, PROJECT_ROOT, OUTPUTS_DIR, INPUTS_DIR
from ...core.exceptions import ProcessingError


class ImageProcessor(BaseImageProcessor):
    """
    Advanced image processor with object detection, OCR, and color analysis.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize image processor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config or get_config('yolo'))
        
        # Initialize components
        self.detector = YOLODetector()
        self.text_extractor = TextExtractor()
        self.color_detector = ColorDetector()
        
        # Processing settings
        self.enable_ocr = self.config.get('enable_ocr', True)
        self.enable_colors = self.config.get('enable_colors', True)
        self.show_labels = self.config.get('show_labels', True)
        self.show_confidence = self.config.get('show_confidence', True)
        
        print("[INFO] Image Processor initialized")
        print(f"[INFO] OCR enabled: {self.enable_ocr}")
        print(f"[INFO] Color detection enabled: {self.enable_colors}")
    
    def process(self, 
                image: np.ndarray, 
                conf_threshold: float = None,
                iou_threshold: float = None,
                imgsz: int = None,
                **kwargs) -> Tuple[Image.Image, Dict]:
        """
        Process image with object detection, OCR, and color analysis.
        
        Args:
            image: Input image in BGR format
            conf_threshold: Confidence threshold for detection
            iou_threshold: IOU threshold for detection
            imgsz: Image size for inference
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (processed_image, results_dict)
        """
        if not self.validate_input(image):
            raise ProcessingError("Invalid input image")
        
        start_time = time.time()
        timestamp = int(time.time())
        
        print(f"[INFO] Starting image processing...")
        
        # Save original image
        input_path = self._save_input_image(image, timestamp)
        
        # Detect objects
        detections = self.detector.detect_objects(
            image, conf_threshold, iou_threshold, imgsz
        )
        
        # Extract text if enabled
        text_results = {}
        if self.enable_ocr:
            print("[INFO] Extracting text...")
            text_results = self.text_extractor.extract_text_comprehensive(
                image, f"img_{timestamp}"
            )
        
        # Analyze colors if enabled
        color_results = {}
        if self.enable_colors:
            print("[INFO] Analyzing colors...")
            color_results = self.color_detector.analyze_image_colors(image, detections)
        
        # Create annotated image
        annotated_image = self._create_annotated_image(image, detections, text_results, color_results)
        
        # Generate results summary
        results = self._generate_results(
            detections, text_results, color_results, time.time() - start_time
        )
        
        # Save processed image
        output_path = self._save_processed_image(annotated_image, timestamp)
        
        results['paths'] = {
            'input': input_path,
            'output': output_path
        }
        
        print(f"[INFO] Image processing completed in {results['processing_time']:.2f}s")
        
        return annotated_image, results
    
    def _create_annotated_image(self, 
                               image: np.ndarray, 
                               detections: List[Dict],
                               text_results: Dict,
                               color_results: Dict) -> Image.Image:
        """
        Create annotated image with detections, text, and colors.
        
        Args:
            image: Original image in BGR format
            detections: Object detection results
            text_results: Text extraction results
            color_results: Color analysis results
            
        Returns:
            Annotated PIL Image
        """
        annotated = image.copy()
        
        # Draw object detections
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Choose color for this class
            color = self._get_class_color(class_name)
            
            # Draw bounding box
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            if self.show_labels:
                label = f"{class_name}"
                if self.show_confidence:
                    label += f" {confidence:.2f}"
                
                # Calculate text position
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = bbox[0]
                text_y = bbox[1] - 10 if bbox[1] > 30 else bbox[1] + text_size[1] + 10
                
                # Draw background for text
                cv2.rectangle(annotated, 
                            (text_x, text_y - text_size[1] - 5),
                            (text_x + text_size[0], text_y + 5),
                            color, -1)
                
                # Draw text
                cv2.putText(annotated, label, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add text annotations
        if text_results and text_results.get('general_text'):
            y_offset = 30
            for text_item in text_results['general_text'][:5]:  # Show top 5
                text = text_item['text']
                confidence = text_item['confidence']
                method = text_item['method']
                
                # Color code by method
                if 'gpu' in method.lower():
                    color = (255, 0, 255)  # Magenta for GPU
                    prefix = "🔥"
                else:
                    color = (0, 255, 255)  # Cyan for CPU
                    prefix = "💻"
                
                text_label = f"{prefix} {text} ({confidence:.2f})"
                
                # Draw background
                text_size = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(annotated, (10, y_offset - 20),
                            (10 + text_size[0] + 5, y_offset + 5), (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(annotated, text_label, (12, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                y_offset += 25
                if y_offset > annotated.shape[0] - 50:
                    break
        
        # Add color annotations
        if color_results and color_results.get('dominant_colors'):
            color_x = annotated.shape[1] - 200
            color_y = 30
            
            cv2.putText(annotated, "Colors:", (color_x, color_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            color_y += 20
            
            for i, color_info in enumerate(color_results['dominant_colors'][:5]):
                color_name = color_info['color']
                percentage = color_info['percentage']
                color_label = f"{color_name} ({percentage:.0f}%)"
                
                cv2.putText(annotated, color_label, (color_x, color_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                color_y += 15
        
        # Convert to PIL Image
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_rgb)
    
    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for a specific class."""
        # Predefined colors for common classes
        class_colors = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),         # Blue
            'truck': (0, 0, 255),       # Red
            'motorcycle': (255, 255, 0), # Cyan
            'bicycle': (255, 0, 255),   # Magenta
            'bus': (0, 255, 255),       # Yellow
            'dog': (128, 0, 128),       # Purple
            'cat': (255, 165, 0),       # Orange
        }
        
        return class_colors.get(class_name.lower(), (128, 128, 128))  # Gray default
    
    def _generate_results(self, 
                          detections: List[Dict],
                          text_results: Dict,
                          color_results: Dict,
                          processing_time: float) -> Dict:
        """Generate comprehensive results summary."""
        results = {
            'processing_time': processing_time,
            'timestamp': time.time(),
            'detection_summary': {
                'total_objects': len(detections),
                'objects_by_class': {}
            },
            'text_summary': {},
            'color_summary': {},
            'all_detections': detections
        }
        
        # Count objects by class
        for detection in detections:
            class_name = detection['class_name']
            if class_name not in results['detection_summary']['objects_by_class']:
                results['detection_summary']['objects_by_class'][class_name] = 0
            results['detection_summary']['objects_by_class'][class_name] += 1
        
        # Add text summary
        if text_results:
            results['text_summary'] = text_results.get('summary', {})
            results['all_text_results'] = text_results.get('text_results', [])
            results['license_plates'] = text_results.get('license_plates', [])
            results['general_text'] = text_results.get('general_text', [])
        
        # Add color summary
        if color_results:
            results['color_summary'] = color_results.get('summary', {})
            results['dominant_colors'] = color_results.get('dominant_colors', [])
        
        return results
    
    def _save_input_image(self, image: np.ndarray, timestamp: int) -> str:
        """Save original input image."""
        INPUTS_DIR.mkdir(exist_ok=True)
        filename = f"input_image_{timestamp}.jpg"
        path = INPUTS_DIR / filename
        
        # Convert BGR to RGB for PIL
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        pil_image.save(path)
        
        print(f"[INFO] Input image saved: {path}")
        return str(path)
    
    def _save_processed_image(self, image: Image.Image, timestamp: int) -> str:
        """Save processed image."""
        OUTPUTS_DIR.mkdir(exist_ok=True)
        filename = f"processed_image_{timestamp}.jpg"
        path = OUTPUTS_DIR / filename
        
        image.save(path)
        print(f"[INFO] Processed image saved: {path}")
        return str(path)
    
    def get_info(self) -> Dict:
        """Get processor information."""
        info = super().get_info()
        info.update({
            'detector': self.detector.get_info(),
            'text_extractor': self.text_extractor.get_info(),
            'color_detector': self.color_detector.get_info(),
            'settings': {
                'ocr_enabled': self.enable_ocr,
                'colors_enabled': self.enable_colors,
                'show_labels': self.show_labels,
                'show_confidence': self.show_confidence
            }
        })
        return info
