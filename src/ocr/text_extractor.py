"""
Text Extractor Module
High-level text extraction with multiple OCR methods.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import time
import hashlib

from .base_ocr import BaseOCR
from .optimized_paddleocr_gpu import extract_text_optimized as PaddleOCRProcessor
from ..config.settings import get_config

# Try absolute import first, then fallback
class OCRError(Exception):
    """OCR related errors"""
    pass


class TextExtractor:
    """
    High-level text extractor with multiple OCR methods and intelligent fallback.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize text extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config('ocr')
        self.ocr_processors = {}
        self.cache = {}
        self.cache_size = self.config.get('cache_size', 100)
        
        # Initialize OCR processors
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize available OCR processors."""
        try:
            # Primary: PaddleOCR GPU - use the function directly
            from .optimized_paddleocr_gpu import extract_text_optimized
            self.ocr_processors['paddleocr_gpu'] = extract_text_optimized
            print("[INFO] PaddleOCR GPU initialized successfully")
        except Exception as e:
            print(f"[WARNING] Failed to initialize PaddleOCR GPU: {e}")
        
        # Add fallback processors as needed
        # TODO: Add other OCR processors here
    
    def extract_text_comprehensive(self, 
                                 image: np.ndarray, 
                                 image_id: str = None,
                                 **kwargs) -> Dict:
        """
        Extract text using multiple methods with intelligent fallback.
        
        Args:
            image: Input image in BGR format
            image_id: Unique image ID for caching
            **kwargs: Additional parameters
            
        Returns:
            Comprehensive text extraction results
        """
        if image_id is None:
            image_id = self._generate_image_id(image)
        
        # Check cache
        if self.config.get('cache_enabled', True) and image_id in self.cache:
            print(f"[DEBUG] Using cached text extraction for {image_id}")
            return self.cache[image_id]
        
        print(f"[INFO] Starting comprehensive text extraction for {image_id}")
        
        result = {
            "image_id": image_id,
            "timestamp": time.time(),
            "text_results": [],
            "license_plates": [],
            "general_text": [],
            "summary": {
                "total_text_instances": 0,
                "license_plates_found": 0,
                "general_text_found": 0,
                "methods_used": []
            },
            "processing_time": 0,
            "device_used": "unknown"
        }
        
        start_time = time.time()
        
        try:
            # Try each OCR processor in order of preference
            for method_name, processor in self.ocr_processors.items():
                try:
                    print(f"[DEBUG] Trying OCR method: {method_name}")
                    
                    # Extract text
                    if callable(processor):
                        # Function-based processor
                        ocr_result = processor(image, **kwargs)
                    else:
                        # Class-based processor
                        ocr_result = processor.extract_text(image, **kwargs)
                    
                    if ocr_result and ocr_result.get('text'):
                        result['text_results'].append({
                            'method': method_name,
                            'text': ocr_result['text'],
                            'confidence': ocr_result.get('confidence', 0.0),
                            'device': ocr_result.get('device', 'unknown'),
                            'processing_time': ocr_result.get('processing_time', 0.0)
                        })
                        
                        result['summary']['methods_used'].append(method_name)
                        result['device_used'] = ocr_result.get('device', 'unknown')
                        
                        print(f"[DEBUG] {method_name} found text: {ocr_result['text']}")
                        
                        # If we got good results, we can stop (unless we want multiple methods)
                        if ocr_result.get('confidence', 0) > 0.8:
                            break
                
                except Exception as e:
                    print(f"[WARNING] OCR method {method_name} failed: {e}")
                    continue
            
            # Process results
            self._process_results(result)
            
            # Calculate processing time
            result['processing_time'] = time.time() - start_time
            
            # Cache results
            if self.config.get('cache_enabled', True):
                self._cache_result(image_id, result)
            
            print(f"[INFO] Text extraction completed in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            print(f"[ERROR] Comprehensive text extraction failed: {e}")
            raise OCRError(f"Text extraction failed: {e}")
    
    def extract_text_from_region(self, 
                                image: np.ndarray, 
                                bbox: List[int],
                                region_id: str = None,
                                **kwargs) -> Dict:
        """
        Extract text from a specific region of the image.
        
        Args:
            image: Input image in BGR format
            bbox: Bounding box [x1, y1, x2, y2]
            region_id: Region ID for identification
            **kwargs: Additional parameters
            
        Returns:
            Text extraction results for the region
        """
        x1, y1, x2, y2 = bbox
        
        # Validate bbox
        if x2 <= x1 or y2 <= y1:
            return {"error": "Invalid bounding box"}
        
        # Crop region
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return {"error": "Empty region"}
        
        # Generate region ID
        if region_id is None:
            region_id = f"region_{x1}_{y1}_{x2}_{y2}"
        
        # Extract text with lower confidence threshold for license plates
        result = self.extract_text_comprehensive(region, region_id, **kwargs)
        
        # Add region information
        result['region_info'] = {
            'bbox': bbox,
            'region_id': region_id,
            'area': (x2 - x1) * (y2 - y1)
        }
        
        return result
    
    def _process_results(self, result: Dict):
        """Process and categorize text extraction results."""
        from .license_plate_detector import LicensePlateDetector
        
        license_detector = LicensePlateDetector()
        
        for text_result in result['text_results']:
            text = text_result['text']
            confidence = text_result['confidence']
            method = text_result['method']
            
            # Check if it's a license plate
            if license_detector.is_license_plate(text):
                result['license_plates'].append({
                    'text': text,
                    'confidence': confidence,
                    'method': method,
                    'device': text_result['device']
                })
                result['summary']['license_plates_found'] += 1
            else:
                # It's general text
                result['general_text'].append({
                    'text': text,
                    'confidence': confidence,
                    'method': method,
                    'device': text_result['device']
                })
                result['summary']['general_text_found'] += 1
            
            result['summary']['total_text_instances'] += 1
    
    def _generate_image_id(self, image: np.ndarray) -> str:
        """Generate unique ID for image based on content."""
        # Create hash of image content
        image_bytes = image.tobytes()
        hash_obj = hashlib.md5(image_bytes)
        return f"img_{hash_obj.hexdigest()[:12]}_{int(time.time())}"
    
    def _cache_result(self, image_id: str, result: Dict):
        """Cache extraction result."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = list(self.cache.keys())[0]
            del self.cache[oldest_key]
        
        self.cache[image_id] = result
    
    def clear_cache(self):
        """Clear extraction cache."""
        self.cache.clear()
        print("[INFO] Text extraction cache cleared")
    
    def get_info(self) -> Dict:
        """Get text extractor information."""
        return {
            "available_processors": list(self.ocr_processors.keys()),
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "config": self.config
        }
