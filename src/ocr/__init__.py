"""
YOLO26 OCR Module
Contains OCR functionality for text extraction and license plate detection.
"""

from .base_ocr import BaseOCR
from .license_plate_detector import LicensePlateDetector
from .text_extractor import TextExtractor

# Import PaddleOCR functions
try:
    from .optimized_paddleocr_gpu import extract_text_optimized, initialize_gpu_environment
    
    class PaddleOCRProcessor(BaseOCR):
        """Wrapper class for PaddleOCR functions to match expected interface."""
        
        def __init__(self, config=None):
            self.config = config or {}
            self.initialized = False
        
        def initialize(self):
            """Initialize the PaddleOCR processor."""
            try:
                initialize_gpu_environment()
                self.initialized = True
                return True
            except Exception as e:
                print(f"[ERROR] Failed to initialize PaddleOCR: {e}")
                return False
        
        def extract_text(self, image, **kwargs):
            """Extract text from image using PaddleOCR."""
            if not self.initialized:
                if not self.initialize():
                    raise Exception("PaddleOCR not initialized")
            
            return extract_text_optimized(image, **kwargs)
    
except ImportError as e:
    print(f"[WARNING] PaddleOCR not available: {e}")
    PaddleOCRProcessor = None

__all__ = [
    'BaseOCR',
    'PaddleOCRProcessor', 
    'LicensePlateDetector',
    'TextExtractor'
]
