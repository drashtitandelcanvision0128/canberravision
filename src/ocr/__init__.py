"""
YOLO26 OCR Module
Contains OCR functionality for text extraction and license plate detection.
"""

from .base_ocr import BaseOCR
from .paddleocr_processor import PaddleOCRProcessor
from .license_plate_detector import LicensePlateDetector
from .text_extractor import TextExtractor

__all__ = [
    'BaseOCR',
    'PaddleOCRProcessor', 
    'LicensePlateDetector',
    'TextExtractor'
]
