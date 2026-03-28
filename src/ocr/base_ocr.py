"""
Base OCR Module
Abstract base class for OCR processors.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np


class BaseOCR(ABC):
    """
    Abstract base class for OCR processors.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize OCR processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the OCR processor.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def extract_text(self, image: np.ndarray, **kwargs) -> Dict:
        """
        Extract text from image.
        
        Args:
            image: Input image in BGR format
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with extracted text and metadata
        """
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Preprocessed image
        """
        import cv2
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Bilateral filter to reduce noise
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return filtered
    
    def validate_input(self, image: np.ndarray) -> bool:
        """
        Validate input image.
        
        Args:
            image: Input image
            
        Returns:
            True if valid, False otherwise
        """
        if image is None:
            return False
        
        if isinstance(image, np.ndarray):
            return image.size > 0
        
        return False
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep alphanumeric
        import re
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def get_info(self) -> Dict:
        """
        Get OCR processor information.
        
        Returns:
            Dictionary with processor info
        """
        return {
            "name": self.name,
            "initialized": self.is_initialized,
            "config": self.config,
            "type": "base_ocr"
        }
