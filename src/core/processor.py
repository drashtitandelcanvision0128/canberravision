"""
Base Processor Module
Abstract base class for all processors in YOLO26.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseProcessor(ABC):
    """
    Abstract base class for all processors.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize processor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """
        Process input data.
        
        Args:
            input_data: Input data to process
            **kwargs: Additional processing parameters
            
        Returns:
            Processed data
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        return input_data is not None
    
    def get_info(self) -> Dict:
        """
        Get processor information.
        
        Returns:
            Dictionary with processor info
        """
        return {
            "name": self.name,
            "config": self.config,
            "type": "base_processor"
        }


class ImageProcessor(BaseProcessor):
    """
    Base class for image processors.
    """
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate image input."""
        if not super().validate_input(input_data):
            return False
        
        if isinstance(input_data, np.ndarray):
            return len(input_data.shape) == 3 and input_data.shape[2] == 3
        
        return False
    
    def get_info(self) -> Dict:
        """Get image processor info."""
        info = super().get_info()
        info["type"] = "image_processor"
        return info


class VideoProcessor(BaseProcessor):
    """
    Base class for video processors.
    """
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate video input."""
        if not super().validate_input(input_data):
            return False
        
        # Check if it's a valid video file path
        if isinstance(input_data, str):
            import os
            return os.path.exists(input_data) and input_data.lower().endswith(
                ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
            )
        
        return False
    
    def get_info(self) -> Dict:
        """Get video processor info."""
        info = super().get_info()
        info["type"] = "video_processor"
        return info


class TextProcessor(BaseProcessor):
    """
    Base class for text processors.
    """
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate text input."""
        if not super().validate_input(input_data):
            return False
        
        if isinstance(input_data, str):
            return len(input_data.strip()) > 0
        
        if isinstance(input_data, np.ndarray):
            return len(input_data.shape) == 3 and input_data.shape[2] == 3
        
        return False
    
    def get_info(self) -> Dict:
        """Get text processor info."""
        info = super().get_info()
        info["type"] = "text_processor"
        return info
