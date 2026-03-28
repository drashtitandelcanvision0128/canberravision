"""
YOLO26 Processors Module
Contains processors for different types of media.
"""

from .image_processor import ImageProcessor
from .video_processor import VideoProcessor
from .webcam_processor import WebcamProcessor

__all__ = [
    'ImageProcessor',
    'VideoProcessor', 
    'WebcamProcessor'
]
