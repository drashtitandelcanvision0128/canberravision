"""
Frame Extractor Module for Unified Detection System
Handles input from webcam, video files, and static images
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional, Union
from pathlib import Path
from enum import Enum
import time


class InputSourceType(Enum):
    """Types of input sources"""
    WEBCAM = "WEBCAM"
    VIDEO = "VIDEO"
    IMAGE = "IMAGE"


class FrameExtractor:
    """
    Frame Extractor for Unified Detection System
    Provides a unified interface for extracting frames from various sources
    """
    
    def __init__(self, source: Union[str, int], source_type: Optional[InputSourceType] = None):
        """
        Initialize Frame Extractor
        
        Args:
            source: Input source (file path, URL, or camera index)
            source_type: Type of input source (auto-detected if None)
        """
        self.source = source
        self.source_type = source_type or self._detect_source_type(source)
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.is_opened = False
        
        # For image sources
        self.static_image = None
        
        self._initialize_source()
    
    def _detect_source_type(self, source: Union[str, int]) -> InputSourceType:
        """Auto-detect the type of input source"""
        if isinstance(source, int):
            return InputSourceType.WEBCAM
        
        source_str = str(source).lower()
        
        # Check for image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        if any(source_str.endswith(ext) for ext in image_extensions):
            return InputSourceType.IMAGE
        
        # Check for video extensions
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        if any(source_str.endswith(ext) for ext in video_extensions):
            return InputSourceType.VIDEO
        
        # Check for URL (rtsp, http)
        if source_str.startswith(('rtsp://', 'http://', 'https://')):
            return InputSourceType.VIDEO
        
        # Default to video for file paths
        return InputSourceType.VIDEO
    
    def _initialize_source(self):
        """Initialize the input source"""
        try:
            if self.source_type == InputSourceType.IMAGE:
                self._initialize_image()
            else:
                self._initialize_video_capture()
        except Exception as e:
            print(f"[ERROR] Failed to initialize source: {e}")
            self.is_opened = False
    
    def _initialize_image(self):
        """Initialize static image source"""
        try:
            self.static_image = cv2.imread(str(self.source))
            if self.static_image is None:
                raise ValueError(f"Could not load image: {self.source}")
            
            self.height, self.width = self.static_image.shape[:2]
            self.fps = 1  # Single frame
            self.total_frames = 1
            self.is_opened = True
            
            print(f"[INFO] Image loaded: {self.width}x{self.height}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load image: {e}")
            self.is_opened = False
    
    def _initialize_video_capture(self):
        """Initialize video/webcam capture"""
        try:
            # Convert source to appropriate format
            if self.source_type == InputSourceType.WEBCAM:
                source_id = int(self.source) if isinstance(self.source, str) else self.source
            else:
                source_id = str(self.source)
            
            # Open capture
            self.cap = cv2.VideoCapture(source_id)
            
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video source: {source_id}")
            
            # Get video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Handle webcam specifics
            if self.source_type == InputSourceType.WEBCAM:
                # Set buffer size for lower latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.total_frames = float('inf')  # Webcam is continuous
            
            self.is_opened = True
            
            source_name = "Webcam" if self.source_type == InputSourceType.WEBCAM else "Video"
            print(f"[INFO] {source_name} opened: {self.width}x{self.height} @ {self.fps:.1f} FPS")
            
        except Exception as e:
            print(f"[ERROR] Failed to open video: {e}")
            self.is_opened = False
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the next frame from the source
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_opened:
            return False, None
        
        try:
            if self.source_type == InputSourceType.IMAGE:
                return self._get_image_frame()
            else:
                return self._get_video_frame()
        except Exception as e:
            print(f"[ERROR] Failed to get frame: {e}")
            return False, None
    
    def _get_image_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get frame from static image"""
        if self.current_frame < self.total_frames:
            self.current_frame += 1
            return True, self.static_image.copy()
        return False, None
    
    def _get_video_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get frame from video/webcam"""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
        return ret, frame
    
    def get_frames(self, skip_frames: int = 0) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields frames from the source
        
        Args:
            skip_frames: Number of frames to skip between each yielded frame
            
        Yields:
            Tuple of (frame_number, frame)
        """
        frame_count = 0
        skipped = 0
        
        while True:
            ret, frame = self.get_frame()
            
            if not ret:
                break
            
            # Skip frames if requested
            if skip_frames > 0:
                skipped += 1
                if skipped <= skip_frames:
                    continue
                skipped = 0
            
            yield frame_count, frame
            frame_count += 1
            
            # Stop after single image frame
            if self.source_type == InputSourceType.IMAGE:
                break
    
    def get_batch(self, batch_size: int = 4) -> Tuple[bool, list]:
        """
        Get a batch of frames
        
        Args:
            batch_size: Number of frames to get
            
        Returns:
            Tuple of (success, frames_list)
        """
        frames = []
        
        for _ in range(batch_size):
            ret, frame = self.get_frame()
            if not ret:
                break
            frames.append(frame)
        
        return len(frames) > 0, frames
    
    def seek(self, frame_number: int) -> bool:
        """
        Seek to a specific frame (video only)
        
        Args:
            frame_number: Frame number to seek to
            
        Returns:
            True if successful
        """
        if self.source_type != InputSourceType.VIDEO or not self.cap:
            return False
        
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
            return True
        except Exception as e:
            print(f"[ERROR] Failed to seek: {e}")
            return False
    
    def get_properties(self) -> dict:
        """Get source properties"""
        return {
            'source': self.source,
            'source_type': self.source_type.value,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames if self.total_frames != float('inf') else -1,
            'current_frame': self.current_frame,
            'is_opened': self.is_opened
        }
    
    def release(self):
        """Release the source"""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.static_image = None
        self.is_opened = False
        print("[INFO] Frame extractor released")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def __del__(self):
        """Destructor"""
        self.release()


def create_frame_extractor(source: Union[str, int], 
                           source_type: Optional[str] = None) -> FrameExtractor:
    """
    Factory function to create a FrameExtractor
    
    Args:
        source: Input source (file path, URL, or camera index)
        source_type: Type of source ('webcam', 'video', 'image', or None for auto)
        
    Returns:
        FrameExtractor instance
    """
    # Convert string type to enum
    if source_type:
        source_type = InputSourceType(source_type.upper())
    
    return FrameExtractor(source, source_type)


# Common source helpers
def webcam(camera_index: int = 0) -> FrameExtractor:
    """Create a webcam frame extractor"""
    return FrameExtractor(camera_index, InputSourceType.WEBCAM)


def video_file(path: str) -> FrameExtractor:
    """Create a video file frame extractor"""
    return FrameExtractor(path, InputSourceType.VIDEO)


def image_file(path: str) -> FrameExtractor:
    """Create an image file frame extractor"""
    return FrameExtractor(path, InputSourceType.IMAGE)


if __name__ == "__main__":
    print("[INFO] Frame Extractor Module")
    print("[INFO] Usage examples:")
    print("  - Webcam: FrameExtractor(0)")
    print("  - Video: FrameExtractor('path/to/video.mp4')")
    print("  - Image: FrameExtractor('path/to/image.jpg')")
