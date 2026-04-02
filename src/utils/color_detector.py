"""
Color Detector Module for YOLO26
Provides color detection functionality for detected objects.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter


class ColorDetector:
    """
    Simple color detector for object color analysis.
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        """
        Initialize color detector.
        
        Args:
            confidence_threshold: Minimum confidence for color detection
        """
        self.confidence_threshold = confidence_threshold
        
        # Define color families with their HSV ranges
        self.color_families = {
            'Red': {
                'hsv_ranges': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
                'rgb_values': [(255, 0, 0), (220, 20, 60), (178, 34, 34), (139, 0, 0)]
            },
            'Blue': {
                'hsv_ranges': [(100, 50, 50), (130, 255, 255)],
                'rgb_values': [(0, 0, 255), (65, 105, 225), (0, 0, 128)]
            },
            'Green': {
                'hsv_ranges': [(35, 50, 50), (85, 255, 255)],
                'rgb_values': [(0, 128, 0), (34, 139, 34), (50, 205, 50)]
            },
            'Yellow': {
                'hsv_ranges': [(20, 50, 50), (35, 255, 255)],
                'rgb_values': [(255, 255, 0), (255, 215, 0), (184, 134, 11)]
            },
            'Purple': {
                'hsv_ranges': [(130, 50, 50), (170, 255, 255)],
                'rgb_values': [(128, 0, 128), (75, 0, 130), (148, 0, 211)]
            },
            'Orange': {
                'hsv_ranges': [(10, 50, 50), (20, 255, 255)],
                'rgb_values': [(255, 165, 0), (255, 140, 0), (255, 69, 0)]
            },
            'White': {
                'hsv_ranges': [(0, 0, 200), (180, 30, 255)],
                'rgb_values': [(255, 255, 255), (240, 248, 255)]
            },
            'Black': {
                'hsv_ranges': [(0, 0, 0), (180, 255, 50)],
                'rgb_values': [(0, 0, 0), (54, 54, 54)]
            },
            'Gray': {
                'hsv_ranges': [(0, 0, 50), (180, 30, 200)],
                'rgb_values': [(128, 128, 128), (192, 192, 192), (211, 211, 211)]
            }
        }
        
        print("[INFO] ColorDetector initialized")
    
    def detect_dominant_color(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Detect the dominant color in an image region.
        
        Args:
            image: BGR image region
            
        Returns:
            Tuple of (color_name, confidence)
        """
        if image is None or image.size == 0:
            return ('unknown', 0.0)
        
        try:
            # Convert to RGB for analysis
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Reshape image to be a list of pixels
            pixels = rgb_image.reshape(-1, 3)
            
            # Calculate average color
            avg_color = np.mean(pixels, axis=0)
            r, g, b = avg_color
            
            # Simple color classification based on RGB values
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            diff = max_val - min_val
            
            # Determine color
            if max_val < 50:
                color = "black"
            elif min_val > 200:
                color = "white"
            elif diff < 30:
                color = "gray"
            elif r > g and r > b:
                color = "red" if r > 150 else "brown"
            elif g > r and g > b:
                color = "green"
            elif b > r and b > g:
                color = "blue"
            elif r > 150 and g > 150:
                color = "yellow"
            elif r > 150 and b > 150:
                color = "magenta"
            elif g > 150 and b > 150:
                color = "cyan"
            elif r > 100 and g > 50 and b < 50:
                color = "orange"
            elif r > 100 and b > 100 and g < 100:
                color = "purple"
            else:
                color = "unknown"
            
            # Calculate confidence based on color saturation
            confidence = min(1.0, diff / 255.0 + 0.5)
            
            return (color, confidence)
            
        except Exception as e:
            print(f"[WARNING] Color detection failed: {e}")
            return ('unknown', 0.0)
    
    def analyze_object_color(self, image: np.ndarray, bbox: List[int]) -> Dict:
        """
        Analyze the color of an object in a bounding box.
        
        Args:
            image: Full BGR image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary with color information
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure valid coordinates
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return {'color': 'unknown', 'confidence': 0.0}
            
            # Extract region
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return {'color': 'unknown', 'confidence': 0.0}
            
            # Get dominant color
            color_name, confidence = self.detect_dominant_color(roi)
            
            return {
                'color': color_name,
                'confidence': confidence,
                'rgb': None  # Could be computed if needed
            }
            
        except Exception as e:
            print(f"[WARNING] Object color analysis failed: {e}")
            return {'color': 'unknown', 'confidence': 0.0}
    
    def get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """
        Get color name from RGB values.
        
        Args:
            rgb: RGB color tuple
            
        Returns:
            Color name
        """
        r, g, b = rgb
        
        # Simple classification
        if max(r, g, b) < 50:
            return "black"
        elif min(r, g, b) > 200:
            return "white"
        elif max(r, g, b) - min(r, g, b) < 30:
            return "gray"
        elif r > g and r > b:
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        elif r > 150 and g > 150:
            return "yellow"
        elif r > 150 and b > 150:
            return "purple"
        elif g > 150 and b > 150:
            return "cyan"
        else:
            return "unknown"
