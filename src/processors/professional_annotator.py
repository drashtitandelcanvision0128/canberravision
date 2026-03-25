"""
Professional Annotation System for YOLO26
Handles non-overlapping bounding box labels with professional design
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import math
import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class ProfessionalAnnotator:
    """
    Professional annotation system that prevents overlapping labels
    and provides a clean, modern design.
    """
    
    def __init__(self):
        # Professional color palette
        self.class_colors = {
            'person': (0, 255, 127),      # Light green
            'car': (255, 99, 71),         # Tomato red
            'truck': (70, 130, 180),      # Steel blue
            'motorcycle': (255, 215, 0),  # Gold
            'bicycle': (255, 105, 180),   # Hot pink
            'bus': (255, 165, 0),         # Orange
            'dog': (147, 112, 219),       # Medium purple
            'cat': (255, 182, 193),       # Light pink
            'bottle': (0, 191, 255),      # Deep sky blue
            'cup': (50, 205, 50),         # Lime green
            'cell phone': (255, 20, 147), # Deep pink
            'laptop': (106, 90, 205),     # Slate blue
            'book': (210, 180, 140),      # Tan
            'license plate': (255, 255, 0), # Yellow
        }
        
        # Default color for unknown classes
        self.default_color = (200, 200, 200)
        
        # Label positioning system
        self.occupied_regions = []
        self.label_margin = 5
        self.min_label_spacing = 25
        
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get professional color for a specific class."""
        return self.class_colors.get(class_name.lower(), self.default_color)
    
    def _calculate_label_position(self, 
                                 bbox: Tuple[int, int, int, int],
                                 label_size: Tuple[int, int],
                                 image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate optimal label position to avoid overlaps.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            label_size: Label size (width, height)
            image_shape: Image shape (height, width)
            
        Returns:
            Optimal label position (x, y)
        """
        x1, y1, x2, y2 = bbox
        label_width, label_height = label_size
        img_h, img_w = image_shape
        
        # Preferred positions (in order of preference)
        positions = [
            # Above the box (most common)
            (x1, y1 - label_height - self.label_margin),
            # Below the box
            (x1, y2 + self.label_margin),
            # Left of the box
            (x1 - label_width - self.label_margin, y1),
            # Right of the box
            (x2 + self.label_margin, y1),
            # Inside top-left
            (x1 + self.label_margin, y1 + label_height + self.label_margin),
        ]
        
        # Check each position for availability
        for pos_x, pos_y in positions:
            if self._is_position_available(pos_x, pos_y, label_width, label_height, img_w, img_h):
                return pos_x, pos_y
        
        # If no preferred position works, find the best available position
        return self._find_best_available_position(bbox, label_size, image_shape)
    
    def _is_position_available(self, 
                              x: int, y: int, 
                              width: int, height: int,
                              img_w: int, img_h: int) -> bool:
        """Check if a position is available (no overlaps and within bounds)."""
        # Check if within image bounds
        if x < 0 or y < 0 or x + width > img_w or y + height > img_h:
            return False
        
        # Check for overlaps with existing labels
        label_rect = (x, y, x + width, y + height)
        
        for occupied_rect in self.occupied_regions:
            if self._rectangles_overlap(label_rect, occupied_rect):
                return False
        
        return True
    
    def _rectangles_overlap(self, rect1: Tuple[int, int, int, int], 
                           rect2: Tuple[int, int, int, int]) -> bool:
        """Check if two rectangles overlap."""
        x1_min, y1_min, x1_max, y1_max = rect1
        x2_min, y2_min, x2_max, y2_max = rect2
        
        # Add small margin between labels
        margin = self.min_label_spacing
        
        return not (x1_max + margin < x2_min or 
                   x2_max + margin < x1_min or 
                   y1_max + margin < y2_min or 
                   y2_max + margin < y1_min)
    
    def _find_best_available_position(self, 
                                    bbox: Tuple[int, int, int, int],
                                    label_size: Tuple[int, int],
                                    image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Find the best available position using a spiral search pattern."""
        x1, y1, x2, y2 = bbox
        label_width, label_height = label_size
        img_h, img_w = image_shape
        
        # Start from the center of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Spiral search parameters
        max_radius = max(img_w, img_h)
        step_size = 10
        
        for radius in range(0, max_radius, step_size):
            for angle in range(0, 360, 45):  # Check 8 directions
                # Calculate position
                x = center_x + int(radius * math.cos(math.radians(angle)))
                y = center_y + int(radius * math.sin(math.radians(angle)))
                
                # Adjust for label size (top-left corner)
                x -= label_width // 2
                y -= label_height // 2
                
                if self._is_position_available(x, y, label_width, label_height, img_w, img_h):
                    return x, y
        
        # Fallback: position at top-left with minimal overlap handling
        return max(0, x1), max(0, y1 - label_height - self.label_margin)
    
    def _add_occupied_region(self, x: int, y: int, width: int, height: int):
        """Add a region to the occupied regions list."""
        self.occupied_regions.append((x, y, x + width, y + height))
    
    def draw_professional_box(self, 
                             image: np.ndarray,
                             bbox: Tuple[int, int, int, int],
                             label: str,
                             confidence: float = None,
                             class_name: str = None) -> np.ndarray:
        """
        Draw a professional bounding box with non-overlapping label.
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            label: Label text
            confidence: Confidence score (optional)
            class_name: Class name for color selection (optional)
            
        Returns:
            Image with professional annotation
        """
        annotated = image.copy()
        x1, y1, x2, y2 = bbox
        
        # Get color for this class
        color = self.get_class_color(class_name or label.split()[0])
        
        # Draw bounding box with rounded corners effect
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Add subtle inner border for depth
        cv2.rectangle(annotated, (x1+1, y1+1), (x2-1, y2-1), color, 1)
        
        # Prepare label text
        if confidence is not None:
            label_text = f"{label} {confidence:.2f}"
        else:
            label_text = label
        
        # Get label size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (label_width, label_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Calculate optimal label position
        label_x, label_y = self._calculate_label_position(
            (x1, y1, x2, y2), 
            (label_width, label_height + baseline),
            image.shape[:2]
        )
        
        # Draw label background with rounded effect
        padding = 4
        bg_x1, bg_y1 = label_x - padding, label_y - label_height - padding
        bg_x2, bg_y2 = label_x + label_width + padding, label_y + baseline + padding
        
        # Ensure background is within image bounds
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(image.shape[1], bg_x2)
        bg_y2 = min(image.shape[0], bg_y2)
        
        # Draw semi-transparent background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)
        
        # Draw border
        cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
        
        # Draw text with shadow effect
        shadow_color = (0, 0, 0)
        cv2.putText(annotated, label_text, (label_x + 1, label_y + 1), 
                   font, font_scale, shadow_color, thickness + 1)
        cv2.putText(annotated, label_text, (label_x, label_y), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # Add this region to occupied list
        self._add_occupied_region(bg_x1, bg_y1, bg_x2 - bg_x1, bg_y2 - bg_y1)
        
        return annotated
    
    def draw_professional_info_panel(self, 
                                   image: np.ndarray,
                                   info_lines: List[str],
                                   position: str = 'top-left') -> np.ndarray:
        """
        Draw a professional information panel.
        
        Args:
            image: Input image
            info_lines: List of information lines
            position: Position ('top-left', 'top-right', 'bottom-left', 'bottom-right')
            
        Returns:
            Image with info panel
        """
        annotated = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20
        padding = 10
        
        # Calculate panel size
        max_line_width = 0
        for line in info_lines:
            (line_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_line_width = max(max_line_width, line_width)
        
        panel_width = max_line_width + (padding * 2)
        panel_height = (len(info_lines) * line_height) + (padding * 2)
        
        # Determine panel position
        img_h, img_w = image.shape[:2]
        
        if position == 'top-left':
            panel_x, panel_y = 10, 10
        elif position == 'top-right':
            panel_x, panel_y = img_w - panel_width - 10, 10
        elif position == 'bottom-left':
            panel_x, panel_y = 10, img_h - panel_height - 10
        else:  # bottom-right
            panel_x, panel_y = img_w - panel_width - 10, img_h - panel_height - 10
        
        # Ensure panel is within bounds
        panel_x = max(0, min(panel_x, img_w - panel_width))
        panel_y = max(0, min(panel_y, img_h - panel_height))
        
        # Draw panel background with gradient effect
        panel_overlay = annotated.copy()
        cv2.rectangle(panel_overlay, 
                     (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (40, 40, 40), -1)
        cv2.addWeighted(panel_overlay, 0.9, annotated, 0.1, 0, annotated)
        
        # Draw panel border
        cv2.rectangle(annotated, 
                     (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (100, 100, 100), 1)
        
        # Draw text
        for i, line in enumerate(info_lines):
            text_y = panel_y + padding + (i + 1) * line_height
            cv2.putText(annotated, line, 
                       (panel_x + padding, text_y),
                       font, font_scale, (255, 255, 255), thickness)
        
        return annotated
    
    def reset_occupied_regions(self):
        """Reset the occupied regions for new frame."""
        self.occupied_regions = []
    
    def annotate_detections(self, 
                           image: np.ndarray,
                           detections: List[Dict],
                           show_confidence: bool = True,
                           show_info_panel: bool = True) -> np.ndarray:
        """
        Annotate all detections with professional design.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            show_confidence: Whether to show confidence scores
            show_info_panel: Whether to show information panel
            
        Returns:
            Professionally annotated image
        """
        # Reset occupied regions for new frame
        self.reset_occupied_regions()
        
        annotated = image.copy()
        
        # Sort detections by confidence (highest first) for better label placement
        sorted_detections = sorted(detections, 
                                 key=lambda x: x.get('confidence', 0), 
                                 reverse=True)
        
        # Draw each detection
        for detection in sorted_detections:
            bbox = detection.get('bbox', detection.get('bounding_box', [0, 0, 0, 0]))
            if isinstance(bbox, dict):
                bbox = [bbox.get('x1', 0), bbox.get('y1', 0), 
                       bbox.get('x2', 0), bbox.get('y2', 0)]
            
            class_name = detection.get('class_name', 'unknown')
            confidence = detection.get('confidence')
            
            # Build enhanced label
            label_parts = [class_name]
            
            # Add color information if available
            if 'color' in detection:
                label_parts.append(f"({detection['color']})")
            
            # Add license plate if available
            if 'license_plate' in detection:
                label_parts.append(f"[{detection['license_plate']}]")
            
            label = " ".join(label_parts)
            
            # Draw the detection
            conf_to_show = confidence if show_confidence else None
            annotated = self.draw_professional_box(
                annotated, tuple(bbox), label, conf_to_show, class_name
            )
        
        # Add information panel
        if show_info_panel:
            info_lines = [
                f"Objects: {len(detections)}",
                f"FPS: {detection.get('fps', 'N/A')}" if any('fps' in d for d in detections) else None,
                f"Mode: Professional"
            ]
            info_lines = [line for line in info_lines if line]  # Remove None values
            
            if info_lines:
                annotated = self.draw_professional_info_panel(
                    annotated, info_lines, 'top-left'
                )
        
        return annotated


# Global annotator instance
professional_annotator = ProfessionalAnnotator()


def annotate_image_professional(image: np.ndarray, 
                               detections: List[Dict],
                               **kwargs) -> np.ndarray:
    """
    Convenience function to annotate image with professional design.
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        **kwargs: Additional arguments for annotation
        
    Returns:
        Professionally annotated image
    """
    return professional_annotator.annotate_detections(image, detections, **kwargs)


if __name__ == "__main__":
    # Test the professional annotator
    print("Testing Professional Annotator...")
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (50, 50, 50)  # Dark gray background
    
    # Create some test detections with overlapping potential
    detections = [
        {'bbox': [100, 100, 200, 200], 'confidence': 0.85, 'class_name': 'person', 'class_id': 0},
        {'bbox': [120, 120, 220, 220], 'confidence': 0.92, 'class_name': 'car', 'class_id': 1, 'color': 'red'},
        {'bbox': [300, 150, 400, 250], 'confidence': 0.78, 'class_name': 'bicycle', 'class_id': 2, 'color': 'blue'},
        {'bbox': [150, 300, 250, 400], 'confidence': 0.91, 'class_name': 'dog', 'class_id': 3, 'color': 'brown'},
        {'bbox': [320, 320, 420, 420], 'confidence': 0.73, 'class_name': 'cat', 'class_id': 4, 'color': 'white'},
    ]
    
    # Test annotation
    result = professional_annotator.annotate_detections(test_image, detections)
    
    print(f"✅ Test completed successfully!")
    print(f"✅ Input shape: {test_image.shape}")
    print(f"✅ Output shape: {result.shape}")
    print(f"✅ Processed {len(detections)} detections")
    
    # Save test result
    cv2.imwrite("test_professional_annotation.jpg", result)
    print("✅ Test result saved to 'test_professional_annotation.jpg'")
