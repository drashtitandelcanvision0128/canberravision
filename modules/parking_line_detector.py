"""
Parking Line Detector Module
=============================

Detects parking slots dynamically from white/yellow parking line markings.
Filters out roads, driving lanes, and non-parking areas.
Only analyzes clearly marked rectangular parking slots.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class DetectedParkingSlot:
    """Represents a parking slot detected from line markings"""
    slot_id: str
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    polygon: np.ndarray  # Four corner points
    confidence: float
    line_color: str  # 'white' or 'yellow'


class ParkingLineDetector:
    """
    Detects parking slots from white/yellow line markings in parking lots.
    Ignores roads, empty areas, and non-parking regions.
    """
    
    def __init__(self, min_slot_area: int = 5000, max_slot_area: int = 50000):
        self.min_slot_area = min_slot_area  # Minimum slot area in pixels
        self.max_slot_area = max_slot_area  # Maximum slot area in pixels
        self.aspect_ratio_min = 1.2  # Min width/height ratio
        self.aspect_ratio_max = 4.0  # Max width/height ratio
        
    def detect_parking_slots(self, frame: np.ndarray) -> List[DetectedParkingSlot]:
        """
        Main method to detect parking slots from line markings
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect white lines
        white_slots = self._detect_colored_lines(frame, gray, 'white')
        print(f"[LINES] Detected {len(white_slots)} white-line slots")
        
        # Detect yellow lines
        yellow_slots = self._detect_colored_lines(frame, gray, 'yellow')
        print(f"[LINES] Detected {len(yellow_slots)} yellow-line slots")
        
        # Combine and filter
        all_slots = white_slots + yellow_slots
        filtered_slots = self._filter_valid_slots(all_slots)
        print(f"[SLOTS] {len(filtered_slots)} valid parking slots after filtering")
        
        # Assign IDs
        for i, slot in enumerate(filtered_slots):
            slot.slot_id = f"S{i+1:03d}"
        
        return filtered_slots
    
    def _detect_colored_lines(self, frame: np.ndarray, gray: np.ndarray, 
                             color_type: str) -> List[DetectedParkingSlot]:
        """Detect parking lines of specific color (white or yellow)"""
        
        if color_type == 'white':
            # Detect white lines (high value, low saturation)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # White lines: high V, low S
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
        else:
            # Detect yellow lines
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Yellow lines
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([40, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Morphological operations to clean up lines
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Edge detection on mask
        edges = cv2.Canny(mask, 50, 150)
        
        # Dilate edges to connect nearby lines
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        slots = []
        for contour in contours:
            # Approximate polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a quadrilateral (4 sides = parking slot)
            if len(approx) == 4:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(approx)
                area = w * h
                
                # Filter by area
                if self.min_slot_area < area < self.max_slot_area:
                    # Check aspect ratio (parking slots are rectangular)
                    aspect_ratio = w / h if h > 0 else 0
                    if self.aspect_ratio_min < aspect_ratio < self.aspect_ratio_max:
                        slot = DetectedParkingSlot(
                            slot_id="",
                            bounding_box=(x, y, x + w, y + h),
                            polygon=approx.reshape(-1, 2),
                            confidence=0.8,
                            line_color=color_type
                        )
                        slots.append(slot)
        
        return slots
    
    def _filter_valid_slots(self, slots: List[DetectedParkingSlot]) -> List[DetectedParkingSlot]:
        """Filter out invalid slots (overlapping, too small, not aligned)"""
        if not slots:
            return []
        
        # Sort by area (largest first)
        slots.sort(key=lambda s: self._get_box_area(s.bounding_box), reverse=True)
        
        filtered = []
        for slot in slots:
            # Check if this slot overlaps too much with already accepted slots
            is_valid = True
            for accepted_slot in filtered:
                overlap = self._calculate_overlap(slot.bounding_box, accepted_slot.bounding_box)
                if overlap > 0.3:  # More than 30% overlap = same slot
                    is_valid = False
                    break
            
            if is_valid:
                filtered.append(slot)
        
        return filtered
    
    def _get_box_area(self, bbox: Tuple[int, int, int, int]) -> int:
        """Calculate area of bounding box"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def _calculate_overlap(self, box1: Tuple[int, int, int, int], 
                          box2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        return intersection / min(area1, area2)
    
    def check_vehicle_in_slot(self, vehicle_bbox: Tuple[int, int, int, int], 
                             slot: DetectedParkingSlot,
                             overlap_threshold: float = 0.4) -> bool:
        """
        Check if a vehicle is inside a parking slot.
        Returns True if vehicle overlaps slot by at least threshold (default 40%)
        """
        overlap = self._calculate_overlap(vehicle_bbox, slot.bounding_box)
        return overlap >= overlap_threshold
    
    def draw_slots(self, frame: np.ndarray, slots: List[DetectedParkingSlot],
                   occupied_slots: List[bool]) -> np.ndarray:
        """Draw parking slots on frame"""
        annotated = frame.copy()
        
        for slot, is_occupied in zip(slots, occupied_slots):
            # Color: RED for occupied, GREEN for unoccupied
            if is_occupied:
                color = (0, 0, 255)  # Red (BGR)
                status = "OCCUPIED"
            else:
                color = (0, 255, 0)  # Green (BGR)
                status = "UNOCCUPIED"
            
            # Draw thin rectangle
            x1, y1, x2, y2 = slot.bounding_box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw slot ID label (small, minimal)
            label = slot.slot_id
            font_scale = 0.5
            thickness = 1
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                  font_scale, thickness)
            
            # Position above the box
            label_x = x1
            label_y = y1 - 3
            if label_y < text_h + 5:
                label_y = y2 + text_h + 3
            
            # Small background
            cv2.rectangle(annotated,
                         (label_x, label_y - text_h - 2),
                         (label_x + text_w + 4, label_y + 2),
                         color, -1)
            
            # Text
            cv2.putText(annotated, label, (label_x + 2, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return annotated


def get_parking_line_detector() -> ParkingLineDetector:
    """Factory function to get parking line detector instance"""
    return ParkingLineDetector()
