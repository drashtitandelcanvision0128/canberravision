"""
Enhanced Smart Parking Detection System
=========================================

Advanced parking slot detection with:
- Parking type classification
- Slot categorization (Car, Bike, EV, Disabled, VIP)
- Temporal smoothing for stable detection
- Entry/exit event tracking
- Structured JSON output
- Confidence scoring

"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class ParkingType(Enum):
    """Classification of parking environments"""
    ON_STREET = "On-Street Parking"
    OPEN_LOT = "Open Parking Lot"
    BASEMENT = "Basement Parking"
    MULTI_LEVEL = "Multi-Level Parking"
    PRIVATE = "Private Parking"


class SlotCategory(Enum):
    """Types of parking slots"""
    CAR = "Car"
    BIKE = "Bike"
    EV_CHARGING = "EV Charging"
    DISABLED = "Disabled"
    VIP = "VIP"
    GENERAL = "General"


class SlotStatus(Enum):
    """Status of parking slots"""
    OCCUPIED = "occupied"
    UNOCCUPIED = "unoccupied"
    UNKNOWN = "unknown"


@dataclass
class ParkingEvent:
    """Represents entry or exit events"""
    event_type: str  # 'entry' or 'exit'
    slot_id: str
    vehicle_type: str
    timestamp: str
    confidence: float


@dataclass
class EnhancedParkingSpot:
    """Enhanced parking spot with all metadata"""
    slot_id: str
    zone_id: str
    camera_id: str
    category: SlotCategory
    status: SlotStatus
    confidence: float
    vehicle_type: Optional[str] = None
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    occupancy_history: deque = field(default_factory=lambda: deque(maxlen=10))
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output"""
        return {
            "slotId": self.slot_id,
            "type": self.category.value,
            "status": self.status.value,
            "confidence": round(self.confidence, 3),
            "vehicleType": self.vehicle_type,
            "boundingBox": self.bounding_box,
            "entryTime": self.entry_time,
            "exitTime": self.exit_time,
            "lastUpdated": self.last_updated
        }


@dataclass
class ParkingAnalysis:
    """Complete parking analysis result"""
    parking_type: ParkingType
    total_slots: int
    occupied_slots: int
    free_slots: int
    slots: List[EnhancedParkingSpot]
    events: List[ParkingEvent] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        data = {
            "parkingType": self.parking_type.value,
            "totalSlots": self.total_slots,
            "occupiedSlots": self.occupied_slots,
            "freeSlots": self.free_slots,
            "timestamp": self.timestamp,
            "slots": [slot.to_dict() for slot in self.slots],
            "events": [
                {
                    "eventType": e.event_type,
                    "slotId": e.slot_id,
                    "vehicleType": e.vehicle_type,
                    "timestamp": e.timestamp,
                    "confidence": e.confidence
                } for e in self.events
            ]
        }
        return json.dumps(data, indent=2)


class EnhancedParkingDetector:
    """
    Enhanced parking detection system with advanced capabilities
    """
    
    def __init__(self, config_path: str = "parking_dataset/config/parking_zones.yaml"):
        self.config_path = config_path
        self.spot_history: Dict[str, deque] = {}  # For temporal smoothing
        self.event_log: List[ParkingEvent] = []
        self.previous_states: Dict[str, SlotStatus] = {}
        self.parking_type: Optional[ParkingType] = None
        
        # Detection thresholds
        self.temporal_window = 5  # frames for smoothing
        self.confidence_threshold = 0.75
        self.occupancy_threshold = 0.3  # IoU threshold
        
    def classify_parking_type(self, frame: np.ndarray, zone_config: Dict) -> ParkingType:
        """Automatically classify the type of parking environment"""
        height, width = frame.shape[:2]
        
        # Analyze frame characteristics
        # Check if it's multi-level (has structural elements)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        structural_density = np.sum(edges > 0) / (height * width)
        
        # Check for ceiling (indicates basement or multi-level)
        upper_region = frame[:height//3, :]
        brightness_upper = np.mean(cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY))
        
        # Classify based on characteristics
        if structural_density > 0.15 and brightness_upper < 80:
            return ParkingType.BASEMENT
        elif structural_density > 0.2:
            return ParkingType.MULTI_LEVEL
        elif zone_config.get('total_spots', 0) > 50:
            return ParkingType.OPEN_LOT
        elif 'private' in zone_config.get('name', '').lower():
            return ParkingType.PRIVATE
        else:
            return ParkingType.OPEN_LOT
    
    def classify_slot_category(self, spot_id: str, zone_config: Dict) -> SlotCategory:
        """Classify slot category based on ID and configuration"""
        spot_id_upper = spot_id.upper()
        
        # Check for EV charging indicators
        if 'EV' in spot_id_upper or 'CHARGE' in spot_id_upper:
            return SlotCategory.EV_CHARGING
        
        # Check for disabled/reserved indicators
        if 'DISABLED' in spot_id_upper or 'HANDICAP' in spot_id_upper or 'RESERVED' in spot_id_upper:
            return SlotCategory.DISABLED
        
        # Check for VIP indicators
        if 'VIP' in spot_id_upper:
            return SlotCategory.VIP
        
        # Check for bike indicators
        if 'BIKE' in spot_id_upper or 'MOTORCYCLE' in spot_id_upper or 'TWO' in spot_id_upper:
            return SlotCategory.BIKE
        
        # Default to car for standard slots
        return SlotCategory.CAR
    
    def apply_temporal_smoothing(self, slot_id: str, current_status: SlotStatus) -> SlotStatus:
        """Apply temporal smoothing to prevent flickering"""
        if slot_id not in self.spot_history:
            self.spot_history[slot_id] = deque(maxlen=self.temporal_window)
        
        history = self.spot_history[slot_id]
        history.append(current_status)
        
        # Require majority consensus
        if len(history) >= 3:
            occupied_count = sum(1 for s in history if s == SlotStatus.OCCUPIED)
            unoccupied_count = sum(1 for s in history if s == SlotStatus.UNOCCUPIED)
            
            if occupied_count > unoccupied_count:
                return SlotStatus.OCCUPIED
            elif unoccupied_count > occupied_count:
                return SlotStatus.UNOCCUPIED
        
        # Default to previous state or current
        return current_status
    
    def detect_events(self, slot_id: str, current_status: SlotStatus, 
                      vehicle_type: Optional[str], confidence: float) -> Optional[ParkingEvent]:
        """Detect entry and exit events"""
        if slot_id not in self.previous_states:
            self.previous_states[slot_id] = SlotStatus.UNOCCUPIED
            return None
        
        previous_status = self.previous_states[slot_id]
        event = None
        timestamp = datetime.now().isoformat()
        
        # Entry event: unoccupied -> occupied
        if previous_status == SlotStatus.UNOCCUPIED and current_status == SlotStatus.OCCUPIED:
            event = ParkingEvent(
                event_type='entry',
                slot_id=slot_id,
                vehicle_type=vehicle_type or 'unknown',
                timestamp=timestamp,
                confidence=confidence
            )
            print(f"[EVENT] Vehicle entered slot {slot_id}")
        
        # Exit event: occupied -> unoccupied
        elif previous_status == SlotStatus.OCCUPIED and current_status == SlotStatus.UNOCCUPIED:
            event = ParkingEvent(
                event_type='exit',
                slot_id=slot_id,
                vehicle_type=vehicle_type or 'unknown',
                timestamp=timestamp,
                confidence=confidence
            )
            print(f"[EVENT] Vehicle exited slot {slot_id}")
        
        # Update previous state
        self.previous_states[slot_id] = current_status
        
        return event
    
    def calculate_confidence(self, is_occupied: bool, overlap_ratio: float, 
                           vehicle_confidence: float = 0.0) -> float:
        """Calculate overall confidence score for detection"""
        if is_occupied:
            # For occupied: combine vehicle detection confidence and overlap
            base_confidence = vehicle_confidence * 0.7 + overlap_ratio * 0.3
        else:
            # For unoccupied: based on lack of overlap
            base_confidence = 0.9 - (overlap_ratio * 0.5)
        
        return min(1.0, max(0.0, base_confidence))
    
    def draw_enhanced_detections(self, frame: np.ndarray, 
                               spots: List[EnhancedParkingSpot]) -> np.ndarray:
        """Draw enhanced visualizations with all metadata"""
        annotated = frame.copy()
        
        for spot in spots:
            x1, y1, x2, y2 = spot.bounding_box
            
            # Color based on status
            if spot.status == SlotStatus.OCCUPIED:
                color = (0, 0, 255)  # Red
                status_text = "OCCUPIED"
            else:
                color = (0, 255, 0)  # Green
                status_text = "UNOCCUPIED"
            
            # Draw bounding box with thickness based on confidence
            thickness = max(2, int(spot.confidence * 6))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw status label
            label = f"{spot.slot_id}: {status_text}"
            font_scale = 0.6
            text_thickness = 2
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )
            
            # Draw label background
            label_y = y1 - 10 if y1 > 30 else y1 + text_height + 10
            cv2.rectangle(annotated,
                         (x1, label_y - text_height - 5),
                         (x1 + text_width + 10, label_y + 5),
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1 + 5, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
            
            # Draw category indicator
            category_text = spot.category.value
            cat_y = label_y + text_height + 15
            cv2.putText(annotated, category_text, (x1, cat_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw confidence score
            conf_text = f"Conf: {spot.confidence:.2f}"
            conf_y = y2 + 15 if y2 < frame.shape[0] - 20 else y1 - 20
            cv2.putText(annotated, conf_text, (x1, conf_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated
    
    def generate_summary_overlay(self, frame: np.ndarray, 
                                analysis: ParkingAnalysis) -> np.ndarray:
        """Generate summary overlay on frame"""
        annotated = frame.copy()
        
        # Create semi-transparent overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        # Draw summary information
        y_offset = 35
        line_height = 25
        
        # Title
        cv2.putText(annotated, f"Parking Type: {analysis.parking_type.value}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        # Statistics
        cv2.putText(annotated, f"Total: {analysis.total_slots} | Occupied: {analysis.occupied_slots} | Free: {analysis.free_slots}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Occupancy rate with color
        rate = analysis.occupied_slots / analysis.total_slots if analysis.total_slots > 0 else 0
        rate_color = (0, 255, 0) if rate < 0.5 else (0, 165, 255) if rate < 0.8 else (0, 0, 255)
        cv2.putText(annotated, f"Occupancy Rate: {rate*100:.1f}%",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rate_color, 2)
        y_offset += line_height
        
        # Recent events
        if analysis.events:
            recent_events = analysis.events[-3:]  # Last 3 events
            cv2.putText(annotated, "Recent Events:",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            for event in recent_events:
                event_text = f"  {event.event_type.upper()}: Slot {event.slot_id}"
                cv2.putText(annotated, event_text,
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 18
        
        return annotated


# Factory function for backward compatibility
def get_enhanced_parking_detector():
    """Get or create enhanced parking detector instance"""
    return EnhancedParkingDetector()
