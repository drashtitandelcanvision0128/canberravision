"""
Unified Multi-Detection Engine for Canberra Vision
Integrates PPE Detection, Vehicle Detection, Number Plate Recognition, and Parking Detection
"""

import cv2
import numpy as np
import torch
import time
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

# Try importing existing modules
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] YOLO not available")

try:
    from modules.ppe_detection import PPEDetector, get_ppe_detector
    PPE_AVAILABLE = True
except ImportError:
    PPE_AVAILABLE = False
    print("[WARNING] PPE detection not available")

# try:
#     from modules.parking_detection import ParkingDetector
#     PARKING_AVAILABLE = True
# except ImportError:
#     PARKING_AVAILABLE = False
#     print("[WARNING] Parking detection not available")

# Parking detection commented out as requested
PARKING_AVAILABLE = False

try:
    from src.ocr.license_plate_detector import LicensePlateDetector
    PLATE_AVAILABLE = True
except ImportError:
    PLATE_AVAILABLE = False
    print("[WARNING] License plate detector not available")


@dataclass
class VehicleInfo:
    """Vehicle detection data"""
    vehicle_id: str
    vehicle_type: str  # bike, car, truck, bus
    color: str
    confidence: float
    bbox: List[float]
    associated_persons: List[str] = field(default_factory=list)


@dataclass
class PPEInfo:
    """PPE detection data"""
    person_id: str
    helmet: bool
    seatbelt: bool
    vest: bool
    confidence: float
    bbox: List[float]
    vehicle_type: str
    associated_vehicle_id: Optional[str] = None


@dataclass
class PlateInfo:
    """Number plate detection data"""
    plate_id: str
    text: str
    confidence: float
    bbox: List[float]
    associated_vehicle_id: Optional[str] = None


# @dataclass
# class ParkingSlotInfo:
#     """Parking slot detection data"""
#     slot_id: int
#     occupied: bool
#     confidence: float
#     bbox: List[float]
#     associated_vehicle_id: Optional[str] = None

# Parking detection data class commented out as requested


@dataclass
class UnifiedDetectionResult:
    """Complete detection result for a frame"""
    timestamp: str
    source: str
    frame_number: int
    ppe_detections: List[PPEInfo] = field(default_factory=list)
    vehicle_detections: List[VehicleInfo] = field(default_factory=list)
    plate_detections: List[PlateInfo] = field(default_factory=list)
    # parking_detections: List[ParkingSlotInfo] = field(default_factory=list)  # Commented out parking detection
    processing_time_ms: float = 0.0


class UnifiedDetector:
    """
    Unified Multi-Detection Engine
    Performs simultaneous detection of PPE, Vehicles, Number Plates, and Parking Slots
    """
    
    # Vehicle type classification
    TWO_WHEELERS = {'bike', 'motorcycle', 'scooter', 'bicycle', '2-wheeler', 'twowheeler'}
    FOUR_WHEELERS = {'car', 'truck', 'bus', 'van', 'suv', '4-wheeler', 'fourwheeler', 'auto', 'rickshaw'}
    
    # Color ranges for vehicle color detection (HSV)
    COLOR_RANGES = {
        'white': ((0, 0, 200), (180, 30, 255)),
        'black': ((0, 0, 0), (180, 255, 50)),
        'gray': ((0, 0, 50), (180, 30, 200)),
        'red': ((0, 100, 100), (10, 255, 255)),
        'red2': ((160, 100, 100), (180, 255, 255)),  # Red wraps around
        'blue': ((100, 100, 100), (140, 255, 255)),
        'green': ((40, 100, 100), (80, 255, 255)),
        'yellow': ((20, 100, 100), (40, 255, 255)),
        'orange': ((10, 100, 100), (20, 255, 255)),
        'purple': ((140, 100, 100), (160, 255, 255)),
        'brown': ((10, 50, 50), (30, 150, 150)),
        'silver': ((0, 0, 150), (180, 20, 220)),
    }
    
    def __init__(self, 
                 model_path: str = "yolo26n.pt",
                 use_gpu: bool = True,
                 enable_ppe: bool = True,
                 enable_vehicles: bool = True,
                 enable_plates: bool = True,
                 # enable_parking: bool = True  # Commented out parking detection
                 ):
        """
        Initialize Unified Detector
        
        Args:
            model_path: Path to YOLO model
            use_gpu: Whether to use GPU acceleration
            enable_ppe: Enable PPE detection
            enable_vehicles: Enable vehicle detection
            enable_plates: Enable number plate recognition
            # enable_parking: Enable parking slot detection  # Commented out parking detection
        """
        self.model_path = model_path
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cuda:0' if self.use_gpu else 'cpu'
        
        # Detection toggles
        self.enable_ppe = enable_ppe and PPE_AVAILABLE
        self.enable_vehicles = enable_vehicles
        self.enable_plates = enable_plates and PLATE_AVAILABLE
        # self.enable_parking = enable_parking and PARKING_AVAILABLE  # Commented out parking detection
        self.enable_parking = False  # Force disabled
        
        # Initialize components
        self.model = None
        self.ppe_detector = None
        self.plate_detector = None
        # self.parking_detector = None  # Commented out parking detection
        
        # Tracking for consistency across frames
        self.vehicle_trackers = {}
        self.person_trackers = {}
        self.plate_trackers = {}
        self.next_vehicle_id = 1
        self.next_person_id = 1
        self.next_plate_id = 1
        
        # Statistics
        self.stats = {
            'total_frames_processed': 0,
            'total_vehicles_detected': 0,
            'total_persons_detected': 0,
            'total_plates_recognized': 0,
            # 'total_parking_slots': 0,  # Commented out parking detection
            'avg_processing_time_ms': 0
        }
        
        self._initialize_components()
        
        print(f"[INFO] Unified Detector initialized")
        print(f"[INFO] GPU: {'Enabled' if self.use_gpu else 'Disabled'}")
        print(f"[INFO] PPE: {'Enabled' if self.enable_ppe else 'Disabled'}")
        print(f"[INFO] Vehicles: {'Enabled' if self.enable_vehicles else 'Disabled'}")
        print(f"[INFO] Plates: {'Enabled' if self.enable_plates else 'Disabled'}")
        # print(f"[INFO] Parking: {'Enabled' if self.enable_parking else 'Disabled'}")  # Commented out parking detection
        print(f"[INFO] Parking: Disabled (commented out)")
    
    def _initialize_components(self):
        """Initialize all detection components"""
        # Initialize YOLO model for vehicle detection
        if YOLO_AVAILABLE and self.enable_vehicles:
            try:
                self.model = YOLO(self.model_path)
                if self.use_gpu:
                    self.model.to(self.device)
                print(f"[INFO] YOLO model loaded on {self.device}")
            except Exception as e:
                print(f"[ERROR] Failed to load YOLO model: {e}")
                self.model = None
        
        # Initialize PPE detector
        if self.enable_ppe:
            try:
                self.ppe_detector = get_ppe_detector(self.model_path)
                print("[INFO] PPE detector initialized")
            except Exception as e:
                print(f"[ERROR] Failed to initialize PPE detector: {e}")
                self.ppe_detector = None
        
        # Initialize license plate detector
        if self.enable_plates:
            try:
                self.plate_detector = LicensePlateDetector()
                print("[INFO] License plate detector initialized")
            except Exception as e:
                print(f"[ERROR] Failed to initialize plate detector: {e}")
                self.plate_detector = None
        
        # Initialize parking detector - Commented out as requested
        # if self.enable_parking:
        #     try:
        #         self.parking_detector = ParkingDetector()
        #         print("[INFO] Parking detector initialized")
        #     except Exception as e:
        #         print(f"[ERROR] Failed to initialize parking detector: {e}")
        #         self.parking_detector = None
    
    def detect_frame(self, frame: np.ndarray, frame_number: int = 0, 
                     source: str = "unknown") -> UnifiedDetectionResult:
        """
        Perform unified multi-detection on a single frame
        
        Args:
            frame: Input frame (BGR format)
            frame_number: Frame number for tracking
            source: Source identifier (WEBCAM, VIDEO, IMAGE)
            
        Returns:
            UnifiedDetectionResult with all detections
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        result = UnifiedDetectionResult(
            timestamp=timestamp,
            source=source,
            frame_number=frame_number
        )
        
        # Step 1: Detect vehicles (foundation for other detections)
        vehicles = []
        if self.enable_vehicles and self.model:
            vehicles = self._detect_vehicles(frame)
            result.vehicle_detections = vehicles
        
        # Step 2: Detect PPE (persons and their safety equipment)
        persons = []
        if self.enable_ppe and self.ppe_detector:
            persons = self._detect_ppe(frame, vehicles)
            result.ppe_detections = persons
        
        # Step 3: Associate persons with vehicles
        self._associate_persons_with_vehicles(persons, vehicles)
        
        # Step 4: Apply strict PPE rules based on vehicle type
        self._apply_ppe_rules(persons, vehicles)
        
        # Step 5: Detect license plates for 4-wheelers
        plates = []
        if self.enable_plates and self.plate_detector:
            plates = self._detect_plates(frame, vehicles)
            result.plate_detections = plates
        
        # Step 6: Detect parking slots - Commented out as requested
        # parking_slots = []
        # if self.enable_parking and self.parking_detector:
        #     parking_slots = self._detect_parking(frame, vehicles)
        #     result.parking_detections = parking_slots
        
        # Step 7: Associate parking slots with vehicles - Commented out as requested
        # self._associate_parking_with_vehicles(parking_slots, vehicles)
        
        # Calculate processing time
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        # Update statistics
        self._update_stats(result)
        
        return result
    
    def _detect_vehicles(self, frame: np.ndarray) -> List[VehicleInfo]:
        """Detect vehicles and identify their type and color"""
        vehicles = []
        
        try:
            results = self.model.predict(
                source=frame,
                conf=0.5,
                iou=0.5,
                device=self.device,
                verbose=False
            )
            
            if results and len(results) > 0:
                detection = results[0]
                
                if hasattr(detection, 'boxes') and detection.boxes is not None:
                    boxes = detection.boxes
                    
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = detection.names.get(class_id, f"class_{class_id}").lower()
                        
                        # Check if it's a vehicle
                        vehicle_type = self._classify_vehicle_type(class_name)
                        if vehicle_type:
                            # Detect color
                            vehicle_region = frame[int(y1):int(y2), int(x1):int(x2)]
                            color = self._detect_vehicle_color(vehicle_region)
                            
                            # Generate or track vehicle ID
                            vehicle_id = self._get_or_create_vehicle_id(
                                [float(x1), float(y1), float(x2), float(y2)],
                                vehicle_type
                            )
                            
                            vehicle = VehicleInfo(
                                vehicle_id=vehicle_id,
                                vehicle_type=vehicle_type,
                                color=color,
                                confidence=confidence,
                                bbox=[float(x1), float(y1), float(x2), float(y2)]
                            )
                            vehicles.append(vehicle)
        
        except Exception as e:
            print(f"[ERROR] Vehicle detection failed: {e}")
        
        return vehicles
    
    def _classify_vehicle_type(self, class_name: str) -> Optional[str]:
        """Classify detected class into vehicle type"""
        class_lower = class_name.lower()
        
        if class_lower in self.TWO_WHEELERS or 'bike' in class_lower or 'motor' in class_lower:
            return 'bike'
        elif class_lower in self.FOUR_WHEELERS:
            if 'truck' in class_lower or 'lorry' in class_lower:
                return 'truck'
            elif 'bus' in class_lower:
                return 'bus'
            else:
                return 'car'
        
        return None
    
    def _detect_vehicle_color(self, vehicle_region: np.ndarray) -> str:
        """Detect the dominant color of a vehicle"""
        if vehicle_region.size == 0:
            return 'unknown'
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2HSV)
            
            color_scores = {}
            
            for color_name, (lower, upper) in self.COLOR_RANGES.items():
                if color_name == 'red2':  # Skip secondary red range
                    continue
                    
                lower = np.array(lower)
                upper = np.array(upper)
                
                # Handle red's wrap-around
                if color_name == 'red':
                    lower2 = np.array(self.COLOR_RANGES['red2'][0])
                    upper2 = np.array(self.COLOR_RANGES['red2'][1])
                    mask1 = cv2.inRange(hsv, lower, upper)
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    mask = cv2.inRange(hsv, lower, upper)
                
                # Calculate color score
                color_score = np.sum(mask > 0)
                base_color = color_name.replace('2', '')
                color_scores[base_color] = color_scores.get(base_color, 0) + color_score
            
            # Get dominant color
            if color_scores:
                dominant_color = max(color_scores, key=color_scores.get)
                total_score = sum(color_scores.values())
                
                if total_score > 0 and color_scores[dominant_color] / total_score > 0.1:
                    return dominant_color
            
            return 'unknown'
            
        except Exception as e:
            return 'unknown'
    
    def _detect_ppe(self, frame: np.ndarray, vehicles: List[VehicleInfo]) -> List[PPEInfo]:
        """Detect PPE (helmet, seatbelt, vest) for persons"""
        persons = []
        
        try:
            # Use PPE detector
            ppe_result = self.ppe_detector.detect(frame, confidence_threshold=0.3)
            
            # Convert PPE results to PPEInfo
            if hasattr(ppe_result, 'persons'):
                for person in ppe_result.persons:
                    person_id = self._get_or_create_person_id(person.bbox)
                    
                    # Determine vehicle context from person's position
                    vehicle_type = self._infer_vehicle_type_from_position(
                        person.bbox, vehicles
                    )
                    
                    ppe_info = PPEInfo(
                        person_id=person_id,
                        helmet=person.helmet.present if hasattr(person, 'helmet') else False,
                        seatbelt=person.seatbelt.present if hasattr(person, 'seatbelt') else False,
                        vest=person.vest.present if hasattr(person, 'vest') else False,
                        confidence=person.confidence if hasattr(person, 'confidence') else 0.5,
                        bbox=person.bbox if hasattr(person, 'bbox') else [0, 0, 0, 0],
                        vehicle_type=vehicle_type
                    )
                    persons.append(ppe_info)
        
        except Exception as e:
            print(f"[ERROR] PPE detection failed: {e}")
        
        return persons
    
    def _apply_ppe_rules(self, persons: List[PPEInfo], vehicles: List[VehicleInfo]):
        """
        Apply strict PPE detection rules:
        1. Helmet = true → seatbelt MUST be false
        2. Seatbelt = true → helmet MUST be false
        3. 2-wheeler → ONLY helmet detection allowed
        4. 4-wheeler → ONLY seatbelt detection allowed
        """
        for person in persons:
            # Rule 1 & 2: Mutual exclusivity of helmet and seatbelt
            if person.helmet and person.seatbelt:
                # If both detected, keep the one with higher confidence
                # or prioritize based on vehicle context
                if person.vehicle_type == 'bike':
                    person.seatbelt = False
                else:
                    person.helmet = False
            
            # Rule 3: 2-wheeler → ONLY helmet
            if person.vehicle_type == 'bike':
                person.seatbelt = False  # Force seatbelt to false for bikes
            
            # Rule 4: 4-wheeler → ONLY seatbelt
            elif person.vehicle_type in ['car', 'truck', 'bus']:
                person.helmet = False  # Force helmet to false for 4-wheelers
    
    def _associate_persons_with_vehicles(self, persons: List[PPEInfo], vehicles: List[VehicleInfo]):
        """Associate detected persons with their nearest vehicles"""
        for person in persons:
            person_center = self._get_bbox_center(person.bbox)
            
            nearest_vehicle = None
            min_distance = float('inf')
            
            for vehicle in vehicles:
                vehicle_center = self._get_bbox_center(vehicle.bbox)
                
                # Calculate distance between person and vehicle
                distance = np.sqrt(
                    (person_center[0] - vehicle_center[0])**2 + 
                    (person_center[1] - vehicle_center[1])**2
                )
                
                # Check if person is within or near vehicle bounding box
                if self._is_point_in_bbox(person_center, vehicle.bbox, margin=50):
                    if distance < min_distance:
                        min_distance = distance
                        nearest_vehicle = vehicle
            
            if nearest_vehicle:
                person.associated_vehicle_id = nearest_vehicle.vehicle_id
                person.vehicle_type = nearest_vehicle.vehicle_type
                
                if person.person_id not in nearest_vehicle.associated_persons:
                    nearest_vehicle.associated_persons.append(person.person_id)
    
    def _detect_plates(self, frame: np.ndarray, vehicles: List[VehicleInfo]) -> List[PlateInfo]:
        """Detect license plates for 4-wheeler vehicles"""
        plates = []
        
        try:
            for vehicle in vehicles:
                # Only detect plates for 4-wheelers
                if vehicle.vehicle_type not in ['car', 'truck', 'bus']:
                    continue
                
                # Extract vehicle region
                x1, y1, x2, y2 = map(int, vehicle.bbox)
                vehicle_region = frame[y1:y2, x1:x2]
                
                if vehicle_region.size == 0:
                    continue
                
                # Try to detect plate in vehicle region
                plate_result = self.plate_detector.detect_plate(vehicle_region)
                
                if plate_result and plate_result.get('text'):
                    # Adjust plate bbox to global coordinates
                    plate_bbox = plate_result.get('bbox', [0, 0, 0, 0])
                    global_bbox = [
                        plate_bbox[0] + x1,
                        plate_bbox[1] + y1,
                        plate_bbox[2] + x1,
                        plate_bbox[3] + y1
                    ]
                    
                    plate_id = f"PLATE_{self.next_plate_id:04d}"
                    self.next_plate_id += 1
                    
                    plate = PlateInfo(
                        plate_id=plate_id,
                        text=plate_result['text'],
                        confidence=plate_result.get('confidence', 0.5),
                        bbox=global_bbox,
                        associated_vehicle_id=vehicle.vehicle_id
                    )
                    plates.append(plate)
        
        except Exception as e:
            print(f"[ERROR] Plate detection failed: {e}")
        
        return plates
    
    # def _detect_parking(self, frame: np.ndarray, vehicles: List[VehicleInfo]) -> List[ParkingSlotInfo]:
    #     """Detect parking slots and their occupancy - Commented out as requested"""
    #     slots = []
    #     
    #     try:
    #         parking_result = self.parking_detector.detect(frame)
    #         
    #         if hasattr(parking_result, 'slots'):
    #             for i, slot in enumerate(parking_result.slots):
    #                 occupied = slot.get('occupied', False)
    #                 
    #                 # Check if any vehicle is in this slot
    #                 associated_vehicle = None
    #                 slot_bbox = slot.get('bbox', [0, 0, 0, 0])
    #                 
    #                 for vehicle in vehicles:
    #                     vehicle_center = self._get_bbox_center(vehicle.bbox)
    #                     if self._is_point_in_bbox(vehicle_center, slot_bbox):
    #                         associated_vehicle = vehicle.vehicle_id
    #                         occupied = True
    #                         break
    #                 
    #                 slot_info = ParkingSlotInfo(
    #                     slot_id=i + 1,
    #                     occupied=occupied,
    #                     confidence=slot.get('confidence', 0.5),
    #                     bbox=slot.get('bbox', [0, 0, 0, 0]),
    #                     associated_vehicle_id=associated_vehicle
    #                 )
    #                 slots.append(slot_info)
    #     
    #     except Exception as e:
    #         print(f"[ERROR] Parking detection failed: {e}")
    #     
    #     return slots
    
    # def _associate_parking_with_vehicles(self, slots: List[ParkingSlotInfo], vehicles: List[VehicleInfo]):
    #     """Associate parking slots with vehicles inside them - Commented out as requested"""
    #     for slot in slots:
    #         if slot.occupied and not slot.associated_vehicle_id:
    #             # Find vehicle in this slot
    #             for vehicle in vehicles:
    #                 vehicle_center = self._get_bbox_center(vehicle.bbox)
    #                 if self._is_point_in_bbox(vehicle_center, slot.bbox):
    #                     slot.associated_vehicle_id = vehicle.vehicle_id
    #                     break
    
    def _infer_vehicle_type_from_position(self, person_bbox: List[float], 
                                          vehicles: List[VehicleInfo]) -> str:
        """Infer vehicle type based on person's position relative to vehicles"""
        person_center = self._get_bbox_center(person_bbox)
        
        for vehicle in vehicles:
            if self._is_point_in_bbox(person_center, vehicle.bbox, margin=30):
                return vehicle.vehicle_type
        
        return 'unknown'
    
    def _get_or_create_vehicle_id(self, bbox: List[float], vehicle_type: str) -> str:
        """Get existing vehicle ID or create new one based on tracking"""
        center = self._get_bbox_center(bbox)
        
        # Check if this matches an existing tracked vehicle
        for vid, tracker in self.vehicle_trackers.items():
            tracker_center = self._get_bbox_center(tracker['bbox'])
            distance = np.sqrt(
                (center[0] - tracker_center[0])**2 + 
                (center[1] - tracker_center[1])**2
            )
            
            # If close enough and same type, consider it the same vehicle
            if distance < 100 and tracker['type'] == vehicle_type:
                tracker['bbox'] = bbox
                tracker['last_seen'] = time.time()
                return vid
        
        # Create new vehicle ID
        vehicle_id = f"VEH_{self.next_vehicle_id:04d}"
        self.next_vehicle_id += 1
        
        self.vehicle_trackers[vehicle_id] = {
            'bbox': bbox,
            'type': vehicle_type,
            'last_seen': time.time()
        }
        
        return vehicle_id
    
    def _get_or_create_person_id(self, bbox: List[float]) -> str:
        """Get existing person ID or create new one based on tracking"""
        center = self._get_bbox_center(bbox)
        
        # Check if this matches an existing tracked person
        for pid, tracker in self.person_trackers.items():
            tracker_center = self._get_bbox_center(tracker['bbox'])
            distance = np.sqrt(
                (center[0] - tracker_center[0])**2 + 
                (center[1] - tracker_center[1])**2
            )
            
            # If close enough, consider it the same person
            if distance < 80:
                tracker['bbox'] = bbox
                tracker['last_seen'] = time.time()
                return pid
        
        # Create new person ID
        person_id = f"PER_{self.next_person_id:04d}"
        self.next_person_id += 1
        
        self.person_trackers[person_id] = {
            'bbox': bbox,
            'last_seen': time.time()
        }
        
        return person_id
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        return (0, 0)
    
    def _is_point_in_bbox(self, point: Tuple[float, float], bbox: List[float], 
                          margin: float = 0) -> bool:
        """Check if a point is inside a bounding box with optional margin"""
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            px, py = point
            return (x1 - margin <= px <= x2 + margin and 
                    y1 - margin <= py <= y2 + margin)
        return False
    
    def _update_stats(self, result: UnifiedDetectionResult):
        """Update detection statistics"""
        self.stats['total_frames_processed'] += 1
        self.stats['total_vehicles_detected'] += len(result.vehicle_detections)
        self.stats['total_persons_detected'] += len(result.ppe_detections)
        self.stats['total_plates_recognized'] += len(result.plate_detections)
        # self.stats['total_parking_slots'] += len(result.parking_detections)  # Commented out parking detection
        
        # Update average processing time
        n = self.stats['total_frames_processed']
        current_avg = self.stats['avg_processing_time_ms']
        self.stats['avg_processing_time_ms'] = (
            (current_avg * (n - 1) + result.processing_time_ms) / n
        )
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        return self.stats.copy()
    
    def reset_trackers(self):
        """Reset all trackers (useful when switching sources)"""
        self.vehicle_trackers.clear()
        self.person_trackers.clear()
        self.plate_trackers.clear()
        self.next_vehicle_id = 1
        self.next_person_id = 1
        self.next_plate_id = 1


# Global detector instance
_unified_detector = None


def get_unified_detector(model_path: str = "yolo26n.pt", **kwargs) -> UnifiedDetector:
    """Get or create global unified detector instance"""
    global _unified_detector
    if _unified_detector is None:
        _unified_detector = UnifiedDetector(model_path, **kwargs)
    return _unified_detector


if __name__ == "__main__":
    print("[INFO] Unified Detection Engine ready")
    print("[INFO] Available components:")
    print(f"  - YOLO: {YOLO_AVAILABLE}")
    print(f"  - PPE: {PPE_AVAILABLE}")
    # print(f"  - Parking: {PARKING_AVAILABLE}")  # Commented out parking detection
    print("  - Parking: Disabled (commented out)")
    print(f"  - Plates: {PLATE_AVAILABLE}")
