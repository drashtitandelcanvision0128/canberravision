"""
Parking Space Detection Module for YOLO26
Handles real-time parking space occupancy detection with zone management
"""

import cv2
import numpy as np
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from ultralytics import YOLO
import time
import threading
from collections import defaultdict

# Import parking line detector for slot-based detection
from modules.parking_line_detector import ParkingLineDetector, DetectedParkingSlot

@dataclass
class ParkingSpot:
    """Represents a single parking spot detection result"""
    spot_id: str
    zone_id: str
    camera_id: str
    status: str  # OCCUPIED or EMPTY
    confidence: float
    vehicle_type: Optional[str] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    timestamp: str = ""

@dataclass
class ZoneResult:
    """Represents parking zone detection results"""
    zone_id: str
    zone_name: str
    total_spots: int
    occupied_spots: int
    empty_spots: int
    occupancy_rate: float
    spot_details: List[ParkingSpot]
    timestamp: str

class ParkingDetector:
    """Main parking detection system with real-time processing"""
    
    def __init__(self, config_path: str = "parking_dataset/config/parking_zones.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.spot_coordinates = self._parse_spot_coordinates()
        self.model = None
        self.device = self._get_device()
        self.detection_cache = {}
        self.last_detection_time = defaultdict(float)
        self.processing_interval = self.config.get('detection_config', {}).get('processing_interval', 1)
        
        # Initialize parking line detector for dynamic slot detection
        self.line_detector = ParkingLineDetector()
        
        # Performance metrics
        self.detection_times = []
        self.total_detections = 0
        
    def _load_config(self) -> Dict:
        """Load parking configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[INFO] Loaded parking configuration: {len(config['zones'])} zones")
            return config
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            raise
            
    def _parse_spot_coordinates(self) -> Dict[str, Dict]:
        """Parse spot coordinates into accessible format"""
        spot_coords = {}
        
        for zone_id, zone_config in self.config['zones'].items():
            spot_coords[zone_id] = {}
            for camera_id, camera_config in zone_config['coordinates'].items():
                spot_coords[zone_id][camera_id] = camera_config['spots']
                
        return spot_coords
        
    def _get_device(self) -> str:
        """Get optimal device for inference"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
                print(f"[INFO] Using CUDA: {torch.cuda.get_device_name()}")
        except ImportError:
            pass
        return "cpu"
        
    def load_model(self, model_path: str = "yolov8n.pt"):
        """Load YOLO model for vehicle detection"""
        try:
            print(f"[INFO] Loading model: {model_path}")
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"[INFO] Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
            
    def detect_vehicles(self, frame: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """Detect vehicles in frame using YOLO - LOWER threshold for high recall"""
        if self.model is None:
            self.load_model()
            
        try:
            results = self.model(
                frame, 
                conf=confidence_threshold,
                iou=0.3,  # Lower IOU for more detections
                device=self.device,
                verbose=False
            )
            
            vehicles = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[cls]
                        
                        # Filter for vehicle types
                        if class_name.lower() in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                            vehicles.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': conf,
                                'class': class_name
                            })
                            print(f"[VEHICLE] Detected {class_name} at ({int(x1)},{int(y1)},{int(x2)},{int(y2)}) conf={conf:.2f}")
                            
            print(f"[DETECT] Total vehicles found: {len(vehicles)}")
            return vehicles
            
        except Exception as e:
            print(f"[ERROR] Vehicle detection failed: {e}")
            return []
            
    def check_parking_spot_occupancy(self, spot_coords: Tuple[int, int, int, int], 
                                   vehicles: List[Dict], 
                                   overlap_threshold: float = 0.2) -> Tuple[bool, Optional[Dict], float]:
        """Check if a parking spot is occupied - RELAXED matching for HIGH sensitivity"""
        best_vehicle = None
        max_overlap = 0.0
        
        # Expand slot boundaries by 15% for relaxed matching
        x1_s, y1_s, x2_s, y2_s = spot_coords
        w_s = x2_s - x1_s
        h_s = y2_s - y1_s
        
        # Expanded slot (15% larger)
        expand_x = int(w_s * 0.15)
        expand_y = int(h_s * 0.15)
        expanded_slot = (
            x1_s - expand_x,
            y1_s - expand_y,
            x2_s + expand_x,
            y2_s + expand_y
        )
        
        for vehicle in vehicles:
            x1_v, y1_v, x2_v, y2_v = vehicle['bbox']
            
            # Condition 1: Overlap ratio (20% threshold)
            overlap = self._calculate_overlap_ratio(expanded_slot, vehicle['bbox'])
            
            # Condition 2: Center point inside slot
            center_x = (x1_v + x2_v) // 2
            center_y = (y1_v + y2_v) // 2
            center_inside = (x1_s <= center_x <= x2_s) and (y1_s <= center_y <= y2_s)
            
            # Condition 3: Any touch/intersection
            touches = self._boxes_intersect(expanded_slot, vehicle['bbox'])
            
            # RELAXED: Mark as match if ANY condition met
            is_match = overlap >= overlap_threshold or center_inside or touches
            
            if is_match and overlap > max_overlap:
                max_overlap = overlap
                best_vehicle = vehicle
                
            # Debug output
            if overlap > 0.05 or center_inside or touches:
                print(f"[MATCH-CHECK] Spot {spot_coords} vs {vehicle['class']} at {vehicle['bbox']}")
                print(f"              Overlap: {overlap:.1%}, Center inside: {center_inside}, Touches: {touches}")
                if is_match:
                    print(f"              ✓ MATCH FOUND!")
        
        is_occupied = max_overlap >= overlap_threshold or best_vehicle is not None
        
        if is_occupied and best_vehicle:
            print(f"[OCCUPIED] Spot occupied with {max_overlap:.1%} overlap by {best_vehicle['class']}")
        elif is_occupied:
            print(f"[OCCUPIED] Spot occupied (relaxed match)")
            
        return is_occupied, best_vehicle if is_occupied else None, max_overlap
        
    def _boxes_intersect(self, box1: Tuple[int, int, int, int], 
                        box2: Tuple[int, int, int, int]) -> bool:
        """Check if two boxes intersect/touch"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)
        
    def _calculate_overlap_ratio(self, spot_bbox: Tuple[int, int, int, int], 
                               vehicle_bbox: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between spot and vehicle bounding boxes"""
        x1_spot, y1_spot, x2_spot, y2_spot = spot_bbox
        x1_veh, y1_veh, x2_veh, y2_veh = vehicle_bbox
        
        # Calculate intersection
        x1_intersect = max(x1_spot, x1_veh)
        y1_intersect = max(y1_spot, y1_veh)
        x2_intersect = min(x2_spot, x2_veh)
        y2_intersect = min(y2_spot, y2_veh)
        
        if x2_intersect <= x1_intersect or y2_intersect <= y1_intersect:
            return 0.0
            
        intersection_area = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)
        spot_area = (x2_spot - x1_spot) * (y2_spot - y1_spot)
        
        return intersection_area / spot_area if spot_area > 0 else 0.0
        
    def detect_all_cars_in_frame(self, frame: np.ndarray) -> List[ParkingSpot]:
        """Detect ALL cars in the frame dynamically, not just predefined spots"""
        try:
            # Detect vehicles using YOLO
            vehicles = self.detect_vehicles(frame)
            detected_cars = []
            car_count = 0
            
            for vehicle in vehicles:
                x1, y1, x2, y2 = vehicle['bbox']
                confidence = vehicle['confidence']
                class_name = vehicle['class']
                
                # Only process cars and similar vehicles
                if class_name.lower() in ['car', 'truck', 'bus', 'van', 'suv']:
                    car_count += 1
                    spot_id = f"CAR-{car_count:03d}"  # Dynamic ID like CAR-001, CAR-002, etc.
                    
                    # Create a ParkingSpot for each detected car
                    car_spot = ParkingSpot(
                        spot_id=spot_id,
                        zone_id="dynamic_detection",
                        camera_id="all_cameras",
                        status="OCCUPIED",
                        confidence=confidence,
                        vehicle_type=class_name,
                        bounding_box=(int(x1), int(y1), int(x2), int(y2)),
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    detected_cars.append(car_spot)
            
            print(f"[INFO] Detected {len(detected_cars)} cars in frame")
            return detected_cars
            
        except Exception as e:
            print(f"[ERROR] Failed to detect all cars: {e}")
            return []

    def process_camera_frame(self, frame: np.ndarray, camera_id: str, zone_id: str, force_refresh: bool = True) -> List[ParkingSpot]:
        """Process a single camera frame and detect parking spot occupancy - DISABLED CACHE for debugging"""
        start_time = time.time()
        
        current_time = time.time()
        camera_key = f"{zone_id}_{camera_id}"
        
        # DISABLED: Check cache only if not forcing refresh
        if not force_refresh and current_time - self.last_detection_time[camera_key] < self.processing_interval:
            cached = self.detection_cache.get(camera_key, [])
            occupied = len([s for s in cached if s.status == "OCCUPIED"])
            empty = len([s for s in cached if s.status == "EMPTY"])
            print(f"[CACHE] Returning {len(cached)} cached spots for {camera_key}: {occupied} occupied, {empty} empty")
            return cached
            
        try:
            # Detect vehicles
            vehicles = self.detect_vehicles(frame)
            print(f"[DETECT] Found {len(vehicles)} vehicles in frame")
            
            # Get spot coordinates for this camera
            if zone_id not in self.spot_coordinates or camera_id not in self.spot_coordinates[zone_id]:
                print(f"[WARNING] No coordinates found for {zone_id}_{camera_id}")
                print(f"[DEBUG] Available zones: {list(self.spot_coordinates.keys())}")
                if zone_id in self.spot_coordinates:
                    print(f"[DEBUG] Available cameras in {zone_id}: {list(self.spot_coordinates[zone_id].keys())}")
                return []
                
            spot_coords = self.spot_coordinates[zone_id][camera_id]
            print(f"[CONFIG] Found {len(spot_coords)} parking spots configured for {zone_id}_{camera_id}")
            detected_spots = []
            occupied_count = 0
            empty_count = 0
            
            # Get frame dimensions for coordinate scaling
            frame_height, frame_width = frame.shape[:2]
            print(f"[FRAME] Video resolution: {frame_width}x{frame_height}")
            
            # Get configured resolution for this zone/camera - with safe navigation
            try:
                zones_config = self.config.get('zones', {})
                zone_config = zones_config.get(zone_id, {})
                cameras_config = zone_config.get('cameras', {})
                camera_config = cameras_config.get(camera_id, {})
                config_resolution = camera_config.get('resolution', [1920, 1080])
            except Exception as e:
                print(f"[WARNING] Could not get config resolution, using default: {e}")
                config_resolution = [1920, 1080]
            
            config_width, config_height = config_resolution
            
            # Calculate scaling factors
            scale_x = frame_width / config_width
            scale_y = frame_height / config_height
            print(f"[SCALE] Scaling coordinates by factor: x={scale_x:.3f}, y={scale_y:.3f}")
            
            # Check each parking spot
            for spot_id, coords in spot_coords.items():
                # Scale coordinates to match actual video resolution
                x1, y1, x2, y2 = coords
                scaled_coords = (
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                )
                
                is_occupied, vehicle, overlap_pct = self.check_parking_spot_occupancy(scaled_coords, vehicles)
                
                # Debug: Print overlap info for first few spots
                if int(spot_id.split('-')[1]) <= 5:  # First 5 spots
                    print(f"[DEBUG] Spot {spot_id}: coords={scaled_coords}, occupied={is_occupied}, overlap={overlap_pct:.1%}, vehicles={len(vehicles)}")
                    if vehicle:
                        print(f"[DEBUG]   -> Vehicle matched: {vehicle['class']} at {vehicle['bbox']}")
                
                spot = ParkingSpot(
                    spot_id=spot_id,
                    zone_id=zone_id,
                    camera_id=camera_id,
                    status="OCCUPIED" if is_occupied else "EMPTY",
                    confidence=vehicle['confidence'] if vehicle else 0.95,
                    vehicle_type=vehicle['class'] if vehicle else None,
                    bounding_box=vehicle['bbox'] if vehicle else scaled_coords,
                    timestamp=datetime.now().isoformat()
                )
                
                detected_spots.append(spot)
                if is_occupied:
                    occupied_count += 1
                else:
                    empty_count += 1
                
            print(f"[RESULT] Detected {len(detected_spots)} spots: {occupied_count} occupied, {empty_count} empty")
                
            # Cache results
            self.detection_cache[camera_key] = detected_spots
            self.last_detection_time[camera_key] = current_time
            
            # Update performance metrics
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            self.total_detections += 1
            
            if self.total_detections % 10 == 0:
                avg_time = np.mean(self.detection_times[-100:])
                print(f"[INFO] Avg detection time: {avg_time:.3f}s (total: {self.total_detections})")
                
            return detected_spots
            
        except Exception as e:
            print(f"[ERROR] Frame processing failed for {camera_id}: {e}")
            return []

    def process_all_detections(self, frame: np.ndarray, camera_id: str = "main", zone_id: str = "all") -> List[ParkingSpot]:
        """Process ALL predefined parking spots from config - FIXED to always include empty slots"""
        try:
            all_detections = []
            
            print(f"[DEBUG] Starting detection for ALL configured parking slots")
            print(f"[DEBUG] Available zones in config: {list(self.config.get('zones', {}).keys())}")
            print(f"[DEBUG] Spot coordinates loaded: {list(self.spot_coordinates.keys())}")
            
            # ALWAYS process ALL configured zones and cameras
            # This ensures ALL parking slots from config are included (both occupied and empty)
            for config_zone_id in self.config.get('zones', {}).keys():
                zone_config = self.config['zones'][config_zone_id]
                camera_ids = zone_config.get('camera_ids', [])
                
                print(f"[DEBUG] Processing zone '{config_zone_id}' with cameras: {camera_ids}")
                
                for config_camera_id in camera_ids:
                    try:
                        print(f"[DEBUG] Detecting spots for {config_zone_id}_{config_camera_id}")
                        zone_spots = self.process_camera_frame(frame, config_camera_id, config_zone_id)
                        
                        if zone_spots:
                            occupied = len([s for s in zone_spots if s.status == 'OCCUPIED'])
                            empty = len([s for s in zone_spots if s.status == 'EMPTY'])
                            print(f"[INFO] Zone {config_zone_id}_{config_camera_id}: {len(zone_spots)} spots ({occupied} occupied, {empty} empty)")
                            all_detections.extend(zone_spots)
                        else:
                            print(f"[WARNING] No spots detected for {config_zone_id}_{config_camera_id}")
                            
                    except Exception as e:
                        print(f"[ERROR] Failed to process {config_zone_id}_{config_camera_id}: {e}")
                        continue
            
            # Add dynamic car detection for any cars not covered by predefined spots
            try:
                print(f"[DEBUG] Checking for additional dynamic cars...")
                dynamic_cars = self.detect_all_cars_in_frame(frame)
                
                # Get boxes of already detected occupied spots
                predefined_car_boxes = [spot.bounding_box for spot in all_detections if spot.status == "OCCUPIED"]
                
                new_cars = 0
                for car in dynamic_cars:
                    car_already_covered = False
                    car_box = car.bounding_box
                    
                    for predefined_box in predefined_car_boxes:
                        overlap = self.calculate_overlap_ratio(car_box, predefined_box)
                        if overlap > 0.3:
                            car_already_covered = True
                            break
                    
                    if not car_already_covered:
                        all_detections.append(car)
                        new_cars += 1
                
                if new_cars > 0:
                    print(f"[INFO] Added {new_cars} dynamic cars not in predefined spots")
                        
            except Exception as e:
                print(f"[WARNING] Could not process dynamic cars: {e}")
            
            # Final summary
            occupied_count = len([s for s in all_detections if s.status == "OCCUPIED"])
            empty_count = len([s for s in all_detections if s.status == "EMPTY"])
            print(f"[INFO] TOTAL DETECTIONS: {len(all_detections)} spots ({occupied_count} occupied + {empty_count} empty)")
            
            return all_detections
            
        except Exception as e:
            print(f"[ERROR] Failed to process all detections: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to dynamic car detection only
            return self.detect_all_cars_in_frame(frame)
    
    def detect_slots_from_lines(self, frame: np.ndarray, 
                                overlap_threshold: float = 0.4) -> List[ParkingSpot]:
        """
        Detect parking slots from white/yellow line markings.
        Only valid parking slots with clear boundaries are detected.
        Roads and non-parking areas are ignored.
        
        Args:
            frame: Input video frame
            overlap_threshold: Minimum overlap (40%) for vehicle to count as occupied
            
        Returns:
            List of ParkingSpot objects with status
        """
        try:
            print("[LINE-DETECT] Detecting parking slots from line markings...")
            
            # Step 1: Detect parking slots from lines
            detected_slots = self.line_detector.detect_parking_slots(frame)
            
            if not detected_slots:
                print("[LINE-DETECT] No parking slots detected from lines")
                return []
            
            print(f"[LINE-DETECT] Found {len(detected_slots)} valid parking slots")
            
            # Step 2: Detect vehicles
            vehicles = self.detect_vehicles(frame)
            print(f"[LINE-DETECT] Found {len(vehicles)} vehicles in frame")
            
            # Step 3: Check occupancy for each slot using 40% overlap
            parking_spots = []
            occupied_count = 0
            empty_count = 0
            
            for slot in detected_slots:
                is_occupied = False
                best_vehicle = None
                max_overlap = 0.0
                
                # Check if any vehicle overlaps this slot by at least 40%
                for vehicle in vehicles:
                    overlap = self.line_detector.check_vehicle_in_slot(
                        vehicle['bbox'], slot, overlap_threshold
                    )
                    
                    # Calculate actual overlap ratio for confidence
                    actual_overlap = self.line_detector._calculate_overlap(
                        vehicle['bbox'], slot.bounding_box
                    )
                    
                    if actual_overlap > max_overlap:
                        max_overlap = actual_overlap
                        if overlap:  # Meets 40% threshold
                            is_occupied = True
                            best_vehicle = vehicle
                
                # Create ParkingSpot result
                spot = ParkingSpot(
                    spot_id=slot.slot_id,
                    zone_id="line_detected",
                    camera_id="auto",
                    status="OCCUPIED" if is_occupied else "EMPTY",
                    confidence=max_overlap if is_occupied else (1.0 - max_overlap),
                    vehicle_type=best_vehicle['class'] if best_vehicle else None,
                    bounding_box=slot.bounding_box,
                    timestamp=datetime.now().isoformat()
                )
                
                parking_spots.append(spot)
                
                if is_occupied:
                    occupied_count += 1
                else:
                    empty_count += 1
            
            print(f"[LINE-DETECT] Results: {occupied_count} occupied, {empty_count} empty out of {len(parking_spots)} slots")
            
            return parking_spots
            
        except Exception as e:
            print(f"[ERROR] Line-based detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []
            
    def process_zone(self, frames: Dict[str, np.ndarray], zone_id: str) -> ZoneResult:
        """Process all cameras for a specific zone"""
        zone_config = self.config['zones'][zone_id]
        all_spot_details = []
        
        # Process each camera in the zone
        for camera_id in zone_config['camera_ids']:
            if camera_id in frames:
                spots = self.process_camera_frame(frames[camera_id], camera_id, zone_id)
                all_spot_details.extend(spots)
                
        # Calculate zone statistics
        total_spots = len(all_spot_details)
        occupied_spots = sum(1 for spot in all_spot_details if spot.status == "OCCUPIED")
        empty_spots = total_spots - occupied_spots
        occupancy_rate = (occupied_spots / total_spots * 100) if total_spots > 0 else 0
        
        # Create zone result
        zone_result = ZoneResult(
            zone_id=zone_id,
            zone_name=zone_config['name'],
            total_spots=total_spots,
            occupied_spots=occupied_spots,
            empty_spots=empty_spots,
            occupancy_rate=round(occupancy_rate, 1),
            spot_details=all_spot_details,
            timestamp=datetime.now().isoformat()
        )
        
        return zone_result
        
    def process_all_zones(self, frames: Dict[str, np.ndarray]) -> Dict[str, ZoneResult]:
        """Process all parking zones"""
        results = {}
        
        for zone_id in self.config['zones'].keys():
            # Filter frames for this zone
            zone_frames = {cam_id: frame for cam_id, frame in frames.items() 
                          if cam_id in self.config['zones'][zone_id]['camera_ids']}
            
            if zone_frames:
                zone_result = self.process_zone(zone_frames, zone_id)
                results[zone_id] = zone_result
                
        return results
        
    def get_json_output(self, results: Dict[str, ZoneResult]) -> str:
        """Convert results to JSON format as specified"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "total_zones": len(results),
            "zones": {}
        }
        
        for zone_id, zone_result in results.items():
            zone_data = {
                "zone_id": zone_result.zone_id,
                "total_spots": zone_result.total_spots,
                "occupied_spots": zone_result.occupied_spots,
                "empty_spots": zone_result.empty_spots,
                "occupancy_rate": zone_result.occupancy_rate,
                "spot_details": []
            }
            
            for spot in zone_result.spot_details:
                spot_data = {
                    "spot_id": spot.spot_id,
                    "status": spot.status,
                    "confidence": round(spot.confidence, 3),
                    "vehicle_type": spot.vehicle_type,
                    "bounding_box": spot.bounding_box
                }
                zone_data["spot_details"].append(spot_data)
                
            output["zones"][zone_id] = zone_data
            
        return json.dumps(output, indent=2)
        
    def draw_detections(self, frame: np.ndarray, spots: List[ParkingSpot]) -> np.ndarray:
        """Draw clean minimal parking spot detections - no overlapping text"""
        annotated_frame = frame.copy()
        
        # Count for summary only
        occupied_count = len([s for s in spots if s.status == "OCCUPIED"])
        empty_count = len([s for s in spots if s.status == "EMPTY"])
        print(f"[DRAW] Drawing {len(spots)} spots: {occupied_count} occupied, {empty_count} empty")
        
        for spot in spots:
            x1, y1, x2, y2 = spot.bounding_box
            
            # Colors: Red=Occupied, Green=Empty
            if spot.status == "OCCUPIED":
                box_color = (0, 0, 255)      # Red
                text_color = (255, 255, 255)  # White text
                status_char = "X"  # X = occupied
            else:
                box_color = (0, 255, 0)      # Green
                text_color = (255, 255, 255)  # White text
                status_char = "O"  # O = empty/open
            
            # THIN border (2 pixels) - not thick
            border_thickness = 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, border_thickness)
            
            # SINGLE minimal label: "A-01" or "A-01 (car)" if occupied
            if spot.status == "OCCUPIED" and spot.vehicle_type:
                label = f"{spot.spot_id} ({spot.vehicle_type[0].upper()})"  # Short vehicle type
            else:
                label = spot.spot_id
            
            # Small font, thin text
            font_scale = 0.5
            text_thickness = 1
            
            # Get text size for background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            
            # Label position: above the box
            label_x = x1
            label_y = y1 - 3
            
            # If too close to top, put below box
            if label_y < text_h + 5:
                label_y = y2 + text_h + 3
            
            # Small background rectangle for readability
            cv2.rectangle(annotated_frame,
                         (label_x, label_y - text_h - 2),
                         (label_x + text_w + 4, label_y + 2),
                         box_color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (label_x + 2, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)
        
        return annotated_frame

class RealTimeParkingSystem:
    """Real-time parking monitoring system with multi-camera support"""
    
    def __init__(self, config_path: str = None):
        self.detector = ParkingDetector(config_path)
        self.camera_connections = {}
        self.is_running = False
        self.results_cache = {}
        self.monitoring_thread = None
        
    def connect_cameras(self, camera_configs: Dict[str, str]):
        """Connect to all cameras"""
        for camera_id, camera_source in camera_configs.items():
            try:
                cap = cv2.VideoCapture(camera_source)
                if cap.isOpened():
                    self.camera_connections[camera_id] = cap
                    print(f"[INFO] Connected to camera: {camera_id}")
                else:
                    print(f"[ERROR] Failed to connect to camera: {camera_id}")
            except Exception as e:
                print(f"[ERROR] Camera connection error {camera_id}: {e}")
                
    def start_monitoring(self, callback=None):
        """Start real-time monitoring"""
        if self.is_running:
            print("[WARNING] Monitoring already running")
            return
            
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, args=(callback,))
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        print("[INFO] Started real-time parking monitoring")
        
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            
        # Close camera connections
        for cap in self.camera_connections.values():
            cap.release()
        self.camera_connections.clear()
        print("[INFO] Stopped parking monitoring")
        
    def _monitoring_loop(self, callback=None):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Capture frames from all cameras
                frames = {}
                for camera_id, cap in self.camera_connections.items():
                    ret, frame = cap.read()
                    if ret:
                        frames[camera_id] = frame
                        
                if frames:
                    # Process all zones
                    results = self.detector.process_all_zones(frames)
                    self.results_cache = results
                    
                    # Call callback if provided
                    if callback:
                        callback(results)
                        
                time.sleep(1)  # Process every second
                
            except Exception as e:
                print(f"[ERROR] Monitoring loop error: {e}")
                time.sleep(5)  # Wait before retry
                
    def get_current_results(self) -> Dict[str, ZoneResult]:
        """Get current parking results"""
        return self.results_cache.copy()
        
    def get_json_results(self) -> str:
        """Get current results in JSON format"""
        return self.detector.get_json_output(self.results_cache)

def main():
    """Test the parking detection system"""
    print("=== YOLO26 Parking Detection System ===")
    
    # Initialize detector
    detector = ParkingDetector()
    
    # Test with a single image
    test_image_path = "test_parking_image.jpg"  # Replace with actual test image
    
    if Path(test_image_path).exists():
        frame = cv2.imread(test_image_path)
        if frame is not None:
            # Simulate camera frame
            frames = {"cam_01": frame}
            
            # Process zone A
            results = detector.process_zone(frames, "zone_a")
            
            # Get JSON output
            json_output = detector.get_json_output({"zone_a": results})
            print("Detection Results:")
            print(json_output)
            
            # Draw and save annotated image
            annotated = detector.draw_detections(frame, results.spot_details)
            cv2.imwrite("annotated_result.jpg", annotated)
            print("[INFO] Annotated image saved as 'annotated_result.jpg'")
        else:
            print(f"[ERROR] Cannot load test image: {test_image_path}")
    else:
        print(f"[WARNING] Test image not found: {test_image_path}")
        print("Please provide a test image to verify the detection system")

if __name__ == "__main__":
    main()
