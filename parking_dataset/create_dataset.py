"""
Parking Dataset Creator for YOLO26 Parking Detection System
Creates training dataset for parking space occupancy detection
"""

import os
import cv2
import numpy as np
import yaml
import json
from pathlib import Path
from datetime import datetime
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ultralytics import YOLO

@dataclass
class ParkingSpot:
    """Represents a single parking spot with its coordinates and status"""
    spot_id: str
    zone_id: str
    camera_id: str
    coordinates: Tuple[int, int, int, int]  # x1, y1, x2, y2
    status: str = "EMPTY"  # EMPTY or OCCUPIED
    vehicle_type: Optional[str] = None
    confidence: float = 0.0

class ParkingDatasetCreator:
    """Creates and manages parking space occupancy dataset"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "parking_dataset/config/parking_zones.yaml"
        self.config = self._load_config()
        self.spots = self._initialize_spots()
        self.dataset_path = Path("parking_dataset")
        self.images_path = self.dataset_path / "images"
        self.labels_path = self.dataset_path / "labels"
        
        # Create directories
        self._create_directories()
        
    def _load_config(self) -> Dict:
        """Load parking zone configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[INFO] Loaded parking configuration from {self.config_path}")
            return config
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            raise
            
    def _initialize_spots(self) -> Dict[str, ParkingSpot]:
        """Initialize all parking spots from configuration"""
        spots = {}
        
        for zone_id, zone_config in self.config['zones'].items():
            for camera_id, camera_config in zone_config['coordinates'].items():
                for spot_id, coords in camera_config['spots'].items():
                    spots[spot_id] = ParkingSpot(
                        spot_id=spot_id,
                        zone_id=zone_id,
                        camera_id=camera_id,
                        coordinates=coords
                    )
                    
        print(f"[INFO] Initialized {len(spots)} parking spots across {len(self.config['zones'])} zones")
        return spots
        
    def _create_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.dataset_path / "images" / "train",
            self.dataset_path / "images" / "val",
            self.dataset_path / "labels" / "train", 
            self.dataset_path / "labels" / "val",
            self.dataset_path / "models",
            self.dataset_path / "logs",
            self.dataset_path / "results"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def capture_training_images(self, camera_source: str, num_images: int = 100) -> List[str]:
        """Capture training images from camera source"""
        captured_images = []
        
        try:
            cap = cv2.VideoCapture(camera_source)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open camera: {camera_source}")
                return captured_images
                
            print(f"[INFO] Capturing {num_images} training images from {camera_source}")
            
            for i in range(num_images):
                ret, frame = cap.read()
                if not ret:
                    print(f"[WARNING] Failed to capture frame {i}")
                    continue
                    
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"training_{camera_source}_{timestamp}.jpg"
                filepath = self.images_path / "train" / filename
                cv2.imwrite(str(filepath), frame)
                captured_images.append(str(filepath))
                
                # Add small delay between captures
                cv2.waitKey(100)
                
            cap.release()
            print(f"[INFO] Successfully captured {len(captured_images)} images")
            
        except Exception as e:
            print(f"[ERROR] Camera capture failed: {e}")
            
        return captured_images
        
    def annotate_parking_spots(self, image_path: str, model_path: str = "yolov8n.pt") -> Dict[str, ParkingSpot]:
        """Annotate parking spots in an image using YOLO model"""
        try:
            # Load YOLO model
            model = YOLO(model_path)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] Cannot load image: {image_path}")
                return {}
                
            # Get camera ID from image path
            camera_id = self._extract_camera_id(image_path)
            if not camera_id:
                print(f"[WARNING] Cannot determine camera ID from {image_path}")
                return {}
                
            # Get spots for this camera
            camera_spots = {spot_id: spot for spot_id, spot in self.spots.items() 
                          if spot.camera_id == camera_id}
            
            # Run YOLO detection
            results = model(image, conf=0.5)
            detected_vehicles = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = model.names[cls]
                        
                        detected_vehicles.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': float(conf),
                            'class': class_name
                        })
            
            # Check each parking spot for occupancy
            annotated_spots = {}
            for spot_id, spot in camera_spots.items():
                spot_copy = ParkingSpot(
                    spot_id=spot.spot_id,
                    zone_id=spot.zone_id,
                    camera_id=spot.camera_id,
                    coordinates=spot.coordinates
                )
                
                # Check if any vehicle overlaps with this spot
                is_occupied = False
                best_confidence = 0.0
                best_vehicle_type = None
                
                for vehicle in detected_vehicles:
                    if self._check_overlap(spot.coordinates, vehicle['bbox']):
                        is_occupied = True
                        if vehicle['confidence'] > best_confidence:
                            best_confidence = vehicle['confidence']
                            best_vehicle_type = vehicle['class']
                            
                spot_copy.status = "OCCUPIED" if is_occupied else "EMPTY"
                spot_copy.confidence = best_confidence
                spot_copy.vehicle_type = best_vehicle_type
                annotated_spots[spot_id] = spot_copy
                
            return annotated_spots
            
        except Exception as e:
            print(f"[ERROR] Annotation failed for {image_path}: {e}")
            return {}
            
    def _extract_camera_id(self, image_path: str) -> Optional[str]:
        """Extract camera ID from image path"""
        try:
            filename = Path(image_path).stem
            if "cam_" in filename.lower():
                parts = filename.split('_')
                for i, part in enumerate(parts):
                    if part.lower() == "cam" and i + 1 < len(parts):
                        return f"cam_{parts[i+1].zfill(2)}"
            return None
        except:
            return None
            
    def _check_overlap(self, spot_bbox: Tuple[int, int, int, int], 
                      vehicle_bbox: Tuple[int, int, int, int], 
                      threshold: float = 0.3) -> bool:
        """Check if vehicle bbox overlaps with parking spot bbox"""
        x1_spot, y1_spot, x2_spot, y2_spot = spot_bbox
        x1_veh, y1_veh, x2_veh, y2_veh = vehicle_bbox
        
        # Calculate intersection area
        x1_intersect = max(x1_spot, x1_veh)
        y1_intersect = max(y1_spot, y1_veh)
        x2_intersect = min(x2_spot, x2_veh)
        y2_intersect = min(y2_spot, y2_veh)
        
        if x2_intersect <= x1_intersect or y2_intersect <= y1_intersect:
            return False
            
        intersection_area = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)
        spot_area = (x2_spot - x1_spot) * (y2_spot - y1_spot)
        
        overlap_ratio = intersection_area / spot_area if spot_area > 0 else 0
        return overlap_ratio >= threshold
        
    def create_yolo_labels(self, annotated_spots: Dict[str, ParkingSpot], 
                          image_shape: Tuple[int, int]) -> List[str]:
        """Create YOLO format labels for annotated spots"""
        height, width = image_shape[:2]
        labels = []
        
        for spot in annotated_spots.values():
            if spot.status == "OCCUPIED":
                # Class 0: EMPTY, Class 1: OCCUPIED
                class_id = 1
                
                # Convert to YOLO format (normalized coordinates)
                x1, y1, x2, y2 = spot.coordinates
                x_center = (x1 + x2) / 2.0 / width
                y_center = (y1 + y2) / 2.0 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                
                label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                labels.append(label_line)
                
        return labels
        
    def generate_dataset(self, camera_sources: Dict[str, str], 
                        images_per_camera: int = 1000) -> bool:
        """Generate complete training dataset"""
        try:
            print("[INFO] Starting dataset generation...")
            
            all_images = []
            
            # Capture images from all cameras
            for camera_id, camera_source in camera_sources.items():
                print(f"[INFO] Capturing images from {camera_id}")
                images = self.capture_training_images(camera_source, images_per_camera)
                all_images.extend(images)
                
            if not all_images:
                print("[ERROR] No images captured")
                return False
                
            print(f"[INFO] Processing {len(all_images)} images for annotation...")
            
            # Process and annotate images
            processed_count = 0
            for image_path in all_images:
                annotated_spots = self.annotate_parking_spots(image_path)
                
                if annotated_spots:
                    # Load image for shape
                    image = cv2.imread(image_path)
                    image_shape = image.shape
                    
                    # Create YOLO labels
                    labels = self.create_yolo_labels(annotated_spots, image_shape)
                    
                    # Save labels
                    label_filename = Path(image_path).stem + ".txt"
                    label_path = self.labels_path / "train" / label_filename
                    
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(labels))
                        
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        print(f"[INFO] Processed {processed_count}/{len(all_images)} images")
                        
            # Split dataset
            self._split_dataset()
            
            # Create dataset.yaml for YOLO training
            self._create_dataset_yaml()
            
            print(f"[INFO] Dataset generation completed. Processed {processed_count} images.")
            return True
            
        except Exception as e:
            print(f"[ERROR] Dataset generation failed: {e}")
            return False
            
    def _split_dataset(self, val_ratio: float = 0.2):
        """Split dataset into train and validation sets"""
        try:
            # Get all image files
            train_images = list((self.images_path / "train").glob("*.jpg"))
            random.shuffle(train_images)
            
            # Calculate split
            val_count = int(len(train_images) * val_ratio)
            val_images = train_images[:val_count]
            train_images = train_images[val_count:]
            
            # Move validation files
            for img_path in val_images:
                # Move image
                val_img_path = self.images_path / "val" / img_path.name
                img_path.rename(val_img_path)
                
                # Move corresponding label
                label_path = self.labels_path / "train" / (img_path.stem + ".txt")
                if label_path.exists():
                    val_label_path = self.labels_path / "val" / label_path.name
                    label_path.rename(val_label_path)
                    
            print(f"[INFO] Dataset split: {len(train_images)} train, {len(val_images)} validation")
            
        except Exception as e:
            print(f"[ERROR] Dataset split failed: {e}")
            
    def _create_dataset_yaml(self):
        """Create dataset.yaml file for YOLO training"""
        dataset_config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 2,  # Number of classes: EMPTY, OCCUPIED
            'names': {
                0: 'EMPTY',
                1: 'OCCUPIED'
            }
        }
        
        yaml_path = self.dataset_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
            
        print(f"[INFO] Created dataset configuration: {yaml_path}")

def main():
    """Main function to create parking dataset"""
    print("=== YOLO26 Parking Dataset Creator ===")
    
    # Initialize dataset creator
    creator = ParkingDatasetCreator()
    
    # Define camera sources (adjust these based on your setup)
    camera_sources = {
        'cam_01': 0,  # Webcam or RTSP stream
        'cam_02': 'rtsp://camera2_url',
        'cam_03': 'rtsp://camera3_url', 
        'cam_04': 'rtsp://camera4_url',
        'cam_05': 'rtsp://camera5_url',
        'cam_06': 'rtsp://camera6_url'
    }
    
    # Generate dataset
    success = creator.generate_dataset(camera_sources, images_per_camera=500)
    
    if success:
        print("[SUCCESS] Parking dataset created successfully!")
        print(f"[INFO] Dataset location: {creator.dataset_path}")
        print("[INFO] You can now start training the model with:")
        print("python parking_dataset/train_parking_model.py")
    else:
        print("[ERROR] Failed to create dataset")

if __name__ == "__main__":
    main()
