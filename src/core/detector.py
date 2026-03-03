"""
YOLO Detector Core Module
Handles YOLO model loading and object detection.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from ultralytics import YOLO

from ..config.settings import get_config, PROJECT_ROOT
from .exceptions import ModelNotFoundError, GPUError


class YOLODetector:
    """
    YOLO-based object detector with GPU support and caching.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize YOLO detector.
        
        Args:
            model_name: Name of the YOLO model to use
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.config = get_config('yolo')
        self.gpu_config = get_config('gpu')
        
        self.model_name = model_name or self.config['default_model']
        self.device = self._setup_device(device)
        self.model = None
        self._model_cache = {}
        
        print(f"[INFO] YOLO Detector initialized with model: {self.model_name}")
        print(f"[INFO] Using device: {self.device}")
    
    def _setup_device(self, device: str = None) -> str:
        """Setup and return the best available device."""
        if device:
            return device
            
        if self.gpu_config['use_gpu'] and torch.cuda.is_available():
            device_id = self.gpu_config['device_id']
            torch.cuda.set_device(device_id)
            print(f"[INFO] GPU detected: {torch.cuda.get_device_name(device_id)}")
            return f"cuda:{device_id}"
        
        print("[INFO] Using CPU for processing")
        return "cpu"
    
    def load_model(self, model_name: str = None) -> YOLO:
        """
        Load YOLO model with caching.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded YOLO model
        """
        model_name = model_name or self.model_name
        
        # Check cache first
        if model_name in self._model_cache:
            print(f"[INFO] Using cached model: {model_name}")
            return self._model_cache[model_name]
        
        # Prepare model path
        if not model_name.endswith('.pt'):
            model_path = PROJECT_ROOT / f"{model_name}.pt"
        else:
            model_path = PROJECT_ROOT / model_name
        
        if not model_path.exists():
            raise ModelNotFoundError(f"Model file not found: {model_path}")
        
        try:
            print(f"[INFO] Loading model: {model_path}")
            model = YOLO(str(model_path))
            
            # Move to device if GPU
            if "cuda" in self.device:
                model.to(self.device)
                print(f"[INFO] Model moved to {self.device}")
            
            # Cache the model
            self._model_cache[model_name] = model
            print(f"[INFO] Model {model_name} loaded successfully")
            
            return model
            
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model {model_name}: {e}")
    
    def detect_objects(self, 
                      image: np.ndarray, 
                      conf_threshold: float = None,
                      iou_threshold: float = None,
                      imgsz: int = None) -> List[Dict]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image in BGR format
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold
            imgsz: Image size for inference
            
        Returns:
            List of detected objects with their properties
        """
        if self.model is None:
            self.model = self.load_model()
        
        # Use config values if not provided
        conf_threshold = conf_threshold or self.config['confidence_threshold']
        iou_threshold = iou_threshold or self.config['iou_threshold']
        imgsz = imgsz or self.config['default_image_size']
        
        try:
            # Run inference
            results = self.model.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
                half=self.gpu_config['mixed_precision'] if "cuda" in self.device else False
            )
            
            # Process results
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Get bounding box
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        
                        # Get confidence
                        confidence = float(boxes.conf[i].cpu().numpy())
                        
                        # Get class
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = result.names.get(class_id, f"class_{class_id}")
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                            'area': int((x2 - x1) * (y2 - y1))
                        }
                        
                        detections.append(detection)
            
            print(f"[INFO] Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            raise RuntimeError(f"Object detection failed: {e}")
    
    def detect_vehicles(self, image: np.ndarray) -> List[Dict]:
        """
        Detect only vehicles in the image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detected vehicles
        """
        vehicle_classes = [
            'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'van',
            'taxi', 'ambulance', 'police', 'fire truck', 'tractor',
            'scooter', 'bike', 'auto', 'rickshaw', 'lorry'
        ]
        
        all_detections = self.detect_objects(image)
        
        # Filter for vehicles only
        vehicles = [
            detection for detection in all_detections
            if detection['class_name'].lower() in vehicle_classes
        ]
        
        print(f"[INFO] Found {len(vehicles)} vehicles out of {len(all_detections)} objects")
        return vehicles
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "device": self.device,
            "classes": list(self.model.names.values()) if hasattr(self.model, 'names') else [],
            "gpu_enabled": "cuda" in self.device,
            "mixed_precision": self.gpu_config['mixed_precision']
        }
    
    def clear_cache(self):
        """Clear model cache."""
        self._model_cache.clear()
        print("[INFO] Model cache cleared")
