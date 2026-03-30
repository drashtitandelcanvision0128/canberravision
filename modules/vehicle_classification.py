"""
Vehicle Make and Model Classification Module
Enhanced vehicle recognition using deep learning models.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

class VehicleClassifier:
    """Advanced vehicle make and model classifier."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.vehicle_labels = self._load_vehicle_labels()
        self._initialize_model()
    
    def _load_vehicle_labels(self) -> Dict:
        """Load vehicle make and model labels."""
        return {
            0: {'make': 'BMW', 'model': '3 Series'},
            1: {'make': 'Audi', 'model': 'A4'},
            2: {'make': 'Mercedes', 'model': 'C-Class'},
            3: {'make': 'Toyota', 'model': 'Camry'},
            4: {'make': 'Honda', 'model': 'Civic'},
            5: {'make': 'Ford', 'model': 'Mustang'},
            6: {'make': 'Chevrolet', 'model': 'Malibu'},
            7: {'make': 'Nissan', 'model': 'Altima'},
            8: {'make': 'Hyundai', 'model': 'Elantra'},
            9: {'make': 'Volkswagen', 'model': 'Jetta'},
            10: {'make': 'Tesla', 'model': 'Model 3'},
            11: {'make': 'Mazda', 'model': 'Mazda3'},
            12: {'make': 'Kia', 'model': 'Optima'},
            13: {'make': 'Subaru', 'model': 'Impreza'},
            14: {'make': 'Lexus', 'model': 'IS'}
        }
    
    def _initialize_model(self):
        """Initialize the vehicle classification model."""
        try:
            # Load a pre-trained ResNet50 for better accuracy
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            
            # Replace the final layer for our vehicle classes
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, len(self.vehicle_labels))
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Define transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            print("[INFO] Vehicle classifier initialized successfully")
            
        except Exception as e:
            print(f"[ERROR] Vehicle classifier initialization failed: {e}")
            self.model = None
    
    def classify_vehicle(self, vehicle_image: np.ndarray) -> Dict:
        """
        Classify vehicle make and model.
        
        Args:
            vehicle_image: Vehicle crop in BGR format
            
        Returns:
            Dictionary with make, model, and confidence
        """
        if self.model is None:
            return {'make': 'Unknown', 'model': 'Unknown', 'confidence': 0.0}
        
        try:
            # Preprocess image
            if len(vehicle_image.shape) == 3 and vehicle_image.shape[2] == 3:
                # Convert BGR to RGB
                vehicle_rgb = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2RGB)
            else:
                vehicle_rgb = vehicle_image
            
            # Apply transforms
            input_tensor = self.transform(vehicle_rgb).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            # Get vehicle info
            vehicle_info = self.vehicle_labels[predicted.item()]
            
            return {
                'make': vehicle_info['make'],
                'model': vehicle_info['model'],
                'confidence': confidence.item()
            }
            
        except Exception as e:
            print(f"[ERROR] Vehicle classification failed: {e}")
            return {'make': 'Unknown', 'model': 'Unknown', 'confidence': 0.0}
    
    def get_top_predictions(self, vehicle_image: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Get top-k vehicle predictions.
        
        Args:
            vehicle_image: Vehicle crop in BGR format
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with make, model, and confidence
        """
        if self.model is None:
            return []
        
        try:
            # Preprocess image
            if len(vehicle_image.shape) == 3 and vehicle_image.shape[2] == 3:
                vehicle_rgb = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2RGB)
            else:
                vehicle_rgb = vehicle_image
            
            # Apply transforms
            input_tensor = self.transform(vehicle_rgb).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # Format results
            predictions = []
            for i in range(top_k):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                vehicle_info = self.vehicle_labels[idx]
                
                predictions.append({
                    'make': vehicle_info['make'],
                    'model': vehicle_info['model'],
                    'confidence': prob
                })
            
            return predictions
            
        except Exception as e:
            print(f"[ERROR] Top-k predictions failed: {e}")
            return []

class VehicleColorDetector:
    """Advanced vehicle color detection using multiple methods."""
    
    def __init__(self):
        self.color_ranges = self._define_color_ranges()
    
    def _define_color_ranges(self) -> Dict:
        """Define HSV color ranges for different vehicle colors."""
        return {
            'White': [(0, 0, 200), (180, 30, 255)],
            'Black': [(0, 0, 0), (180, 255, 50)],
            'Silver': [(0, 0, 100), (180, 30, 200)],
            'Gray': [(0, 0, 50), (180, 30, 100)],
            'Red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
            'Blue': [(100, 50, 50), (130, 255, 255)],
            'Green': [(40, 50, 50), (80, 255, 255)],
            'Yellow': [(20, 50, 50), (40, 255, 255)],
            'Brown': [(10, 50, 50), (20, 255, 255)],
            'Purple': [(130, 50, 50), (170, 255, 255)],
            'Orange': [(10, 50, 50), (20, 255, 255)]
        }
    
    def detect_color(self, vehicle_image: np.ndarray) -> Dict:
        """
        Detect vehicle color using HSV analysis.
        
        Args:
            vehicle_image: Vehicle crop in BGR format
            
        Returns:
            Dictionary with color name and confidence
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2HSV)
            
            # Calculate color scores
            color_scores = {}
            
            for color_name, ranges in self.color_ranges.items():
                if color_name == 'Red':
                    # Red has two ranges
                    mask1 = cv2.inRange(hsv, ranges[0], ranges[1])
                    mask2 = cv2.inRange(hsv, ranges[2], ranges[3])
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    mask = cv2.inRange(hsv, ranges[0], ranges[1])
                
                # Calculate percentage of pixels matching this color
                score = cv2.countNonZero(mask) / (vehicle_image.shape[0] * vehicle_image.shape[1])
                color_scores[color_name] = score
            
            # Get the best matching color
            best_color = max(color_scores, key=color_scores.get)
            confidence = color_scores[best_color]
            
            # Apply minimum confidence threshold
            if confidence < 0.05:
                best_color = 'Unknown'
                confidence = 0.0
            
            return {
                'color': best_color,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"[ERROR] Color detection failed: {e}")
            return {'color': 'Unknown', 'confidence': 0.0}
    
    def detect_color_kmeans(self, vehicle_image: np.ndarray, k: int = 5) -> Dict:
        """
        Detect vehicle color using K-means clustering.
        
        Args:
            vehicle_image: Vehicle crop in BGR format
            k: Number of clusters for K-means
            
        Returns:
            Dictionary with color name and confidence
        """
        try:
            # Resize for faster processing
            small_img = cv2.resize(vehicle_image, (100, 100))
            
            # Reshape for clustering
            pixel_values = small_img.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            
            # Apply K-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert centers to uint8
            centers = np.uint8(centers)
            
            # Find the most dominant cluster
            unique_labels, counts = np.unique(labels, return_counts=True)
            dominant_cluster = unique_labels[np.argmax(counts)]
            dominant_color = centers[dominant_cluster]
            
            # Convert BGR to HSV for better color classification
            dominant_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2HSV)[0][0]
            
            # Classify the color
            color_name = self._classify_hsv_color(dominant_hsv)
            
            return {
                'color': color_name,
                'confidence': np.max(counts) / len(labels)
            }
            
        except Exception as e:
            print(f"[ERROR] K-means color detection failed: {e}")
            return {'color': 'Unknown', 'confidence': 0.0}
    
    def _classify_hsv_color(self, hsv_color: np.ndarray) -> str:
        """Classify HSV color to color name."""
        h, s, v = hsv_color
        
        if v < 50:
            return 'Black'
        elif v > 200 and s < 30:
            return 'White'
        elif v > 100 and v <= 200 and s < 30:
            return 'Silver' if v > 150 else 'Gray'
        elif s < 30:
            return 'Gray'
        elif h < 10 or h > 170:
            return 'Red'
        elif 10 <= h < 25:
            return 'Orange'
        elif 25 <= h < 35:
            return 'Yellow'
        elif 35 <= h < 85:
            return 'Green'
        elif 85 <= h < 130:
            return 'Blue'
        elif 130 <= h < 170:
            return 'Purple'
        else:
            return 'Unknown'

# Test the modules
if __name__ == "__main__":
    # Initialize classifiers
    vehicle_classifier = VehicleClassifier()
    color_detector = VehicleColorDetector()
    
    print("[INFO] Vehicle classification modules loaded successfully")
    print(f"[INFO] Available vehicle models: {len(vehicle_classifier.vehicle_labels)}")
    print(f"[INFO] Available colors: {list(color_detector.color_ranges.keys())}")
