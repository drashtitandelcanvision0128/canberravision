"""
Vehicle Classification Enhancement Module
Improves vehicle detection accuracy with angle-invariant classification.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import re

class VehicleClassifier:
    """
    Enhanced vehicle classifier that improves detection accuracy
    for vehicles at different angles and perspectives.
    """
    
    def __init__(self):
        """Initialize the vehicle classifier."""
        # Enhanced vehicle class mappings with angle-invariant characteristics
        self.vehicle_classes = {
            'car': {
                'aspect_ratio_range': (0.3, 3.0),  # Expanded for all angles
                'size_range': (800, 80000),  # Wider range for different distances
                'confidence_boost': 0.15,
                'keywords': ['car', 'sedan', 'hatchback', 'coupe', 'convertible', 'suv', 'jeep'],
                'angle_invariant_features': {
                    'typical_width_ratio': (0.1, 0.8),  # Relative to image width
                    'typical_height_ratio': (0.1, 0.6),  # Relative to image height
                    'shape_complexity': 'medium'
                }
            },
            'truck': {
                'aspect_ratio_range': (0.4, 4.0),
                'size_range': (2000, 120000),
                'confidence_boost': 0.1,
                'keywords': ['truck', 'pickup', 'lorry', 'van', 'delivery'],
                'angle_invariant_features': {
                    'typical_width_ratio': (0.15, 0.9),
                    'typical_height_ratio': (0.15, 0.7),
                    'shape_complexity': 'high'
                }
            },
            'bus': {
                'aspect_ratio_range': (0.6, 6.0),  # Buses can be very long
                'size_range': (4000, 200000),
                'confidence_boost': 0.1,
                'keywords': ['bus', 'coach', 'minibus', 'van'],
                'angle_invariant_features': {
                    'typical_width_ratio': (0.2, 0.95),
                    'typical_height_ratio': (0.2, 0.8),
                    'shape_complexity': 'high'
                }
            },
            'motorcycle': {
                'aspect_ratio_range': (0.1, 5.0),  # Very flexible for different angles
                'size_range': (300, 15000),
                'confidence_boost': 0.2,
                'keywords': ['motorcycle', 'bike', 'scooter', 'moped'],
                'angle_invariant_features': {
                    'typical_width_ratio': (0.05, 0.4),
                    'typical_height_ratio': (0.05, 0.5),
                    'shape_complexity': 'low'
                }
            },
            'bicycle': {
                'aspect_ratio_range': (0.1, 6.0),
                'size_range': (100, 8000),
                'confidence_boost': 0.2,
                'keywords': ['bicycle', 'bike'],
                'angle_invariant_features': {
                    'typical_width_ratio': (0.03, 0.3),
                    'typical_height_ratio': (0.03, 0.4),
                    'shape_complexity': 'low'
                }
            }
        }
        
        # Class ID mappings for YOLO models
        self.yolo_class_mapping = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
        }
        
        print("[INFO] Enhanced Vehicle Classifier initialized with angle-invariant detection")
    
    def enhance_vehicle_detection(self, detections: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Enhance vehicle detection results with angle-invariant classification.
        
        Args:
            detections: List of detection dictionaries
            image_shape: (height, width) of the image
            
        Returns:
            Enhanced detections with improved vehicle classification
        """
        enhanced_detections = []
        
        for detection in detections:
            class_name = detection['class_name'].lower()
            
            # Only process vehicle detections
            if class_name in self.vehicle_classes:
                enhanced_detection = self._enhance_single_vehicle_angle_invariant(detection, image_shape)
                enhanced_detections.append(enhanced_detection)
            else:
                enhanced_detections.append(detection)
        
        return enhanced_detections
    
    def _enhance_single_vehicle_angle_invariant(self, detection: Dict, image_shape: Tuple[int, int]) -> Dict:
        """Enhance a single vehicle detection with angle-invariant analysis."""
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name'].lower()
        
        # Calculate enhanced bounding box properties
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Calculate image-relative ratios (angle-invariant features)
        img_height, img_width = image_shape
        width_ratio = width / img_width
        height_ratio = height / img_height
        
        # Get vehicle characteristics
        vehicle_info = self.vehicle_classes.get(class_name, {})
        
        # Enhanced characteristic score with angle-invariant features
        characteristic_score = self._calculate_angle_invariant_score(
            area, aspect_ratio, width_ratio, height_ratio, vehicle_info
        )
        
        # Apply confidence adjustment based on angle-invariant analysis
        adjusted_confidence = confidence
        if characteristic_score > 0.6:
            # Boost confidence if characteristics match well
            adjusted_confidence += vehicle_info.get('confidence_boost', 0)
            adjusted_confidence = min(adjusted_confidence, 1.0)  # Cap at 1.0
        
        # Enhanced angle-invariant classification
        suggested_class = self._suggest_correct_class_angle_invariant(
            area, aspect_ratio, width_ratio, height_ratio, confidence
        )
        
        # Smart reclassification with angle consideration
        if confidence < 0.5 and suggested_class != class_name:
            suggested_info = self.vehicle_classes.get(suggested_class, {})
            suggested_score = self._calculate_angle_invariant_score(
                area, aspect_ratio, width_ratio, height_ratio, suggested_info
            )
            
            current_score = self._calculate_angle_invariant_score(
                area, aspect_ratio, width_ratio, height_ratio, vehicle_info
            )
            
            # More lenient reclassification for challenging angles
            if suggested_score > current_score + 0.2:
                print(f"[DEBUG] Angle-invariant reclassification: {class_name} -> {suggested_class} (conf: {confidence:.3f})")
                class_name = suggested_class
                adjusted_confidence = max(adjusted_confidence, 0.4)
        
        # Create enhanced detection
        enhanced_detection = detection.copy()
        enhanced_detection.update({
            'class_name': class_name,
            'confidence': adjusted_confidence,
            'original_confidence': confidence,
            'characteristic_score': characteristic_score,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'width_ratio': width_ratio,
            'height_ratio': height_ratio,
            'reclassified': class_name != detection['class_name'].lower(),
            'angle_invariant': True
        })
        
        return enhanced_detection
    
    def _calculate_angle_invariant_score(self, area: int, aspect_ratio: float, 
                                       width_ratio: float, height_ratio: float, 
                                       vehicle_info: Dict) -> float:
        """Calculate angle-invariant characteristic matching score."""
        if not vehicle_info:
            return 0.5
        
        # Area score (more flexible for different distances)
        min_area, max_area = vehicle_info.get('size_range', (0, float('inf')))
        area_score = 0.0
        if min_area <= area <= max_area:
            area_mid = (min_area + max_area) / 2
            area_range = max_area - min_area
            area_diff = abs(area - area_mid)
            area_score = max(0, 1 - (area_diff / (area_range / 1.5)))  # More lenient
        
        # Aspect ratio score (expanded ranges for all angles)
        min_ratio, max_ratio = vehicle_info.get('aspect_ratio_range', (0, float('inf')))
        ratio_score = 0.0
        if min_ratio <= aspect_ratio <= max_ratio:
            ratio_mid = (min_ratio + max_ratio) / 2
            ratio_range = max_ratio - min_ratio
            ratio_diff = abs(aspect_ratio - ratio_mid)
            ratio_score = max(0, 1 - (ratio_diff / (ratio_range / 1.5)))  # More lenient
        
        # Width ratio score (angle-invariant)
        width_features = vehicle_info.get('angle_invariant_features', {})
        min_width_ratio, max_width_ratio = width_features.get('typical_width_ratio', (0, 1))
        width_score = 0.0
        if min_width_ratio <= width_ratio <= max_width_ratio:
            width_mid = (min_width_ratio + max_width_ratio) / 2
            width_range = max_width_ratio - min_width_ratio
            width_diff = abs(width_ratio - width_mid)
            width_score = max(0, 1 - (width_diff / (width_range / 1.5)))
        
        # Height ratio score (angle-invariant)
        min_height_ratio, max_height_ratio = width_features.get('typical_height_ratio', (0, 1))
        height_score = 0.0
        if min_height_ratio <= height_ratio <= max_height_ratio:
            height_mid = (min_height_ratio + max_height_ratio) / 2
            height_range = max_height_ratio - min_height_ratio
            height_diff = abs(height_ratio - height_mid)
            height_score = max(0, 1 - (height_diff / (height_range / 1.5)))
        
        # Combined score with angle-invariant features
        return (area_score + ratio_score + width_score + height_score) / 4
    
    def _suggest_correct_class_angle_invariant(self, area: int, aspect_ratio: float,
                                             width_ratio: float, height_ratio: float,
                                             confidence: float) -> str:
        """Suggest the most likely correct vehicle class using angle-invariant analysis."""
        if confidence > 0.7:  # High confidence detections are likely correct
            return ''
        
        best_class = ''
        best_score = 0.0
        
        for class_name, info in self.vehicle_classes.items():
            score = self._calculate_angle_invariant_score(
                area, aspect_ratio, width_ratio, height_ratio, info
            )
            if score > best_score:
                best_score = score
                best_class = class_name
        
        return best_class if best_score > 0.6 else ''  # Lower threshold for angle challenges
    
    def _calculate_characteristic_score(self, area: int, aspect_ratio: float, vehicle_info: Dict) -> float:
        """Calculate how well detection matches vehicle characteristics."""
        if not vehicle_info:
            return 0.5
        
        # Check area range
        min_area, max_area = vehicle_info.get('size_range', (0, float('inf')))
        area_score = 0.0
        if min_area <= area <= max_area:
            # Better score if closer to middle of range
            area_mid = (min_area + max_area) / 2
            area_range = max_area - min_area
            area_diff = abs(area - area_mid)
            area_score = max(0, 1 - (area_diff / (area_range / 2)))
        
        # Check aspect ratio range
        min_ratio, max_ratio = vehicle_info.get('aspect_ratio_range', (0, float('inf')))
        ratio_score = 0.0
        if min_ratio <= aspect_ratio <= max_ratio:
            # Better score if closer to middle of range
            ratio_mid = (min_ratio + max_ratio) / 2
            ratio_range = max_ratio - min_ratio
            ratio_diff = abs(aspect_ratio - ratio_mid)
            ratio_score = max(0, 1 - (ratio_diff / (ratio_range / 2)))
        
        # Combined score
        return (area_score + ratio_score) / 2
    
    def _suggest_correct_class(self, area: int, aspect_ratio: float, confidence: float) -> str:
        """Suggest the most likely correct vehicle class based on characteristics."""
        if confidence > 0.6:
            # High confidence detections are likely correct
            return ''
        
        best_class = ''
        best_score = 0.0
        
        for class_name, info in self.vehicle_classes.items():
            score = self._calculate_characteristic_score(area, aspect_ratio, info)
            if score > best_score:
                best_score = score
                best_class = class_name
        
        return best_class if best_score > 0.7 else ''
    
    def filter_low_confidence_vehicles(self, detections: List[Dict], min_confidence: float = 0.3) -> List[Dict]:
        """
        Filter out low confidence vehicle detections that are likely incorrect.
        
        Args:
            detections: List of detection dictionaries
            min_confidence: Minimum confidence threshold for vehicles
            
        Returns:
            Filtered detections
        """
        filtered_detections = []
        
        for detection in detections:
            class_name = detection['class_name'].lower()
            confidence = detection['confidence']
            
            # Keep non-vehicle detections as they are
            if class_name not in self.vehicle_classes:
                filtered_detections.append(detection)
                continue
            
            # For vehicles, apply stricter filtering
            if confidence >= min_confidence:
                filtered_detections.append(detection)
            elif 'characteristic_score' in detection and detection['characteristic_score'] > 0.8:
                # Keep if characteristics match very well even with low confidence
                filtered_detections.append(detection)
            else:
                print(f"[DEBUG] Filtering low confidence vehicle: {class_name} (conf: {confidence:.3f})")
        
        return filtered_detections
    
    def remove_duplicate_vehicles(self, detections: List[Dict]) -> List[Dict]:
        """
        Remove duplicate vehicle detections (same object detected as multiple vehicle types).
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered detections with duplicates removed
        """
        if not detections:
            return detections
        
        # Group detections by bounding box similarity
        vehicle_detections = [d for d in detections if d['class_name'].lower() in self.vehicle_classes]
        non_vehicle_detections = [d for d in detections if d['class_name'].lower() not in self.vehicle_classes]
        
        if len(vehicle_detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        vehicle_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_vehicles = []
        
        for detection in vehicle_detections:
            bbox = detection['bbox']
            is_duplicate = False
            
            # Check if this detection overlaps significantly with any already accepted detection
            for accepted in filtered_vehicles:
                accepted_bbox = accepted['bbox']
                
                # Calculate Intersection over Union (IoU)
                iou = self._calculate_iou(bbox, accepted_bbox)
                
                # If IoU > 0.7, consider it a duplicate
                if iou > 0.7:
                    print(f"[DEBUG] Removing duplicate {detection['class_name']} (conf: {detection['confidence']:.3f}) - overlaps with {accepted['class_name']} (conf: {accepted['confidence']:.3f})")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_vehicles.append(detection)
        
        # Combine filtered vehicles with non-vehicle detections
        return filtered_vehicles + non_vehicle_detections
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Calculate intersection area
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def get_vehicle_summary(self, detections: List[Dict]) -> Dict:
        vehicle_counts = {}
        total_vehicles = 0
        
        for detection in detections:
            class_name = detection['class_name'].lower()
            if class_name in self.vehicle_classes:
                vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1
                total_vehicles += 1
        
        return {
            'total_vehicles': total_vehicles,
            'vehicle_counts': vehicle_counts,
            'vehicle_types': list(vehicle_counts.keys())
        }
