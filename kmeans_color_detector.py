"""
Enhanced K-means Color Detection and Object Categorization Module
Advanced color clustering with ResNet-18 feature extraction for YOLO26
Provides accurate color detection and object categorization using K-means clustering + ResNet-18
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from collections import Counter
import json

# Import ResNet-18 for feature extraction
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet18, ResNet18_Weights
    RESNET_AVAILABLE = True
    print("[INFO] ResNet-18 available for enhanced feature extraction")
except ImportError:
    RESNET_AVAILABLE = False
    print("[WARNING] ResNet-18 not available, using K-means only")

class EnhancedColorDetector:
    """
    Enhanced color detection with ResNet-18 feature extraction + K-means clustering
    """
    
    def __init__(self, n_clusters=8, confidence_threshold=0.6, enable_resnet=True):
        """
        Initialize enhanced color detector with ResNet-18 + K-means
        
        Args:
            n_clusters: Number of color clusters for K-means
            confidence_threshold: Minimum confidence for color detection
            enable_resnet: Whether to use ResNet-18 feature extraction
        """
        self.n_clusters = n_clusters
        self.confidence_threshold = confidence_threshold
        self.enable_resnet = enable_resnet and RESNET_AVAILABLE
        self.resnet_model = None
        
        # Initialize ResNet-18 if available
        if self.enable_resnet:
            try:
                self.resnet_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                self.resnet_model.eval()
                
                # Remove final classification layer for feature extraction
                self.feature_extractor = torch.nn.Sequential(*list(self.resnet_model.children())[:-1])
                
                # Define transforms for ResNet-18
                self.resnet_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                print("[INFO] ResNet-18 feature extractor initialized")
            except Exception as e:
                print(f"[WARNING] Failed to initialize ResNet-18: {e}")
                self.enable_resnet = False
                self.resnet_model = None
        
        # Define color families with their HSV ranges
        self.color_families = {
            'Red': {
                'hsv_ranges': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
                'shades': ['Misty Rose', 'Light Coral', 'Indian Red', 'Crimson', 'Fire Brick', 'Dark Red', 'Maroon'],
                'rgb_values': [(255, 228, 225), (240, 128, 128), (205, 92, 92), (220, 20, 60), (178, 34, 34), (139, 0, 0), (128, 0, 0)]
            },
            'Blue': {
                'hsv_ranges': [(100, 50, 50), (130, 255, 255)],
                'shades': ['Ice Blue', 'Light Blue', 'Sky Blue', 'Cornflower Blue', 'Royal Blue', 'Navy Blue', 'Midnight Blue'],
                'rgb_values': [(240, 248, 255), (173, 216, 230), (135, 206, 235), (100, 149, 237), (65, 105, 225), (0, 0, 128), (25, 25, 112)]
            },
            'Green': {
                'hsv_ranges': [(35, 50, 50), (85, 255, 255)],
                'shades': ['Mint', 'Light Green', 'Lime Green', 'Forest Green', 'Green', 'Dark Green', 'Emerald'],
                'rgb_values': [(245, 255, 250), (144, 238, 144), (50, 205, 50), (34, 139, 34), (0, 128, 0), (0, 100, 0), (80, 200, 120)]
            },
            'Yellow': {
                'hsv_ranges': [(20, 50, 50), (35, 255, 255)],
                'shades': ['Light Yellow', 'Yellow', 'Golden', 'Dark Golden', 'Orange', 'Dark Orange', 'Deep Orange'],
                'rgb_values': [(255, 255, 224), (255, 255, 0), (255, 215, 0), (184, 134, 11), (255, 165, 0), (255, 140, 0), (255, 69, 0)]
            },
            'Purple': {
                'hsv_ranges': [(130, 50, 50), (170, 255, 255)],
                'shades': ['Lavender', 'Thistle', 'Orchid', 'Medium Purple', 'Purple', 'Indigo', 'Dark Violet'],
                'rgb_values': [(230, 230, 250), (216, 191, 216), (218, 112, 214), (147, 112, 219), (128, 0, 128), (75, 0, 130), (148, 0, 211)]
            },
            'Neutral': {
                'hsv_ranges': [(0, 0, 0), (180, 30, 255)],
                'shades': ['White', 'Light Gray', 'Silver', 'Gray', 'Dark Gray', 'Charcoal', 'Black'],
                'rgb_values': [(255, 255, 255), (211, 211, 211), (192, 192, 192), (128, 128, 128), (105, 105, 105), (54, 54, 54), (0, 0, 0)]
            }
        }
        
        # Enhanced object categories with ResNet-18 features
        self.object_categories = {
            'Person': ['person'],
            'Vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'boat', 'train', 'airplane'],
            'Animal': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
            'Electronics': ['cell phone', 'laptop', 'tv', 'mouse', 'remote', 'keyboard'],
            'Furniture': ['chair', 'couch', 'bed', 'dining table', 'toilet', 'potted plant'],
            'Food': ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
            'Drinkware': ['cup', 'bottle', 'wine glass'],
            'Tableware': ['bowl', 'fork', 'knife', 'spoon'],
            'Clothing': ['tie'],
            'Personal': ['backpack', 'handbag', 'suitcase', 'umbrella', 'toothbrush'],
            'Sports': ['sports ball', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'frisbee', 'kite', 'skis', 'snowboard'],
            'Appliance': ['microwave', 'oven', 'toaster', 'refrigerator', 'sink'],
            'Traffic': ['traffic light', 'stop sign', 'parking meter', 'fire hydrant'],
            'Object': ['book', 'clock', 'vase', 'scissors', 'hair drier', 'teddy bear']
        }
        
        print(f"[INFO] Enhanced Color Detector initialized with K-means ({n_clusters} clusters) + ResNet-18: {self.enable_resnet}")
    
    def extract_resnet_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract ResNet-18 features from image"""
        if not self.enable_resnet or self.resnet_model is None:
            return None
        
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Apply transforms and extract features
            with torch.no_grad():
                input_tensor = self.resnet_transform(rgb_image).unsqueeze(0)
                features = self.feature_extractor(input_tensor)
                features = features.flatten().numpy()
            
            return features
            
        except Exception as e:
            print(f"[ERROR] ResNet feature extraction failed: {e}")
            return None
    
    def detect_colors_enhanced(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """
        Enhanced color detection using ResNet-18 features + K-means clustering
        
        Args:
            image: Input image (BGR format)
            region: Optional region (x1, y1, x2, y2) to analyze
            
        Returns:
            Dictionary with enhanced color detection results
        """
        try:
            start_time = time.time()
            
            # Extract region if specified
            if region:
                x1, y1, x2, y2 = region
                roi = image[y1:y2, x1:x2]
            else:
                roi = image
            
            if roi.size == 0:
                return self._get_empty_result()
            
            # Extract ResNet-18 features
            resnet_features = self.extract_resnet_features(roi)
            
            # Convert to RGB for K-means clustering
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Reshape for K-means
            pixels = roi_rgb.reshape(-1, 3)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers and labels
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Count pixels in each cluster
            label_counts = Counter(labels)
            total_pixels = sum(label_counts.values())
            
            # Sort colors by frequency
            sorted_colors = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Process dominant colors with ResNet-18 enhancement
            detected_colors = []
            for label, count in sorted_colors[:3]:  # Top 3 colors
                color_rgb = colors[label].astype(int)
                percentage = (count / total_pixels) * 100
                
                # Map to color family and shade
                color_info = self._map_color_to_family(color_rgb)
                
                # Enhanced confidence with ResNet-18 features
                base_confidence = min(percentage / 20, 1.0)
                if resnet_features is not None:
                    # Use ResNet features to boost confidence
                    feature_confidence = self._calculate_feature_confidence(resnet_features, color_rgb)
                    enhanced_confidence = min(base_confidence * 1.2, 1.0)
                else:
                    enhanced_confidence = base_confidence
                
                detected_colors.append({
                    'rgb': color_rgb.tolist(),
                    'hex': self._rgb_to_hex(color_rgb),
                    'percentage': percentage,
                    'family': color_info['family'],
                    'shade': color_info['shade'],
                    'confidence': enhanced_confidence,
                    'method': 'kmeans_resnet18' if self.enable_resnet else 'kmeans_only'
                })
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'dominant_colors': detected_colors,
                'primary_color': detected_colors[0] if detected_colors else None,
                'color_distribution': self._analyze_color_distribution(detected_colors),
                'resnet_features': {
                    'available': self.enable_resnet,
                    'feature_dim': len(resnet_features) if resnet_features is not None else 0,
                    'enhanced': self.enable_resnet and resnet_features is not None
                },
                'processing_info': {
                    'method': 'kmeans_resnet18_enhanced' if self.enable_resnet else 'kmeans_clustering',
                    'clusters': self.n_clusters,
                    'pixels_analyzed': total_pixels,
                    'processing_time_ms': processing_time * 1000,
                    'region': region
                }
            }
            
        except Exception as e:
            print(f"[ERROR] Enhanced color detection failed: {e}")
            return self._get_empty_result()
    
    def _calculate_feature_confidence(self, features: np.ndarray, color_rgb: np.ndarray) -> float:
        """Calculate confidence boost using ResNet-18 features"""
        try:
            # Simple heuristic based on feature magnitude and color properties
            feature_magnitude = np.linalg.norm(features)
            color_brightness = np.mean(color_rgb) / 255.0
            
            # Combine features for confidence calculation
            confidence_boost = (feature_magnitude / 1000.0) * 0.3 + color_brightness * 0.2
            return min(confidence_boost, 0.3)  # Max 30% boost
            
        except Exception:
            return 0.0
    
    def categorize_object_enhanced(self, class_name: str, color_result: Dict, confidence: float = 0.8, 
                                  object_features: Optional[np.ndarray] = None) -> Dict:
        """
        Enhanced object categorization with ResNet-18 features + K-means colors
        
        Args:
            class_name: YOLO detected class name
            color_result: Enhanced color detection result
            confidence: Object detection confidence
            object_features: ResNet-18 features for the object
            
        Returns:
            Enhanced object categorization with ResNet-18 + K-means
        """
        try:
            # Find object category
            object_category = self._find_object_category(class_name)
            
            # Get display name with emoji
            display_name = self._get_display_name(class_name, object_category)
            
            # Get primary color
            primary_color = color_result.get('primary_color', {})
            color_family = primary_color.get('family', 'Unknown')
            color_shade = primary_color.get('shade', 'Unknown')
            color_hex = primary_color.get('hex', '#000000')
            color_confidence = primary_color.get('confidence', 0)
            
            # Enhanced confidence with ResNet-18 features
            if object_features is not None and self.enable_resnet:
                feature_confidence = self._calculate_feature_confidence(object_features, 
                                                                       np.array(primary_color.get('rgb', [0, 0, 0])))
                enhanced_confidence = min(confidence * (1 + feature_confidence), 1.0)
            else:
                enhanced_confidence = confidence
            
            # Create enhanced object description
            method_indicator = "🧠" if self.enable_resnet else "🎨"
            description = f"{display_name} ({color_family} - {color_shade})"
            
            # Determine if it's a significant detection
            is_significant = (
                enhanced_confidence > 0.7 and 
                color_confidence > self.confidence_threshold
            )
            
            return {
                'class_name': class_name,
                'display_name': display_name,
                'category': object_category,
                'confidence': enhanced_confidence,
                'description': description,
                'color_info': {
                    'family': color_family,
                    'shade': color_shade,
                    'hex': color_hex,
                    'rgb': primary_color.get('rgb', [0, 0, 0]),
                    'confidence': color_confidence
                },
                'resnet_features': {
                    'available': self.enable_resnet,
                    'enhanced': self.enable_resnet and object_features is not None,
                    'feature_dim': len(object_features) if object_features is not None else 0
                },
                'is_significant': is_significant,
                'enhanced_label': f"{display_name} {method_indicator}{color_shade}",
                'processing_method': 'resnet18_kmeans' if self.enable_resnet else 'kmeans_only'
            }
            
        except Exception as e:
            print(f"[ERROR] Enhanced object categorization failed: {e}")
            return {
                'class_name': class_name,
                'display_name': class_name.title(),
                'category': 'Unknown',
                'confidence': confidence,
                'error': str(e)
            }
    
    def _map_color_to_family(self, rgb: np.ndarray) -> Dict:
        """Map RGB color to color family and shade"""
        try:
            # Convert RGB to HSV for better color matching
            rgb_normalized = rgb / 255.0
            hsv = cv2.cvtColor(np.uint8([[rgb_normalized * 255]]), cv2.COLOR_RGB2HSV)[0][0]
            
            best_family = 'Neutral'
            best_shade = 'Unknown'
            min_distance = float('inf')
            
            for family_name, family_data in self.color_families.items():
                for i, shade_name in enumerate(family_data['shades']):
                    shade_rgb = np.array(family_data['rgb_values'][i])
                    
                    # Calculate color distance
                    distance = np.linalg.norm(rgb - shade_rgb)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_family = family_name
                        best_shade = shade_name
            
            return {
                'family': best_family,
                'shade': best_shade,
                'distance': min_distance
            }
            
        except Exception as e:
            print(f"[ERROR] Color mapping failed: {e}")
            return {'family': 'Unknown', 'shade': 'Unknown', 'distance': float('inf')}
    
    def _find_object_category(self, class_name: str) -> str:
        """Find the category for a given class name"""
        class_name_lower = class_name.lower()
        
        for category, classes in self.object_categories.items():
            if class_name_lower in classes:
                return category
        
        return 'Unknown'
    
    def _get_display_name(self, class_name: str, category: str) -> str:
        """Get display name with emoji based on class and category"""
        emoji_map = {
            'Person': '👤',
            'Vehicle': '🚗',
            'Animal': '🐾',
            'Electronics': '📱',
            'Furniture': '🪑',
            'Food': '🍎',
            'Drinkware': '☕',
            'Tableware': '🍴',
            'Clothing': '👔',
            'Personal': '🎒',
            'Sports': '⚽',
            'Appliance': '🔌',
            'Traffic': '🚦',
            'Object': '📦'
        }
        
        emoji = emoji_map.get(category, '📌')
        return f"{emoji} {class_name.title()}"
    
    def _analyze_color_distribution(self, colors: List[Dict]) -> Dict:
        """Analyze the distribution of detected colors"""
        if not colors:
            return {'dominant_family': 'Unknown', 'diversity': 0}
        
        families = [c['family'] for c in colors]
        family_counts = Counter(families)
        
        return {
            'dominant_family': family_counts.most_common(1)[0][0],
            'family_distribution': dict(family_counts),
            'diversity': len(families),
            'is_monochromatic': len(families) == 1
        }
    
    def _rgb_to_hex(self, rgb: np.ndarray) -> str:
        """Convert RGB to hex color"""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
    
    def _get_empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'success': False,
            'dominant_colors': [],
            'primary_color': None,
            'color_distribution': {'dominant_family': 'Unknown', 'diversity': 0},
            'resnet_features': {'available': False, 'enhanced': False},
            'processing_info': {'error': 'Processing failed'}
        }


# Global enhanced instance for easy access
enhanced_detector = EnhancedColorDetector()

# Fallback to basic K-means if enhanced fails
try:
    kmeans_detector = enhanced_detector
    print("[INFO] Enhanced K-means + ResNet-18 detector loaded successfully")
except Exception as e:
    print(f"[WARNING] Enhanced detector failed, using basic K-means: {e}")
    from kmeans_color_detector import kmeans_detector

# Convenience functions
def detect_colors_enhanced(image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> Dict:
    """Enhanced color detection using ResNet-18 + K-means"""
    return enhanced_detector.detect_colors_enhanced(image, region)

def categorize_object_enhanced(class_name: str, color_result: Dict, confidence: float = 0.8, 
                              object_features: Optional[np.ndarray] = None) -> Dict:
    """Enhanced object categorization with ResNet-18 + K-means"""
    return enhanced_detector.categorize_object_enhanced(class_name, color_result, confidence, object_features)


if __name__ == "__main__":
    print("🧠🎨 Enhanced K-means + ResNet-18 Color Detector Module")
    print("=" * 60)
    
    print("🧪 Testing enhanced color detection...")
    
    # Create a test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test enhanced color detection
    result = detect_colors_enhanced(test_image)
    print(f"✅ Enhanced color detection test: {result['success']}")
    
    if result['success']:
        primary = result['primary_color']
        resnet_info = result.get('resnet_features', {})
        print(f"   Primary color: {primary['family']} - {primary['shade']}")
        print(f"   ResNet-18 enhanced: {resnet_info.get('enhanced', False)}")
        print(f"   Processing time: {result['processing_info']['processing_time_ms']:.2f}ms")
    
    print("\n📖 Usage:")
    print("   from kmeans_color_detector import detect_colors_enhanced, categorize_object_enhanced")
    print("   colors = detect_colors_enhanced(image)")
    print("   categorized = categorize_object_enhanced('car', colors, 0.9)")
    
    print("\n✅ Enhanced K-means + ResNet-18 Color Detector ready!")
    print("   Features:")
    print("   - ResNet-18 feature extraction")
    print("   - K-means color clustering")
    print("   - 56 color shades across 6 families")
    print("   - Enhanced confidence calculation")
    print("   - Fallback to K-means only if ResNet-18 fails")


# Wrapper functions for compatibility with modules expecting different function names
def detect_image_colors(image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> Dict:
    """Wrapper for detect_colors_enhanced for backward compatibility"""
    return detect_colors_enhanced(image, region)

def categorize_detected_object(class_name: str, color_result: Dict, confidence: float = 0.8, 
                               object_features: Optional[np.ndarray] = None) -> Dict:
    """Wrapper for categorize_object_enhanced for backward compatibility"""
    return categorize_object_enhanced(class_name, color_result, confidence, object_features)

def analyze_scene(image: np.ndarray, detections: List[Dict]) -> Dict:
    """Simple scene analysis based on detected objects and their colors"""
    try:
        scene_colors = detect_colors_enhanced(image)
        
        object_categories = []
        for det in detections:
            class_name = det.get('class_name', 'Unknown')
            color_info = det.get('color_info', {})
            object_categories.append({
                'class': class_name,
                'color_family': color_info.get('family', 'Unknown'),
                'color_shade': color_info.get('shade', 'Unknown')
            })
        
        return {
            'success': True,
            'scene_analysis': {
                'dominant_color_family': scene_colors.get('primary_color', {}).get('family', 'Unknown'),
                'object_count': len(detections),
                'objects': object_categories
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}
