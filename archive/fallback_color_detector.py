"""
Fallback Color Detection System for YOLO26
Multi-level fallback system ensures color detection always works
Level 1: Enhanced ResNet-18 + K-means
Level 2: Basic K-means only
Level 3: Simple HSV color detection
Level 4: Basic color classification
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import Counter

class FallbackColorDetector:
    """
    Multi-level fallback color detection system
    Ensures color detection always works regardless of dependencies
    """
    
    def __init__(self):
        self.enhanced_detector = None
        self.kmeans_detector = None
        self.fallback_level = 0
        
        # Try to initialize enhanced detector (Level 1)
        try:
            from kmeans_color_detector import enhanced_detector
            self.enhanced_detector = enhanced_detector
            self.fallback_level = 1
            print("[INFO] Level 1: Enhanced ResNet-18 + K-means detector available")
        except ImportError:
            print("[WARNING] Level 1: Enhanced detector not available")
        
        # Try to initialize basic K-means (Level 2)
        try:
            from sklearn.cluster import KMeans
            self.kmeans_available = True
            if self.fallback_level == 0:
                self.fallback_level = 2
                print("[INFO] Level 2: Basic K-means detector available")
        except ImportError:
            self.kmeans_available = False
            print("[WARNING] Level 2: K-means not available")
        
        # Always available fallbacks (Level 3 & 4)
        print("[INFO] Level 3: Simple HSV detection always available")
        print("[INFO] Level 4: Basic color classification always available")
        
        # Define color families for all fallback levels
        self.color_families = {
            'Red': {
                'hsv_ranges': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
                'shades': ['Red', 'Dark Red', 'Light Red'],
                'rgb_values': [(255, 0, 0), (139, 0, 0), (255, 102, 102)]
            },
            'Blue': {
                'hsv_ranges': [(100, 50, 50), (130, 255, 255)],
                'shades': ['Blue', 'Dark Blue', 'Light Blue'],
                'rgb_values': [(0, 0, 255), (0, 0, 139), (102, 102, 255)]
            },
            'Green': {
                'hsv_ranges': [(35, 50, 50), (85, 255, 255)],
                'shades': ['Green', 'Dark Green', 'Light Green'],
                'rgb_values': [(0, 255, 0), (0, 100, 0), (102, 255, 102)]
            },
            'Yellow': {
                'hsv_ranges': [(20, 50, 50), (35, 255, 255)],
                'shades': ['Yellow', 'Orange', 'Gold'],
                'rgb_values': [(255, 255, 0), (255, 165, 0), (255, 215, 0)]
            },
            'Purple': {
                'hsv_ranges': [(130, 50, 50), (170, 255, 255)],
                'shades': ['Purple', 'Violet', 'Magenta'],
                'rgb_values': [(128, 0, 128), (238, 130, 238), (255, 0, 255)]
            },
            'Neutral': {
                'hsv_ranges': [(0, 0, 0), (180, 30, 255)],
                'shades': ['White', 'Gray', 'Black'],
                'rgb_values': [(255, 255, 255), (128, 128, 128), (0, 0, 0)]
            }
        }
        
        print(f"[INFO] Fallback Color Detector initialized (Level {self.fallback_level})")
    
    def detect_colors_with_fallback(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """
        Detect colors with multi-level fallback system
        
        Args:
            image: Input image (BGR format)
            region: Optional region (x1, y1, x2, y2) to analyze
            
        Returns:
            Dictionary with color detection results and fallback info
        """
        start_time = time.time()
        
        # Extract region if specified
        if region:
            x1, y1, x2, y2 = region
            roi = image[y1:y2, x1:x2]
        else:
            roi = image
        
        if roi.size == 0:
            return self._get_empty_result()
        
        # Level 1: Enhanced ResNet-18 + K-means
        if self.fallback_level >= 1 and self.enhanced_detector:
            try:
                from kmeans_color_detector import detect_colors_enhanced
                result = detect_colors_enhanced(image, region)
                if result.get('success'):
                    result['fallback_level'] = 1
                    result['fallback_method'] = 'Enhanced ResNet-18 + K-means'
                    result['processing_time_ms'] = (time.time() - start_time) * 1000
                    return result
            except Exception as e:
                print(f"[WARNING] Level 1 failed: {e}, falling back to Level 2")
        
        # Level 2: Basic K-means only
        if self.fallback_level >= 2 and self.kmeans_available:
            try:
                result = self._basic_kmeans_detection(roi)
                if result.get('success'):
                    result['fallback_level'] = 2
                    result['fallback_method'] = 'Basic K-means'
                    result['processing_time_ms'] = (time.time() - start_time) * 1000
                    return result
            except Exception as e:
                print(f"[WARNING] Level 2 failed: {e}, falling back to Level 3")
        
        # Level 3: Simple HSV color detection
        try:
            result = self._simple_hsv_detection(roi)
            if result.get('success'):
                result['fallback_level'] = 3
                result['fallback_method'] = 'Simple HSV Detection'
                result['processing_time_ms'] = (time.time() - start_time) * 1000
                return result
        except Exception as e:
            print(f"[WARNING] Level 3 failed: {e}, falling back to Level 4")
        
        # Level 4: Basic color classification (always works)
        try:
            result = self._basic_color_classification(roi)
            result['fallback_level'] = 4
            result['fallback_method'] = 'Basic Color Classification'
            result['processing_time_ms'] = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            print(f"[ERROR] All fallback levels failed: {e}")
            return self._get_empty_result()
    
    def _basic_kmeans_detection(self, roi: np.ndarray) -> Dict:
        """Level 2: Basic K-means color detection"""
        try:
            from sklearn.cluster import KMeans
            
            # Convert to RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pixels = roi_rgb.reshape(-1, 3)
            
            # Apply K-means with fewer clusters for speed
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=5)
            kmeans.fit(pixels)
            
            # Get dominant colors
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            label_counts = Counter(labels)
            
            # Sort by frequency
            sorted_colors = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            
            detected_colors = []
            for label, count in sorted_colors[:3]:
                color_rgb = colors[label].astype(int)
                percentage = (count / len(labels)) * 100
                
                # Map to color family
                color_info = self._map_rgb_to_family(color_rgb)
                
                detected_colors.append({
                    'rgb': color_rgb.tolist(),
                    'hex': self._rgb_to_hex(color_rgb),
                    'percentage': percentage,
                    'family': color_info['family'],
                    'shade': color_info['shade'],
                    'confidence': min(percentage / 25, 1.0)
                })
            
            return {
                'success': True,
                'dominant_colors': detected_colors,
                'primary_color': detected_colors[0] if detected_colors else None,
                'color_distribution': self._analyze_color_distribution(detected_colors)
            }
            
        except Exception as e:
            raise Exception(f"Basic K-means detection failed: {e}")
    
    def _simple_hsv_detection(self, roi: np.ndarray) -> Dict:
        """Level 3: Simple HSV color detection"""
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            color_counts = {}
            
            # Check each color family
            for family_name, family_data in self.color_families.items():
                count = 0
                for i in range(0, len(family_data['hsv_ranges']), 2):
                    if i + 1 < len(family_data['hsv_ranges']):
                        lower = np.array(family_data['hsv_ranges'][i])
                        upper = np.array(family_data['hsv_ranges'][i + 1])
                        mask = cv2.inRange(hsv, lower, upper)
                        count += cv2.countNonZero(mask)
                
                color_counts[family_name] = count
            
            # Find dominant color
            if color_counts:
                dominant_family = max(color_counts, key=color_counts.get)
                if color_counts[dominant_family] > 0:
                    shade = family_data['shades'][0]
                    rgb = family_data['rgb_values'][0]
                    
                    detected_color = {
                        'rgb': rgb,
                        'hex': self._rgb_to_hex(np.array(rgb)),
                        'percentage': (color_counts[dominant_family] / (roi.shape[0] * roi.shape[1])) * 100,
                        'family': dominant_family,
                        'shade': shade,
                        'confidence': 0.7
                    }
                    
                    return {
                        'success': True,
                        'dominant_colors': [detected_color],
                        'primary_color': detected_color,
                        'color_distribution': {'dominant_family': dominant_family, 'diversity': 1}
                    }
            
            raise Exception("No dominant color found")
            
        except Exception as e:
            raise Exception(f"HSV detection failed: {e}")
    
    def _basic_color_classification(self, roi: np.ndarray) -> Dict:
        """Level 4: Basic color classification (always works)"""
        try:
            # Calculate average color
            avg_color = np.mean(roi, axis=(0, 1))
            
            # Simple color classification based on average
            b, g, r = avg_color
            
            # Determine dominant color channel
            max_channel = max(r, g, b)
            
            if max_channel < 50:
                family = 'Neutral'
                shade = 'Black'
                rgb = [0, 0, 0]
            elif max_channel > 200:
                if r > g and r > b:
                    family = 'Red'
                    shade = 'Red'
                    rgb = [255, 0, 0]
                elif g > r and g > b:
                    family = 'Green'
                    shade = 'Green'
                    rgb = [0, 255, 0]
                elif b > r and b > g:
                    family = 'Blue'
                    shade = 'Blue'
                    rgb = [0, 0, 255]
                else:
                    family = 'Neutral'
                    shade = 'White'
                    rgb = [255, 255, 255]
            else:
                # Medium intensity - classify based on ratios
                if r > g * 1.5 and r > b * 1.5:
                    family = 'Red'
                    shade = 'Red'
                    rgb = [255, 0, 0]
                elif g > r * 1.5 and g > b * 1.5:
                    family = 'Green'
                    shade = 'Green'
                    rgb = [0, 255, 0]
                elif b > r * 1.5 and b > g * 1.5:
                    family = 'Blue'
                    shade = 'Blue'
                    rgb = [0, 0, 255]
                elif r > 150 and g > 150:
                    family = 'Yellow'
                    shade = 'Yellow'
                    rgb = [255, 255, 0]
                elif r > 150 and b > 150:
                    family = 'Purple'
                    shade = 'Purple'
                    rgb = [128, 0, 128]
                else:
                    family = 'Neutral'
                    shade = 'Gray'
                    rgb = [128, 128, 128]
            
            detected_color = {
                'rgb': rgb,
                'hex': self._rgb_to_hex(np.array(rgb)),
                'percentage': 100.0,
                'family': family,
                'shade': shade,
                'confidence': 0.5
            }
            
            return {
                'success': True,
                'dominant_colors': [detected_color],
                'primary_color': detected_color,
                'color_distribution': {'dominant_family': family, 'diversity': 1}
            }
            
        except Exception as e:
            raise Exception(f"Basic color classification failed: {e}")
    
    def _map_rgb_to_family(self, rgb: np.ndarray) -> Dict:
        """Map RGB color to color family and shade"""
        try:
            best_family = 'Neutral'
            best_shade = 'Unknown'
            min_distance = float('inf')
            
            for family_name, family_data in self.color_families.items():
                for i, shade_name in enumerate(family_data['shades']):
                    shade_rgb = np.array(family_data['rgb_values'][i])
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
            
        except Exception:
            return {'family': 'Unknown', 'shade': 'Unknown', 'distance': float('inf')}
    
    def _analyze_color_distribution(self, colors: List[Dict]) -> Dict:
        """Analyze color distribution"""
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
            'fallback_level': 0,
            'fallback_method': 'None',
            'processing_info': {'error': 'All fallback levels failed'}
        }
    
    def get_fallback_status(self) -> Dict:
        """Get current fallback system status"""
        return {
            'current_level': self.fallback_level,
            'enhanced_available': self.enhanced_detector is not None,
            'kmeans_available': self.kmeans_available,
            'levels_available': [
                'Enhanced ResNet-18 + K-means' if self.enhanced_detector else None,
                'Basic K-means' if self.kmeans_available else None,
                'Simple HSV Detection',
                'Basic Color Classification'
            ],
            'recommended_usage': self._get_usage_recommendation()
        }
    
    def _get_usage_recommendation(self) -> str:
        """Get usage recommendation based on available levels"""
        if self.fallback_level >= 1:
            return "Full enhanced detection available - use for best accuracy"
        elif self.fallback_level >= 2:
            return "Basic K-means available - good for most use cases"
        else:
            return "Basic detection only - suitable for simple applications"


# Global fallback detector instance
fallback_detector = FallbackColorDetector()

# Convenience functions
def detect_colors_fallback(image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> Dict:
    """Detect colors with automatic fallback"""
    return fallback_detector.detect_colors_with_fallback(image, region)

def get_fallback_status() -> Dict:
    """Get fallback system status"""
    return fallback_detector.get_fallback_status()


if __name__ == "__main__":
    print("🛡️ Fallback Color Detection System")
    print("=" * 50)
    
    # Show fallback status
    status = get_fallback_status()
    print(f"Current Level: {status['current_level']}")
    print(f"Enhanced Available: {status['enhanced_available']}")
    print(f"K-means Available: {status['kmeans_available']}")
    print(f"Recommendation: {status['recommended_usage']}")
    
    print("\n📊 Available Fallback Levels:")
    for i, level in enumerate(status['levels_available'], 1):
        if level:
            print(f"   Level {i}: ✅ {level}")
        else:
            print(f"   Level {i}: ❌ Not Available")
    
    # Test fallback system
    print("\n🧪 Testing fallback system...")
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = detect_colors_fallback(test_image)
    
    if result['success']:
        print(f"✅ Fallback system working (Level {result['fallback_level']})")
        print(f"   Method: {result['fallback_method']}")
        print(f"   Primary color: {result['primary_color']['family']} - {result['primary_color']['shade']}")
        print(f"   Processing time: {result['processing_time_ms']:.2f}ms")
    else:
        print("❌ Fallback system failed")
    
    print("\n🛡️ Fallback Color Detection System Ready!")
    print("   - Always works regardless of dependencies")
    print("   - Automatic level selection")
    print("   - Graceful degradation")
    print("   - Performance monitoring")
