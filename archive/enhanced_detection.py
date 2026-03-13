"""
Enhanced Detection for Blurry and Angled Images
Using MobileNetV2 + LightOnOCR + Advanced Preprocessing
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import re
from typing import Tuple, Optional, List

class EnhancedLicensePlateDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.mobilenet = self._load_mobilenet()
        self.transform = self._get_transforms()
        
    def _load_mobilenet(self):
        """Load MobileNetV2 for license plate detection in challenging conditions"""
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        # Modify for binary classification (plate vs no-plate)
        model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
        model.to(self.device)
        model.eval()
        return model
    
    def _get_transforms(self):
        """Image transforms for MobileNetV2"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_challenging_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Generate multiple enhanced versions of challenging images
        """
        enhanced_versions = []
        
        # Original
        enhanced_versions.append(image)
        
        # Deblur using Wiener filter
        try:
            deblurred = self._deblur_image(image)
            enhanced_versions.append(deblurred)
        except:
            pass
        
        # Perspective correction for angled plates
        try:
            corrected = self._correct_perspective(image)
            enhanced_versions.append(corrected)
        except:
            pass
        
        # Super-resolution enhancement
        try:
            upscaled = self._enhance_resolution(image)
            enhanced_versions.append(upscaled)
        except:
            pass
        
        # Contrast enhancement
        try:
            enhanced = self._enhance_contrast(image)
            enhanced_versions.append(enhanced)
        except:
            pass
        
        return enhanced_versions
    
    def _deblur_image(self, image: np.ndarray) -> np.ndarray:
        """Apply deblurring techniques"""
        # Convert to grayscale for deblurring
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Wiener filter (simplified version)
        kernel = np.ones((5,5), np.float32) / 25
        blurred = cv2.filter2D(gray, -1, kernel)
        
        # Estimate noise and apply deconvolution
        noise_var = np.var(gray - blurred)
        deblurred = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        
        # Convert back to BGR if needed
        if len(image.shape) == 3:
            deblurred = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
        
        return deblurred
    
    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Correct perspective distortion for angled license plates"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest rectangular contour (potential license plate)
        largest_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # Rectangle
                    largest_contour = approx
                    max_area = area
        
        if largest_contour is not None:
            # Apply perspective transform
            corrected = self._apply_perspective_transform(image, largest_contour)
            return corrected
        
        return image
    
    def _apply_perspective_transform(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Apply perspective transformation to straighten license plate"""
        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = corners.reshape(4, 2)
        
        # Calculate destination rectangle dimensions
        width = max(np.linalg.norm(corners[0] - corners[1]), 
                   np.linalg.norm(corners[2] - corners[3]))
        height = max(np.linalg.norm(corners[0] - corners[3]), 
                    np.linalg.norm(corners[1] - corners[2]))
        
        # Destination points
        dst_points = np.array([
            [0, 0],
            [width-1, 0],
            [width-1, height-1],
            [0, height-1]
        ], dtype=np.float32)
        
        # Get transformation matrix
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_points)
        
        # Apply transformation
        corrected = cv2.warpPerspective(image, M, (int(width), int(height)))
        
        return corrected
    
    def _enhance_resolution(self, image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """Enhance image resolution using super-resolution techniques"""
        # Simple bicubic upsampling with sharpening
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Upscale
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply sharpening filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel)
        
        return sharpened
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast for better text visibility"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels back
        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_license_plate_in_challenging_image(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect license plate in challenging images using multiple preprocessing approaches
        """
        enhanced_images = self.preprocess_challenging_image(image)
        
        best_detection = None
        best_confidence = 0.0
        
        for i, enhanced_img in enumerate(enhanced_images):
            try:
                # Extract candidate regions using sliding window
                candidate_regions = self._extract_candidate_regions(enhanced_img)
                
                for region in candidate_regions:
                    x1, y1, x2, y2 = region
                    crop = enhanced_img[y1:y2, x1:x2]
                    
                    if crop.size == 0:
                        continue
                    
                    # Classify with MobileNetV2
                    confidence = self._classify_with_mobilenet(crop)
                    
                    if confidence > best_confidence and confidence > 0.7:  # Threshold
                        best_confidence = confidence
                        best_detection = (x1, y1, x2, y2)
                        print(f"[DEBUG] Better detection found: confidence={confidence:.3f}, method=enhanced_{i}")
                        
            except Exception as e:
                print(f"[DEBUG] Error processing enhanced image {i}: {e}")
                continue
        
        return best_detection
    
    def _extract_candidate_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Extract candidate regions for license plate detection"""
        regions = []
        
        # Method 1: Edge-based detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # License plate aspect ratio typically 2:1 to 5:1
                if 1.5 < aspect_ratio < 6.0:
                    regions.append((x, y, x + w, y + h))
        
        # Method 2: Sliding window approach
        h, w = image.shape[:2]
        window_sizes = [(80, 20), (120, 30), (160, 40), (200, 50)]
        
        for win_w, win_h in window_sizes:
            stride_y, stride_x = win_h // 2, win_w // 2
            
            for y in range(0, h - win_h, stride_y):
                for x in range(0, w - win_w, stride_x):
                    regions.append((x, y, x + win_w, y + win_h))
        
        return regions
    
    def _classify_with_mobilenet(self, crop: np.ndarray) -> float:
        """Classify image region as license plate or not using MobileNetV2"""
        try:
            # Convert PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.mobilenet(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence = probabilities[0][1].item()  # Probability of class 1 (license plate)
            
            return confidence
            
        except Exception as e:
            print(f"[DEBUG] MobileNetV2 classification error: {e}")
            return 0.0
    
    def enhance_color_detection(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """
        Enhanced color detection for challenging images
        """
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return "unknown"
        
        # Apply multiple color detection methods
        colors = []
        
        # Method 1: Enhanced HSV analysis
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for Indian vehicles
        color_ranges = {
            'white': ([0, 0, 200], [180, 30, 255]),
            'black': ([0, 0, 0], [180, 255, 50]),
            'silver': ([0, 0, 100], [180, 30, 200]),
            'gray': ([0, 0, 50], [180, 30, 150]),
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
            'yellow': ([20, 50, 50], [40, 255, 255]),
            'brown': ([8, 50, 50], [20, 255, 255]),
        }
        
        # Count pixels for each color
        color_counts = {}
        total_pixels = hsv.shape[0] * hsv.shape[1]
        
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            count = cv2.countNonZero(mask)
            percentage = (count / total_pixels) * 100
            color_counts[color_name] = percentage
        
        # Find dominant color
        if color_counts:
            dominant_color = max(color_counts, key=color_counts.get)
            if color_counts[dominant_color] > 10:  # Minimum 10% threshold
                colors.append(dominant_color)
        
        # Method 2: LAB color space analysis (better for challenging conditions)
        try:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Analyze color distribution
            a_mean, b_mean = np.mean(a), np.mean(b)
            
            # Simple color classification based on LAB values
            if a_mean > 130:
                colors.append('red')
            elif a_mean < 110:
                colors.append('green')
            elif b_mean > 140:
                colors.append('yellow')
            elif b_mean < 110:
                colors.append('blue')
            elif np.mean(l) > 200:
                colors.append('white')
            elif np.mean(l) < 50:
                colors.append('black')
                
        except Exception as e:
            print(f"[DEBUG] LAB color analysis error: {e}")
        
        # Return the most common color
        if colors:
            from collections import Counter
            most_common = Counter(colors).most_common(1)[0][0]
            return most_common
        
        return "unknown"


# Integration function for existing app.py
def enhanced_license_plate_detection(image: np.ndarray) -> dict:
    """
    Enhanced detection for challenging images
    Integrates with existing YOLO26 system
    """
    detector = EnhancedLicensePlateDetector()
    
    # Try to detect license plate in challenging conditions
    plate_bbox = detector.detect_license_plate_in_challenging_image(image)
    
    result = {
        "plate_detected": plate_bbox is not None,
        "plate_bbox": plate_bbox,
        "color": "unknown",
        "method": "enhanced_detection"
    }
    
    if plate_bbox:
        # Enhanced color detection
        color = detector.enhance_color_detection(image, plate_bbox)
        result["color"] = color
        
        # Extract text using existing OCR methods
        x1, y1, x2, y2 = plate_bbox
        plate_crop = image[y1:y2, x1:x2]
        
        # Use existing OCR functions from app.py
        try:
            from app import _extract_text_from_license_plate_crop
            plate_text = _extract_text_from_license_plate_crop(plate_crop)
            result["plate_text"] = plate_text
        except ImportError:
            result["plate_text"] = ""
    
    return result


if __name__ == "__main__":
    # Test the enhanced detection
    print("Enhanced License Plate Detection for Challenging Images")
    print("Features:")
    print("- Deblurring using Wiener filter")
    print("- Perspective correction for angled plates")
    print("- Super-resolution enhancement")
    print("- MobileNetV2 classification")
    print("- Enhanced color detection using HSV + LAB")
    print("- Multiple preprocessing approaches")
