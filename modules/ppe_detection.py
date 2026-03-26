"""
PPE Detection Module - Enhanced with Fallback and Auto-Recovery
Ensures PPE detection ALWAYS produces results, even if models fail
"""

import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from ultralytics import YOLO
import time
import threading
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class PPEItem:
    name: str
    present: bool
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    detection_method: str = "color"


@dataclass
class PersonPPE:
    person_id: str
    bbox: Tuple[int, int, int, int]
    vehicle_type: str = "unknown"  # bike, car, truck, unknown
    head_bbox: Optional[Tuple[int, int, int, int]] = None
    helmet: PPEItem = field(default_factory=lambda: PPEItem("helmet", False, 0.0))
    vest: PPEItem = field(default_factory=lambda: PPEItem("vest", False, 0.0))
    seatbelt: PPEItem = field(default_factory=lambda: PPEItem("seatbelt", False, 0.0))
    status: str = "violation"  # compliant or violation
    confidence: float = 0.0
    debug_info: Dict = field(default_factory=dict)


@dataclass
class PPEResult:
    total_persons: int
    helmet_detected: int
    no_helmet: int
    seatbelt_detected: int
    no_seatbelt: int
    persons: List[PersonPPE]
    timestamp: str
    processing_time: float = 0.0
    debug_mode: bool = False
    model_loaded: bool = False
    fallback_used: bool = False
    error_message: str = ""


class PPEDetector:
    """
    Robust PPE Detector with automatic fallback and recovery
    NEVER returns empty - always provides minimum detection
    """

    # Helmet colors in HSV: (name, lower, upper)
    HELMET_COLORS = [
        ("yellow", [20, 100, 100], [35, 255, 255]),
        ("white", [0, 0, 180], [180, 30, 255]),
        ("blue", [90, 50, 50], [130, 255, 255]),
        ("red1", [0, 100, 100], [10, 255, 255]),
        ("red2", [160, 100, 100], [180, 255, 255]),
        ("orange", [10, 100, 100], [20, 255, 255]),
    ]

    COLORS = {
        "compliant": (0, 255, 0),       # Green
        "non_compliant": (0, 0, 255),   # Red
        "head_region": (255, 0, 255),   # Magenta
        "fallback": (0, 255, 255),      # Yellow
    }

    def __init__(self, model_path="yolov8n.pt", device=None, debug=False, auto_recovery=True):
        self.model_path = model_path
        self.model = None
        self.device = device or self._get_device()
        self.debug = debug
        self.auto_recovery = auto_recovery
        self.model_load_attempts = 0
        self.max_load_attempts = 3

        # LOW thresholds for maximum detection sensitivity
        self.person_threshold = 0.25
        self.helmet_threshold = 0.15  # Lowered from 0.20 for better detection
        self.fallback_threshold = 0.10  # Even lower for fallback

        self._ensure_model_loaded()
        print(f"[PPE] Initialized - helmet_threshold={self.helmet_threshold}, fallback_enabled=True")

    def _get_device(self):
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except:
            pass
        return "cpu"

    def _ensure_model_loaded(self):
        """Ensure model is loaded with retry mechanism"""
        while self.model is None and self.model_load_attempts < self.max_load_attempts:
            try:
                self.model_load_attempts += 1
                print(f"[PPE] Loading model (attempt {self.model_load_attempts}/{self.max_load_attempts})...")
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                print(f"[PPE] Model loaded successfully on {self.device}")
                self.model_load_attempts = 0
                return True
            except Exception as e:
                print(f"[PPE-WARNING] Model load attempt {self.model_load_attempts} failed: {e}")
                time.sleep(0.5)

        if self.model is None:
            print(f"[PPE-ERROR] Failed to load model after {self.max_load_attempts} attempts")
            return False
        return True

    def _reload_model(self):
        """Force reload model"""
        self.model = None
        self.model_load_attempts = 0
        return self._ensure_model_loaded()

    def get_head_region(self, person_bbox):
        """Extract head region - top 25% of person bbox"""
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1
        width = x2 - x1
        # BALANCED: Top 25% - good helmet detection area
        head_height = int(height * 0.25)
        # Small width expansion for better coverage
        expand_x = int(width * 0.05)
        return (max(0, x1 - expand_x), y1, min(x2, x2 + expand_x), y1 + head_height)

    def detect_helmet_by_color(self, head_roi, threshold=None):
        """Detect helmet color - ONLY as supporting evidence, NEVER primary"""
        if head_roi.size == 0:
            return False, 0.0, "none"

        thresh = threshold or self.helmet_threshold

        try:
            hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
            best_conf = 0.0
            best_color = "none"

            for i, (color_name, lower, upper) in enumerate(self.HELMET_COLORS):
                if color_name == "red1":
                    mask1 = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    mask2 = cv2.inRange(hsv, np.array(self.HELMET_COLORS[4][1]), np.array(self.HELMET_COLORS[4][2]))
                    mask = cv2.bitwise_or(mask1, mask2)
                    color_name = "red"
                elif color_name == "red2":
                    continue
                else:
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    roi_area = head_roi.shape[0] * head_roi.shape[1]
                    if 0.10 * roi_area < area < 0.70 * roi_area:
                        coverage = area / roi_area
                        conf = min(coverage * 2.0, 0.8)  # MAX 0.8 - color alone never enough
                        if conf > best_conf:
                            best_conf = conf
                            best_color = color_name

            if self.debug and best_conf > 0:
                print(f"[PPE-DEBUG] Color: {best_color}, conf: {best_conf:.2f} (SUPPORTING ONLY)")

            return best_conf >= thresh, best_conf, best_color
        except Exception as e:
            if self.debug:
                print(f"[PPE-DEBUG] Color detection error: {e}")
            return False, 0.0, "error"

    def _check_dome_shape(self, gray_roi):
        """Check if the shape resembles a helmet dome"""
        try:
            # Blur to reduce noise
            blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

            # Find contours
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return False, 0.0

            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            roi_area = gray_roi.shape[0] * gray_roi.shape[1]

            if area < roi_area * 0.15:  # Must cover at least 15% of region
                return False, 0.0

            # Check convexity - helmets are convex
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                convexity = area / hull_area
                if convexity < 0.75:  # Must be fairly convex
                    return False, 0.0

            # Check for dome-like top (parabola shape)
            x, y, w, h = cv2.boundingRect(largest)
            if h > 0:
                aspect_ratio = w / h
                # Helmet dome typically wider than tall or equal
                if aspect_ratio < 0.8 or aspect_ratio > 3.0:
                    return False, 0.0

            # Calculate dome confidence
            dome_conf = min(area / (roi_area * 0.6), 1.0) * convexity
            return True, dome_conf
        except:
            return False, 0.0

    def _check_hair_texture(self, head_roi):
        """Detect if hair is visible (rejects helmet if hair dominates)"""
        try:
            gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)

            # Hair detection: high texture, dark, non-uniform
            # Use Laplacian for texture detection
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_score = np.var(laplacian)

            # Dark regions (potential hair)
            dark_mask = gray < 60
            dark_ratio = np.sum(dark_mask) / dark_mask.size

            # Hair has high texture AND dark color
            if texture_score > 500 and dark_ratio > 0.3:
                return True  # Hair detected

            # Check for fine texture patterns (hair strands)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # High edge density suggests hair texture
            if edge_density > 0.15 and texture_score > 300:
                return True

            return False
        except:
            return False

    def _check_smooth_surface(self, head_roi):
        """Check if surface is smooth (helmet) vs textured (hair/hat)"""
        try:
            gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Calculate local variance (smoothness metric)
            mean = cv2.blur(blurred.astype(np.float32), (5, 5))
            sq_mean = cv2.blur((blurred.astype(np.float32))**2, (5, 5))
            variance = sq_mean - mean**2

            # Helmets are smooth - low local variance
            avg_variance = np.mean(variance)

            # Lower variance = smoother surface
            smoothness = max(0, min(1, 1 - (avg_variance / 1000)))
            return smoothness > 0.6, smoothness
        except:
            return False, 0.0

    def detect_helmet_in_head(self, frame, head_bbox, threshold=None):
        """BALANCED helmet detection - requires 2 of 3 strong conditions"""
        x1, y1, x2, y2 = head_bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return False, 0.0, "invalid_bbox"

        head_roi = frame[y1:y2, x1:x2]
        if head_roi.size == 0:
            return False, 0.0, "empty_roi"

        # STEP 1: Check for hair (strong negative indicator)
        has_hair = self._check_hair_texture(head_roi)
        if has_hair:
            if self.debug:
                print("[PPE-DEBUG] Hair detected - likely no helmet")
            # Hair is strong negative, but don't immediately reject
            hair_penalty = 0.3

        # STEP 2: Check smooth surface (strong positive)
        is_smooth, smooth_conf = self._check_smooth_surface(head_roi)
        smooth_score = smooth_conf if is_smooth else 0.0

        # STEP 3: Check dome shape (strong positive)
        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
        has_dome, dome_conf = self._check_dome_shape(gray)
        dome_score = dome_conf if has_dome else 0.0

        # STEP 4: Check position (must be ON TOP of head)
        position_score = self._check_top_position(head_roi)

        # STEP 5: Color check (supporting only)
        color_detected, color_conf, color_name = self.detect_helmet_by_color(head_roi, threshold)
        color_score = color_conf * 0.5 if color_detected else 0.0  # Reduce color importance

        if self.debug:
            print(f"[PPE-DEBUG] Detection scores:")
            print(f"  - Hair detected: {has_hair}")
            print(f"  - Smooth surface: {is_smooth} (conf: {smooth_conf:.2f})")
            print(f"  - Dome shape: {has_dome} (conf: {dome_conf:.2f})")
            print(f"  - Top position: {position_score:.2f}")
            print(f"  - Color detected: {color_detected} ({color_name}, conf: {color_conf:.2f})")

        # BALANCED DECISION LOGIC:
        # Need 2 of 3 strong conditions OR 1 strong + position + color
        
        strong_conditions = 0
        if is_smooth and smooth_conf > 0.4:  # Lowered from 0.5
            strong_conditions += 1
        if has_dome and dome_conf > 0.3:  # Lowered from 0.4
            strong_conditions += 1
        if position_score > 0.5:  # Lowered from 0.6
            strong_conditions += 1

        # Calculate combined confidence
        combined_conf = (dome_score * 0.35 + smooth_score * 0.35 + 
                        position_score * 0.20 + color_score * 0.10)

        # Apply hair penalty if hair detected
        if has_hair:
            combined_conf *= hair_penalty

        # Decision logic
        if strong_conditions >= 2:
            # Strong case - definitely helmet
            reason = f"valid_helmet_shape_{color_name}" if color_detected else "valid_helmet_shape"
            if combined_conf >= 0.4:  # Lowered from 0.5
                return True, combined_conf, reason

        elif strong_conditions == 1 and color_detected:
            # One strong + color support
            if combined_conf >= 0.5:  # Lowered from 0.6
                return True, combined_conf, f"uncertain_but_positive_{color_name}"

        elif strong_conditions == 1 and not has_hair:
            # One strong, no hair, no color - borderline
            if combined_conf >= 0.6:  # Lowered from 0.7
                return True, combined_conf, "uncertain_but_positive"

        # No helmet cases
        if has_hair:
            return False, combined_conf, "hair_visible"
        elif strong_conditions == 0:
            return False, combined_conf, "no_helmet"
        else:
            return False, combined_conf, "insufficient_evidence"

    def _check_top_position(self, head_roi):
        """Check if helmet-like object is on top of head region"""
        try:
            gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Check top 30% of head region for solid object
            top_region = gray[:int(h*0.3), :]
            
            # Look for solid object at top (helmet should be here)
            _, thresh = cv2.threshold(top_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Calculate coverage in top region
            top_coverage = np.sum(thresh > 0) / (top_region.shape[0] * top_region.shape[1])
            
            # Helmet should have good coverage at top
            if top_coverage > 0.2:
                return min(top_coverage * 2, 1.0)
            return 0.0
        except:
            return 0.0

    def _detect_seatbelt(self, frame, person_bbox):
        """Detect seatbelt - more sensitive for clear seatbelts"""
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1
        width = x2 - x1
        
        # Focus on upper torso and diagonal chest area
        shoulder_y1 = y1 + int(height * 0.10)  # Start higher
        shoulder_y2 = y1 + int(height * 0.55)  # Extend lower
        chest_x1 = x1 + int(width * 0.15)     # Wider area
        chest_x2 = x2 - int(width * 0.05)     # Wider area
        
        h, w = frame.shape[:2]
        # Clamp values
        shoulder_y1 = max(0, min(h, shoulder_y1))
        shoulder_y2 = max(0, min(h, shoulder_y2))
        chest_x1 = max(0, min(w, chest_x1))
        chest_x2 = max(0, min(w, chest_x2))
        
        if shoulder_y2 <= shoulder_y1 or chest_x2 <= chest_x1:
            return False, 0.0, "invalid_region"
        
        try:
            # Extract shoulder and chest regions
            shoulder_roi = frame[shoulder_y1:shoulder_y2, chest_x1:chest_x2]
            
            if shoulder_roi.size == 0:
                return False, 0.0, "empty_roi"
            
            # METHOD 1: Look for diagonal strap patterns (more sensitive)
            hsv = cv2.cvtColor(shoulder_roi, cv2.COLOR_BGR2HSV)
            
            # Dark colors for seatbelt (black, dark grey) - expanded range
            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([180, 60, 120])  # Increased range
            dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
            
            # Calculate dark coverage - lowered threshold
            dark_pixels = np.sum(dark_mask > 0)
            total_pixels = shoulder_roi.shape[0] * shoulder_roi.shape[1]
            dark_coverage = dark_pixels / total_pixels
            
            # METHOD 2: Look for diagonal lines (more sensitive)
            gray = cv2.cvtColor(shoulder_roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 20, 80)  # Lowered thresholds
            
            # Find lines with more lenient criteria
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=8,  # Lowered threshold
                                  minLineLength=15, maxLineGap=15)  # More lenient
            
            diagonal_lines = 0
            best_diagonal_line = None
            
            if lines is not None:
                for line in lines:
                    x1_l, y1_l, x2_l, y2_l = line[0]
                    if x2_l != x1_l:
                        angle = np.arctan2(y2_l - y1_l, x2_l - x1_l) * 180 / np.pi
                        # Wider diagonal strap angle (15-75 degrees)
                        if 15 < abs(angle) < 75:
                            diagonal_lines += 1
                            # Store the best (longest) diagonal line
                            line_length = ((x2_l - x1_l)**2 + (y2_l - y1_l)**2)**0.5
                            if best_diagonal_line is None or line_length > best_diagonal_line[2]:
                                best_diagonal_line = (x1_l, y1_l, x2_l, y2_l, line_length)
            
            # METHOD 3: Check for strap pattern (more sensitive)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # Smaller kernel
            morph = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in the morphological result
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            strap_like_contours = 0
            best_strap_contour = None
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:  # Lowered minimum area
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    aspect_ratio = cw / ch
                    # More lenient aspect ratio for strap
                    if aspect_ratio > 2.0:  # Lowered requirement
                        strap_like_contours += 1
                        # Store the best contour
                        if best_strap_contour is None or area > cv2.contourArea(best_strap_contour):
                            best_strap_contour = cnt
            
            # METHOD 4: Additional validation - check for strap continuity
            strap_continuity = 0.0
            if best_diagonal_line is not None and best_strap_contour is not None:
                # Check if the diagonal line aligns with the strap contour
                x1_l, y1_l, x2_l, y2_l, line_length = best_diagonal_line
                
                # Sample points along the diagonal line
                num_samples = 8  # Fewer samples for more leniency
                aligned_points = 0
                for i in range(num_samples):
                    t = i / (num_samples - 1)
                    sample_x = int(x1_l + t * (x2_l - x1_l))
                    sample_y = int(y1_l + t * (y2_l - y1_l))
                    
                    # Check if sample point is within the strap contour
                    if (0 <= sample_x < dark_mask.shape[1] and 
                        0 <= sample_y < dark_mask.shape[0]):
                        if dark_mask[sample_y, sample_x] > 0:
                            aligned_points += 1
                
                strap_continuity = aligned_points / num_samples
            
            # Calculate confidence with more lenient criteria
            color_conf = min(dark_coverage * 5, 0.8)  # Increased multiplier
            line_conf = min(diagonal_lines / 2, 0.7)  # More lenient
            strap_conf = min(strap_like_contours / 1.5, 0.6)  # More lenient
            continuity_conf = strap_continuity * 0.4
            
            # Combined confidence with more weight on basic features
            combined_conf = (color_conf * 0.4 + line_conf * 0.3 + strap_conf * 0.2 + continuity_conf * 0.1)
            
            if self.debug:
                print(f"[PPE-DEBUG] Sensitive seatbelt detection:")
                print(f"  - Dark coverage: {dark_coverage:.2f} (need >0.08)")
                print(f"  - Diagonal lines: {diagonal_lines} (need >=1)")
                print(f"  - Strap-like contours: {strap_like_contours} (need >=1)")
                print(f"  - Strap continuity: {strap_continuity:.2f}")
                print(f"  - Combined conf: {combined_conf:.2f}")
            
            # More lenient detection criteria
            if (dark_coverage > 0.08 and  # Lowered dark coverage requirement
                diagonal_lines >= 1 and   # Need at least 1 diagonal line
                strap_like_contours >= 1 and # Need at least 1 strap-like contour
                combined_conf > 0.15):     # Lowered confidence threshold
                
                reason = "seatbelt_detected"
                if diagonal_lines > 2:
                    reason = "seatbelt_multiple_diagonal_straps"
                elif dark_coverage > 0.20:
                    reason = "seatbelt_strong_dark_strap"
                elif strap_continuity > 0.5:
                    reason = "seatbelt_continuous_strap"
                
                return True, combined_conf, reason
            
            return False, combined_conf, "no_seatbelt"
            
        except Exception as e:
            if self.debug:
                print(f"[PPE-DEBUG] Seatbelt detection error: {e}")
            return False, 0.0, "error"

    def _detect_vehicle_type(self, frame, person_bbox):
        """Detect if person is on 2-wheeler or 4-wheeler by actually detecting vehicles"""
        x1, y1, x2, y2 = person_bbox
        h, w = frame.shape[:2]
        
        try:
            # Method 1: Actually detect vehicles in the frame using YOLO
            if self.model is not None:
                try:
                    # Detect vehicles (car=2, motorcycle=3, bus=5, truck=7 in COCO)
                    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
                    vehicle_results = self.model(frame, conf=0.3, iou=0.45,
                                               device=self.device, verbose=False, classes=vehicle_classes)
                    
                    detected_vehicles = []
                    for result in vehicle_results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                vx1, vy1, vx2, vy2 = box.xyxy[0].cpu().numpy()
                                vclass = int(box.cls[0].cpu().numpy())
                                vconf = float(box.conf[0].cpu().numpy())
                                detected_vehicles.append({
                                    "bbox": (int(vx1), int(vy1), int(vx2), int(vy2)),
                                    "class": vclass,
                                    "confidence": vconf
                                })
                    
                    if self.debug:
                        print(f"[PPE-DEBUG] Detected {len(detected_vehicles)} vehicles")
                    
                    # Check if person is inside or on any detected vehicle
                    person_center_x = (x1 + x2) / 2
                    person_center_y = (y1 + y2) / 2
                    
                    for vehicle in detected_vehicles:
                        vx1, vy1, vx2, vy2 = vehicle["bbox"]
                        vclass = vehicle["class"]
                        
                        # Check if person center is within vehicle bounds
                        if (vx1 <= person_center_x <= vx2 and vy1 <= person_center_y <= vy2):
                            if vclass == 3:  # motorcycle
                                return "2-wheeler"
                            elif vclass in [2, 5, 7]:  # car, bus, truck
                                return "4-wheeler"
                        
                        # Check if person is riding/on top of motorcycle (close proximity)
                        if vclass == 3:  # motorcycle
                            # Calculate distance between person and motorcycle
                            vehicle_center_x = (vx1 + vx2) / 2
                            vehicle_center_y = (vy1 + vy2) / 2
                            distance = ((person_center_x - vehicle_center_x)**2 + 
                                      (person_center_y - vehicle_center_y)**2)**0.5
                            
                            # If person is very close to motorcycle, likely riding it
                            if distance < max(x2-x1, y2-y1) * 1.5:  # Within 1.5x person size
                                return "2-wheeler"
                
                except Exception as e:
                    if self.debug:
                        print(f"[PPE-DEBUG] Vehicle detection failed: {e}")
            
            # Method 2: Fallback - Check for clear vehicle indicators in context
            # Only use if no vehicles were detected above
            
            # Look for vehicle-specific features around the person
            expand = 50
            context_x1 = max(0, x1 - expand)
            context_y1 = max(0, y1 - expand)
            context_x2 = min(w, x2 + expand)
            context_y2 = min(h, y2 + expand)
            
            context_roi = frame[context_y1:context_y2, context_x1:context_x2]
            
            if context_roi.size > 0:
                # Look for vehicle interior features (windows, steering wheel, etc.)
                gray_context = cv2.cvtColor(context_roi, cv2.COLOR_BGR2GRAY)
                
                # Check for rectangular shapes (windows, doors)
                edges = cv2.Canny(gray_context, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                rectangular_shapes = 0
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 500:  # Minimum area for meaningful shapes
                        x, y, cw, ch = cv2.boundingRect(cnt)
                        aspect_ratio = cw / ch
                        # Look for rectangular shapes (windows, doors)
                        if 0.5 < aspect_ratio < 3.0:
                            rectangular_shapes += 1
                
                # If we see multiple rectangular shapes, likely vehicle interior
                if rectangular_shapes >= 2:
                    # Additional check for vehicle-specific patterns
                    # Look for horizontal lines (car hood, roof)
                    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15,
                                          minLineLength=30, maxLineGap=10)
                    
                    if lines is not None:
                        horizontal_lines = 0
                        for line in lines:
                            x1_l, y1_l, x2_l, y2_l = line[0]
                            angle = np.arctan2(y2_l - y1_l, x2_l - x1_l) * 180 / np.pi
                            if abs(angle) < 15:  # Nearly horizontal
                                horizontal_lines += 1
                        
                        # Multiple horizontal lines suggest vehicle structure
                        if horizontal_lines >= 3:
                            return "4-wheeler"
            
            # Method 3: Position-based fallback (only if no clear vehicle evidence)
            # This is the least reliable method
            person_height = y2 - y1
            person_center_y = (y1 + y2) / 2
            image_height = h
            person_width = x2 - x1
            aspect_ratio = person_width / person_height
            
            # Very specific conditions for vehicle detection
            # Person must be in lower portion AND have sitting posture AND have structured background
            if (person_center_y > image_height * 0.7 and  # Very low in frame
                aspect_ratio > 0.45 and  # Wide (sitting)
                person_height < image_height * 0.4):  # Not too tall (sitting)
                
                # One final check - look for wheel-like shapes
                wheel_region = frame[y2:int(y2 + person_height*0.3), x1:x2]
                if wheel_region.size > 0:
                    gray_wheel = cv2.cvtColor(wheel_region, cv2.COLOR_BGR2GRAY)
                    circles = cv2.HoughCircles(gray_wheel, cv2.HOUGH_GRADIENT, 1, 20,
                                             param1=50, param2=30, minRadius=10, maxRadius=30)
                    if circles is not None:
                        return "4-wheeler"
            
            # Default to "unknown" if no clear vehicle evidence
            return "unknown"
            
        except Exception as e:
            if self.debug:
                print(f"[PPE-DEBUG] Vehicle detection error: {e}")
            return "unknown"

    def detect_persons_with_fallback(self, frame):
        persons = []
        model_worked = False

        # Try primary model
        if self.model is not None:
            try:
                results = self.model(frame, conf=self.person_threshold, iou=0.45,
                                   device=self.device, verbose=False, classes=[0])
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            persons.append({"bbox": (int(x1), int(y1), int(x2), int(y2)), "confidence": conf})
                model_worked = True
                if self.debug:
                    print(f"[PPE-DEBUG] Primary model detected {len(persons)} persons")
            except Exception as e:
                print(f"[PPE-WARNING] Primary model failed: {e}")
                if self.auto_recovery:
                    print("[PPE] Attempting model recovery...")
                    if self._reload_model():
                        return self.detect_persons_with_fallback(frame)

        # Fallback: Use general object detection if primary failed
        if not model_worked or len(persons) == 0:
            print("[PPE] Using fallback person detection...")
            try:
                # Use OpenCV HOG detector as ultimate fallback
                persons = self._detect_persons_hog(frame)
                if self.debug:
                    print(f"[PPE-DEBUG] HOG fallback detected {len(persons)} persons")
            except Exception as e:
                print(f"[PPE-ERROR] Fallback detection also failed: {e}")

        return persons, model_worked

    def _detect_persons_hog(self, frame):
        """Ultimate fallback using OpenCV HOG detector"""
        persons = []
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            # Detect people
            boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8),
                                                  padding=(4, 4), scale=1.05)

            for i, (x, y, w, h) in enumerate(boxes):
                conf = float(weights[i]) if i < len(weights) else 0.3
                if conf > 0.1:  # Very low threshold for fallback
                    persons.append({
                        "bbox": (int(x), int(y), int(x+w), int(y+h)),
                        "confidence": conf
                    })
        except Exception as e:
            if self.debug:
                print(f"[PPE-DEBUG] HOG detection error: {e}")

        return persons

    def _create_minimum_detection(self, frame):
        """Create minimum working detection when all else fails"""
        persons = []
        h, w = frame.shape[:2]

        # Look for large vertical objects (potential persons)
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            edges = cv2.Canny(blurred, 50, 150)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                # Look for tall objects (person-like)
                if ch > h * 0.3 and cw > w * 0.1 and ch > cw * 1.5:
                    persons.append({
                        "bbox": (int(x), int(y), int(x+cw), int(y+ch)),
                        "confidence": 0.2
                    })
        except:
            pass

        return persons

    def _detect_vest(self, frame, person_bbox):
        """Simple vest detection using high-vis colors in torso region"""
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1
        torso_y1 = y1 + int(height * 0.30)
        torso_y2 = y1 + int(height * 0.70)
        h, w = frame.shape[:2]
        torso_y1 = max(0, min(h, torso_y1))
        torso_y2 = max(0, min(h, torso_y2))

        if torso_y2 <= torso_y1 or x2 <= x1:
            return False, 0.0

        torso_roi = frame[torso_y1:torso_y2, x1:x2]
        if torso_roi.size == 0:
            return False, 0.0

        try:
            hsv = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2HSV)
            yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
            orange = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([20, 255, 255]))
            green = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([75, 255, 255]))
            combined = cv2.bitwise_or(yellow, orange)
            combined = cv2.bitwise_or(combined, green)
            coverage = np.sum(combined > 0) / (torso_roi.shape[0] * torso_roi.shape[1])
            conf = min(coverage * 2.5, 1.0)
            return coverage > 0.08, conf
        except:
            return False, 0.0

    def _create_emergency_result(self, frame, timestamp, error_msg):
        """Emergency result when everything fails"""
        h, w = frame.shape[:2]

        # Create at least one person detection in center
        center_bbox = (int(w*0.3), int(h*0.2), int(w*0.7), int(h*0.8))

        emergency_person = PersonPPE(
            person_id="P1",
            bbox=center_bbox,
            vehicle_type="unknown",
            head_bbox=None,
            helmet=PPEItem("helmet", False, 0.0, None, "emergency"),
            vest=PPEItem("vest", False, 0.0),
            seatbelt=PPEItem("seatbelt", False, 0.0, None, "emergency"),
            status="violation",
            confidence=0.0,
            debug_info={"emergency": True, "error": error_msg}
        )

        return PPEResult(
            total_persons=1,
            helmet_detected=0,
            no_helmet=1,
            seatbelt_detected=0,
            no_seatbelt=1,
            persons=[emergency_person],
            timestamp=timestamp,
            processing_time=0.0,
            debug_mode=self.debug,
            model_loaded=False,
            fallback_used=True,
            error_message=f"Emergency mode: {error_msg}"
        )

    def detect(self, frame, debug=None):
        """
        Main detection with guaranteed results and enhanced debug mode
        NEVER returns empty - always provides minimum detection
        """
        if debug is not None:
            self.debug = debug

        start = time.time()
        timestamp = datetime.now().isoformat()
        fallback_used = False
        error_msg = ""

        try:
            if self.debug:
                print(f"\n{'='*60}")
                print(f"[PPE] 🎯 AI SAFETY DETECTION SYSTEM STARTED")
                print(f"[PPE] Frame: {frame.shape}")
                print(f"[PPE] Debug Mode: ENABLED")
                print(f"{'='*60}")

            # Ensure model is loaded
            model_loaded = self.model is not None or self._ensure_model_loaded()

            # Detect persons with fallback
            persons, model_worked = self.detect_persons_with_fallback(frame)

            if not model_worked:
                fallback_used = True
                print("[PPE] Using fallback detection mode")

            # If still no persons, create minimum working detection
            if len(persons) == 0:
                print("[PPE-WARNING] No persons detected, checking for any human-like shapes...")
                persons = self._create_minimum_detection(frame)
                if len(persons) > 0:
                    fallback_used = True

            person_results = []
            helmet_count = 0
            no_helmet_count = 0
            seatbelt_count = 0
            no_seatbelt_count = 0

            if self.debug:
                print(f"\n[PPE] 👥 PROCESSING {len(persons)} PERSONS...")
                print(f"{'-'*60}")

            for idx, person in enumerate(persons):
                person_id = f"P{idx + 1}"
                person_bbox = person["bbox"]
                
                # STEP 1: Detect vehicle type FIRST (before PPE checks)
                vehicle_type = self._detect_vehicle_type(frame, person_bbox)
                
                # STEP 2: Check PPE based on vehicle type
                # Rule: For 4-wheelers, ONLY check seatbelt (skip helmet to avoid false positives)
                # Rule: For 2-wheelers/workers, check helmet first (highest priority)
                # Rule: If helmet detected, NEVER detect/report seatbelt
                # Rule: Only detect seatbelt for 4-wheelers if NO helmet present
                
                helmet_present, helmet_conf, helmet_method = False, 0.0, "not_checked"
                seatbelt_present, seatbelt_conf, seatbelt_method = False, 0.0, "not_applicable"
                
                if vehicle_type == "4-wheeler":
                    # 4-WHEELER: Skip helmet, check seatbelt only
                    seatbelt_present, seatbelt_conf, seatbelt_method = self._detect_seatbelt(frame, person_bbox)
                else:
                    # 2-WHEELER or UNKNOWN: Check helmet first
                    head_bbox = self.get_head_region(person_bbox)
                    threshold = self.fallback_threshold if fallback_used else self.helmet_threshold
                    helmet_present, helmet_conf, helmet_method = self.detect_helmet_in_head(frame, head_bbox, threshold)
                    
                    if helmet_present:
                        # STRICT RULE: Helmet detected = ignore seatbelt
                        seatbelt_present, seatbelt_conf, seatbelt_method = False, 0.0, "not_applicable"
                    else:
                        # No helmet, check seatbelt for any vehicle type
                        seatbelt_present, seatbelt_conf, seatbelt_method = self._detect_seatbelt(frame, person_bbox)
                
                # STEP 4: Update counts based on strict output rules
                if helmet_present:
                    helmet_count += 1
                    # Seatbelt is NEVER counted when helmet is present
                elif seatbelt_present:
                    seatbelt_count += 1
                else:
                    # No PPE detected
                    if vehicle_type == "4-wheeler":
                        no_seatbelt_count += 1
                    else:
                        no_helmet_count += 1
                
                vest_present, vest_conf = self._detect_vest(frame, person_bbox)

                # STEP 5: Determine status and label based on strict priority
                if helmet_present:
                    # STRICT: Only "Helmet Detected" label
                    status = "compliant"
                    compliance_reason = "helmet_detected"
                    output_label = "Helmet Detected"
                elif seatbelt_present:
                    # Only for 4-wheelers without helmet
                    status = "compliant"
                    compliance_reason = "seatbelt_detected"
                    output_label = "Seatbelt Detected"
                else:
                    status = "violation"
                    compliance_reason = "no_ppe_detected"
                    output_label = "No PPE Detected"

                debug_info = {}
                if self.debug:
                    debug_info = {
                        "vehicle_type": vehicle_type,
                        "helmet_method": helmet_method,
                        "helmet_confidence": f"{helmet_conf:.3f}",
                        "seatbelt_method": seatbelt_method,
                        "seatbelt_confidence": f"{seatbelt_conf:.3f}",
                        "compliance_reason": compliance_reason,
                        "fallback": fallback_used
                    }

                person_ppe = PersonPPE(
                    person_id=person_id,
                    bbox=person_bbox,
                    vehicle_type=vehicle_type,
                    head_bbox=head_bbox if helmet_present else None,
                    helmet=PPEItem("helmet", helmet_present, helmet_conf, head_bbox if helmet_present else None, helmet_method),
                    vest=PPEItem("vest", vest_present, vest_conf),
                    seatbelt=PPEItem("seatbelt", seatbelt_present, seatbelt_conf, None, seatbelt_method),
                    status=status,
                    confidence=person["confidence"],
                    debug_info=debug_info
                )
                person_results.append(person_ppe)

                if self.debug:
                    safety_status = "🟩 COMPLIANT" if status == "compliant" else "🟥 VIOLATION"
                    print(f"\n👤 {person_id} [{vehicle_type.upper()}]: {safety_status}")
                    print(f"  🎯 Output Label: {output_label}")
                    
                    # Show detection details based on priority
                    if helmet_present:
                        print(f"  🪖 Helmet: ✅ DETECTED (conf: {helmet_conf:.3f})")
                        print(f"  🚗 Seatbelt: Not checked (helmet has priority)")
                    elif seatbelt_present:
                        print(f"  🪖 Helmet: Not detected")
                        print(f"  🚗 Seatbelt: ✅ DETECTED (conf: {seatbelt_conf:.3f})")
                    else:
                        print(f"  🪖 Helmet: ❌ Not detected (conf: {helmet_conf:.3f})")
                        if vehicle_type == "4-wheeler":
                            print(f"  🚗 Seatbelt: ❌ Not detected (conf: {seatbelt_conf:.3f})")
                        else:
                            print(f"  🚗 Seatbelt: Not applicable (2-wheeler/worker)")
                    
                    print(f"  🔍 Reason: {compliance_reason}")

            proc_time = time.time() - start

            if self.debug:
                print(f"\n{'='*60}")
                print(f"[PPE] 📊 DETECTION SUMMARY")
                print(f"[PPE] 👥 Total Persons: {len(persons)}")
                print(f"[PPE] 🎯 Priority-based Output:")
                print(f"     - Helmet Detected: {helmet_count} ✅")
                print(f"     - Seatbelt Detected: {seatbelt_count} ✅")
                print(f"     - No PPE Detected: {no_helmet_count + no_seatbelt_count} ❌")
                print(f"[PPE] 🔧 System Status:")
                print(f"     - Model Loaded: {'✅ YES' if model_loaded else '❌ NO'}")
                print(f"     - Fallback Used: {'✅ YES' if fallback_used else '❌ NO'}")
                print(f"     - Processing Time: {proc_time:.3f}s")
                print(f"{'='*60}\n")

            return PPEResult(
                total_persons=len(persons),
                helmet_detected=helmet_count,
                no_helmet=no_helmet_count,
                seatbelt_detected=seatbelt_count,
                no_seatbelt=no_seatbelt_count,
                persons=person_results,
                timestamp=timestamp,
                processing_time=proc_time,
                debug_mode=self.debug,
                model_loaded=model_loaded,
                fallback_used=fallback_used,
                error_message=error_msg
            )

        except Exception as e:
            error_msg = str(e)
            print(f"[PPE-CRITICAL] Detection failed: {e}")
            print("[PPE] Creating emergency fallback result...")

            # Emergency fallback - always return something
            return self._create_emergency_result(frame, timestamp, error_msg)

    def visualize(self, frame, result, show_labels=True, show_head_region=False):
        """Draw detection results - GREEN for compliant, RED for violation"""
        img = frame.copy()

        for person in result.persons:
            x1, y1, x2, y2 = person.bbox

            # SIMPLE COLOR LOGIC: GREEN = any safety equipment present, RED = missing
            # For 2-wheeler: check helmet, for 4-wheeler: check seatbelt, for unknown: check either
            is_compliant = False
            if person.vehicle_type == "2-wheeler":
                is_compliant = person.helmet.present
            elif person.vehicle_type == "4-wheeler":
                is_compliant = person.seatbelt.present
            else:
                # Unknown vehicle type (workers without vehicles): compliant only if helmet present
                is_compliant = person.helmet.present
            
            if is_compliant:
                color = self.COLORS["compliant"]  # GREEN
                status_emoji = "🟩"
            else:
                color = self.COLORS["non_compliant"]  # RED
                status_emoji = "🟥"

            # Create status text showing ONLY ONE label per strict rules
            if person.helmet.present:
                # STRICT RULE: Helmet has priority - only show helmet
                safety_text = "Helmet Detected"
            elif person.seatbelt.present:
                # Only for 4-wheelers without helmet
                safety_text = "Seatbelt Detected"
            else:
                safety_text = "No PPE Detected"

            # Use fallback color if in fallback mode
            if result.fallback_used:
                color = self.COLORS["fallback"]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            if show_head_region or result.debug_mode:
                if person.head_bbox:
                    hx1, hy1, hx2, hy2 = person.head_bbox
                    cv2.rectangle(img, (hx1, hy1), (hx2, hy2), self.COLORS["head_region"], 2)

            if show_labels:
                # Label with person ID, vehicle type, and safety status
                label = f"{person.person_id} [{person.vehicle_type.upper()}] {safety_text}"
                if result.fallback_used:
                    label += " (FB)"

                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = x1
                text_y = y1 - 10 if y1 > 30 else y1 + text_size[1] + 10

                # Background rectangle for text
                cv2.rectangle(img, (text_x, text_y - text_size[1] - 5),
                            (text_x + text_size[0], text_y + 5), color, -1)
                cv2.putText(img, label, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Summary with all counts
        summary = f"Persons: {result.total_persons} | Helmets: {result.helmet_detected} | Seatbelts: {result.seatbelt_detected} | Model: {'OK' if result.model_loaded else 'FB'}"
        cv2.rectangle(img, (10, 5), (700, 35), (0, 0, 0), -1)
        cv2.putText(img, summary, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if result.error_message:
            cv2.putText(img, f"Error: {result.error_message[:50]}", (15, 55),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return img

    def to_dict(self, result):
        """Convert to JSON-serializable dict - ALWAYS include both helmet and seatbelt"""
        persons_list = []
        for p in result.persons:
            # ALWAYS include both fields - NEVER skip
            person_data = {
                "id": p.person_id,
                "helmet": p.helmet.present,  # ALWAYS included
                "seatbelt": p.seatbelt.present,  # ALWAYS included
                "vehicle_type": p.vehicle_type,
                "status": p.status,
                "confidence": p.confidence,
                "detection_method": {
                    "helmet": p.helmet.detection_method,
                    "seatbelt": p.seatbelt.detection_method
                },
                "confidence_scores": {
                    "helmet": round(p.helmet.confidence, 3),
                    "seatbelt": round(p.seatbelt.confidence, 3)
                }
            }
            
            # Apply strict priority rules for output label
            # Rule: Only ONE label - Helmet has priority over Seatbelt
            if p.helmet.present:
                person_data["label"] = "Helmet Detected"
                person_data["detectedPPE"] = "helmet"
            elif p.seatbelt.present:
                person_data["label"] = "Seatbelt Detected"
                person_data["detectedPPE"] = "seatbelt"
            else:
                person_data["label"] = "No PPE Detected"
                person_data["detectedPPE"] = "none"
            
            # Include detection details for reference (but not as primary labels)
            person_data["helmetDetected"] = p.helmet.present
            person_data["seatbeltDetected"] = p.seatbelt.present
            person_data["vehicleType"] = p.vehicle_type
            
            persons_list.append(person_data)
        
        return {
            "totalPersons": result.total_persons,
            "helmetDetected": result.helmet_detected,
            "noHelmet": result.no_helmet,
            "seatbeltDetected": result.seatbelt_detected,
            "noSeatbelt": result.no_seatbelt,
            "persons": persons_list,
            "debug": {
                "modelLoaded": result.model_loaded,
                "fallbackUsed": result.fallback_used,
                "processingTime": result.processing_time,
                "timestamp": result.timestamp
            }
        }

    def get_summary_text(self, result):
        """Generate markdown summary with strict priority-based output"""
        lines = [
            "PPE Detection Results",
            "",
            "System Status:",
            f"* Model Loaded: {'Yes' if result.model_loaded else 'No'}",
            f"* Fallback Used: {'Yes' if result.fallback_used else 'No'}",
            "",
            "Detection Summary (Priority-based):",
            f"* Total Persons: {result.total_persons}",
            f"* Helmet Detected: {result.helmet_detected} (Priority: Highest)",
            f"* Seatbelt Detected: {result.seatbelt_detected} (Only if no helmet in 4-wheeler)",
            f"* No PPE Detected: {result.no_helmet + result.no_seatbelt}",
            "",
            "Person Details:",
        ]

        for person in result.persons:
            # Determine single output label based on strict priority
            if person.helmet.present:
                output_label = "Helmet Detected"
                emoji = "🟩"
                details = f"Helmet: ✅ Present (conf: {person.helmet.confidence:.2f})"
            elif person.seatbelt.present:
                output_label = "Seatbelt Detected"
                emoji = "🟩"
                details = f"Seatbelt: ✅ Present (conf: {person.seatbelt.confidence:.2f})"
            else:
                output_label = "No PPE Detected"
                emoji = "🟥"
                if person.vehicle_type == "4-wheeler":
                    details = f"No Seatbelt detected (conf: {person.seatbelt.confidence:.2f})"
                else:
                    details = f"No Helmet detected (conf: {person.helmet.confidence:.2f})"
            
            lines.append(f"Person {person.person_id} {emoji}")
            lines.append(f"  - Output: {output_label}")
            lines.append(f"  - Vehicle Type: {person.vehicle_type}")
            lines.append(f"  - {details}")
            
            # Add debug information if available
            if person.debug_info and result.debug_mode:
                lines.append(f"  - Debug: {person.debug_info}")

        if result.error_message:
            lines.append(f"\n⚠️ Warning: {result.error_message}")

        return "\n".join(lines)


# Global instance with auto-recovery
_ppe_detector = None
_lock = threading.Lock()


def get_ppe_detector(model_path="yolov8n.pt", debug=False, auto_recovery=True):
    """Get or create PPE detector with auto-recovery"""
    global _ppe_detector
    with _lock:
        if _ppe_detector is None:
            _ppe_detector = PPEDetector(model_path, debug=debug, auto_recovery=auto_recovery)
        return _ppe_detector


def reset_ppe_detector():
    """Reset detector"""
    global _ppe_detector
    with _lock:
        _ppe_detector = None
    print("[PPE] Detector reset")


if __name__ == "__main__":
    print("[PPE] Robust Helmet Detection Module Ready")
    print("Features:")
    print("  - Auto-recovery on model failure")
    print("  - OpenCV HOG fallback detection")
    print("  - Emergency minimum detection")
    print("  - Never returns empty results")
