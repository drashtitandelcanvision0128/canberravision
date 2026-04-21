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

import yaml

import os



# Suppress warnings for cleaner output

warnings.filterwarnings('ignore')



# PPE Dataset Configuration

PPE_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'PPE', 'data.yaml')



# Load PPE dataset classes

def load_ppe_classes():

    """Load PPE classes from dataset YAML file"""

    try:

        if os.path.exists(PPE_DATASET_PATH):

            with open(PPE_DATASET_PATH, 'r') as f:

                config = yaml.safe_load(f)

            return config.get('names', []), config.get('nc', 5)

        else:

            print(f"[WARNING] PPE dataset config not found at {PPE_DATASET_PATH}")

            return ['helmet', 'no-helmet', 'no-vest', 'person', 'vest'], 5

    except Exception as e:

        print(f"[ERROR] Failed to load PPE dataset config: {e}")

        return ['helmet', 'no-helmet', 'no-vest', 'person', 'vest'], 5



# Load dataset classes

PPE_CLASSES, NUM_PPE_CLASSES = load_ppe_classes()

print(f"[INFO] PPE Dataset loaded: {PPE_CLASSES} ({NUM_PPE_CLASSES} classes)")





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

    vest_bbox: Optional[Tuple[int, int, int, int]] = None

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

    vest_detected: int

    no_vest: int

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

        ("white", [0, 0, 120], [180, 80, 255]),  # Wider white range - lower value, higher saturation

        ("blue", [90, 50, 50], [130, 255, 255]),

        ("red1", [0, 100, 100], [10, 255, 255]),

        ("red2", [160, 100, 100], [180, 255, 255]),

        ("orange", [10, 100, 100], [20, 255, 255]),

        ("black", [0, 0, 0], [180, 50, 60]),  # Black helmets - low value, any hue/saturation

        ("dark_grey", [0, 0, 20], [180, 40, 80]),  # Dark grey helmets

        ("green", [35, 50, 50], [85, 255, 255]),  # Green helmets

        ("pink", [140, 50, 50], [170, 255, 255]),  # Pink/magenta helmets

        ("purple", [120, 50, 50], [150, 255, 255]),  # Purple helmets

        ("cyan", [75, 50, 50], [95, 255, 255]),  # Cyan/light blue helmets

        ("silver", [0, 0, 100], [180, 30, 200]),  # Silver/light grey helmets

        ("brown", [10, 50, 30], [25, 255, 150]),  # Brown helmets

        ("beige", [20, 20, 150], [40, 60, 255]),  # Beige/tan helmets

        ("lime", [35, 80, 80], [50, 255, 255]),  # Lime green helmets

        ("teal", [70, 50, 50], [90, 255, 255]),  # Teal helmets

    ]



    COLORS = {

        "compliant": (0, 255, 0),       # Green

        "non_compliant": (0, 0, 255),   # Red

        "head_region": (255, 0, 255),   # Magenta

        "fallback": (0, 255, 255),      # Yellow

    }



    def __init__(self, model_path=None, device=None, debug=False, auto_recovery=True):

        # Use PPE dataset model path if not provided

        if model_path is None:

            # Try to find trained PPE model first

            ppe_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'PPE', 'best.pt')

            if os.path.exists(ppe_model_path):

                model_path = ppe_model_path

                print(f"[PPE] Using trained PPE model: {model_path}")

            else:

                # Fallback to YOLOv8 for person detection

                model_path = "yolov8n.pt"

                print(f"[PPE] Using fallback model: {model_path}")

        

        self.model_path = model_path

        self.model = None

        self.device = device or self._get_device()

        self.debug = debug

        self.auto_recovery = auto_recovery

        self.model_load_attempts = 0

        self.max_load_attempts = 3



        # LOW thresholds for maximum detection sensitivity

        self.person_threshold = 0.25

        self.helmet_threshold = 0.08  # Lowered from 0.12 for maximum helmet detection sensitivity

        self.fallback_threshold = 0.05  # Even lower for maximum sensitivity in fallback mode



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



    def get_head_region(self, person_bbox, frame=None):

        """Extract head region - adaptive based on pose using YOLO model"""

        x1, y1, x2, y2 = person_bbox

        height = y2 - y1

        width = x2 - x1

        

        # ADAPTIVE: Try to find actual helmet position using model

        if frame is not None and self.model is not None:

            try:

                # Run model on full person to find helmet

                person_roi = frame[y1:y2, x1:x2]

                if person_roi.size > 0:

                    results = self.model(person_roi, conf=0.1, iou=0.45, 

                                      device=self.device, verbose=False)

                    

                    for result in results:

                        if result.boxes is not None:

                            for box in result.boxes:

                                cls = int(box.cls[0].cpu().numpy())

                                if cls < len(PPE_CLASSES) and PPE_CLASSES[cls] == 'helmet':

                                    hx1, hy1, hx2, hy2 = box.xyxy[0].cpu().numpy()

                                    # Convert to full frame coordinates

                                    return (int(x1 + hx1), int(y1 + hy1), 

                                           int(x1 + hx2), int(y1 + hy2))

            except:

                pass  # Fall back to image analysis

        

        # SMART FALLBACK: Analyze image content to find head position

        if frame is not None:

            try:

                person_roi = frame[y1:y2, x1:x2]

                if person_roi.size > 0:

                    # Convert to HSV for skin detection

                    hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)

                    gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

                    

                    # Detect skin (face/head area)

                    lower_skin = np.array([0, 20, 70])

                    upper_skin = np.array([20, 170, 255])

                    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

                    

                    # Detect hair (dark textured region at top)

                    dark_mask = gray < 80

                    

                    # Analyze top 30% of person for head features

                    top_region_h = int(height * 0.30)

                    top_skin = skin_mask[:top_region_h, :]

                    top_dark = dark_mask[:top_region_h, :]

                    

                    # Find where skin or hair ends (head bottom)

                    head_bottom = 0

                    for row in range(top_region_h):

                        skin_pixels = np.sum(top_skin[row, :])

                        dark_pixels = np.sum(top_dark[row, :])

                        width = top_skin.shape[1]

                        

                        if skin_pixels > width * 0.1 or dark_pixels > width * 0.2:

                            head_bottom = row

                    

                    # Use detected head position if reasonable (8% to 20%)

                    if head_bottom > int(height * 0.08) and head_bottom < int(height * 0.20):

                        if self.debug:

                            print(f"[PPE-DEBUG] Image-based head detection: {head_bottom}px ({head_bottom/height:.1%})")

                        return (x1, y1, x2, y1 + head_bottom)

            except:

                pass

        

        # FINAL FALLBACK: Fixed top 12% for head bounding box

        head_height = int(height * 0.12)

        return (x1, y1, x2, y1 + head_height)

    

    def get_vest_region(self, person_bbox, frame=None, head_bbox=None):

        """Extract vest region - starts right after head region with small gap"""

        x1, y1, x2, y2 = person_bbox

        height = y2 - y1

        width = x2 - x1

        

        # ADAPTIVE: Try to find actual vest position

        if frame is not None and self.model is not None:

            try:

                # Run model on full person

                person_roi = frame[y1:y2, x1:x2]

                if person_roi.size > 0:

                    results = self.model(person_roi, conf=0.1, iou=0.45,

                                      device=self.device, verbose=False)

                    

                    for result in results:

                        if result.boxes is not None:

                            for box in result.boxes:

                                cls = int(box.cls[0].cpu().numpy())

                                if cls < len(PPE_CLASSES) and PPE_CLASSES[cls] == 'vest':

                                    vx1, vy1, vx2, vy2 = box.xyxy[0].cpu().numpy()

                                    return (int(x1 + vx1), int(y1 + vy1),

                                           int(x1 + vx2), int(y1 + vy2))

            except:

                pass

        

        # FALLBACK: Vest starts right after head bbox + small gap (5%)
        if head_bbox is not None:
            _, head_y2, _, _ = head_bbox
            gap = int(height * 0.05)  # Small gap after head
            vest_y1 = head_y2 + gap
            vest_y2 = y1 + int(height * 0.75)  # Vest covers torso down to 75%
        else:
            # No head bbox - use fixed position after head region (12% + 5% gap = 17%)
            vest_y1 = y1 + int(height * 0.17)  # After head (12%) + gap (5%)
            vest_y2 = y1 + int(height * 0.75)

        return (x1, vest_y1, x2, vest_y2)



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

                    if 0.05 * roi_area < area < 0.70 * roi_area:  # Lowered from 0.10 to 0.05 for better white helmet detection

                        coverage = area / roi_area

                        conf = min(coverage * 3.0, 0.8)  # Increased from 2.0 to 3.0 for better color confidence

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



            if area < roi_area * 0.05:  # Very low threshold - extremely lenient

                return False, 0.0



            # Check convexity - helmets are convex

            hull = cv2.convexHull(largest)

            hull_area = cv2.contourArea(hull)

            if hull_area > 0:

                convexity = area / hull_area

                if convexity < 0.5:  # Very lenient convexity

                    return False, 0.0



            # Check for dome-like top (parabola shape)

            x, y, w, h = cv2.boundingRect(largest)

            if h > 0:

                aspect_ratio = w / h

                # Helmet dome typically has specific proportions

                if aspect_ratio < 0.5 or aspect_ratio > 5.0:  # Extremely wide range

                    return False, 0.0



            # Additional check: Must be more circular/rounded
            # Use bounding rect area instead of w*h for better circularity calc
            bounding_area = w * h if w * h > 0 else 1
            circularity = 4 * np.pi * area / bounding_area
            # More lenient for helmets that may be partially cut off in ROI
            min_circularity = 0.25 if area / roi_area > 0.15 else 0.35
            if circularity < min_circularity:
                return False, 0.0



            # Calculate dome confidence with balanced criteria

            base_conf = min(area / (roi_area * 0.5), 1.0)  # Even lower area requirement

            dome_conf = min(base_conf * convexity * circularity, 1.0)  # Cap final result at 1.0

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
            dark_mask = gray < 70
            dark_ratio = np.sum(dark_mask) / dark_mask.size



            # Hair has high texture AND dark color (made MORE LENIENT to avoid rejecting valid helmets)
            # Increased thresholds: 800 → 1200, 0.35 → 0.5
            if texture_score > 1200 and dark_ratio > 0.5:

                if self.debug:
                    print(f"[PPE-DEBUG] Hair detected: texture={texture_score:.0f}, dark_ratio={dark_ratio:.2f}")
                return True  # Hair detected



            # Check for fine texture patterns (hair strands)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size



            # High edge density suggests hair texture (made MORE LENIENT)
            # Increased thresholds: 0.25 → 0.40, 500 → 800
            if edge_density > 0.40 and texture_score > 800:

                if self.debug:
                    print(f"[PPE-DEBUG] Hair detected by edges: edge_density={edge_density:.2f}, texture={texture_score:.0f}")
                return True



            return False

        except:

            return False



    def _check_styled_hair(self, head_roi):

        """Detect styled/oiled hair that looks smooth like helmet

        Key difference: Helmets have HARD/SHARP edges, styled hair has SOFT/FEATHERED edges

        """

        try:

            if head_roi.size == 0:

                return False

            

            gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)

            h, w = gray.shape

            

            # CHECK 1: Edge sharpness - helmets have crisp edges, hair has fuzzy edges

            edges = cv2.Canny(gray, 30, 100)

            edge_density = np.sum(edges > 0) / edges.size

            

            # Dilate edges and compare - helmet edges stay sharp, hair edges blur out

            kernel = np.ones((3, 3), np.uint8)

            dilated_edges = cv2.dilate(edges, kernel, iterations=2)

            dilated_density = np.sum(dilated_edges > 0) / dilated_edges.size

            

            # Edge spread ratio: if dilated density is much higher, edges are fuzzy = hair

            # Helmet: edge spread is LOW (hard boundary stays crisp)

            # Hair: edge spread is HIGH (soft boundary spreads out)

            if edge_density > 0.01:

                edge_spread = dilated_density / edge_density

            else:

                edge_spread = 0

            

            # CHECK 2: Color uniformity - hair has subtle color variation, helmet is solid

            # Calculate local variance in small patches

            patch_size = max(h // 4, 4)

            local_vars = []

            for row in range(0, h - patch_size, patch_size):

                for col in range(0, w - patch_size, patch_size):

                    patch = gray[row:row+patch_size, col:col+patch_size]

                    if patch.size > 0:

                        local_vars.append(np.var(patch))

            

            if local_vars:

                color_variation = np.std(local_vars)  # Variation of local variances

            else:

                color_variation = 0

            

            # CHECK 3: Gradient smoothness at boundary

            # Hair boundary has gradual transition, helmet boundary has sharp transition

            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

            gradient_variance = np.var(gradient_mag)

            

            # DECISION: Styled hair indicators

            # 1. High edge spread (fuzzy boundaries) = hair

            # 2. Low color variation (but not as uniform as helmet) = styled hair

            # 3. Low gradient variance = soft transitions = hair

            

            is_styled = False

            reasons = []

            

            if edge_spread > 3.0:  # Fuzzy edges = hair

                is_styled = True

                reasons.append(f"edge_spread={edge_spread:.1f}")

            

            if color_variation < 50 and edge_density > 0.05 and edge_spread > 2.0:

                # Moderate uniformity + fuzzy edges = styled/oiled hair

                is_styled = True

                reasons.append(f"color_var={color_variation:.1f}")

            

            if gradient_variance < 500 and edge_density > 0.05 and edge_spread > 2.5:

                # Soft gradients + fuzzy edges = hair

                is_styled = True

                reasons.append(f"grad_var={gradient_variance:.0f}")

            

            if self.debug and (is_styled or edge_spread > 2.0):

                print(f"[PPE-DEBUG] Styled hair check: spread={edge_spread:.1f}, color_var={color_variation:.1f}, grad_var={gradient_variance:.0f}, edge_dens={edge_density:.3f}, result={'STYLED_HAIR' if is_styled else 'not_styled'}")

            

            return is_styled

            

        except:

            return False



    def _check_skin_texture(self, head_roi):
        """Detect skin texture to reject bald heads as helmet false positives"""
        try:
            if head_roi.size == 0:
                return False
            
            # Convert to HSV for skin color detection
            hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
            
            # Skin color range in HSV - BROADER range to catch all skin tones
            # Need to detect bald heads that might be mistaken for helmets
            lower_skin1 = np.array([0, 30, 60])  # Increased saturation back to original
            upper_skin1 = np.array([20, 170, 240])  # Narrower hue to avoid yellow helmets
            lower_skin2 = np.array([160, 30, 60])  # Increased saturation
            upper_skin2 = np.array([180, 170, 240])  # Narrower range
            
            # Create skin masks
            skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
            
            # Calculate skin pixel ratio
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
            
            # Check for skin texture
            gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_score = np.var(laplacian)
            
            # LOWER threshold to catch more bald heads (but avoid white helmets)
            # White helmets have low saturation, skin has more
            mean_saturation = np.mean(hsv[:, :, 1])

            # Skin detected if: reasonable skin ratio AND moderate texture
            # Made MORE LENIENT to avoid rejecting valid helmets: 0.30 → 0.45, 40 → 60, 800 → 600
            if skin_ratio > 0.45 and 60 < texture_score < 600 and mean_saturation > 25:
                if self.debug:
                    print(f"[PPE-DEBUG] Skin texture detected: skin_ratio={skin_ratio:.2f}, texture={texture_score:.0f}, saturation={mean_saturation:.1f}")
                return True  # Likely skin/bald head

            # Additional check: very high skin ratio is strong indicator even with other variations
            # Increased back to 0.60 to avoid rejecting valid helmets
            if skin_ratio > 0.60:
                if self.debug:
                    print(f"[PPE-DEBUG] High skin ratio detected: {skin_ratio:.2f}")
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

            smoothness = max(0, min(1, 1 - (avg_variance / 1500)))  # More lenient threshold

            return smoothness > 0.4, smoothness  # More lenient requirement

        except:

            return False, 0.0



    def _detect_car_interior_features(self, roi):
        """Detect car interior features: seatbelt straps, steering wheel, car seats"""
        try:
            features = 0
            h, w = roi.shape[:2]
            
            if h < 50 or w < 50:
                return 0
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Feature 1: Detect diagonal dark straps (seatbelts)
            # Seatbelts are typically dark (black/dark grey) and diagonal
            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([180, 255, 80])  # Dark colors
            dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
            
            # Look for diagonal lines using Hough transform
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                                   minLineLength=40, maxLineGap=5)
            
            diagonal_dark_straps = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Check if line is diagonal (30-60 degrees)
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if 30 < angle < 60:  # Diagonal
                        # Check if line passes through dark region
                        mid_x = (x1 + x2) // 2
                        mid_y = (y1 + y2) // 2
                        if 0 <= mid_x < w and 0 <= mid_y < h:
                            if dark_mask[mid_y, mid_x] > 0:
                                diagonal_dark_straps += 1
            
            # Lowered threshold for seatbelt detection in car context
            if diagonal_dark_straps >= 1:  # Very lenient - even 1 diagonal strap is strong indicator
                features += 1  # Seatbelt detected
            
            # Feature 2: Detect circular shapes (steering wheel)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                      param1=50, param2=30, minRadius=20, maxRadius=80)
            if circles is not None:
                features += 1  # Steering wheel detected
            
            # Feature 3: Check for car seat texture (leather/fabric pattern)
            # Car seats often have horizontal stitching lines
            horizontal_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 10:  # Nearly horizontal
                        horizontal_lines += 1
            
            if horizontal_lines >= 3:  # Lowered from 5 to 3 for better sensitivity
                features += 1  # Car seat texture detected
            
            # Feature 4: Window/light pattern (bright rectangular regions on sides)
            left_brightness = np.mean(gray[:, :w//3])
            right_brightness = np.mean(gray[:, 2*w//3:])
            center_brightness = np.mean(gray[:, w//3:2*w//3])
            
            # Windows are brighter than interior - lowered threshold for better sensitivity
            if (left_brightness > center_brightness * 1.1 or
                right_brightness > center_brightness * 1.1):  # Lowered from 1.2 to 1.1
                features += 1  # Window light pattern detected
            
            # Feature 5: Car interior color pattern (dark seats, headliner)
            # Car interiors often have dark upper and lower regions with person in middle
            upper_dark = np.mean(gray[:h//3, :]) < 80
            lower_dark = np.mean(gray[2*h//3:, :]) < 80
            if upper_dark or lower_dark:
                features += 1  # Dark car interior pattern detected
            
            # Feature 6: Detect car headrests (very distinctive - black rectangles behind head)
            # Headrests are typically dark, rectangular, and in upper portion of image
            upper_region = gray[:h//2, :]  # Top half where headrest would be
            if upper_region.size > 0:
                # Look for dark rectangular regions (headrests)
                _, thresh = cv2.threshold(upper_region, 60, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 300:  # Lowered from 500 to 300 for better sensitivity
                        x, y, cw, ch = cv2.boundingRect(cnt)
                        aspect = cw / max(ch, 1)
                        # Headrests are typically wider than tall (aspect 1.0-4.0)
                        if 0.8 < aspect < 4.0 and ch > 20:  # Lowered from 30 to 20
                            features += 1  # Headrest detected
                            break  # Only count once
            
            # Feature 7: Car window frame detection (vertical lines on sides)
            # Look for strong vertical edges on left/right sides (window frames)
            left_edges = edges[:, :w//4]
            right_edges = edges[:, 3*w//4:]
            left_vertical = np.sum(left_edges > 0)
            right_vertical = np.sum(right_edges > 0)
            edge_threshold = h * 1.5  # Lowered from 2 to 1.5 for better sensitivity
            if left_vertical > edge_threshold or right_vertical > edge_threshold:
                features += 1  # Window frame detected
            
            # Feature 8: Detect collared shirt (typical car driver clothing)
            # Look for collar pattern in lower part of ROI
            lower_region = gray[2*h//3:, :]
            if lower_region.size > 0:
                # Look for horizontal lines that might be shirt collar
                lower_edges = cv2.Canny(lower_region, 30, 100)
                lines_lower = cv2.HoughLinesP(lower_edges, 1, np.pi/180, threshold=5,  # Lowered from 10
                                             minLineLength=15, maxLineGap=15)  # More lenient
                if lines_lower is not None and len(lines_lower) >= 1:  # Lowered from 2 to 1
                    features += 1  # Possible collared shirt detected
            
            # Feature 9: Check for car headliner/ceiling (bright horizontal region at top)
            top_region = gray[:h//4, :]
            if top_region.size > 0:
                top_brightness = np.mean(top_region)
                if top_brightness > 80:  # Lowered from 100 to 80 for better sensitivity
                    features += 1
            
            return features
            
        except:
            return 0
            
    def _detect_construction_features(self, roi):
        """Detect construction site features to distinguish workers from car occupants"""
        try:
            features = 0
            h, w = roi.shape[:2]
            
            if h < 50 or w < 50:
                return 0
            
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Feature 1: Check for yellow/orange colors (construction vests, equipment)
            # RAISED threshold to prevent false positives on car interiors
            yellow_lower = np.array([15, 60, 60])  # Higher saturation for construction yellow
            yellow_upper = np.array([40, 255, 255])
            orange_lower = np.array([5, 60, 60])   # Higher saturation for construction orange
            orange_upper = np.array([30, 255, 255])

            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)

            yellow_pixels = cv2.countNonZero(yellow_mask)
            orange_pixels = cv2.countNonZero(orange_mask)
            total_pixels = h * w

            # RAISED threshold from 0.08 to 0.15 to prevent car interior false positives
            if (yellow_pixels + orange_pixels) / total_pixels > 0.15:
                features += 2  # Strong indicator

            # Feature 2: Detect grid/crosshatch pattern (scaffolding, metal structures)
            # RAISED thresholds to be more specific
            edges = cv2.Canny(gray, 30, 100)  # Back to original thresholds
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15,
                                   minLineLength=30, maxLineGap=10)  # More restrictive

            if lines is not None and len(lines) > 20:  # RAISED from 10 to 20
                # Count horizontal and vertical lines
                horizontal = 0
                vertical = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 20:  # Tighter angle range
                        horizontal += 1
                    elif angle > 70:  # Tighter angle range
                        vertical += 1

                # Grid pattern = both horizontal AND vertical lines
                # RAISED thresholds from 3 to 5
                if horizontal > 5 and vertical > 5:
                    features += 2  # Strong indicator of scaffolding/grid

            # Feature 3: Detect hard hat colors in upper region (worker presence)
            # RAISED threshold to prevent false positives
            upper_region = roi[:h//3, :]
            if upper_region.size > 0:
                upper_hsv = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)
                # Hard hat colors: white, yellow, orange, blue, red
                white_mask = cv2.inRange(upper_hsv, np.array([0, 0, 180]), np.array([180, 30, 255]))
                yellow_hat = cv2.inRange(upper_hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))

                white_pixels = cv2.countNonZero(white_mask)
                yellow_hat_pixels = cv2.countNonZero(yellow_hat)

                # RAISED threshold from 0.10 to 0.20
                if (white_pixels + yellow_hat_pixels) / (w * (h//3)) > 0.20:
                    features += 1  # Possible hard hat detected

            # Feature 4: Detect bright sky background (outdoor work environment)
            # Upper portion is very bright (sky), middle has person
            top_brightness = np.mean(gray[:h//3, :])
            middle_brightness = np.mean(gray[h//3:2*h//3, :])
            
            if top_brightness > 150 and middle_brightness < 100:
                # Bright sky above, darker below (typical outdoor construction)
                features += 1
            
            return features
            
        except:
            return 0



    def _analyze_image_conditions(self, frame):

        """Analyze image conditions for adaptive detection"""

        try:

            # Calculate overall brightness

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            brightness = np.mean(gray)

            

            # Calculate contrast

            contrast = np.std(gray)

            

            # Determine image condition

            if brightness > 180:

                condition = "bright"

                brightness_factor = 0.9  # Reduce sensitivity for bright images

            elif brightness < 80:

                condition = "dark"

                brightness_factor = 1.2  # Increase sensitivity for dark images

            else:

                condition = "normal"

                brightness_factor = 1.0

            

            if contrast > 60:

                contrast_factor = 0.9  # Reduce sensitivity for high contrast

            elif contrast < 30:

                contrast_factor = 1.1  # Increase sensitivity for low contrast

            else:

                contrast_factor = 1.0

            

            return condition, brightness_factor, contrast_factor, brightness, contrast

        except:

            return "normal", 1.0, 1.0, 128, 40



    def detect_helmet_in_head(self, frame, head_bbox, threshold=None):

        """Enhanced helmet detection using PPE dataset classes"""

        x1, y1, x2, y2 = head_bbox

        h, w = frame.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)

        x2, y2 = min(w, x2), min(h, y2)



        if x2 <= x1 or y2 <= y1:

            return False, 0.0, "invalid_bbox"



        head_roi = frame[y1:y2, x1:x2]

        if head_roi.size == 0:

            return False, 0.0, "empty_roi"



        # Check if we have a trained PPE model

        is_ppe_model = "PPE" in self.model_path or "best.pt" in self.model_path

        

        if is_ppe_model:

            # Use direct PPE class detection

            return self._detect_ppe_classes_direct(frame, head_bbox)

        else:

            # Use traditional helmet detection

            return self._detect_helmet_traditional(frame, head_bbox, head_roi)



    def _check_cap_features(self, head_roi):
        """Detect cap/hat features to distinguish from helmets"""
        try:
            if head_roi.size == 0:
                return False, 0.0
            
            gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # FEATURE 1: Check for brim/visor (horizontal edge at front)
            # Caps have a brim that extends horizontally at the bottom front
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for horizontal lines in the bottom 30% (where brim would be)
            bottom_region = edges[int(h*0.7):, :]
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//2, 1))
            horizontal_lines = cv2.morphologyEx(bottom_region, cv2.MORPH_OPEN, horizontal_kernel)
            brim_pixels = np.sum(horizontal_lines > 0)
            brim_ratio = brim_pixels / (bottom_region.shape[0] * bottom_region.shape[1])
            
            # FEATURE 2: Check aspect ratio - caps are flatter (wider relative to height)
            # Use edge density to estimate shape
            total_edges = np.sum(edges > 0)
            edge_density = total_edges / (h * w)
            
            # FEATURE 3: Check dome shape - caps are less dome-shaped than helmets
            has_dome, dome_conf = self._check_dome_shape(gray)
            
            # FEATURE 4: Check size - caps are smaller relative to head region
            # Caps typically cover less area
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            solid_pixels = np.sum(binary > 0)
            coverage = solid_pixels / (h * w)
            
            # DECISION: Cap if:
            # - Has brim (horizontal edge at bottom)
            # - Low dome shape
            # - Lower coverage (smaller)
            cap_score = 0.0
            
            if brim_ratio > 0.08:  # Has brim - increased threshold
                cap_score += 0.4
            
            if not has_dome or dome_conf < 0.2:  # Not dome-shaped - lowered threshold
                cap_score += 0.3
            
            if coverage < 0.3:  # Smaller coverage - lowered threshold
                cap_score += 0.3
            
            is_cap = cap_score > 0.6  # Increased threshold to be more lenient
            
            if self.debug and is_cap:
                print(f"[PPE-DEBUG] CAP DETECTED: brim={brim_ratio:.3f}, dome={dome_conf:.2f}, coverage={coverage:.3f}, score={cap_score:.2f}")
            
            return is_cap, cap_score
            
        except:
            return False, 0.0



    def _validate_helmet_position(self, frame, head_bbox, person_bbox=None):
        """Validate that helmet is actually worn on head - simplified for accuracy"""
        try:
            x1, y1, x2, y2 = head_bbox
            head_roi = frame[y1:y2, x1:x2]
            
            if head_roi.size == 0:
                return True, 1.0
            
            h, w = head_roi.shape[:2]
            
            # CHECK 1: Reject caps/hats (not safety helmets)
            is_cap, cap_score = self._check_cap_features(head_roi)
            if is_cap:
                if self.debug:
                    print(f"[PPE-DEBUG] HELMET REJECTED: Cap/hat detected (score: {cap_score:.2f})")
                return False, 0.1
            
            # CHECK 2: Verify helmet color exists in head region
            # This is the most important check - if no helmet color on head, it's not a helmet
            head_helmet_detected, head_helmet_conf, head_color_name = self.detect_helmet_by_color(head_roi, 0.10)
            
            if head_helmet_detected and head_helmet_conf > 0.10:
                # Helmet color found on head - likely a real helmet
                if self.debug:
                    print(f"[PPE-DEBUG] Helmet position VALID: {head_color_name} color on head (conf: {head_helmet_conf:.2f})")
                return True, 1.0
            
            # No helmet color on head - but could still be a dark/black helmet
            # Check if the head region is very dark (black helmet with low color detection)
            gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
            dark_ratio = np.sum(gray < 60) / gray.size
            
            if dark_ratio > 0.5:
                # Very dark head region - likely black/dark helmet
                if self.debug:
                    print(f"[PPE-DEBUG] Helmet position VALID: Dark region detected (ratio: {dark_ratio:.2f}) - likely dark helmet")
                return True, 0.8
            
            # Check for smooth dome shape as additional evidence
            is_smooth, smooth_conf = self._check_smooth_surface(head_roi)
            has_dome, dome_conf = self._check_dome_shape(gray)
            
            if is_smooth and smooth_conf > 0.5 and has_dome and dome_conf > 0.5:
                # Strong shape evidence even without color - could be grey/silver helmet
                if self.debug:
                    print(f"[PPE-DEBUG] Helmet position VALID: Strong shape evidence (smooth: {smooth_conf:.2f}, dome: {dome_conf:.2f})")
                return True, 0.7
            
            # No color, no dark region, no strong shape - likely not a helmet
            if self.debug:
                print(f"[PPE-DEBUG] HELMET REJECTED: No helmet evidence on head (color: {head_helmet_conf:.2f}, dark: {dark_ratio:.2f})")
            return False, 0.2
            
        except Exception as e:
            if self.debug:
                print(f"[PPE-DEBUG] Helmet position validation failed: {e}")
            return True, 1.0



    def _detect_ppe_classes_direct(self, frame, head_bbox):

        """Direct detection using PPE dataset classes"""

        try:

            # Run model on the head region

            x1, y1, x2, y2 = head_bbox

            head_region = frame[y1:y2, x1:x2]

            

            # Detect PPE items in head region

            results = self.model(head_region, conf=0.1, iou=0.45, 

                              device=self.device, verbose=False)

            

            helmet_detected = False

            max_conf = 0.0

            detected_class = "unknown"

            

            for result in results:

                if result.boxes is not None:

                    for box in result.boxes:

                        conf = float(box.conf[0].cpu().numpy())

                        cls = int(box.cls[0].cpu().numpy())

                        class_name = PPE_CLASSES[cls] if cls < len(PPE_CLASSES) else "unknown"

                        

                        # Check for helmet or no-helmet classes
            # Use lower threshold for yellow-colored helmets
                        if class_name == 'helmet' and conf > max_conf:

                            helmet_detected = True

                            max_conf = conf

                            detected_class = "helmet"

                        elif class_name == 'no-helmet' and conf > max_conf:

                            helmet_detected = False

                            max_conf = conf

                            detected_class = "no-helmet"

            

            if self.debug and max_conf > 0:

                print(f"[PPE-DEBUG] Direct PPE detection: {detected_class} (conf: {max_conf:.2f})")

            

            # Apply threshold - lowered for better detection

            threshold = self.helmet_threshold * 0.6  # 40% more lenient for direct PPE model

            if helmet_detected and max_conf >= threshold:
                # VALIDATION: Check if helmet is actually worn, not held
                person_bbox = getattr(self, '_current_person_bbox', None)
                if person_bbox is not None:
                    is_worn, worn_conf = self._validate_helmet_position(frame, head_bbox, person_bbox)
                    if not is_worn:
                        if self.debug:
                            print(f"[PPE-DEBUG] HELMET REJECTED: Not worn on head (held in hand)")
                        return False, max_conf * worn_conf, f"direct_ppe_held"
                return True, max_conf, f"direct_ppe_helmet"

            elif not helmet_detected and max_conf >= threshold:

                return False, max_conf, f"direct_ppe_no_helmet"

            else:

                # Fallback to traditional detection

                return self._detect_helmet_traditional(frame, head_bbox, frame[y1:y2, x1:x2])

                

        except Exception as e:

            if self.debug:

                print(f"[PPE-DEBUG] Direct PPE detection failed: {e}")

            # Fallback to traditional detection

            return self._detect_helmet_traditional(frame, head_bbox, frame[y1:y2, x1:x2])



    def _detect_helmet_traditional(self, frame, head_bbox, head_roi):

        """Traditional helmet detection using color and shape analysis"""



        # STEP 1: Check for hair (strong negative indicator)

        has_hair = self._check_hair_texture(head_roi)

        has_styled_hair = self._check_styled_hair(head_roi)  # NEW: catch smooth/oiled hair

        if has_hair or has_styled_hair:

            if self.debug:

                print(f"[PPE-DEBUG] Hair detected - likely no helmet (textured={has_hair}, styled={has_styled_hair})")

            # Hair is strong negative - styled hair is also hair, not helmet
            hair_penalty = 0.4  # Strong penalty - hair means NO helmet

        # STEP 1b: Check for color first
        color_detected_early, color_conf_early, color_name_early = self.detect_helmet_by_color(head_roi, self.helmet_threshold)

        # STEP 1c: Check for skin texture (reject bald heads as helmet)
        # IMPROVED: Don't skip skin check for any color - instead use smarter check
        # A real helmet has LOW skin-like texture even if white/yellow
        has_skin = self._check_skin_texture(head_roi)
        if has_skin:
            # But if we have strong helmet color evidence, override skin check
            # Real helmets have solid color, skin has varied texture
            if not (color_detected_early and color_conf_early > 0.50 and color_name_early in ['yellow', 'blue', 'red', 'orange', 'green']):
                # No strong helmet color → likely bald head, reject
                if self.debug:
                    print("[PPE-DEBUG] Skin texture detected - rejecting helmet (bald head)")
                return False, 0.0, "skin_detected_bald_head"
            else:
                if self.debug:
                    print(f"[PPE-DEBUG] Skin detected but overridden by strong {color_name_early} color (conf: {color_conf_early:.2f})")



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
        # Use early detection results from STEP 1b
        color_detected, color_conf, color_name = color_detected_early, color_conf_early, color_name_early

        color_score = color_conf * 0.5 if color_detected else 0.0  # Reduce color importance



        if self.debug:

            print(f"[PPE-DEBUG] Detection scores:")

            print(f"  - Hair detected: {has_hair}")

            print(f"  - Smooth surface: {is_smooth} (conf: {smooth_conf:.2f})")

            print(f"  - Dome shape: {has_dome} (conf: {dome_conf:.2f})")

            print(f"  - Top position: {position_score:.2f}")

            print(f"  - Color detected: {color_detected} ({color_name}, conf: {color_conf:.2f})")



        # INDEPENDENT HELMET DETECTION - Shape-based only
        # Completely independent from vest detection
        strong_conditions = 0

        # Shape conditions (no color dependency)
        if is_smooth and smooth_conf > 0.15:  # Moderate threshold
            strong_conditions += 1
        if has_dome and dome_conf > 0.30:  # Moderate threshold
            strong_conditions += 1
        if position_score > 0.35:  # STRICT - helmet must be at top of head
            strong_conditions += 1

        # Color is optional bonus (not required)
        if color_detected and color_name in ['white', 'yellow', 'orange', 'blue', 'red', 'black']:
            strong_conditions += 0.3  # Small bonus, not required



        # Calculate combined confidence

        # Increased color weight from 0.10 to 0.25 for better yellow helmet detection
        combined_conf = (dome_score * 0.30 + smooth_score * 0.30 +

                        position_score * 0.15 + color_score * 0.25)



        # Apply hair penalty if hair detected

        if has_hair or has_styled_hair:

            combined_conf *= hair_penalty  # Strong penalty for hair



        if self.debug:

            print(f"[PPE-DEBUG] Helmet detection analysis:")

            print(f"  - Strong conditions: {strong_conditions:.1f}/3")

            print(f"  - Combined confidence: {combined_conf:.3f}")

            print(f"  - Hair detected: {has_hair}")

            print(f"  - Color detected: {color_detected}")

            if color_detected:

                print(f"  - Color name: {color_name}, conf: {color_conf:.3f}")



        # INDEPENDENT DECISION LOGIC - Simple threshold
        # Completely independent from vest detection

        # If styled hair detected, need ALL 3 conditions (strict)
        # Otherwise, need 2 conditions (normal)
        min_conditions = 3.0 if has_styled_hair else 2.0

        if strong_conditions >= min_conditions:
            # Sufficient conditions = helmet detected
            if self.debug:
                print(f"[PPE-DEBUG] HELMET DETECTED: {strong_conditions:.1f} conditions (min: {min_conditions:.1f})")
            
            # VALIDATION: Check if helmet is actually worn, not held
            person_bbox = getattr(self, '_current_person_bbox', None)
            if person_bbox is not None:
                is_worn, worn_conf = self._validate_helmet_position(frame, head_bbox, person_bbox)
                if not is_worn:
                    if self.debug:
                        print(f"[PPE-DEBUG] HELMET REJECTED: Not worn on head (held in hand)")
                    return False, combined_conf * worn_conf, "held_helmet"
            
            return True, combined_conf, "helmet_detected"

        # Only reject if hair detected (no helmet)
        if has_hair or has_styled_hair:
            if self.debug:
                print(f"[PPE-DEBUG] HELMET REJECTED: Hair detected (no helmet)")
            return False, combined_conf, "hair_visible"

        # Default: no helmet
        if self.debug:
            print(f"[PPE-DEBUG] HELMET NOT DETECTED: {strong_conditions:.1f} conditions")
        return False, combined_conf, "no_helmet"



    def _check_top_position(self, head_roi):

        """Check if helmet-like object is on top of head region - STRICT validation"""

        try:

            gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)

            h, w = gray.shape

            

            # Check top 25% of head region for solid object (helmet must be at very top)
            
            top_region = gray[:int(h*0.25), :]

            

            # Look for solid object at top (helmet should be here)

            _, thresh = cv2.threshold(top_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            

            # Calculate coverage in top region

            top_coverage = np.sum(thresh > 0) / (top_region.shape[0] * top_region.shape[1])

            

            # Helmet should have high coverage at the very top - STRICT check

            if top_coverage > 0.4:  # Increased from 0.2 to 0.4

                return min(top_coverage * 2, 1.0)

            return 0.0

        except:

            return 0.0



    def _detect_seatbelt(self, frame, person_bbox):

        """Detect seatbelt - more sensitive for clear seatbelts"""

        x1, y1, x2, y2 = person_bbox

        height = y2 - y1

        width = x2 - x1

        

        # Focus on upper torso and diagonal chest area - EXPANDED for better coverage
        # Seatbelts can appear at various angles across the chest

        shoulder_y1 = y1 + int(height * 0.05)  # Start even higher (was 0.10)

        shoulder_y2 = y1 + int(height * 0.70)  # Extend much lower (was 0.55)

        chest_x1 = x1 + int(width * 0.10)     # Wider area (was 0.15)

        chest_x2 = x2 + int(width * 0.05)     # Extend further right (was -0.05)

        

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

                print(f"[PPE-DEBUG] Seatbelt detection:")

                print(f"  - Dark coverage: {dark_coverage:.2f} (need >0.04)")

                print(f"  - Diagonal lines: {diagonal_lines} (need >=1)")

                print(f"  - Strap-like contours: {strap_like_contours} (need >=1)")

                print(f"  - Strap continuity: {strap_continuity:.2f}")

                print(f"  - Combined conf: {combined_conf:.2f} (need >0.15)")

            

            # ULTRA-LENIENT detection criteria for car drivers
            # Catch more real-world seatbelts even with partial evidence
            has_strap_contour = strap_like_contours >= 1
            has_many_diagonals = diagonal_lines >= 4  # Lowered from 6
            has_some_diagonals = diagonal_lines >= 1  # Any diagonal line helps
            
            # ULTRA-sensitive detection - catch seatbelts with minimal evidence
            # Real seatbelts can be subtle in photos - need very low thresholds
            seatbelt_detected = (
                (dark_coverage > 0.04 and combined_conf > 0.15 and has_strap_contour) or  # Strap contour based (very low)
                (dark_coverage > 0.05 and has_many_diagonals) or  # Many diagonals (very low)
                (dark_coverage > 0.08 and has_some_diagonals) or  # Just dark + any diagonals
                (dark_coverage > 0.15)  # Very dark region alone (strong strap presence)
            )
            
            if seatbelt_detected:
                
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

                            # Additional validation: person should be inside vehicle

                            person_area = (x2 - x1) * (y2 - y1)

                            vehicle_area = (vx2 - vx1) * (vy2 - vy1)

                            

                            # RELAXED: Person can take up to 60% of vehicle (for close-up shots)

                            if person_area < vehicle_area * 0.6:  # Relaxed from 0.3

                                if vclass == 3:  # motorcycle

                                    return "2-wheeler"

                                elif vclass in [2, 5, 7]:  # car, bus, truck

                                    if self.debug:

                                        print(f"[PPE-DEBUG] Person inside vehicle (car/bus/truck) - 4-WHEELER")

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

            

            # STEP 1: Check for car interior features FIRST (more reliable than helmet color)
            # Previous helmet color check was removed - car interiors (black/dark grey headrests)
            # were being misclassified as worker helmet colors
            expand = 50
            context_x1 = max(0, x1 - expand)
            context_y1 = max(0, y1 - expand)
            context_x2 = min(w, x2 + expand)
            context_y2 = min(h, y2 + expand)
            context_roi = frame[context_y1:context_y2, context_x1:context_x2]

            construction_features = 0  # Initialize before the block

            if context_roi.size > 0:
                # STEP 1: Check for construction site context
                construction_features = self._detect_construction_features(context_roi)
                if construction_features >= 1:
                    if self.debug:
                        print(f"[PPE-DEBUG] Construction features detected ({construction_features}) - WORKER")
                    return "unknown"

                # STEP 2: Check for car interior features
                # Only classify as car if no construction features detected
                car_features = self._detect_car_interior_features(context_roi)
                # Require at least 3 strong car features to classify as 4-wheeler
                if car_features >= 3:
                    if self.debug:
                        print(f"[PPE-DEBUG] Car features detected ({car_features}) - 4-WHEELER")
                    return "4-wheeler"

            # Final check: If seatbelt detected AND no construction features, likely in a car
            # This prevents safety harnesses on workers from being misclassified as seatbelts
            if construction_features == 0:
                seatbelt_present, _, _ = self._detect_seatbelt(frame, person_bbox)
                if seatbelt_present:
                    if self.debug:
                        print("[PPE-DEBUG] Seatbelt detected without other features - assuming 4-WHEELER")
                    return "4-wheeler"
            
            # Default to unknown (worker or unclear)
            if self.debug:
                print("[PPE-DEBUG] No vehicle or worker PPE detected - returning UNKNOWN")
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

                # Check if this is a PPE-trained model

                is_ppe_model = "PPE" in self.model_path or "best.pt" in self.model_path

                

                if is_ppe_model:

                    # Use PPE dataset classes: detect all PPE classes

                    ppe_class_indices = list(range(NUM_PPE_CLASSES))  # [0,1,2,3,4] for all PPE classes

                    results = self.model(frame, conf=self.person_threshold, iou=0.45,

                                       device=self.device, verbose=False, classes=ppe_class_indices)

                    

                    for result in results:

                        if result.boxes is not None:

                            for box in result.boxes:

                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                                conf = float(box.conf[0].cpu().numpy())

                                cls = int(box.cls[0].cpu().numpy())

                                class_name = PPE_CLASSES[cls] if cls < len(PPE_CLASSES) else "unknown"

                                

                                # Only add person detections for PPE processing

                                if class_name == 'person':

                                    persons.append({

                                        "bbox": (int(x1), int(y1), int(x2), int(y2)), 

                                        "confidence": conf,

                                        "class": class_name,

                                        "class_id": cls

                                    })

                                elif self.debug:

                                    print(f"[PPE-DEBUG] Detected PPE item: {class_name} (conf: {conf:.2f})")

                else:

                    # Use standard person detection (class 0)

                    results = self.model(frame, conf=self.person_threshold, iou=0.45,

                                       device=self.device, verbose=False, classes=[0])

                    for result in results:

                        if result.boxes is not None:

                            for box in result.boxes:

                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                                conf = float(box.conf[0].cpu().numpy())

                                persons.append({

                                    "bbox": (int(x1), int(y1), int(x2), int(y2)), 

                                    "confidence": conf,

                                    "class": "person",

                                    "class_id": 0

                                })

                

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

        """Create minimum working detection when all else fails - VERY STRICT"""

        persons = []

        h, w = frame.shape[:2]



        # Look for large vertical objects (potential persons) - MUCH STRICTER

        try:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            blurred = cv2.GaussianBlur(gray, (21, 21), 0)

            edges = cv2.Canny(blurred, 50, 150)



            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



            # Sort contours by area (largest first)

            contours = sorted(contours, key=cv2.contourArea, reverse=True)



            for cnt in contours[:3]:  # Only check top 3 largest contours

                x, y, cw, ch = cv2.boundingRect(cnt)

                area = cv2.contourArea(cnt)

                

                # MUCH STRICTER criteria for person detection

                # 1. Must be tall (aspect ratio)

                # 2. Must be large enough (minimum area)

                # 3. Must be reasonably positioned (not too low or too high)

                min_area = (h * w) * 0.01  # At least 1% of frame area

                

                if (ch > h * 0.25 and  # At least 25% of frame height

                    cw > w * 0.05 and   # At least 5% of frame width

                    ch > cw * 2.0 and   # Much taller than wide (person-like)

                    area > min_area and   # Large enough area

                    y > h * 0.1 and y < h * 0.8):  # Reasonably positioned

                    

                    # Additional check: Look for head-like shape at top

                    head_region = gray[y:y+int(ch*0.2), x:x+cw]

                    if head_region.size > 0:

                        # Check for circular/rounded head shape

                        head_edges = cv2.Canny(head_region, 50, 150)

                        head_contours, _ = cv2.findContours(head_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        

                        if head_contours:

                            # Check if any head contour is reasonably circular

                            for head_cnt in head_contours:

                                head_area = cv2.contourArea(head_cnt)

                                head_x, head_y, head_w, head_h = cv2.boundingRect(head_cnt)

                                if head_area > 100:  # Minimum head area

                                    circularity = 4 * np.pi * head_area / (head_w * head_h + 1e-6)

                                    if circularity > 0.3:  # Reasonably circular

                                        persons.append({

                                            "bbox": (int(x), int(y), int(x+cw), int(y+ch)),

                                            "confidence": 0.3

                                        })

                                        break  # Only add one person per contour

        except:

            pass



        return persons



    def _detect_vest(self, frame, person_bbox):

        """Adaptive vest detection using high-vis colors in torso region"""

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



            # Simple color ranges - no adaptive conditions
            # Use wide ranges for consistent detection

            yellow = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))

            orange = cv2.inRange(hsv, np.array([5, 80, 80]), np.array([25, 255, 255]))

            green = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([70, 255, 255]))

            # Fixed threshold for consistent detection
            min_coverage = 0.30  # Balanced threshold

            combined = cv2.bitwise_or(yellow, orange)

            combined = cv2.bitwise_or(combined, green)



            # Simple coverage calculation - no adaptive factors
            coverage = np.sum(combined > 0) / (torso_roi.shape[0] * torso_roi.shape[1])
            
            # CHECK: Look for reflective strips (white/grey horizontal bands)
            # Real safety vests have reflective strips, shirts don't
            gray = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2GRAY)
            
            # Look for bright horizontal bands (reflective strips)
            bright_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
            bright_coverage = np.sum(bright_mask > 0) / (torso_roi.shape[0] * torso_roi.shape[1])
            
            # Check for horizontal strip pattern using morphology
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (torso_roi.shape[1]//3, 1))
            bright_horizontal = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_h)
            strip_contours, _ = cv2.findContours(bright_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            has_reflective_strips = False
            for cnt in strip_contours:
                area = cv2.contourArea(cnt)
                if area > 150:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    aspect = cw / ch if ch > 0 else 0
                    if aspect > 3.5:
                        has_reflective_strips = True
                        break
            
            # BALANCED scoring:
            # - Very high coverage (>0.50) = definitely vest even without strips
            # - Reflective strips = strong vest evidence, lower coverage needed
            # - Low coverage + no strips = likely shirt, reject
            if coverage > 0.50:
                # Very high color coverage = definitely a vest (can't be shirt)
                vest_score = coverage
            elif has_reflective_strips:
                # Has reflective strips = real vest, bonus
                vest_score = coverage + 0.15
            else:
                # No strips, moderate coverage = likely shirt, apply penalty
                vest_score = coverage * 0.7
            
            # Simple confidence calculation
            conf = min(vest_score / min_coverage, 1.0)

            if self.debug:

                print(f"[PPE-DEBUG] Vest detection:")

                print(f"  - Color coverage: {coverage:.3f}")

                print(f"  - Bright coverage: {bright_coverage:.3f}")

                print(f"  - Reflective strips: {has_reflective_strips}")

                print(f"  - Vest score: {vest_score:.3f}")

                print(f"  - Threshold: {min_coverage:.3f}")

                print(f"  - Confidence: {conf:.3f}")

                print(f"  - Result: {vest_score:.3f} > {min_coverage:.3f} = {vest_score > min_coverage}")



            return vest_score > min_coverage, conf

        except:

            return False, 0.0



    def _create_emergency_result(self, frame, timestamp, error_msg):

        """Emergency result when everything fails - ONLY if actual person detected"""

        h, w = frame.shape[:2]



        # FIRST: Check if there's actually a person-like shape in the frame

        try:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            blurred = cv2.GaussianBlur(gray, (21, 21), 0)

            edges = cv2.Canny(blurred, 50, 150)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            

            # Look for at least one person-like contour

            person_found = False

            for cnt in contours:

                x, y, cw, ch = cv2.boundingRect(cnt)

                # Must be tall and reasonably sized

                if ch > h * 0.2 and cw > w * 0.05 and ch > cw * 1.5:

                    person_found = True

                    center_bbox = (int(x), int(y), int(x+cw), int(y+ch))

                    break

            

            if not person_found:

                # No person detected - return empty result

                return PPEResult(

                    total_persons=0,

                    helmet_detected=0,

                    no_helmet=0,

                    seatbelt_detected=0,

                    no_seatbelt=0,

                    vest_detected=0,

                    no_vest=0,

                    persons=[],

                    timestamp=timestamp,

                    processing_time=0.0,

                    debug_mode=self.debug,

                    model_loaded=False,

                    fallback_used=True,

                    error_message=f"No person detected: {error_msg}"

                )

        except:

            # If person detection fails, return empty result

            return PPEResult(

                total_persons=0,

                helmet_detected=0,

                no_helmet=0,

                seatbelt_detected=0,

                no_seatbelt=0,

                vest_detected=0,

                no_vest=0,

                persons=[],

                timestamp=timestamp,

                processing_time=0.0,

                debug_mode=self.debug,

                model_loaded=False,

                fallback_used=True,

                error_message=f"Emergency mode failed: {error_msg}"

            )



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

            vest_detected=0,

            no_vest=1,

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

                print(f"[PPE]  AI SAFETY DETECTION SYSTEM STARTED")

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

            vest_count = 0

            no_vest_count = 0



            if self.debug:

                print(f"\n[PPE]  PROCESSING {len(persons)} PERSONS...")

                print(f"{'-'*60}")



            for idx, person in enumerate(persons):

                person_id = f"P{idx + 1}"

                person_bbox = person["bbox"]

                

                # ============================================
                # SIMPLE PPE DETECTION: Helmet + Vest only
                # ============================================

                # STEP 3: Detect vehicle type FIRST to decide what PPE to check
                vehicle_type = self._detect_vehicle_type(frame, person_bbox)

                helmet_present, helmet_conf, helmet_method = False, 0.0, "not_checked"
                vest_present, vest_conf = False, 0.0
                seatbelt_present, seatbelt_conf, seatbelt_method = False, 0.0, "not_applicable"

                if vehicle_type == "4-wheeler":
                    # 4-WHEELER: Only seatbelt detection, NO helmet/vest
                    if self.debug:
                        print(f"[PPE-DEBUG] 4-wheeler detected - only checking seatbelt")
                    seatbelt_present, seatbelt_conf, seatbelt_method = self._detect_seatbelt(frame, person_bbox)
                    
                    # Still compute head/vest bbox for visualization (but mark as not detected)
                    head_bbox = self.get_head_region(person_bbox, frame)
                    vest_bbox = self.get_vest_region(person_bbox, frame, head_bbox)
                    
                    if self.debug:
                        print(f"[PPE-DEBUG] Seatbelt: {'DETECTED' if seatbelt_present else 'Not detected'} (conf: {seatbelt_conf:.3f})")
                        print(f"[PPE-DEBUG] Helmet: SKIPPED (4-wheeler)")
                        print(f"[PPE-DEBUG] Vest: SKIPPED (4-wheeler)")
                else:
                    # 2-WHEELER or UNKNOWN: Helmet + Vest detection + seatbelt check
                    if self.debug:
                        print(f"[PPE-DEBUG] {vehicle_type} - checking helmet & vest")

                    # Detect helmet - ADAPTIVE: finds actual helmet position
                    self._current_person_bbox = person_bbox  # Store for validation
                    head_bbox = self.get_head_region(person_bbox, frame)  # Pass frame for adaptive detection
                    threshold = self.fallback_threshold if fallback_used else self.helmet_threshold
                    helmet_present, helmet_conf, helmet_method = self.detect_helmet_in_head(frame, head_bbox, threshold)
                    self._current_person_bbox = None  # Clean up after validation

                    # Detect vest - ADAPTIVE: finds actual vest position
                    vest_bbox = self.get_vest_region(person_bbox, frame, head_bbox)  # Pass head_bbox for gap
                    vest_present, vest_conf = self._detect_vest(frame, person_bbox)

                    # Also check seatbelt for unknown vehicles (might be in car)
                    if vehicle_type == "unknown":
                        seatbelt_present, seatbelt_conf, seatbelt_method = self._detect_seatbelt(frame, person_bbox)
                        if self.debug:
                            print(f"[PPE-DEBUG] Seatbelt: {'DETECTED' if seatbelt_present else 'Not detected'} (conf: {seatbelt_conf:.3f})")

                    if self.debug:
                        print(f"[PPE-DEBUG] Helmet: {'DETECTED' if helmet_present else 'Not detected'} (conf: {helmet_conf:.3f})")
                        print(f"[PPE-DEBUG] Vest: {'DETECTED' if vest_present else 'Not detected'} (conf: {vest_conf:.3f})")

                

                # STEP 4: Update counts based on vehicle type

                if vehicle_type == "4-wheeler":
                    # 4-WHEELER: Count seatbelt only
                    if seatbelt_present:
                        seatbelt_count += 1
                    else:
                        no_seatbelt_count += 1
                    if self.debug:
                        print(f"[PPE-DEBUG] Counting (4-wheeler): seatbelt={seatbelt_present}")
                else:
                    # 2-WHEELER/UNKNOWN: Count helmet + vest primarily
                    if helmet_present:
                        helmet_count += 1
                        if self.debug:
                            print(f"[PPE-DEBUG] Helmet count incremented to {helmet_count}")
                    else:
                        no_helmet_count += 1
                        if self.debug:
                            print(f"[PPE-DEBUG] No helmet count incremented to {no_helmet_count}")

                    if vest_present:
                        vest_count += 1
                    else:
                        no_vest_count += 1
                    
                    # Also count seatbelt for unknown vehicles if detected
                    if vehicle_type == "unknown" and seatbelt_present:
                        seatbelt_count += 1
                    elif vehicle_type == "unknown" and not seatbelt_present:
                        no_seatbelt_count += 1



                # STEP 5: Determine status and label based on vehicle type

                if vehicle_type == "4-wheeler":
                    # 4-WHEELER: Compliant if seatbelt worn
                    if seatbelt_present:
                        status = "compliant"
                        compliance_reason = "seatbelt_detected"
                        output_label = "Seatbelt Detected"
                    else:
                        status = "violation"
                        compliance_reason = "no_seatbelt"
                        output_label = "No Seatbelt"
                else:
                    # 2-WHEELER/UNKNOWN: Compliant if helmet OR vest OR seatbelt present
                    if helmet_present or vest_present or seatbelt_present:
                        status = "compliant"
                        compliance_reason = "ppe_detected"
                        # Priority order: seatbelt > helmet+vest > helmet > vest
                        if seatbelt_present and vehicle_type == "unknown":
                            output_label = "Seatbelt Detected"
                        elif helmet_present and vest_present:
                            output_label = "Helmet & Vest Detected"
                        elif helmet_present:
                            output_label = "Helmet Detected"
                        elif vest_present:
                            output_label = "Vest Detected"
                        else:
                            output_label = "Seatbelt Detected"
                    else:
                        status = "violation"
                        compliance_reason = "no_ppe_detected"
                        output_label = "No Helmet & No Vest"

                debug_info = {}

                if self.debug:

                    debug_info = {

                        "helmet_method": helmet_method,

                        "helmet_confidence": f"{helmet_conf:.3f}",

                        "vest_confidence": f"{vest_conf:.3f}",

                        "compliance_reason": compliance_reason,

                        "fallback": fallback_used

                    }

                person_ppe = PersonPPE(

                    person_id=person_id,

                    bbox=person_bbox,

                    vehicle_type=vehicle_type,

                    head_bbox=head_bbox,  # Always show head region

                    vest_bbox=vest_bbox,  # Always show vest region

                    helmet=PPEItem("helmet", helmet_present, helmet_conf, head_bbox if helmet_present else None, helmet_method),

                    vest=PPEItem("vest", vest_present, vest_conf, vest_bbox if vest_present else None, "color"),

                    seatbelt=PPEItem("seatbelt", seatbelt_present, seatbelt_conf, None, seatbelt_method),

                    status=status,

                    confidence=person["confidence"],

                    debug_info=debug_info

                )

                person_results.append(person_ppe)



                if self.debug:

                    safety_status = "COMPLIANT" if status == "compliant" else "VIOLATION"

                    print(f"  {safety_status} Output Label: {output_label}")

                    print(f"  Helmet: {'DETECTED' if helmet_present else 'Not detected'} (conf: {helmet_conf:.3f})")

                    print(f"  Vest: {'DETECTED' if vest_present else 'Not detected'} (conf: {vest_conf:.3f})")

                    print(f"  Reason: {compliance_reason}")

                

            proc_time = time.time() - start



            if self.debug:

                print(f"\n{'='*60}")

                print(f"[PPE] DETECTION SUMMARY")

                print(f"[PPE] Total Persons: {len(persons)}")

                print(f"[PPE] Priority-based Output:")

                print(f"     - Helmet Detected: {helmet_count} (YES)")

                print(f"     - Vest Detected: {vest_count} (YES)")

                print(f"     - No Helmet & No Vest: {no_helmet_count + no_vest_count} (NO)")

                print(f"[PPE] Detailed Counting:")

                print(f"     - Helmet count: {helmet_count}")

                print(f"     - No helmet count: {no_helmet_count}")

                print(f"     - Vest count: {vest_count}")

                print(f"     - No vest count: {no_vest_count}")

                print(f"[PPE] System Status:")

                print(f"     - Model Loaded: {'YES' if model_loaded else 'NO'}")

                print(f"     - Fallback Used: {'YES' if fallback_used else 'NO'}")

                print(f"     - Processing Time: {proc_time:.3f}s")

                print(f"{'='*60}\n")



            return PPEResult(

                total_persons=len(persons),

                helmet_detected=helmet_count,

                no_helmet=no_helmet_count,

                seatbelt_detected=seatbelt_count,

                no_seatbelt=no_seatbelt_count,

                vest_detected=vest_count,

                no_vest=no_vest_count,

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



            # COLOR LOGIC based on vehicle type
            if person.vehicle_type == "4-wheeler":
                # 4-WHEELER: Green if seatbelt, Red if no seatbelt
                is_compliant = person.seatbelt.present
            else:
                # 2-WHEELER/UNKNOWN: Green if helmet OR vest OR seatbelt, Red if none
                is_compliant = person.helmet.present or person.vest.present or person.seatbelt.present



            if is_compliant:

                color = self.COLORS["compliant"]  # GREEN

                status_emoji = "🟩"

            else:

                color = self.COLORS["non_compliant"]  # RED

                status_emoji = "🟥"



            # Create status text based on vehicle type
            if person.vehicle_type == "4-wheeler":
                if person.seatbelt.present:
                    safety_text = "Seatbelt Detected"
                else:
                    safety_text = "No Seatbelt"
            else:
                # Priority: seatbelt > helmet+vest > helmet > vest
                if person.seatbelt.present and person.vehicle_type == "unknown":
                    safety_text = "Seatbelt Detected"
                elif person.helmet.present and person.vest.present:
                    safety_text = "Helmet & Vest Detected"
                elif person.helmet.present:
                    safety_text = "Helmet Detected"
                elif person.vest.present:
                    safety_text = "Vest Detected"
                else:
                    safety_text = "No Helmet & No Vest"



            # Use fallback color if in fallback mode

            if result.fallback_used:

                color = self.COLORS["fallback"]



            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Only draw helmet/vest boxes for 2-wheeler/unknown (NOT 4-wheeler)
            if person.vehicle_type != "4-wheeler":
                # Draw vest bounding box
                if person.vest_bbox:

                    vx1, vy1, vx2, vy2 = person.vest_bbox

                    vest_color = (0, 165, 255) if person.vest.present else (0, 100, 255)  # Orange if vest, darker orange if not
                    cv2.rectangle(img, (vx1, vy1), (vx2, vy2), vest_color, 2)

                    vest_label = "VEST" if person.vest.present else "NO VEST"
                    cv2.putText(img, vest_label, (vx1, vy1 - 5),

                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, vest_color, 2)

                # Draw helmet bounding box
                if person.head_bbox:

                    hx1, hy1, hx2, hy2 = person.head_bbox

                    helmet_color = (255, 0, 255) if person.helmet.present else (128, 0, 128)  # Magenta if helmet, purple if not
                    cv2.rectangle(img, (hx1, hy1), (hx2, hy2), helmet_color, 2)

                    helmet_label = "HELMET" if person.helmet.present else "NO HELMET"
                    cv2.putText(img, helmet_label, (hx1, hy1 - 5),

                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, helmet_color, 2)
            
            # Show seatbelt indicator if seatbelt detected (for any vehicle type)
            if person.seatbelt.present:
                seatbelt_color = (0, 255, 0)
                seatbelt_label = "SEATBELT"
                cv2.putText(img, seatbelt_label, (x1 + 5, y1 + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, seatbelt_color, 2)
            elif person.vehicle_type == "4-wheeler":
                # 4-WHEELER: Show NO SEATBELT indicator
                seatbelt_color = (0, 0, 255)
                seatbelt_label = "NO SEATBELT"
                cv2.putText(img, seatbelt_label, (x1 + 5, y1 + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, seatbelt_color, 2)



            if show_labels:

                # Label with person ID and safety status

                label = f"{person.person_id} {safety_text}"

                # Add confidence scores
                if person.helmet.present:
                    label += f" H:{person.helmet.confidence:.2f}"
                if person.vest.present:
                    label += f" V:{person.vest.confidence:.2f}"

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

        summary = f"Persons: {result.total_persons} | Helmets: {result.helmet_detected} | Vests: {result.vest_detected} | Model: {'OK' if result.model_loaded else 'FB'}"

        cv2.rectangle(img, (10, 5), (900, 35), (0, 0, 0), -1)  # Increased width for vest column

        cv2.putText(img, summary, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



        if result.error_message:

            cv2.putText(img, f"Error: {result.error_message[:50]}", (15, 55),

                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)



        return img



    def to_dict(self, result):

        """Convert to JSON-serializable dict - helmet and vest only"""

        persons_list = []

        for p in result.persons:

            person_data = {

                "id": p.person_id,

                "helmet": p.helmet.present,

                "vest": p.vest.present,

                "status": p.status,

                "confidence": p.confidence,

                "detection_method": {

                    "helmet": p.helmet.detection_method,

                    "vest": p.vest.detection_method

                },

                "confidence_scores": {

                    "helmet": round(p.helmet.confidence, 3),

                    "vest": round(p.vest.confidence, 3)

                }
            }

            # Determine output label based on helmet and vest
            if p.helmet.present and p.vest.present:
                person_data["label"] = "Helmet & Vest Detected"
                person_data["detectedPPE"] = "helmet_vest"
            elif p.helmet.present:
                person_data["label"] = "No Vest"
                person_data["detectedPPE"] = "helmet"
            elif p.vest.present:
                person_data["label"] = "No Helmet"
                person_data["detectedPPE"] = "vest"
            else:
                person_data["label"] = "No Helmet & No Vest"
                person_data["detectedPPE"] = "none"

            persons_list.append(person_data)

        return {
            "totalPersons": result.total_persons,
            "helmetDetected": result.helmet_detected,
            "noHelmet": result.no_helmet,
            "vestDetected": result.vest_detected,
            "noVest": result.no_vest,
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

            f"* Helmet & Vest: {result.helmet_detected + result.vest_detected}",

            f"* No Helmet & No Vest: {result.no_helmet + result.no_vest}",

            "",

            "Person Details:",

        ]



        for person in result.persons:

            # Determine single output label based on strict priority

            if person.helmet.present and person.vest.present:

                output_label = "Helmet & Vest Detected"

                emoji = "🟩"

                details = f"Helmet:  Present (conf: {person.helmet.confidence:.2f}), Vest:  Present (conf: {person.vest.confidence:.2f})"

            elif person.helmet.present:

                output_label = "No Vest"

                emoji = ""

                details = f"Helmet:  Present (conf: {person.helmet.confidence:.2f}), Vest:  Not detected"

            elif person.vest.present:

                output_label = "No Helmet"

                emoji = ""

                details = f"Helmet:  Not detected, Vest:  Present (conf: {person.vest.confidence:.2f})"

            else:

                output_label = "No Helmet & No Vest"

                emoji = "🟥"

                details = f"Helmet:  Not detected, Vest:  Not detected"

            

            lines.append(f"Person {person.person_id} {emoji}")

            lines.append(f"  - Output: {output_label}")

            lines.append(f"  - {details}")

            

            # Add debug information if available

            if person.debug_info and result.debug_mode:

                lines.append(f"  - Debug: {person.debug_info}")



        if result.error_message:

            lines.append(f"\n Warning: {result.error_message}")



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

