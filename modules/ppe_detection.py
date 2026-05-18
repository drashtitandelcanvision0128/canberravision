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
    """Load PPE classes directly from model"""
    try:
        from ultralytics import YOLO
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'models', 'best_ppe.pt')
        if os.path.exists(model_path):
            m = YOLO(model_path)
            names = list(m.names.values())
            print(f"[INFO] Classes loaded from model: {names}")
            return names, len(names)
    except Exception as e:
        print(f"[ERROR] Model se classes load nahi hui: {e}")
    # Fallback
    return ['boots','gloves','goggles','helmet','mask',
            'no_boots','no_gloves','no_goggle','no_helmet',
            'no_mask','no_vest','person','vest'], 13

# Load dataset classes

PPE_CLASSES, NUM_PPE_CLASSES = load_ppe_classes()

print(f"[INFO] PPE Dataset loaded: {PPE_CLASSES} ({NUM_PPE_CLASSES} classes)")


def _normalize_yolo_class_name(name: str) -> str:

    return str(name).lower().replace('-', '').replace(' ', '').replace('_', '')


# Main PPE forward: only these classes (fewer competing boxes in NMS).
# Names match normalized labels from your data.yaml (e.g. no_helmet → nohelmet).
# Other datasets sometimes use Hardhat/Safety Vest — add here only if your weights use those strings.
_CORE_INFERENCE_CLASS_KEYS = frozenset({
    'helmet', 'nohelmet',
    'vest', 'novest',
    'mask', 'nomask',
    'person',
})


def core_inference_class_ids_from_names(model_names: Dict) -> Optional[List[int]]:

    if not model_names:
        return None
    ids = sorted({int(cid) for cid, cname in model_names.items()
                  if _normalize_yolo_class_name(str(cname)) in _CORE_INFERENCE_CLASS_KEYS})
    return ids if ids else None


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

    gloves_bbox: Optional[Tuple[int, int, int, int]] = None

    goggles_bbox: Optional[Tuple[int, int, int, int]] = None

    mask_bbox: Optional[Tuple[int, int, int, int]] = None

    helmet: PPEItem = field(default_factory=lambda: PPEItem("helmet", False, 0.0))

    vest: PPEItem = field(default_factory=lambda: PPEItem("vest", False, 0.0))

    seatbelt: PPEItem = field(default_factory=lambda: PPEItem("seatbelt", False, 0.0))

    gloves: PPEItem = field(default_factory=lambda: PPEItem("gloves", False, 0.0))

    goggles: PPEItem = field(default_factory=lambda: PPEItem("goggles", False, 0.0))

    mask: PPEItem = field(default_factory=lambda: PPEItem("mask", False, 0.0))

    boots: PPEItem = field(default_factory=lambda: PPEItem("boots", False, 0.0))

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

    gloves_detected: int = 0

    no_gloves: int = 0

    goggles_detected: int = 0

    no_goggles: int = 0

    mask_detected: int = 0

    no_mask: int = 0

    boots_detected: int = 0

    no_boots: int = 0

    accuracy_score: float = 0.0

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

        ("white", [0, 0, 100], [180, 100, 255]),  # Wider white range - improved for better detection

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

    def __init__(self, model_path=None, device=None, debug=False, auto_recovery=True, detection_mode="all"):

        # Use PPE dataset model path if not provided or if using standard YOLO models

        # For PPE detection, always prefer trained PPE model

        # detection_mode: "all" (default), "helmet_vest" (only helmet & vest), "mask" (only mask)
        self.detection_mode = detection_mode
        print(f"[PPE] Initializing with model_path: {repr(model_path)}, detection_mode: {detection_mode}")

        # Normalize model_path for comparison

        normalized_path = str(model_path).strip().lower() if model_path else ""

        if normalized_path.endswith('.pt'):

            normalized_path = normalized_path[:-3]

        standard_models = ['yolov8n', 'yolo26n', 'yolov8m', 'yolov8s', 'yolo11n', 'yolo11s']

        is_ppe_choice = 'ppe' in normalized_path or 'best' in normalized_path

        is_standard_model = normalized_path in standard_models or model_path is None

        if is_standard_model:

            print(f"[PPE] Standard YOLO model detected ({model_path}), switching to PPE model")

            # Try to find trained PPE model first

            ppe_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_ppe.pt')

            print(f"[PPE] Checking PPE model at: {ppe_model_path} (exists={os.path.exists(ppe_model_path)})")

            if os.path.exists(ppe_model_path):

                model_path = ppe_model_path

                print(f"[PPE] Using trained PPE model: {model_path}")

            else:

                # Try absolute path as fallback
                abs_ppe_path = r"c:\Users\Sensepart\canberravision\models\best_ppe.pt"
                print(f"[PPE] Relative path failed, trying absolute: {abs_ppe_path} (exists={os.path.exists(abs_ppe_path)})")
                if os.path.exists(abs_ppe_path):
                    model_path = abs_ppe_path
                    print(f"[PPE] Using trained PPE model (absolute): {model_path}")
                else:
                    # Fallback to best_ppe.pt for PPE detection
                    model_path = "best_ppe.pt"
                    print(f"[PPE] PPE model not found, using fallback: {model_path}")

        elif is_ppe_choice:

            # User selected "best (PPE)" from dropdown - resolve to actual path

            print(f"[PPE] PPE model choice detected ({model_path}), resolving path")

            ppe_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_ppe.pt')

            if os.path.exists(ppe_model_path):

                model_path = ppe_model_path

                print(f"[PPE] Using trained PPE model: {model_path}")

            else:

                abs_ppe_path = r"c:\Users\Sensepart\canberravision\models\best_ppe.pt"
                if os.path.exists(abs_ppe_path):
                    model_path = abs_ppe_path
                    print(f"[PPE] Using trained PPE model (absolute): {model_path}")
                else:
                    model_path = "yolov8n.pt"
                    print(f"[PPE] PPE model file not found, using fallback: {model_path}")

        self.model_path = model_path

        self.model = None

        # Separate person detection model (PPE model doesn't reliably detect Person class)

        self.person_model = None

        _repo = os.path.dirname(os.path.dirname(__file__))
        _yolo_n = os.path.join(_repo, "models", "yolov8n.pt")
        # Prefer repo-local weights so person detection works regardless of cwd (e.g. apps/)
        self.person_model_path = _yolo_n if os.path.exists(_yolo_n) else "yolov8n.pt"

        self.device = device or self._get_device()

        self.debug = debug

        self.auto_recovery = auto_recovery

        self.model_load_attempts = 0

        self.max_load_attempts = 3

        # Optimized thresholds for balanced accuracy and detection

        # Single person box approach - detect PPE within person regions

        # Person (COCO) — balanced for site/crowd scenes (reduces machinery/partial FPs)
        self.person_threshold = 0.22
        self.person_fallback_threshold = 0.14  # second pass when primary finds nobody
        self.person_dedup_iou = 0.50
        self.person_center_merge_ratio = 0.28  # merge boxes whose centers are this close (× avg height)
        self.person_min_height_frac = 0.045  # min box height vs frame (keeps distant workers)
        self.person_min_area_frac = 0.0012  # min box area vs frame
        self.person_min_aspect = 0.12  # width/height — allow tall/narrow
        self.person_max_aspect = 1.05  # reject wider-than-tall (machinery, horizontal blobs)
        self.ppe_threshold = 0.06  # Lower confidence for PPE model inference in video
        self.helmet_threshold = 0.05  # More sensitive helmet fallback threshold for video
        self.fallback_threshold = 0.12
        # When the PPE model reports "helmet" on the head ROI, reject unless confidence is high enough
        # if hair/styled-hair or skin-dominated head cues fire (reduces hair/shine/bald false positives).
        self.model_helmet_min_conf_if_hair = 0.52
        self.model_helmet_min_conf_if_skin = 0.58
        self.model_helmet_skin_color_override_conf = 0.40  # min color score to keep helmet when skin-like ROI
        # Raw YOLO "helmet/hardhat" below this is treated as uncertain — skip assigning helmet_detected
        # (avoids 0.27–0.35 hair FPs when texture heuristics miss). Lower if distant helmets disappear.
        self.model_helmet_accept_min_conf = 0.42
        # Vest: color fallback must see real hi-vis; weak model-only vest on street clothes is rejected
        self.vest_hivis_min_saturation = 95
        self.vest_color_min_coverage = 0.34
        self.vest_color_solid_coverage = 0.52
        self.vest_color_with_strip_coverage = 0.26
        self.model_vest_min_conf_without_color = 0.58

        # Video smoothing state
        self._ppe_history = {} # person_id -> history of results
        self._max_history = 5  # Number of frames to smooth over

        self._ensure_model_loaded()

        print(
            f"[PPE] Initialized - person_threshold={self.person_threshold}, "
            f"helmet_threshold={self.helmet_threshold}, "
            f"model_helmet_accept_min_conf={self.model_helmet_accept_min_conf}, fallback_enabled=True"
        )

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

                # Also load separate person detection model if PPE model is being used

                if self.person_model is None and ("best.pt" in self.model_path or "best_ppe.pt" in self.model_path):

                    try:

                        print("[PPE] Loading separate person detection model (yolov8n)...")

                        self.person_model = YOLO(self.person_model_path)

                        self.person_model.to(self.device)

                        print("[PPE] Person detection model loaded successfully")

                    except Exception as pe:

                        print(f"[PPE-WARNING] Person model load failed: {pe}")

                        self.person_model = None

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

        """Extract head region - more robust fallback based on height proportions"""

        x1, y1, x2, y2 = person_bbox
        height = y2 - y1
        width = x2 - x1

        # Use fixed proportions for robust head region estimation
        # Head is top 20-25% depending on distance (increased for better helmet detection)
        head_h_ratio = 0.22 if height > 400 else 0.28 # Increased from 0.18/0.22 for larger head ROI
        head_top = y1
        head_bottom = y1 + int(height * head_h_ratio)
        
        # Center head horizontally (increased width for better helmet detection)
        head_width = int(width * 0.5)  # Increased from 0.4 to 0.5
        head_x_center = x1 + int(width / 2)
        head_x1 = head_x_center - int(head_width / 2)
        head_x2 = head_x_center + int(head_width / 2)

        return (max(x1, head_x1), max(y1, head_top), min(x2, head_x2), min(y2, head_bottom))

    def _select_helmet_draw_bbox(self, frame, person_bbox, head_anchor_bbox, model_helmet_bbox):
        """Use YOLO helmet box on the input frame when it aligns with person/head; else head_anchor."""
        if model_helmet_bbox is None or frame is None:
            return head_anchor_bbox
        fh, fw = frame.shape[:2]
        try:
            x1, y1, x2, y2 = (int(round(c)) for c in model_helmet_bbox)
        except (TypeError, ValueError):
            return head_anchor_bbox
        x1 = max(0, min(x1, fw - 1))
        x2 = max(0, min(x2, fw - 1))
        y1 = max(0, min(y1, fh - 1))
        y2 = max(0, min(y2, fh - 1))
        if x2 <= x1 or y2 <= y1:
            return head_anchor_bbox
        px1, py1, px2, py2 = person_bbox
        person_area = max(1, (px2 - px1) * (py2 - py1))
        model_area = max(1, (x2 - x1) * (y2 - y1))
        if model_area > 0.72 * person_area:
            return head_anchor_bbox
        hx1, hy1, hx2, hy2 = head_anchor_bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        margin_w = (px2 - px1) * 0.15
        margin_top = (py2 - py1) * 0.40
        if not (px1 - margin_w <= cx <= px2 + margin_w and py1 - margin_top <= cy <= py2 + (py2 - py1) * 0.08):
            return head_anchor_bbox
        ix1 = max(x1, hx1)
        iy1 = max(y1, hy1)
        ix2 = min(x2, hx2)
        iy2 = min(y2, hy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter / model_area < 0.10:
            return head_anchor_bbox
        return (x1, y1, x2, y2)

    def get_face_region(self, person_bbox, frame=None):
        """Extract face/mouth region for mask detection - adaptive based on actual face position"""
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1
        width = x2 - x1
        
        # Try to detect face using skin color in upper portion
        if frame is not None:
            try:
                # Look at upper 50% of person (where face usually is) - larger for better detection
                head_region_h = int(height * 0.50)
                head_roi = frame[y1:y1+head_region_h, x1:x2]
                
                if head_roi.size > 0:
                    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
                    gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
                    
                    # Detect skin color - expanded range for better detection in various conditions
                    lower_skin = np.array([0, 10, 40])
                    upper_skin = np.array([25, 180, 255])
                    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                    
                    # Also detect darker skin tones
                    lower_dark = np.array([0, 0, 20])
                    upper_dark = np.array([180, 100, 100])
                    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
                    
                    # Combine both masks
                    skin_mask = cv2.bitwise_or(skin_mask, dark_mask)
                    
                    # Apply slight blur for noise reduction
                    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
                    
                    # Find skin region center - use larger threshold for small images
                    min_area = max(50, (x2-x1) * (y2-y1) * 0.01)  # At least 1% of person area
                    if np.sum(skin_mask > 0) > min_area:
                        moments = cv2.moments(skin_mask)
                        if moments["m00"] > min_area * 10:
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])
                            
                            # Face region around detected skin center
                            face_h = int(height * 0.15)
                            face_w = int(width * 0.4)
                            
                            fx1 = max(x1, x1 + cx - face_w//2)
                            fy1 = max(y1, y1 + cy - face_h//2)
                            fx2 = min(x2, x1 + cx + face_w//2)
                            fy2 = min(y1 + head_region_h, y1 + cy + face_h//2)
                            
                            if fx2 > fx1 and fy2 > fy1:
                                return (fx1, fy1, fx2, fy2)
            except:
                pass
        
        # Fallback: fixed percentage (20-45% of person height) - larger range for small images
        face_top = max(y1, y1 + int(height * 0.20))
        face_bottom = min(y2, y1 + int(height * 0.45))
        
        return (x1, face_top, x2, face_bottom)

    def get_mask_region_between_helmet_vest(self, person_bbox, head_bbox=None, vest_bbox=None):

        """Extract mask region - face area between helmet (above) and vest (below)"""

        x1, y1, x2, y2 = person_bbox
        height = y2 - y1
        width = x2 - x1

        # If helmet bbox is available, use its bottom as mask top reference
        # If vest bbox is available, use its top as mask bottom reference
        if head_bbox and vest_bbox:
            hx1, hy1, hx2, hy2 = head_bbox
            vx1, vy1, vx2, vy2 = vest_bbox
            
            # Mask should be below helmet and above vest
            mask_top = hy2  # Helmet bottom
            mask_bottom = vy1  # Vest top
            
            # Ensure there's space between them
            if mask_bottom <= mask_top:
                # Fallback to fixed percentages if helmet and vest overlap
                mask_top = y1 + int(height * 0.12)
                mask_bottom = y1 + int(height * 0.35)
            
            # Center horizontally between helmet and vest
            mask_x_center = (hx1 + hx2 + vx1 + vx2) // 4
            mask_width = int(width * 0.35)
            mask_x1 = mask_x_center - int(mask_width / 2)
            mask_x2 = mask_x_center + int(mask_width / 2)
            
        elif head_bbox:
            # Only helmet available - mask below helmet
            hx1, hy1, hx2, hy2 = head_bbox
            mask_top = hy2
            mask_bottom = y1 + int(height * 0.40)  # 40% from top
            
            mask_x_center = (hx1 + hx2) // 2
            mask_width = int(width * 0.35)
            mask_x1 = mask_x_center - int(mask_width / 2)
            mask_x2 = mask_x_center + int(mask_width / 2)
            
        elif vest_bbox:
            # Only vest available - mask above vest
            vx1, vy1, vx2, vy2 = vest_bbox
            mask_top = y1 + int(height * 0.10)  # 10% from top
            mask_bottom = vy1
            
            mask_x_center = (vx1 + vx2) // 2
            mask_width = int(width * 0.35)
            mask_x1 = mask_x_center - int(mask_width / 2)
            mask_x2 = mask_x_center + int(mask_width / 2)
            
        else:
            # No helmet or vest - fallback to fixed percentages
            mask_top = y1 + int(height * 0.12)
            mask_bottom = y1 + int(height * 0.35)
            
            mask_width = int(width * 0.35)
            mask_x_center = x1 + int(width / 2)
            mask_x1 = mask_x_center - int(mask_width / 2)
            mask_x2 = mask_x_center + int(mask_width / 2)

        return (max(x1, mask_x1), max(y1, mask_top), min(x2, mask_x2), min(y2, mask_bottom))

    def get_vest_region(self, person_bbox, frame=None, head_bbox=None):

        """Torso band for vest — always below head, never on face."""

        x1, y1, x2, y2 = person_bbox
        height = max(1, y2 - y1)
        width = x2 - x1

        if head_bbox is not None:
            vest_top = head_bbox[3] + max(2, int(height * 0.02))
        else:
            vest_top = y1 + int(height * 0.22)
        vest_top = max(y1 + int(height * 0.18), vest_top)
        vest_bottom = y1 + int(height * 0.72)

        vest_width = int(width * 0.8)
        vest_x_center = x1 + int(width / 2)
        vest_x1 = vest_x_center - int(vest_width // 2)
        vest_x2 = vest_x_center + int(vest_width // 2)

        return (max(x1, vest_x1), max(y1, vest_top), min(x2, vest_x2), min(y2, vest_bottom))

    def _normalize_vest_bbox(self, vest_bbox, person_bbox, head_bbox=None):
        """Clip model vest boxes to torso; replace head/face false positives with torso region."""
        if person_bbox is None:
            return vest_bbox
        if vest_bbox is None:
            return self.get_vest_region(person_bbox, None, head_bbox)

        px1, py1, px2, py2 = person_bbox
        ph = max(1, py2 - py1)
        pw = max(1, px2 - px1)

        if head_bbox is not None:
            torso_top = head_bbox[3] + max(2, int(ph * 0.02))
        else:
            torso_top = py1 + int(ph * 0.22)
        torso_top = max(py1 + int(ph * 0.18), torso_top)
        torso_bottom = py1 + int(ph * 0.75)

        vx1, vy1, vx2, vy2 = vest_bbox
        vcy = (vy1 + vy2) / 2

        if vcy < torso_top or vy2 <= torso_top + 2:
            if self.debug:
                print(f"[PPE-DEBUG] Vest bbox on head/face — using torso region (was {vest_bbox})")
            return self.get_vest_region(person_bbox, None, head_bbox)

        cy1 = max(int(vy1), torso_top)
        cy2 = min(int(vy2), torso_bottom)
        cx1 = max(int(vx1), px1 + int(pw * 0.08))
        cx2 = min(int(vx2), px2 - int(pw * 0.08))
        if cy2 - cy1 < max(8, int(ph * 0.08)) or cx2 <= cx1:
            return self.get_vest_region(person_bbox, None, head_bbox)
        return (cx1, cy1, cx2, cy2)

    def _get_hand_region(self, person_bbox, frame):

        """Extract hand/arm region for glove detection"""

        x1, y1, x2, y2 = person_bbox

        height = y2 - y1

        width = x2 - x1

        # Hands are typically at mid-torso to lower body

        # Left side of person (x1 to x1 + width*0.3) and right side (x2 - width*0.3 to x2)

        # Check both sides

        h, w = frame.shape[:2]

        # Left hand region - extend outward from left side

        left_x1 = max(0, int(x1 - width * 0.2))

        left_y1 = int(y1 + height * 0.5)  # Mid torso

        left_x2 = int(x1 + width * 0.3)

        left_y2 = int(y1 + height * 0.8)

        # Right hand region - extend outward from right side

        right_x1 = int(x2 - width * 0.3)

        right_y1 = int(y1 + height * 0.5)

        right_x2 = min(w, int(x2 + width * 0.2))

        right_y2 = int(y1 + height * 0.8)

        # Return larger region that covers both potential hand positions

        return (left_x1, left_y1, right_x2, right_y2)

    def _get_eye_region(self, person_bbox, frame, head_bbox=None):

        """Extract eye region for goggles detection"""

        x1, y1, x2, y2 = person_bbox

        height = y2 - y1

        width = x2 - x1

        if head_bbox is not None:

            hx1, hy1, hx2, hy2 = head_bbox

            # Eyes are in upper half of head region

            eye_y1 = hy1 + int((hy2 - hy1) * 0.3)

            eye_y2 = hy1 + int((hy2 - hy1) * 0.6)

            eye_x1 = hx1

            eye_x2 = hx2

        else:

            # Fallback: eyes are in upper head area (~15-25% from top)

            eye_y1 = y1 + int(height * 0.15)

            eye_y2 = y1 + int(height * 0.28)

            eye_x1 = int(x1 + width * 0.2)

            eye_x2 = int(x2 - width * 0.2)

        h, w = frame.shape[:2]

        eye_x1 = max(0, eye_x1)

        eye_y1 = max(0, eye_y1)

        eye_x2 = min(w, eye_x2)

        eye_y2 = min(h, eye_y2)

        if eye_x2 > eye_x1 and eye_y2 > eye_y1:

            return (eye_x1, eye_y1, eye_x2, eye_y2)

        return None

    def _get_face_region(self, person_bbox, frame, head_bbox=None):

        """Extract face region (below eyes) for mask detection"""

        x1, y1, x2, y2 = person_bbox

        height = y2 - y1

        width = x2 - x1

        if head_bbox is not None:

            hx1, hy1, hx2, hy2 = head_bbox

            # Face below eyes is lower 60% of head

            face_y1 = hy1 + int((hy2 - hy1) * 0.4)

            face_y2 = hy2

            face_x1 = hx1

            face_x2 = hx2

        else:

            # Fallback: mask covers nose/mouth area (~25-40% from top)

            face_y1 = y1 + int(height * 0.25)

            face_y2 = y1 + int(height * 0.45)

            face_x1 = int(x1 + width * 0.25)

            face_x2 = int(x2 - width * 0.25)

        h, w = frame.shape[:2]

        face_x1 = max(0, face_x1)

        face_y1 = max(0, face_y1)

        face_x2 = min(w, face_x2)

        face_y2 = min(h, face_y2)

        if face_x2 > face_x1 and face_y2 > face_y1:

            return (face_x1, face_y1, face_x2, face_y2)

        return None

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

    def detect_gloves_by_color(self, hand_roi, threshold=0.25):

        """Detect gloves by color - white, blue, yellow work gloves"""

        if hand_roi.size == 0:

            return False, 0.0, "none"

        try:

            hsv = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)

            # Glove colors: white, blue, yellow, orange, red, black

            GLOVE_COLORS = [

                ("white", [0, 0, 180], [180, 40, 255]),

                ("blue", [100, 50, 50], [130, 255, 255]),

                ("yellow", [20, 100, 100], [35, 255, 255]),

                ("orange", [10, 100, 100], [25, 255, 255]),

                ("red1", [0, 100, 100], [10, 255, 255]),

                ("red2", [160, 100, 100], [180, 255, 255]),

                ("black", [0, 0, 0], [180, 50, 60]),

            ]

            best_conf = 0.0

            best_color = "none"

            for color_name, lower, upper in GLOVE_COLORS:

                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:

                    area = cv2.contourArea(cnt)

                    roi_area = hand_roi.shape[0] * hand_roi.shape[1]

                    if 0.1 * roi_area < area < 0.8 * roi_area:

                        coverage = area / roi_area

                        conf = min(coverage * 2.5, 0.7)

                        if conf > best_conf:

                            best_conf = conf

                            best_color = color_name

            if self.debug and best_conf > 0:

                print(f"[PPE-DEBUG] Glove color: {best_color}, conf: {best_conf:.2f}")

            return best_conf >= threshold, best_conf, best_color

        except Exception as e:

            if self.debug:

                print(f"[PPE-DEBUG] Glove detection error: {e}")

            return False, 0.0, "error"

    def detect_goggles_by_color(self, eye_roi, threshold=0.25, frame_roi=None):

        """Detect safety goggles by reflective/glass properties"""

        if frame_roi is None or frame_roi.size == 0:
            return False, 0.0, "unknown"

        try:
            gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

            # Goggles characteristics: clear/transparent with frame

            # Look for high contrast edges (frame) and uniform clear area (lens)

            edges = cv2.Canny(gray, 50, 150)

            edge_density = np.sum(edges > 0) / (eye_roi.shape[0] * eye_roi.shape[1])

            # Check for uniform brightness (clear lens)

            brightness_std = np.std(gray)

            brightness_mean = np.mean(gray)

            # Goggles: moderate edge density, uniform brightness (clear lens)

            is_clear_lens = 80 < brightness_mean < 200 and brightness_std < 50

            has_frame = 0.05 < edge_density < 0.25

            conf = 0.0

            if is_clear_lens and has_frame:

                conf = 0.5 + (0.25 - abs(edge_density - 0.15)) * 2

                conf = min(conf, 0.7)

            if self.debug and conf > 0:

                print(f"[PPE-DEBUG] Goggles: clear={is_clear_lens}, frame={has_frame}, conf: {conf:.2f}")

            return conf >= threshold, conf, "goggles" if conf >= threshold else "none"

        except Exception as e:

            if self.debug:

                print(f"[PPE-DEBUG] Goggles detection error: {e}")

            return False, 0.0, "error"

    def detect_mask_by_color(self, face_roi, threshold=0.25):

        """Detect face mask by color and position (covers mouth/nose)"""

        if face_roi.size == 0:

            return False, 0.0, "none"

        try:

            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

            # Mask colors: blue surgical, white N95, black cloth

            MASK_COLORS = [

                ("blue", [90, 50, 50], [130, 255, 200]),

                ("white", [0, 0, 150], [180, 50, 255]),

                ("black", [0, 0, 0], [180, 50, 80]),

                ("green", [40, 50, 50], [80, 255, 200]),

            ]

            best_conf = 0.0

            best_color = "none"

            for color_name, lower, upper in MASK_COLORS:

                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

                # Mask should cover lower face (horizontal band)

                h, w = mask.shape

                lower_face = mask[int(h*0.3):, :]

                upper_face = mask[:int(h*0.3), :]

                lower_coverage = np.sum(lower_face > 0) / lower_face.size

                upper_coverage = np.sum(upper_face > 0) / upper_face.size

                # Mask covers lower face more than upper face

                if lower_coverage > 0.3 and upper_coverage < 0.2:

                    conf = min(lower_coverage * 1.5, 0.7)

                    if conf > best_conf:

                        best_conf = conf

                        best_color = color_name

            if self.debug and best_conf > 0:

                print(f"[PPE-DEBUG] Mask color: {best_color}, conf: {best_conf:.2f}")

            return best_conf >= threshold, best_conf, best_color

        except Exception as e:

            if self.debug:

                print(f"[PPE-DEBUG] Mask detection error: {e}")

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

            # Threshold: needs at least 8% yellow/orange pixels (vest, equipment)

            # Lowered from 0.15 to catch more construction workers

            if (yellow_pixels + orange_pixels) / total_pixels > 0.08:

                features += 2  # Strong indicator - construction vest detected

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

                # Threshold: needs at least 10% of upper region to be hard hat colors

            # Lowered from 0.20 to catch more hard hats

                if (white_pixels + yellow_hat_pixels) / (w * (h//3)) > 0.10:

                    features += 1  # Hard hat detected

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

        is_ppe_model = "PPE" in self.model_path or "best.pt" in self.model_path or "best_ppe.pt" in self.model_path

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

            model_names = self.model.names if hasattr(self.model, 'names') and self.model.names else {}

            helmet_cls_ids = sorted({
                int(cid) for cid, cname in model_names.items()
                if _normalize_yolo_class_name(str(cname)) in ('helmet', 'nohelmet')
            })

            pred_h = dict(conf=0.1, iou=0.45, device=self.device, verbose=False)
            if helmet_cls_ids:
                pred_h["classes"] = helmet_cls_ids

            results = self.model(head_region, **pred_h)

            helmet_detected = False

            max_conf = 0.0

            detected_class = "unknown"

            for result in results:

                if result.boxes is not None:

                    for box in result.boxes:

                        conf = float(box.conf[0].cpu().numpy())

                        cls = int(box.cls[0].cpu().numpy())

                        class_name = model_names.get(cls, PPE_CLASSES[cls] if cls < len(PPE_CLASSES) else "unknown")

                        # Check for helmet or no-helmet classes

                        # Support both old names (helmet/no-helmet) and new dataset names (Hardhat/NO-Hardhat)

                        class_lower = class_name.lower().replace('-', '').replace(' ', '').replace('_', '')

                        is_helmet_class = class_lower in ['helmet', 'hardhat']

                        is_no_helmet_class = class_lower in ['nohelmet', 'nohardhat']

                        if is_helmet_class and conf > max_conf:

                            helmet_detected = True

                            max_conf = conf

                            detected_class = class_name

                        elif is_no_helmet_class and conf > max_conf:

                            helmet_detected = False

                            max_conf = conf

                            detected_class = class_name

            if self.debug and max_conf > 0:

                print(f"[PPE-DEBUG] Direct PPE detection: {detected_class} (conf: {max_conf:.2f})")

            # Apply threshold - lowered for better detection

            threshold = self.helmet_threshold * 0.6  # 40% more lenient for direct PPE model

            if helmet_detected and max_conf >= threshold:

                # VALIDATION: Check if helmet is actually worn, not held

                person_bbox = getattr(self, '_current_person_bbox', None)

                if person_bbox is not None:

                    is_worn, worn_conf = self._validate_helmet_position(frame, head_bbox, person_bbox)

                    if self.debug:

                        print(f"[PPE-DEBUG] Direct PPE helmet validation: max_conf={max_conf:.2f}, worn_conf={worn_conf:.2f}, is_worn={is_worn}")

                    # Require stronger validation for low-confidence helmets
                    if max_conf < 0.35 and (not is_worn or worn_conf < 0.65):

                        if self.debug:

                            print(f"[PPE-DEBUG] Weak direct helmet rejected: low confidence and poor head validation")

                        return False, max_conf * worn_conf, f"direct_ppe_rejected_low_conf"

                    if not is_worn:

                        if self.debug:

                            print(f"[PPE-DEBUG] HELMET REJECTED: Not worn on head (held in hand)")

                        return False, max_conf * worn_conf, f"direct_ppe_held"

                return True, max_conf, f"direct_ppe_helmet"

            elif not helmet_detected and max_conf >= threshold:

                return False, max_conf, f"direct_ppe_no_helmet"

            else:

                if self.debug:
                    print(f"[PPE-DEBUG] Direct PPE model inconclusive: helmet_detected={helmet_detected}, max_conf={max_conf:.2f}, threshold={threshold:.2f}. Falling back to traditional detection.")
                    print(f"[PPE-DEBUG] head_bbox={head_bbox}, frame_shape={frame.shape}")
                
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

                    # PPE-trained model doesn't have COCO vehicle classes - skip vehicle detection

                    is_ppe_model = "PPE" in self.model_path or "best.pt" in self.model_path or "best_ppe.pt" in self.model_path

                    if is_ppe_model:

                        # PPE model: no vehicle classes available, use image analysis only

                        if self.debug:

                            print("[PPE-DEBUG] PPE model - skipping vehicle detection, using image analysis")

                    else:

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

                # Require at least 7 strong car features to classify as 4-wheeler

                # Raised from 5 to prevent false positives from safety vest straps

                if car_features >= 7:

                    if self.debug:

                        print(f"[PPE-DEBUG] Car features detected ({car_features}) - 4-WHEELER")

                    return "4-wheeler"

            # NOTE: Removed seatbelt-based 4-wheeler assumption.

            # Seatbelt detection alone is too unreliable (false positives from

            # safety harness straps, diagonal lines, etc.) and was causing

            # construction workers to be wrongly classified as 4-wheeler drivers.

            # Only actual vehicle detection or car interior features should

            # classify someone as 4-wheeler.

            # Default to unknown (worker or unclear)

            if self.debug:

                print("[PPE-DEBUG] No vehicle or worker PPE detected - returning UNKNOWN")

            return "unknown"

        except Exception as e:

            if self.debug:

                print(f"[PPE-DEBUG] Vehicle detection error: {e}")

            return "unknown"

    def _get_ppe_detections_near_person(self, person_bbox):
        """Get PPE detections for THIS specific person only - closest person wins."""

        px1, py1, px2, py2 = person_bbox
        person_w = px2 - px1
        person_h = py2 - py1

        # Skip mask detection if mode is "helmet_vest" only
        include_mask = self.detection_mode != "helmet_vest"

        result = {
            'helmet_detected': False, 'helmet_conf': 0.0, 'helmet_class': '',
            'helmet_bbox': None, 'no_hardhat': False, 'no_hardhat_conf': 0.0,
            'vest_detected': False, 'vest_conf': 0.0, 'vest_class': '',
            'vest_bbox': None, 'no_safetyvest': False, 'no_safetyvest_conf': 0.0,
            'no_safetyvest_bbox': None,
            'mask_detected': False, 'mask_conf': 0.0, 'mask_class': '',
            'mask_bbox': None, 'no_mask': False, 'no_mask_conf': 0.0,
            'no_mask_bbox': None
        }

        ppe_detections = getattr(self, '_last_ppe_detections', [])
        if not ppe_detections:
            return result

        if self.debug:
            print(f"[PPE-DEBUG] Found {len(ppe_detections)} total PPE items in frame")
        
        for det in ppe_detections:
            dx1, dy1, dx2, dy2 = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            class_lower = class_name.lower().replace('-', '').replace(' ', '').replace('_', '')
            if not include_mask and class_lower in ['mask', 'nomask']:
                continue

            det_cx = (dx1 + dx2) / 2
            det_cy = (dy1 + dy2) / 2

            if self.debug:
                print(f"[PPE-DEBUG] Checking PPE: {class_name} at ({dx1},{dy1},{dx2},{dy2}) vs person at ({px1},{py1},{px2},{py2})")

            # ============================================
            # STEP 1: Near person check (margin)
            # ============================================
            # Define item type flags for margin logic
            is_mask_class = class_lower in ['mask', 'nomask']
            is_head_item = class_lower in ['helmet', 'hardhat', 'nohelmet', 'nohardhat']
            is_vest_item = class_lower in ['vest', 'safetyvest', 'novest', 'nosafetyvest']
            
            # Strict margins for items that must be ON the person
            if is_mask_class:
                # Mask must be in the upper middle area
                margin_w = person_w * 0.18
                margin_h = person_h * 0.12
            elif is_head_item:
                # Helmet can sit above/beside head; generous margin for small/crowded person boxes
                margin_w = person_w * 0.38
                margin_h = person_h * 0.38
            elif is_vest_item:
                # Vest must be in the torso area with generous margin
                margin_w = person_w * 0.35
                margin_h = person_h * 0.28
            else:
                margin_w = person_w * 0.20
                margin_h = person_h * 0.20
            
            near_person = (px1 - margin_w <= det_cx <= px2 + margin_w and
                        py1 - margin_h <= det_cy <= py2 + margin_h)

            if not near_person:
                if self.debug:
                    print(f"[PPE-DEBUG] {class_name} not near person (margin={margin_w:.1f},{margin_h:.1f}), skip")
                continue

            if is_vest_item:
                torso_min_y = py1 + person_h * 0.22
                if det_cy < torso_min_y:
                    if self.debug:
                        print(
                            f"[PPE-DEBUG] {class_name} rejected: center above torso "
                            f"(cy={det_cy:.0f} < {torso_min_y:.0f})"
                        )
                    continue

            if self.debug and (is_head_item or is_vest_item):
                print(f"[PPE-DEBUG] {class_name} NEAR person (margin={margin_w:.1f},{margin_h:.1f}), processing...")

            # ============================================
            # STEP 2: IoU overlap check
            # ============================================
            ix1 = max(px1, dx1)
            iy1 = max(py1, dy1)
            ix2 = min(px2, dx2)
            iy2 = min(py2, dy2)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            det_area = (dx2 - dx1) * (dy2 - dy1)
            overlap_ratio = intersection / det_area if det_area > 0 else 0

            # is_mask_class already defined in STEP 1
            # Relax overlap for helmet (often at top of bbox) and mask (small items)
            if is_mask_class:
                min_overlap = 0.01  # Very low for masks
            elif is_head_item:
                min_overlap = 0.0  # Allow helmets that are mostly above the person box
            else:
                min_overlap = 0.05  # Standard for other items

            if det_area > 0 and overlap_ratio < min_overlap:
                if self.debug:
                    print(f"[PPE-DEBUG] {class_name} overlap {overlap_ratio:.3f} < {min_overlap}, skip")
                continue

            # ============================================
            # STEP 3: Detection assign karo
            # ============================================

            # Helmet — require min confidence so weak "hair as hardhat" boxes are not assigned to persons
            if class_lower in ['hardhat', 'helmet']:
                min_helmet_assign = max(self.ppe_threshold, self.model_helmet_accept_min_conf)
                if conf > result['helmet_conf'] and conf >= min_helmet_assign:
                    result['helmet_detected'] = True
                    result['helmet_conf'] = conf
                    result['helmet_class'] = class_name
                    result['helmet_bbox'] = det['bbox']
                    if self.debug:
                        print(f"[PPE-DEBUG] Helmet accepted: conf={conf:.3f} (min_assign={min_helmet_assign:.2f})")
                elif self.debug and conf >= self.ppe_threshold and conf < min_helmet_assign:
                    print(
                        f"[PPE-DEBUG] Helmet REJECTED (weak model score): conf={conf:.3f} "
                        f"< model_helmet_accept_min_conf={self.model_helmet_accept_min_conf:.2f}"
                    )
                elif conf < self.ppe_threshold and self.debug:
                    print(f"[PPE-DEBUG] Helmet REJECTED: conf={conf:.3f} below {self.ppe_threshold:.2f}")

            elif class_lower in ['nohardhat', 'nohelmet']:
                result['no_hardhat'] = True
                if conf > result['no_hardhat_conf']:
                    result['no_hardhat_conf'] = conf
                # Only override helmet if no-helmet confidence is significantly higher
                if result['helmet_detected']:
                    if conf > result['helmet_conf'] + 0.20 and conf >= 0.70:
                        result['helmet_detected'] = False
                        result['helmet_conf'] = conf
                        result['helmet_class'] = class_name
                        if self.debug:
                            print(f"[PPE-DEBUG] NO-Helmet overrides helmet: conf={conf:.3f} > helmet_conf+0.20")
                    else:
                        if self.debug:
                            print(f"[PPE-DEBUG] NO-Helmet ignored: helmet conf={result['helmet_conf']:.3f}, nohelmet conf={conf:.3f}")
                else:
                    if self.debug:
                        print(f"[PPE-DEBUG] NO-Helmet accepted: conf={conf:.3f}")

            # Vest
            if class_lower in ['safetyvest', 'vest']:
                if conf > result['vest_conf'] and conf >= 0.30:
                    result['vest_detected'] = True
                    result['vest_conf'] = conf
                    result['vest_class'] = class_name
                    result['vest_bbox'] = det['bbox']
                    if self.debug:
                        print(f"[PPE-DEBUG] Vest accepted: conf={conf:.3f}")
                elif conf < 0.30 and self.debug:
                    print(f"[PPE-DEBUG] Vest REJECTED: conf={conf:.3f} below 0.30")

            elif class_lower in ['nosafetyvest', 'novest']:
                result['no_safetyvest'] = True
                if conf > result['no_safetyvest_conf']:
                    result['no_safetyvest_conf'] = conf
                    result['no_safetyvest_bbox'] = det['bbox']
                # Only let no-vest override a positive vest when it is strictly more confident.
                # Otherwise low "no_vest" scores (common on orange/unusual vests) erase real vest boxes.
                if result['vest_detected'] and conf > result['vest_conf']:
                    result['vest_detected'] = False
                    result['vest_conf'] = conf
                    result['vest_class'] = class_name
                if self.debug:
                    print(f"[PPE-DEBUG] NO-Vest accepted: conf={conf:.3f}")

            # Mask
            if class_lower in ['mask']:
                if conf > result['mask_conf']:
                    result['mask_detected'] = True
                    result['mask_conf'] = conf
                    result['mask_class'] = class_name
                    result['mask_bbox'] = det['bbox']
                    result['no_mask'] = False
                    result['no_mask_conf'] = 0.0
                    if self.debug:
                        print(f"[PPE-DEBUG] Mask accepted: conf={conf:.3f}")

            elif class_lower in ['nomask']:
                result['no_mask'] = True
                if conf > result['no_mask_conf']:
                    result['no_mask_conf'] = conf
                    result['no_mask_bbox'] = det['bbox']
                if result['mask_detected'] and conf > result['mask_conf'] * 2.5 and conf > 0.70:
                    result['mask_detected'] = False
                    result['mask_conf'] = conf
                    result['mask_class'] = class_name
                    if self.debug:
                        print(f"[PPE-DEBUG] NO-MASK overriding mask: {conf:.3f} > mask*2.5")
                if self.debug:
                    print(f"[PPE-DEBUG] NO-Mask accepted: conf={conf:.3f}")

        return result

    def _refine_person_detections(self, persons, frame_shape):
        """Drop non-person-shaped boxes and merge duplicates (crowded sites)."""
        if not persons:
            return persons

        h, w = frame_shape[:2]
        frame_area = max(h * w, 1)
        min_h = h * self.person_min_height_frac
        min_area = frame_area * self.person_min_area_frac
        filtered = []

        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            if bh < min_h or (bw * bh) < min_area:
                if self.debug:
                    print(f"[PPE-DEBUG] Person filter reject (size): bbox={p['bbox']}")
                continue
            aspect = bw / bh
            if aspect < self.person_min_aspect or aspect > self.person_max_aspect:
                if self.debug:
                    print(
                        f"[PPE-DEBUG] Person filter reject (aspect {aspect:.2f}): bbox={p['bbox']}"
                    )
                continue
            filtered.append(p)

        if len(filtered) < len(persons) and self.debug:
            print(f"[PPE-DEBUG] Person shape filter: {len(persons)} -> {len(filtered)}")

        persons = sorted(filtered, key=lambda p: p.get("confidence", 0), reverse=True)
        deduped = []
        iou_thr = self.person_dedup_iou
        center_ratio = self.person_center_merge_ratio

        for p in persons:
            px1, py1, px2, py2 = p["bbox"]
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            ph = py2 - py1
            is_dup = False

            for d in deduped:
                dx1, dy1, dx2, dy2 = d["bbox"]
                dcx = (dx1 + dx2) / 2
                dcy = (dy1 + dy2) / 2
                dh = dy2 - dy1

                ix1 = max(px1, dx1)
                iy1 = max(py1, dy1)
                ix2 = min(px2, dx2)
                iy2 = min(py2, dy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area1 = (px2 - px1) * (py2 - py1)
                area2 = (dx2 - dx1) * (dy2 - dy1)
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0

                center_dist = ((pcx - dcx) ** 2 + (pcy - dcy) ** 2) ** 0.5
                merge_dist = center_ratio * min(ph, dh)

                if iou > iou_thr or center_dist < merge_dist:
                    is_dup = True
                    if p.get("confidence", 0) > d.get("confidence", 0):
                        deduped.remove(d)
                        deduped.append(p)
                    break

            if not is_dup:
                deduped.append(p)

        if len(deduped) < len(filtered) and self.debug:
            print(f"[PPE-DEBUG] Person dedup: {len(filtered)} -> {len(deduped)} (iou>{iou_thr})")

        return deduped

    def detect_persons_with_fallback(self, frame):

        persons = []

        model_worked = False

        # Initialize PPE detections storage

        self._last_ppe_detections = []

        # Try primary model

        if self.model is not None:

            try:

                # Check if this is a PPE-trained model

                is_ppe_model = "PPE" in self.model_path or "best.pt" in self.model_path or "best_ppe.pt" in self.model_path

                if self.debug:

                    print(f"[PPE-DEBUG] Model path: {self.model_path}")

                    print(f"[PPE-DEBUG] Is PPE model: {is_ppe_model}")

                if is_ppe_model:

                    # PPE model detects PPE items (Hardhat, Vest, etc.) but NOT Person reliably

                    # Use separate person_model (yolov8n) for person detection

                    model_names = self.model.names if hasattr(self.model, 'names') else {}

                    core_infer_ids = core_inference_class_ids_from_names(model_names)

                    mask_class_ids = [
                        int(cid) for cid, cname in model_names.items()
                        if _normalize_yolo_class_name(str(cname)) in ('mask', 'nomask')
                    ]

                    if self.debug and core_infer_ids:

                        print(f"[PPE-DEBUG] Core-focus YOLO class ids (helmet/vest/mask/person): {core_infer_ids}")

                    if self.debug:

                        print(f"[PPE-DEBUG] Model internal names: {model_names}")

                    # Always use separate person model for better person detection
                    # Don't rely on PPE model for person detection even if it has person class
                    has_person_class = any(name.lower() == 'person' for name in model_names.values())

                    if has_person_class and self.person_model is None:

                        if self.debug:

                            print(f"[PPE-DEBUG] Primary model has 'Person' class but no person_model available.")

                    # STEP 1: Detect persons using separate person model (COCO class 0 = person)
                    if self.person_model is not None:
                        if self.debug:
                            print(f"[PPE-DEBUG] Using person_model (yolov8n) for person detection")
                        try:
                            person_results = self.person_model(frame, conf=self.person_threshold, iou=0.45,
                                                              device=self.device, verbose=False, classes=[0])
                            for result in person_results:
                                if result.boxes is not None:
                                    for box in result.boxes:
                                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                        conf = float(box.conf[0].cpu().numpy())
                                        persons.append({
                                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                            "confidence": conf,
                                            "class": "person",
                                            "class_id": 0,
                                            "source": "person_model"
                                        })

                            if self.debug:

                                print(f"[PPE-DEBUG] Person model detected {len(persons)} persons")

                        except Exception as pe:

                            print(f"[PPE-WARNING] Person model failed: {pe}")

                    # STEP 2: Detect PPE items and/or persons using PPE model (best.pt)
                    # Lower IoU + class-aware NMS so overlapping PPE items (helmet+mask) aren't suppressed

                    pred_kw = dict(
                        conf=self.ppe_threshold,
                        iou=0.30,
                        device=self.device,
                        verbose=False,
                    )
                    if core_infer_ids:
                        pred_kw["classes"] = core_infer_ids
                    results = self.model(frame, **pred_kw)

                    # Initialize ppe_detections first
                    ppe_detections = []

                    if mask_class_ids:
                        mask_results = self.model(
                            frame,
                            conf=0.05,
                            iou=0.45,
                            device=self.device,
                            verbose=False,
                            classes=mask_class_ids,
                        )
                    else:
                        mask_results = []
                    
                    # Process mask-only detections and merge with main results
                    for m_result in mask_results:
                        if m_result.boxes is not None:
                            for box in m_result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0].cpu().numpy())
                                cls = int(box.cls[0].cpu().numpy())
                                class_name = model_names.get(cls, PPE_CLASSES[cls] if cls < len(PPE_CLASSES) else "unknown")
                                cn_key = _normalize_yolo_class_name(class_name)

                                # If this is a mask/no_mask detection, add to ppe_detections
                                if cn_key in ('mask', 'nomask'):
                                    ppe_detections.append({
                                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                        "confidence": conf,
                                        "class": class_name,
                                        "class_id": cls
                                    })
                                    if self.debug:
                                        print(f"[PPE-DEBUG] Mask-only pass detected: {class_name} (conf: {conf:.2f})")

                    # Store all PPE detections for direct use

                    for result in results:

                        if result.boxes is not None:

                            for box in result.boxes:

                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                                conf = float(box.conf[0].cpu().numpy())

                                cls = int(box.cls[0].cpu().numpy())

                                # Use model's own class names first, fallback to PPE_CLASSES

                                class_name = model_names.get(cls, PPE_CLASSES[cls] if cls < len(PPE_CLASSES) else "unknown")

                                # Debug: log ALL model detections to see what best.pt finds

                                if self.debug:

                                    print(f"[PPE-DEBUG] Model raw detection: {class_name} (cls_id: {cls}, conf: {conf:.2f}) at ({int(x1)},{int(y1)},{int(x2)},{int(y2)})")

                                if class_name.lower() == 'person':
                                    if self.debug:
                                        print(f"[PPE-DEBUG] PPE model person detected - checking if should add...")
                                    # Only add person from PPE model if person_model didn't find any persons
                                    # This prevents duplicate/fake person detections
                                    person_model_persons = [p for p in persons if p.get('source') == 'person_model']
                                    if self.debug:
                                        print(f"[PPE-DEBUG] person_model persons found: {len(person_model_persons)}")
                                    if len(person_model_persons) == 0:
                                        persons.append({
                                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                            "confidence": conf,
                                            "class": "person",
                                            "class_id": cls,
                                            "source": "ppe_model"
                                        })
                                        if self.debug:
                                            print(f"[PPE-DEBUG] Person detected by PPE model (conf: {conf:.2f})")
                                    else:
                                        if self.debug:
                                            print(f"[PPE-DEBUG] Skipping PPE model person - person_model already found {len(person_model_persons)} persons")

                                else:

                                    nk = _normalize_yolo_class_name(class_name)
                                    if nk not in _CORE_INFERENCE_CLASS_KEYS:
                                        continue

                                    # Store all PPE item detections

                                    ppe_detections.append({

                                        "bbox": (int(x1), int(y1), int(x2), int(y2)),

                                        "confidence": conf,

                                        "class": class_name,

                                        "class_id": cls

                                    })

                                    if self.debug:

                                        print(f"[PPE-DEBUG] PPE item: {class_name} (cls_id: {cls}, conf: {conf:.2f}) at ({int(x1)},{int(y1)},{int(x2)},{int(y2)})")

                    # STEP 2b: Dedicated mask inference when still missing — tight IoU, mask classes only
                    mask_found_in_main = any(d['class_id'] in mask_class_ids for d in ppe_detections)
                    if not mask_found_in_main and mask_class_ids:
                        try:
                            if self.debug:
                                print(f"[PPE-DEBUG] Mask class IDs (dedicated pass): {mask_class_ids}")

                            mask_results = self.model(frame, conf=0.03, iou=0.10,
                                                    device=self.device, verbose=False,
                                                    classes=mask_class_ids)
                            for result in mask_results:
                                if result.boxes is not None:
                                    for box in result.boxes:
                                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                        conf = float(box.conf[0].cpu().numpy())
                                        cls = int(box.cls[0].cpu().numpy())
                                        class_name = model_names.get(cls, PPE_CLASSES[cls] if cls < len(PPE_CLASSES) else "unknown")
                                        if self.debug:
                                            print(f"[PPE-DEBUG] Mask-only detection: {class_name} (cls_id: {cls}, conf: {conf:.2f}) at ({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
                                        ppe_detections.append({
                                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                            "confidence": conf,
                                            "class": class_name,
                                            "class_id": cls
                                        })
                        except Exception as me:
                            if self.debug:
                                print(f"[PPE-DEBUG] Mask-only inference failed: {me}")

                    if self.detection_mode == "helmet_vest":
                        ppe_detections = [
                            det for det in ppe_detections
                            if det.get("class", "").lower().replace('-', '').replace(' ', '').replace('_', '')
                            in ['hardhat', 'helmet', 'nohardhat', 'nohelmet', 'safetyvest', 'vest', 'nosafetyvest', 'novest']
                        ]
                    elif self.detection_mode == "mask":
                        ppe_detections = [
                            det for det in ppe_detections
                            if _normalize_yolo_class_name(det.get("class", "")) in ('mask', 'nomask')
                        ]

                    self._last_ppe_detections = ppe_detections

                    # If no persons yet, do NOT return here — outer fallback retries person_model
                    # at lower conf, PPE-model person class, then HOG (was broken by early return []).
                    if len(persons) == 0 and self.debug:
                        print("[PPE-DEBUG] No persons from primary PPE path; will run fallback chain if needed")

                else:

                    # Non–PPE checkpoint: person class id from names (not hardcoded 11)
                    model_names = self.model.names if hasattr(self.model, 'names') else {}
                    person_ids = [
                        int(cid) for cid, cname in model_names.items()
                        if _normalize_yolo_class_name(str(cname)) == 'person'
                    ]
                    if not person_ids:
                        person_ids = [11]

                    results = self.model(frame, conf=0.15, iou=0.45,

                                       device=self.device, verbose=False, classes=person_ids)

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
                    ppe_model_persons = [p for p in persons if p.get('source') == 'ppe_model']
                    person_model_persons = [p for p in persons if p.get('source') == 'person_model']
                    print(f"[PPE-DEBUG] Person counts - person_model: {len(person_model_persons)}, PPE model: {len(ppe_model_persons)}, Total: {len(persons)}")

            except Exception as e:

                print(f"[PPE-WARNING] Primary model failed: {e}")

                # Even if primary model failed, run PPE detection to populate _last_ppe_detections
                # This ensures fallback person detection can still use PPE info
                try:
                    model_names = self.model.names if hasattr(self.model, 'names') else {}
                    core_fb = core_inference_class_ids_from_names(model_names)
                    ppe_kw = dict(conf=self.ppe_threshold, iou=0.30,
                                  device=self.device, verbose=False)
                    if core_fb:
                        ppe_kw["classes"] = core_fb
                    ppe_results = self.model(frame, **ppe_kw)
                    ppe_detections = []
                    for r in ppe_results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                cls = int(box.cls[0].cpu().numpy())
                                conf = float(box.conf[0].cpu().numpy())
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                class_name = model_names.get(cls, PPE_CLASSES[cls] if cls < len(PPE_CLASSES) else "unknown")
                                nk = _normalize_yolo_class_name(class_name)
                                if nk == 'person' or nk not in _CORE_INFERENCE_CLASS_KEYS:
                                    continue
                                ppe_detections.append({
                                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                        "confidence": conf,
                                        "class": class_name,
                                        "class_id": cls
                                    })
                    self._last_ppe_detections = ppe_detections
                    if self.debug:
                        print(f"[PPE-DEBUG] PPE detections captured despite primary model failure: {len(ppe_detections)}")
                except Exception as ppe_err:
                    if self.debug:
                        print(f"[PPE-DEBUG] PPE detection also failed: {ppe_err}")

                if self.auto_recovery:

                    print("[PPE] Attempting model recovery...")

                    if self._reload_model():

                        return self.detect_persons_with_fallback(frame)

        # Fallback: Try model without classes filter first, then HOG

        if not model_worked or len(persons) == 0:

            print("[PPE] Using fallback person detection...")

            # Try person_model or model without classes filter to catch persons

            if len(persons) == 0:

                # First try dedicated person_model

                if self.person_model is not None:

                    try:

                        person_results = self.person_model(
                            frame,
                            conf=self.person_fallback_threshold,
                            iou=0.50,
                            device=self.device,
                            verbose=False,
                            classes=[0],
                        )

                        for result in person_results:

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

                        if self.debug:

                            print(f"[PPE-DEBUG] Fallback person_model: {len(persons)} persons")

                    except Exception as e:

                        print(f"[PPE-WARNING] Fallback person_model failed: {e}")

                # Then try PPE model for person class

                if len(persons) == 0 and self.model is not None:

                    try:

                        model_names = self.model.names if hasattr(self.model, 'names') else {}

                        person_ids = [
                            int(cid) for cid, cname in model_names.items()
                            if _normalize_yolo_class_name(str(cname)) == 'person'
                        ]
                        pred_fb = dict(
                            conf=self.ppe_threshold,
                            iou=0.30,
                            device=self.device,
                            verbose=False,
                        )
                        if person_ids:
                            pred_fb["classes"] = person_ids
                        results = self.model(frame, **pred_fb)

                        for result in results:

                            if result.boxes is not None:

                                for box in result.boxes:

                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                                    conf = float(box.conf[0].cpu().numpy())

                                    cls = int(box.cls[0].cpu().numpy())

                                    class_name = model_names.get(cls, PPE_CLASSES[cls] if cls < len(PPE_CLASSES) else "unknown")

                                    if class_name.lower() == 'person':

                                        persons.append({

                                            "bbox": (int(x1), int(y1), int(x2), int(y2)),

                                            "confidence": conf,

                                            "class": class_name,

                                            "class_id": cls

                                        })

                        if self.debug:

                            print(f"[PPE-DEBUG] Fallback PPE model retry: {len(persons)} persons")

                    except Exception as e:

                        print(f"[PPE-WARNING] Retry detection failed: {e}")

            # If still no persons, use HOG detector

            if len(persons) == 0:

                try:

                    # Use OpenCV HOG detector as ultimate fallback

                    persons = self._detect_persons_hog(frame)

                    if self.debug:

                        print(f"[PPE-DEBUG] HOG fallback detected {len(persons)} persons")

                except Exception as e:

                    print(f"[PPE-ERROR] Fallback detection also failed: {e}")

        before_refine = len(persons)
        persons = self._refine_person_detections(persons, frame.shape)
        if before_refine != len(persons) and not self.debug:
            print(f"[PPE] Person refine: {before_refine} -> {len(persons)} boxes")

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

        """Hi-vis vest on torso only — rejects normal shirts (plaid, black, tan)."""

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
            s_min = self.vest_hivis_min_saturation
            v_min = 75

            yellow = cv2.inRange(hsv, np.array([15, s_min, v_min]), np.array([38, 255, 255]))
            orange = cv2.inRange(hsv, np.array([5, s_min, v_min]), np.array([22, 255, 255]))
            green = cv2.inRange(hsv, np.array([35, s_min, v_min]), np.array([88, 255, 255]))

            hivis = cv2.bitwise_or(yellow, orange)
            hivis = cv2.bitwise_or(hivis, green)

            roi_area = torso_roi.shape[0] * torso_roi.shape[1]
            coverage = np.sum(hivis > 0) / roi_area

            bright_mask = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 45, 255]))
            bright_coverage = np.sum(bright_mask > 0) / roi_area

            kernel_w = max(1, torso_roi.shape[1] // 3)
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
            bright_horizontal = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_h)
            strip_contours, _ = cv2.findContours(
                bright_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            has_reflective_strips = False
            for cnt in strip_contours:
                area = cv2.contourArea(cnt)
                if area > 150:
                    _x, _y, cw, ch = cv2.boundingRect(cnt)
                    if cw / max(ch, 1) > 3.5:
                        has_reflective_strips = True
                        break

            solid_thr = self.vest_color_solid_coverage
            strip_thr = self.vest_color_with_strip_coverage
            min_thr = self.vest_color_min_coverage

            if coverage >= solid_thr:
                present = True
                conf = min(0.95, 0.55 + coverage * 0.45)
            elif has_reflective_strips and coverage >= strip_thr:
                present = True
                conf = min(0.92, 0.40 + coverage * 0.55)
            elif coverage >= min_thr and has_reflective_strips:
                present = True
                conf = min(0.85, 0.35 + coverage * 0.50)
            else:
                present = False
                conf = min(0.35, coverage * 0.6)

            if self.debug:
                print(f"[PPE-DEBUG] Vest detection (hi-vis):")
                print(f"  - Hi-vis coverage: {coverage:.3f} (need >={strip_thr} w/strips or >={solid_thr} solid)")
                print(f"  - Bright coverage: {bright_coverage:.3f}")
                print(f"  - Reflective strips: {has_reflective_strips}")
                print(f"  - Present: {present}, conf: {conf:.3f}")

            return present, conf

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

                # persons = self._create_minimum_detection(frame)  # DISABLED - causes false positives

                if len(persons) > 0:

                    fallback_used = True

            person_results = []

            helmet_count = 0

            no_helmet_count = 0

            vest_count = 0

            no_vest_count = 0

            mask_count = 0

            no_mask_count = 0

            if self.debug:

                print(f"\n[PPE]  PROCESSING {len(persons)} PERSONS...")

                print(f"{'-'*60}")

            # Track which PPE items have been assigned to prevent duplicates
            used_ppe_bboxes = set()

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

                mask_present, mask_conf, mask_bbox = False, 0.0, None
                head_bbox = None
                vest_bbox = None

                # Helmet + Vest + Mask detection for all

                if self.debug:

                    print(f"[PPE-DEBUG] {vehicle_type} - checking helmet & vest")

                is_ppe_model = self.model_path and ("PPE" in self.model_path or "best.pt" in self.model_path or "best_ppe.pt" in self.model_path)

                # Check if trained PPE model detected PPE items near this person

                ppe_items_near_person = self._get_ppe_detections_near_person(person_bbox)

                # Prevent same PPE from being assigned to multiple persons
                if ppe_items_near_person.get('mask_bbox'):
                    mask_bbox_key = tuple(int(x) for x in ppe_items_near_person['mask_bbox'])
                    used_ppe_bboxes.add(mask_bbox_key)

                if self.debug:
                    print(f"[PPE-DEBUG] PPE items near person: vest_detected={ppe_items_near_person.get('vest_detected')}, vest_conf={ppe_items_near_person.get('vest_conf', 0):.3f}")
                    print(f"[PPE-DEBUG] PPE items near person: mask_detected={ppe_items_near_person.get('mask_detected')}, mask_conf={ppe_items_near_person.get('mask_conf', 0):.3f}, no_mask={ppe_items_near_person.get('no_mask')}")

                # Skip helmet detection if mode is "mask" only
                if self.detection_mode == "mask":
                    helmet_present = False
                    helmet_conf = 0.0
                    helmet_method = "not_applicable"

                elif ppe_items_near_person.get('helmet_detected'):

                    # Trained model detected Hardhat near this person (min 20% confidence)

                    helmet_present = True

                    helmet_conf = ppe_items_near_person['helmet_conf']

                    helmet_method = ppe_items_near_person.get('helmet_class', 'model_detected')

                    # Head anchor: full head ROI for hair/skin post-checks (not the tight draw box)
                    head_anchor = self.get_head_region(person_bbox, frame)
                    head_roi = frame[head_anchor[1]:head_anchor[3], head_anchor[0]:head_anchor[2]]

                    # Post-filter YOLO "helmet" FPs: shiny/short hair and moderate model scores (e.g. 0.34)
                    # used to skip checks because rejection only ran when conf < 0.20.
                    if head_roi.size > 0:
                        has_hair = self._check_hair_texture(head_roi)
                        has_styled = self._check_styled_hair(head_roi)
                        if (has_hair or has_styled) and helmet_conf < self.model_helmet_min_conf_if_hair:
                            helmet_present = False
                            helmet_conf = helmet_conf * 0.5
                            helmet_method = 'rejected_hair_detected'
                            if self.debug:
                                print(
                                    f"[PPE-DEBUG] Helmet REJECTED: hair/styled on head ROI "
                                    f"(hair={has_hair}, styled={has_styled}), model_conf={ppe_items_near_person['helmet_conf']:.2f} "
                                    f"< min_if_hair={self.model_helmet_min_conf_if_hair:.2f}"
                                )
                        if helmet_present:
                            has_skin = self._check_skin_texture(head_roi)
                            if has_skin and helmet_conf < self.model_helmet_min_conf_if_skin:
                                color_ok, color_c, _color_name = self.detect_helmet_by_color(
                                    head_roi, self.helmet_threshold
                                )
                                strong_helmet_color = (
                                    color_ok and color_c >= self.model_helmet_skin_color_override_conf
                                )
                                if not strong_helmet_color:
                                    helmet_present = False
                                    helmet_conf = helmet_conf * 0.45
                                    helmet_method = 'rejected_skin_dominated_head'
                                    if self.debug:
                                        print(
                                            f"[PPE-DEBUG] Helmet REJECTED: skin-dominated head ROI, "
                                            f"model_conf={ppe_items_near_person['helmet_conf']:.2f} "
                                            f"< min_if_skin={self.model_helmet_min_conf_if_skin:.2f}, "
                                            f"color_override={strong_helmet_color} (color_conf={color_c:.2f})"
                                        )

                    if helmet_present:
                        head_bbox = self._select_helmet_draw_bbox(
                            frame, person_bbox, head_anchor, ppe_items_near_person.get('helmet_bbox')
                        )
                    else:
                        head_bbox = head_anchor

                    if self.debug and helmet_present:

                        print(f"[PPE-DEBUG] Model detected helmet: {helmet_method} (conf: {helmet_conf:.2f}) draw_bbox: {head_bbox}")

                elif ppe_items_near_person.get('no_hardhat'):

                    # Model explicitly detected NO-Hardhat - trust it, skip color fallback

                    helmet_present = False

                    helmet_conf = ppe_items_near_person.get('no_hardhat_conf', 0.0)

                    helmet_method = 'NO-Hardhat (model)'

                    head_bbox = self.get_head_region(person_bbox, frame)

                    if self.debug:

                        print(f"[PPE-DEBUG] Model detected NO-Hardhat (conf: {helmet_conf:.2f}) - skipping color fallback")

                else:

                    # Fallback to color-based helmet detection

                    head_bbox = self.get_head_region(person_bbox, frame)

                    if is_ppe_model:

                        # Video frames often make helmets too small for the PPE model.
                        # If the model is silent, fall back to head-region color/shape checks.
                        self._current_person_bbox = person_bbox
                        threshold = self.fallback_threshold if fallback_used else self.helmet_threshold
                        helmet_present, helmet_conf, helmet_method = self.detect_helmet_in_head(frame, head_bbox, threshold)
                        self._current_person_bbox = None

                    else:

                        self._current_person_bbox = person_bbox  # Store for validation

                        threshold = self.fallback_threshold if fallback_used else self.helmet_threshold

                        helmet_present, helmet_conf, helmet_method = self.detect_helmet_in_head(frame, head_bbox, threshold)

                        self._current_person_bbox = None  # Clean up after validation

                if self.detection_mode == "mask":
                    vest_present = False
                    vest_conf = 0.0
                    vest_bbox = None

                elif ppe_items_near_person.get('vest_detected') and ppe_items_near_person['vest_conf'] >= 0.30:

                    model_vest_conf = ppe_items_near_person['vest_conf']
                    color_vest_ok, color_vest_conf = self._detect_vest(frame, person_bbox)

                    if color_vest_ok or model_vest_conf >= self.model_vest_min_conf_without_color:
                        vest_present = True
                        vest_conf = max(model_vest_conf, color_vest_conf) if color_vest_ok else model_vest_conf
                    else:
                        vest_present = False
                        vest_conf = model_vest_conf * 0.4
                        if self.debug:
                            print(
                                f"[PPE-DEBUG] Model vest REJECTED on street clothes: "
                                f"model={model_vest_conf:.2f}, color_ok={color_vest_ok}, "
                                f"need model>={self.model_vest_min_conf_without_color:.2f}"
                            )

                    if vest_present:
                        if ppe_items_near_person.get('vest_bbox'):
                            vest_bbox = ppe_items_near_person['vest_bbox']
                        else:
                            vest_bbox = self.get_vest_region(person_bbox, frame, head_bbox)
                    else:
                        vest_bbox = self.get_vest_region(person_bbox, frame, head_bbox)

                    if self.debug and vest_present:
                        print(
                            f"[PPE-DEBUG] Model vest kept (conf: {vest_conf:.2f}, "
                            f"color_support={color_vest_ok}) bbox: {vest_bbox}"
                        )

                elif (
                    ppe_items_near_person.get('no_safetyvest')
                    and ppe_items_near_person.get('no_safetyvest_conf', 0.0) >= 0.45
                ):

                    # High-confidence NO-Safety Vest: trust model, skip color fallback

                    vest_present = False

                    vest_conf = ppe_items_near_person.get('no_safetyvest_conf', 0.0)

                    # Use actual detected bbox for no_vest if available, otherwise fallback
                    if ppe_items_near_person.get('no_safetyvest_bbox'):
                        vest_bbox = ppe_items_near_person['no_safetyvest_bbox']
                    else:
                        vest_bbox = self.get_vest_region(person_bbox, frame, head_bbox)

                    if self.debug:

                        print(f"[PPE-DEBUG] Model detected NO-Safety Vest (conf: {vest_conf:.2f}) - skipping color fallback")

                else:

                    # Fallback to color-based vest detection (also when no_vest is weak — helps orange vests)
                    # Skip vest detection if mode is "mask" only
                    if self.detection_mode != "mask":
                        vest_bbox = self.get_vest_region(person_bbox, frame, head_bbox)

                        # Always try color-based detection as backup, even with PPE model

                        vest_present, vest_conf = self._detect_vest(frame, person_bbox)

                    if self.debug:

                        print(f"[PPE-DEBUG] Color-based vest detection: {vest_present} (conf: {vest_conf:.3f})")

                if vest_bbox is not None:
                    if head_bbox is None:
                        head_bbox = self.get_head_region(person_bbox, frame)
                    vest_bbox = self._normalize_vest_bbox(vest_bbox, person_bbox, head_bbox)

                # Check for Mask from PPE model first, then fallback to color-based detection
                # Skip mask detection if mode is "helmet_vest" only
                if self.detection_mode != "helmet_vest":
                    mask_present = ppe_items_near_person.get('mask_detected', False)

                    mask_conf = ppe_items_near_person.get('mask_conf', 0.0)

                    mask_bbox = ppe_items_near_person.get('mask_bbox')

                    # If no_mask detected, store its bbox for drawing in mask-capable modes.
                    if ppe_items_near_person.get('no_mask'):
                        no_mask_conf_val = ppe_items_near_person.get('no_mask_conf', 0.0)
                        no_mask_bbox_val = ppe_items_near_person.get('no_mask_bbox')
                        
                        if not mask_present:
                            mask_conf = no_mask_conf_val
                            mask_bbox = no_mask_bbox_val
                        elif no_mask_conf_val > mask_conf * 3.0 and no_mask_conf_val > 0.80:
                            mask_present = False
                            mask_conf = no_mask_conf_val
                            mask_bbox = no_mask_bbox_val
                            if self.debug:
                                print(f"[PPE-DEBUG] NO-MASK overriding mask (significantly more confident): no_mask_conf={no_mask_conf_val:.2f} > mask_conf*3.0={mask_conf*3.0:.2f}")
                        else:
                            if self.debug:
                                print(f"[PPE-DEBUG] Both mask and no_mask detected, keeping mask (conf={mask_conf:.2f} vs no_mask={no_mask_conf_val:.2f})")

                    # Fallback: Color-based mask detection when model doesn't detect mask.
                    if not mask_present and not ppe_items_near_person.get('no_mask') and frame is not None:
                        model_mask_bbox = ppe_items_near_person.get('mask_bbox')
                        model_no_mask_bbox = ppe_items_near_person.get('no_mask_bbox')
                        
                        if model_mask_bbox:
                            mask_bbox = model_mask_bbox
                            if self.debug:
                                print(f"[PPE-DEBUG] Using model mask position for NO MASK bbox: {mask_bbox}")
                        elif model_no_mask_bbox:
                            mask_bbox = model_no_mask_bbox
                            if self.debug:
                                print(f"[PPE-DEBUG] Using model no_mask position for bbox: {mask_bbox}")
                        else:
                            mask_bbox = self.get_mask_region_between_helmet_vest(person_bbox, head_bbox, vest_bbox)
                            if self.debug:
                                print(f"[PPE-DEBUG] Using position between helmet & vest for NO MASK bbox: {mask_bbox}")
                            
                            if mask_bbox:
                                mx1, my1, mx2, my2 = mask_bbox
                                face_roi = frame[max(0,my1):my2, max(0,mx1):mx2]
                                if face_roi.size > 0:
                                    mask_found, mask_color_conf, mask_color = self.detect_mask_by_color(face_roi, threshold=0.20)
                                    if mask_found:
                                        mask_present = True
                                        mask_conf = mask_color_conf
                                        if self.debug:
                                            print(f"[PPE-DEBUG] Mask DETECTED via color fallback: color={mask_color}, conf={mask_color_conf:.2f}")
                else:
                    mask_present = False
                    mask_conf = 0.0
                    mask_bbox = None

                if self.debug:

                    print(f"[PPE-DEBUG] Mask: {'DETECTED' if mask_present else 'Not detected'} (conf: {mask_conf:.2f})")

                    print(f"[PPE-DEBUG] Helmet: {'DETECTED' if helmet_present else 'Not detected'} (conf: {helmet_conf:.3f})")

                    print(f"[PPE-DEBUG] Vest: {'DETECTED' if vest_present else 'Not detected'} (conf: {vest_conf:.3f})")

                # STEP 4: Update counts

                if self.detection_mode != "mask" and helmet_present:

                    helmet_count += 1

                    if self.debug:

                        print(f"[PPE-DEBUG] Helmet count incremented to {helmet_count}")

                elif self.detection_mode != "mask":

                    no_helmet_count += 1

                    if self.debug:

                        print(f"[PPE-DEBUG] No helmet count incremented to {no_helmet_count}")

                if self.detection_mode != "mask" and vest_present:

                    vest_count += 1

                elif self.detection_mode != "mask":

                    no_vest_count += 1

                if self.detection_mode != "helmet_vest" and mask_present:

                    mask_count += 1

                elif self.detection_mode != "helmet_vest":

                    no_mask_count += 1

                # STEP 5: Determine status and label

                # Build label with mask status included

                ppe_items = []
                if self.detection_mode != "mask" and helmet_present:
                    ppe_items.append("Helmet")
                if self.detection_mode != "mask" and vest_present:
                    ppe_items.append("Vest")
                if self.detection_mode != "helmet_vest" and mask_present:
                    ppe_items.append("Mask")

                if ppe_items:
                    status = "compliant"
                    compliance_reason = "ppe_detected"
                    output_label = " & ".join(ppe_items) + " Detected"
                else:
                    status = "violation"
                    compliance_reason = "no_ppe_detected"
                    no_ppe_items = []
                    if self.detection_mode != "mask" and not helmet_present:
                        no_ppe_items.append("No Helmet")
                    if self.detection_mode != "mask" and not vest_present:
                        no_ppe_items.append("No Vest")
                    if self.detection_mode != "helmet_vest" and not mask_present:
                        no_ppe_items.append("No Mask")
                    output_label = " & ".join(no_ppe_items)

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

                    head_bbox=head_bbox,  # Model-aligned helmet box when available, else head anchor

                    vest_bbox=vest_bbox,  # Model vest bbox when available, else estimated torso region

                    mask_bbox=mask_bbox,  # Always pass mask_bbox (can be no_mask bbox too)

                    helmet=PPEItem("helmet", helmet_present, helmet_conf, head_bbox if helmet_present else None, helmet_method),

                    vest=PPEItem("vest", vest_present, vest_conf, vest_bbox if vest_present else None, "color"),

                    seatbelt=PPEItem("seatbelt", False, 0.0, None, "not_applicable"),

                    gloves=PPEItem("gloves", False, 0.0, None, "not_applicable"),

                    goggles=PPEItem("goggles", False, 0.0, None, "not_applicable"),

                    mask=PPEItem("mask", mask_present, mask_conf, mask_bbox if mask_present else None, "model"),

                    boots=PPEItem("boots", False, 0.0, None, "not_applicable"),

                    status=status,

                    confidence=person["confidence"],

                    debug_info=debug_info

                )

                # Temporal Smoothing for Video
                if person_id not in self._ppe_history:
                    self._ppe_history[person_id] = []
                
                self._ppe_history[person_id].append({
                    'helmet': helmet_present,
                    'vest': vest_present,
                    'mask': mask_present
                })
                
                if len(self._ppe_history[person_id]) > self._max_history:
                    self._ppe_history[person_id].pop(0)
                
                # Apply smoothing (majority vote)
                if len(self._ppe_history[person_id]) >= 3:
                    h_votes = sum(1 for x in self._ppe_history[person_id] if x['helmet'])
                    v_votes = sum(1 for x in self._ppe_history[person_id] if x['vest'])
                    m_votes = sum(1 for x in self._ppe_history[person_id] if x['mask'])
                    
                    smoothed_h = self.detection_mode != "mask" and h_votes > len(self._ppe_history[person_id]) / 2
                    smoothed_v = self.detection_mode != "mask" and v_votes > len(self._ppe_history[person_id]) / 2
                    smoothed_m = self.detection_mode != "helmet_vest" and m_votes > len(self._ppe_history[person_id]) / 2
                    
                    # Update status if smoothed result differs (only for more stability)
                    if (
                        smoothed_h != helmet_present
                        or smoothed_v != vest_present
                        or (self.detection_mode != "helmet_vest" and smoothed_m != mask_present)
                    ):
                        helmet_present = smoothed_h
                        vest_present = smoothed_v
                        mask_present = smoothed_m
                        
                        # Re-calculate status and label
                        ppe_items = []
                        if self.detection_mode != "mask" and helmet_present: ppe_items.append("Helmet")
                        if self.detection_mode != "mask" and vest_present: ppe_items.append("Vest")
                        if self.detection_mode != "helmet_vest" and mask_present: ppe_items.append("Mask")
                        
                        if ppe_items:
                            status = "compliant"
                            output_label = " & ".join(ppe_items) + " Detected"
                        else:
                            status = "violation"
                            no_ppe_items = []
                            if self.detection_mode != "mask" and not helmet_present: no_ppe_items.append("No Helmet")
                            if self.detection_mode != "mask" and not vest_present: no_ppe_items.append("No Vest")
                            if self.detection_mode != "helmet_vest" and not mask_present: no_ppe_items.append("No Mask")
                            output_label = " & ".join(no_ppe_items)
                        
                        # Update the object
                        person_ppe.helmet.present = helmet_present
                        person_ppe.vest.present = vest_present
                        person_ppe.mask.present = mask_present
                        person_ppe.status = status

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

                print(f"     - Mask count: {mask_count}")

                print(f"     - No mask count: {no_mask_count}")

                total_ppe_items = helmet_count + no_helmet_count + vest_count + no_vest_count + mask_count + no_mask_count

                ppe_detected_items = helmet_count + vest_count + mask_count

                acc = (ppe_detected_items / total_ppe_items * 100) if total_ppe_items > 0 else 0.0

                print(f"     - Accuracy Score: {acc:.1f}%")

                print(f"[PPE] System Status:")

                print(f"     - Model Loaded: {'YES' if model_loaded else 'NO'}")

                print(f"     - Fallback Used: {'YES' if fallback_used else 'NO'}")

                print(f"     - Processing Time: {proc_time:.3f}s")

                print(f"{'='*60}\n")

            # Calculate accuracy score: ratio of PPE detected to total PPE checks

            total_ppe_items = helmet_count + no_helmet_count + vest_count + no_vest_count + mask_count + no_mask_count

            ppe_detected_items = helmet_count + vest_count + mask_count

            accuracy_score = (ppe_detected_items / total_ppe_items * 100) if total_ppe_items > 0 else 0.0

            return PPEResult(

                total_persons=len(persons),

                helmet_detected=helmet_count,

                no_helmet=no_helmet_count,

                seatbelt_detected=0,

                no_seatbelt=0,

                vest_detected=vest_count,

                no_vest=no_vest_count,

                mask_detected=mask_count,

                no_mask=no_mask_count,

                accuracy_score=round(accuracy_score, 1),

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

        h, w = img.shape[:2]

        # Top summary bar occupies ~y 5–35; keep person text strip below it when show_labels
        summary_band_bottom = 40

        def _clamp_bbox(bbox):
            if not bbox:
                return None
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))
            if x2 <= x1 or y2 <= y1:
                return None
            return (x1, y1, x2, y2)

        for person in result.persons:

            clamped_person_bbox = _clamp_bbox(person.bbox)
            if not clamped_person_bbox:
                continue
            x1, y1, x2, y2 = clamped_person_bbox

            # COLOR LOGIC based on vehicle type

            if self.detection_mode == "mask":

                is_compliant = person.mask.present

            elif person.vehicle_type == "4-wheeler":

                # 4-WHEELER: Green if helmet or vest, Red if none

                is_compliant = person.helmet.present or person.vest.present

            else:

                # 2-WHEELER/UNKNOWN: Green if helmet OR vest, Red if none

                is_compliant = person.helmet.present or person.vest.present

            if is_compliant:

                color = self.COLORS["compliant"]  # GREEN

                status_emoji = "🟩"

            else:

                color = self.COLORS["non_compliant"]  # RED

                status_emoji = "🟥"

            # Create status text - Helmet, Vest, Mask only

            if self.detection_mode == "mask":

                safety_text = "Mask Detected" if person.mask.present else "No Mask"

            elif person.helmet.present and person.vest.present:

                safety_text = "Helmet & Vest Detected"

            elif person.helmet.present:

                safety_text = "Helmet Detected"

            elif person.vest.present:

                safety_text = "Vest Detected"

            else:

                if self.detection_mode == "helmet_vest":
                    safety_text = "No Helmet & No Vest"
                elif person.mask.present:
                    safety_text = "No Helmet & No Vest"
                else:
                    safety_text = "No Helmet & No Vest & No Mask"

            # Use fallback color if in fallback mode

            if result.fallback_used:

                color = self.COLORS["fallback"]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # When both helmet and vest are OK, use only the person box (no inner helmet/vest rects)
            merge_helmet_vest = (
                self.detection_mode != "mask"
                and person.helmet.present
                and person.vest.present
            )
            # Mask-only UI: always use person bbox only (no inner mask / NO MASK face box)
            merge_mask_into_person = self.detection_mode == "mask"

            # Draw helmet/vest/mask bounding boxes ONLY if at least one PPE is present
            # If no helmet, no vest, no mask - only show red person bbox
            any_ppe_present = (
                (self.detection_mode != "mask" and person.helmet.present)
                or (self.detection_mode != "mask" and person.vest.present)
                or (self.detection_mode != "helmet_vest" and person.mask.present)
                or (self.detection_mode == "mask" and person.mask_bbox is not None)
            )

            if any_ppe_present:

                # Draw vest bounding box

                if self.detection_mode != "mask" and person.vest_bbox and not merge_helmet_vest:

                    clamped_vest_bbox = _clamp_bbox(person.vest_bbox)
                    if clamped_vest_bbox:
                        vx1, vy1, vx2, vy2 = clamped_vest_bbox

                        vest_color = (0, 165, 255) if person.vest.present else (0, 100, 255)  # Orange if vest, darker orange if not

                        cv2.rectangle(img, (vx1, vy1), (vx2, vy2), vest_color, 2)

                        vest_label = "VEST" if person.vest.present else "NO VEST"

                        cv2.putText(img, vest_label, (vx1, vy1 - 5),

                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, vest_color, 2)
                    else:
                        if self.debug:
                            print(f"[PPE-DEBUG] No vest bbox to draw for person {person.person_id}")

                else:

                    if self.debug and not merge_helmet_vest:

                        print(f"[PPE-DEBUG] No vest bbox to draw for person {person.person_id}")

                # Draw helmet bounding box

                if self.detection_mode != "mask" and person.head_bbox and not merge_helmet_vest:

                    clamped_head_bbox = _clamp_bbox(person.head_bbox)
                    if clamped_head_bbox:
                        hx1, hy1, hx2, hy2 = clamped_head_bbox

                        helmet_color = (255, 0, 255) if person.helmet.present else (128, 0, 128)  # Magenta if helmet, purple if not

                        cv2.rectangle(img, (hx1, hy1), (hx2, hy2), helmet_color, 2)

                        helmet_label = "HELMET" if person.helmet.present else "NO HELMET"

                        cv2.putText(img, helmet_label, (hx1, hy1 - 5),

                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, helmet_color, 2)

                # Draw mask bounding box

                if self.detection_mode != "helmet_vest" and person.mask_bbox and not merge_mask_into_person:
                    if self.debug:
                        print(f"[PPE-DEBUG] Drawing mask bbox: {person.mask_bbox}, present={person.mask.present}")
                    clamped_mask_bbox = _clamp_bbox(person.mask_bbox)
                    if clamped_mask_bbox:
                        mx1, my1, mx2, my2 = clamped_mask_bbox

                        mask_color = (0, 255, 0) if person.mask.present else (0, 100, 200)  # Bright green for MASK, blue for NO MASK

                        cv2.rectangle(img, (mx1, my1), (mx2, my2), mask_color, 2)  # Thinner line for NO MASK

                        mask_label = "MASK" if person.mask.present else "NO MASK"

                        cv2.putText(img, mask_label, (mx1, my1 - 5),

                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, mask_color, 2)
                    else:
                        if self.debug:
                            print(f"[PPE-DEBUG] No mask bbox to draw for person {person.person_id}")

                else:

                    if self.debug and not merge_mask_into_person:

                        print(f"[PPE-DEBUG] No mask bbox to draw for person {person.person_id}")


            if show_labels:

                # Label with person ID and safety status

                label = f"{person.person_id} {safety_text}"

                # Add confidence scores

                if self.detection_mode != "mask" and person.helmet.present:

                    label += f" H:{person.helmet.confidence:.2f}"

                if self.detection_mode != "mask" and person.vest.present:

                    label += f" V:{person.vest.confidence:.2f}"

                if self.detection_mode != "helmet_vest" and person.mask.present:

                    label += f" M:{person.mask.confidence:.2f}"

                if result.fallback_used:

                    label += " (FB)"

                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                text_x = x1

                text_y = y1 - 10 if y1 > 30 else y1 + text_size[1] + 10

                label_top = text_y - text_size[1] - 5
                if label_top < summary_band_bottom:
                    text_y = summary_band_bottom + text_size[1] + 6

                # Background rectangle for text

                cv2.rectangle(img, (text_x, text_y - text_size[1] - 5),

                            (text_x + text_size[0], text_y + 5), color, -1)

                cv2.putText(img, label, (text_x, text_y),

                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Summary with all counts

        if self.detection_mode == "helmet_vest":
            summary = f"Persons: {result.total_persons} | Helmets: {result.helmet_detected} | Vests: {result.vest_detected} | Model: {'OK' if result.model_loaded else 'FB'}"
        elif self.detection_mode == "mask":
            summary = f"Persons: {result.total_persons} | Masks: {result.mask_detected} | Model: {'OK' if result.model_loaded else 'FB'}"
        else:
            summary = f"Persons: {result.total_persons} | Helmets: {result.helmet_detected} | Vests: {result.vest_detected} | Masks: {result.mask_detected} | Model: {'OK' if result.model_loaded else 'FB'}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 2
        (tw, th), _baseline = cv2.getTextSize(summary, font, font_scale, font_thick)
        bar_x1, bar_y1 = 10, 5
        pad_right = 12
        bar_x2 = min(w - 1, 15 + tw + pad_right)
        bar_y2 = 35
        cv2.rectangle(img, (bar_x1, bar_y1), (bar_x2, bar_y2), (0, 0, 0), -1)

        cv2.putText(img, summary, (15, 25), font, font_scale, (255, 255, 255), font_thick)

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

            # Add mask details if present

            if person.mask.present:

                lines.append(f"  - Mask: Present (conf: {person.mask.confidence:.2f})")

            # Add debug information if available

            if person.debug_info and result.debug_mode:

                lines.append(f"  - Debug: {person.debug_info}")

        if result.error_message:

            lines.append(f"\n Warning: {result.error_message}")

        return "\n".join(lines)

    @staticmethod
    def _max_person_id_suffix(tracker, matched_ids=None):
        """Largest numeric suffix from P1, P2, ... seen in tracker (for stable new IDs)."""
        matched_ids = matched_ids or {}
        nums = []
        for k in tracker.get('track_history', {}).keys():
            if isinstance(k, str) and len(k) >= 2 and k[0].upper() == 'P':
                try:
                    nums.append(int(k[1:]))
                except ValueError:
                    pass
        for p in tracker.get('prev_persons', []):
            pid = p.get('person_id')
            if isinstance(pid, str) and len(pid) >= 2 and pid[0].upper() == 'P':
                try:
                    nums.append(int(pid[1:]))
                except ValueError:
                    pass
        for pid in matched_ids.values():
            if isinstance(pid, str) and len(pid) >= 2 and pid[0].upper() == 'P':
                try:
                    nums.append(int(pid[1:]))
                except ValueError:
                    pass
        return max(nums) if nums else 0

    # ==================== VIDEO-OPTIMIZED DETECTION ====================

    def detect_video(self, frame, frame_number=0, debug=None):
        """
        Video-optimized detection with temporal smoothing and person tracking.
        
        Key improvements over detect():
        1. Person tracking via IoU - consistent person IDs across frames
        2. Temporal smoothing - averages PPE status over recent frames
        3. Frame pre-processing - resizes large frames for faster inference
        4. Reduced inference - skips re-detection on stable frames
        
        Args:
            frame: Video frame (BGR)
            frame_number: Current frame number
            debug: Override debug mode
            
        Returns:
            PPEResult with smoothed/stable detections
        """
        if debug is not None:
            self.debug = debug
        
        start = time.time()
        timestamp = datetime.now().isoformat()
        
        # Initialize video tracking state
        if not hasattr(self, '_video_tracker'):
            self._video_tracker = {
                'prev_persons': [],       # List of {bbox, person_id, helmet, vest, mask, frame}
                'track_history': {},       # person_id -> list of recent PPE states
                'last_process_frame': -1,
                'smooth_window': 2,        # Number of frames to average over
                'resize_width': 960,       # Keep small helmets visible while still reducing large frames
                'skip_stable_frames': 2,    # Reuse results if person bboxes haven't changed much
                'stable_count': 0,
                'last_result': None,
                'ghost_ttl': 15,           # keep last bbox for missed frames so IDs can rematch
            }
        tracker = self._video_tracker
        
        # Frame pre-processing: resize if too large for faster inference
        original_frame = frame
        scale = 1.0
        h, w = frame.shape[:2]
        target_w = tracker['resize_width']
        if w > target_w:
            scale = target_w / w
            new_w = target_w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            if self.debug:
                print(f"[PPE-VIDEO] Resized {w}x{h} -> {new_w}x{new_h} (scale={scale:.2f})")
        
        # Run base detection on (possibly resized) frame
        result = self.detect(frame, debug=False)
        
        # Scale bboxes back to original frame size if we resized
        if scale != 1.0:
            for person in result.persons:
                person.bbox = tuple(int(v / scale) for v in person.bbox)
                if person.head_bbox:
                    person.head_bbox = tuple(int(v / scale) for v in person.head_bbox)
                if person.vest_bbox:
                    person.vest_bbox = tuple(int(v / scale) for v in person.vest_bbox)
                if person.mask_bbox:
                    person.mask_bbox = tuple(int(v / scale) for v in person.mask_bbox)
        
        # Match detected persons to tracked persons via IoU
        current_detections = []
        for person in result.persons:
            current_detections.append({
                'bbox': person.bbox,
                'helmet': person.helmet.present,
                'helmet_conf': person.helmet.confidence,
                'vest': person.vest.present,
                'vest_conf': person.vest.confidence,
                'mask': person.mask.present,
                'mask_conf': person.mask.confidence,
                'person_obj': person,
            })
        
        # IoU-based matching
        matched_ids = {}
        used_prev = set()
        
        for curr in current_detections:
            best_iou = 0.0
            best_id = None
            best_idx = None
            
            for idx, prev in enumerate(tracker['prev_persons']):
                if idx in used_prev:
                    continue
                iou = self._compute_iou(curr['bbox'], prev['bbox'])
                if iou > best_iou and iou > 0.3:  # 30% IoU threshold for same person
                    best_iou = iou
                    best_id = prev['person_id']
                    best_idx = idx
            
            if best_id is not None:
                matched_ids[id(curr)] = best_id
                used_prev.add(best_idx)
            else:
                # Truly new track: next id = max(P*) + 1 (never add frame_number — that caused P3, P4 jumps)
                n = self._max_person_id_suffix(tracker, matched_ids) + 1
                matched_ids[id(curr)] = f"P{n}"
        
        # Apply temporal smoothing
        for curr in current_detections:
            pid = matched_ids[id(curr)]
            curr['person_id'] = pid
            
            # Initialize track history for new person
            if pid not in tracker['track_history']:
                tracker['track_history'][pid] = []
            
            history = tracker['track_history'][pid]
            
            # Add current detection to history
            history.append({
                'helmet': curr['helmet'],
                'helmet_conf': curr['helmet_conf'],
                'vest': curr['vest'],
                'vest_conf': curr['vest_conf'],
                'mask': curr['mask'],
                'mask_conf': curr['mask_conf'],
                'frame': frame_number,
            })
            
            # Keep only recent history (sliding window)
            window = tracker['smooth_window']
            if len(history) > window:
                history[:] = history[-window:]
            
            # Smooth PPE status: majority vote over window
            if len(history) >= 2:
                helmet_votes = sum(1 for h in history if h['helmet'])
                vest_votes = sum(1 for h in history if h['vest'])
                mask_votes = sum(1 for h in history if h['mask'])
                
                # Majority vote (more than half)
                curr['helmet'] = helmet_votes > len(history) / 2
                curr['vest'] = vest_votes > len(history) / 2
                curr['mask'] = self.detection_mode != "helmet_vest" and mask_votes > len(history) / 2
                
                # Average confidence
                if curr['helmet']:
                    curr['helmet_conf'] = sum(h['helmet_conf'] for h in history if h['helmet']) / max(helmet_votes, 1)
                if curr['vest']:
                    curr['vest_conf'] = sum(h['vest_conf'] for h in history if h['vest']) / max(vest_votes, 1)
                if curr['mask']:
                    curr['mask_conf'] = sum(h['mask_conf'] for h in history if h['mask']) / max(mask_votes, 1)
            
            # Apply smoothed values to person object
            person = curr['person_obj']
            person.person_id = pid
            person.helmet.present = curr['helmet']
            person.helmet.confidence = curr['helmet_conf']
            person.vest.present = curr['vest']
            person.vest.confidence = curr['vest_conf']
            person.mask.present = curr['mask']
            person.mask.confidence = curr['mask_conf']
            
            # Update status based on smoothed values
            ppe_items = []
            if person.helmet.present:
                ppe_items.append("Helmet")
            if person.vest.present:
                ppe_items.append("Vest")
            if self.detection_mode != "helmet_vest" and person.mask.present:
                ppe_items.append("Mask")
            person.status = "compliant" if ppe_items else "violation"
        
        # Update tracker: live detections + short-lived "ghosts" for missed persons (stable P2, etc.)
        new_prev = []
        for curr in current_detections:
            new_prev.append({
                'bbox': curr['bbox'],
                'person_id': curr['person_id'],
                'helmet': curr['helmet'],
                'vest': curr['vest'],
                'mask': curr['mask'],
                'ghost_age': 0,
            })
        ghost_ttl = int(tracker.get('ghost_ttl', 15))
        ghosts = []
        for idx, prev in enumerate(tracker['prev_persons']):
            if idx in used_prev:
                continue
            age = int(prev.get('ghost_age', 0)) + 1
            if age <= ghost_ttl:
                ghosts.append({
                    'bbox': prev['bbox'],
                    'person_id': prev['person_id'],
                    'helmet': prev.get('helmet', False),
                    'vest': prev.get('vest', False),
                    'mask': prev.get('mask', False),
                    'ghost_age': age,
                })
        tracker['prev_persons'] = new_prev + ghosts
        
        # Recalculate result counts based on smoothed values
        result.helmet_detected = sum(1 for p in result.persons if p.helmet.present)
        result.no_helmet = sum(1 for p in result.persons if not p.helmet.present)
        result.vest_detected = sum(1 for p in result.persons if p.vest.present)
        result.no_vest = sum(1 for p in result.persons if not p.vest.present)
        if self.detection_mode == "mask":
            for person in result.persons:
                person.helmet.present = False
                person.helmet.confidence = 0.0
                person.head_bbox = None
                person.vest.present = False
                person.vest.confidence = 0.0
                person.vest_bbox = None
            result.helmet_detected = 0
            result.no_helmet = 0
            result.vest_detected = 0
            result.no_vest = 0
            result.mask_detected = sum(1 for p in result.persons if p.mask.present)
            result.no_mask = sum(1 for p in result.persons if not p.mask.present)
        elif self.detection_mode == "helmet_vest":
            for person in result.persons:
                person.mask.present = False
                person.mask.confidence = 0.0
                person.mask_bbox = None
            result.mask_detected = 0
            result.no_mask = 0
        else:
            result.mask_detected = sum(1 for p in result.persons if p.mask.present)
            result.no_mask = sum(1 for p in result.persons if not p.mask.present)
        result.processing_time = time.time() - start
        
        return result
    
    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reset_video_tracker(self):
        """Reset video tracking state - call when starting a new video"""
        if hasattr(self, '_video_tracker'):
            self._video_tracker = {
                'prev_persons': [],
                'track_history': {},
                'last_process_frame': -1,
                'smooth_window': 3,
                'resize_width': 960,
                'skip_stable_frames': 2,
                'stable_count': 0,
                'last_result': None,
                'ghost_ttl': 15,
            }

# Global instance with auto-recovery

_ppe_detector = None

_lock = threading.Lock()

def get_ppe_detector(model_path="yolov8n.pt", debug=False, auto_recovery=True, force_new=False, detection_mode="all"):

    """Get or create PPE detector with auto-recovery"""

    global _ppe_detector

    with _lock:

        detector_mode = getattr(_ppe_detector, 'detection_mode', None) if _ppe_detector is not None else None
        detector_model_path = getattr(_ppe_detector, 'model_path', None) if _ppe_detector is not None else None
        requested_model_path = model_path
        mode_changed = detector_mode != detection_mode
        model_changed = detector_model_path is not None and requested_model_path is not None and str(detector_model_path) != str(requested_model_path)

        if _ppe_detector is None or force_new or mode_changed or model_changed:

            _ppe_detector = PPEDetector(model_path, debug=debug, auto_recovery=auto_recovery, detection_mode=detection_mode)
        else:
            _ppe_detector.debug = debug
            _ppe_detector.detection_mode = detection_mode

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
