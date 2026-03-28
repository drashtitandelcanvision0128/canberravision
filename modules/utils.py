"""
Utility functions module for YOLO26 object detection.
Contains all helper functions, model management, color detection, and annotation utilities.
"""

import os
import sys
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO
import PIL.Image as Image
try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from optimized_paddleocr_gpu import extract_text_optimized
    OPTIMIZED_PADDLEOCR_AVAILABLE = True
except Exception:
    extract_text_optimized = None
    OPTIMIZED_PADDLEOCR_AVAILABLE = False

# Import enhanced detection for challenging images
try:
    from enhanced_detection import enhanced_license_plate_detection
    ENHANCED_DETECTION_AVAILABLE = True
    print("[INFO] Enhanced detection for challenging images loaded")
except ImportError:
    ENHANCED_DETECTION_AVAILABLE = False
    print("[WARNING] Enhanced detection not available")

# Import text extraction functions
try:
    from .text_extraction import _extract_text_ocr, _clean_general_text, _clean_license_plate_text, _is_valid_indian_license_plate
except ImportError:
    # Fallback functions if text_extraction is not available
    def _extract_text_ocr(crop_bgr: np.ndarray) -> str:
        return ""
    def _clean_general_text(text: str) -> str:
        return text if text else ""
    def _clean_license_plate_text(text: str) -> str:
        return text if text else ""
    def _is_valid_indian_license_plate(text: str) -> bool:
        return False

# Cache model for streaming performance
_model_cache = {}

# Cache ResNet for classification
_resnet_cache = {"model": None, "weights": None, "device": None, "transforms": None, "categories": None}
_resnet_stream_state = {"frame_idx": 0, "labels": {}}

# Cache gender classifier and face detector
_gender_cache = {"pipeline": None, "model_id": "dima806/fairface_gender_image_detection"}
_face_cache = {"cascade": None}


def _get_device():
    """Get the best available device for processing."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"[INFO] CUDA available with {device_count} GPU(s)")
        for i in range(device_count):
            print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)}")
        return 0  # Use first GPU
    else:
        print("[WARNING] CUDA not available, using CPU (slower performance)")
        return "cpu"


def get_model(model_name):
    """Get or create a cached model instance with CUDA support."""
    if "+" in str(model_name):
        # Return a list of models for ensemble-style inference
        parts = [p.strip() for p in str(model_name).split("+") if p.strip()]
        return [get_model(p) for p in parts]

    if model_name not in _model_cache:
        # Add .pt extension if not present
        if not model_name.endswith('.pt'):
            model_path = f"{model_name}.pt"
        else:
            model_path = model_name
            
        print(f"[INFO] Loading model: {model_path}")
        model = YOLO(model_path)
        
        # Print model info to verify it loaded correctly
        print(f"[INFO] Model loaded: {type(model)}")
        print(f"[INFO] Model names: {model.names}")
        
        # Move model to CUDA device if available
        device = _get_device()
        if device != "cpu":
            model.to(device)
            print(f"[INFO] Model moved to CUDA device: {device}")
        
        _model_cache[model_name] = model
        print(f"[INFO] Model {model_name} loaded and cached successfully")
    
    return _model_cache[model_name]


def _get_resnet18_classifier():
    if _resnet_cache["model"] is not None:
        return _resnet_cache

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    device = _get_device()
    if device != "cpu":
        model.to(device)

    _resnet_cache["model"] = model
    _resnet_cache["weights"] = weights
    _resnet_cache["device"] = device
    _resnet_cache["transforms"] = weights.transforms()
    _resnet_cache["categories"] = list(weights.meta.get("categories", []))
    return _resnet_cache


def _get_face_cascade():
    if _face_cache["cascade"] is not None:
        return _face_cache["cascade"]
    try:
        xml_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(xml_path)
        if cascade.empty():
            cascade = None
        _face_cache["cascade"] = cascade
        return cascade
    except Exception:
        _face_cache["cascade"] = None
        return None


def _get_gender_pipeline():
    if _gender_cache["pipeline"] is not None:
        return _gender_cache["pipeline"]
    try:
        from transformers import pipeline as hf_pipeline

        device = _get_device()
        device_arg = 0 if device != "cpu" else -1
        _gender_cache["pipeline"] = hf_pipeline(
            task="image-classification",
            model=_gender_cache["model_id"],
            device=device_arg,
        )
        return _gender_cache["pipeline"]
    except Exception as e:
        print(f"[WARNING] Gender pipeline not available: {e}")
        _gender_cache["pipeline"] = None
        return None


def _find_largest_face_bbox(face_cascade, img_bgr: np.ndarray):
    if face_cascade is None or img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        return None
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if faces is None or len(faces) == 0:
            return None
        # choose largest area
        best = None
        best_area = -1
        for (x, y, w, h) in faces:
            area = int(w) * int(h)
            if area > best_area:
                best_area = area
                best = (int(x), int(y), int(w), int(h))
        return best
    except Exception:
        return None


def _predict_boy_girl_from_person_crop(person_crop_bgr: np.ndarray) -> str | None:
    if person_crop_bgr is None or not isinstance(person_crop_bgr, np.ndarray) or person_crop_bgr.size == 0:
        return None

    face_cascade = _get_face_cascade()
    face_box = _find_largest_face_bbox(face_cascade, person_crop_bgr)
    if face_box is None:
        return None

    x, y, w, h = face_box
    ih, iw = person_crop_bgr.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(iw, x + w)
    y2 = min(ih, y + h)
    if x2 <= x1 or y2 <= y1:
        return None

    face_bgr = person_crop_bgr[y1:y2, x1:x2]
    if face_bgr.size == 0:
        return None

    pipe = _get_gender_pipeline()
    if pipe is None:
        return None

    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        out = pipe(pil_img)
        if not out:
            return None
        label = str(out[0].get("label", "")).strip().lower()
        # Common labels: 'Male', 'Female' or 'male', 'female'
        if "female" in label:
            return "girl"
        if "male" in label:
            return "boy"
        return None
    except Exception:
        return None


def _classify_color_bgr(crop_bgr: np.ndarray) -> str:
    """
    Enhanced color classification using MobileNetV2 + traditional methods
    Integrates with YOLO26 for high accuracy color detection
    """
    try:
        # Try advanced color detection first
        from advanced_color_detection import detect_object_color_advanced
        advanced_result = detect_object_color_advanced(crop_bgr, use_hybrid=True)
        
        if advanced_result and isinstance(advanced_result, dict):
            confidence = advanced_result.get("confidence", 0)
            if isinstance(confidence, (int, float)) and confidence > 0.7:
                detected_color = advanced_result.get("final_color", "unknown")
                confidence = advanced_result.get("confidence", 0)
                print(f"[DEBUG] Advanced color detection: {detected_color} (conf: {confidence:.3f})")
                return detected_color
            else:
                conf_val = advanced_result.get('confidence', 0) if isinstance(advanced_result.get('confidence', 0), (int, float)) else 0
                print(f"[DEBUG] Advanced detection confidence low ({conf_val:.3f}), using fallback")
            
    except ImportError:
        print("[DEBUG] Advanced color detection not available, using traditional method")
    except Exception as e:
        print(f"[DEBUG] Advanced color detection failed: {e}, using traditional method")
    
    # Fallback to traditional HSV-based detection
    return _classify_color_traditional_fallback(crop_bgr)


def _classify_color_traditional_fallback(crop_bgr: np.ndarray) -> str:
    """
    Traditional HSV-based color classification as fallback
    """
    if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
        return "unknown"

    h, w = crop_bgr.shape[:2]
    if h < 2 or w < 2:
        return "unknown"

    # Reduce background influence: focus on the center region of the bbox
    margin_y = int(max(0, round(h * 0.2)))
    margin_x = int(max(0, round(w * 0.2)))
    if (h - 2 * margin_y) >= 2 and (w - 2 * margin_x) >= 2:
        crop_bgr = crop_bgr[margin_y : h - margin_y, margin_x : w - margin_x]
        h, w = crop_bgr.shape[:2]

    max_side = 64
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        crop_bgr = cv2.resize(crop_bgr, (max(2, int(w * scale)), max(2, int(h * scale))))

    # Light blur helps reduce sensor noise / compression artifacts in video.
    crop_bgr = cv2.GaussianBlur(crop_bgr, (3, 3), 0)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hsv_reshaped = hsv.reshape(-1, 3)

    s = hsv_reshaped[:, 1].astype(np.float32)
    v = hsv_reshaped[:, 2].astype(np.float32)

    # Enhanced black detection - more strict criteria
    v_mean = float(np.mean(v))
    s_mean = float(np.mean(s))
    
    # Check for black first (most common issue)
    if v_mean < 45:  # Lowered threshold for better black detection
        return "black"
    
    # Very dark but not pure black
    if v_mean < 55 and s_mean < 25:
        return "black"

    # Ignore very dark pixels and low-saturation pixels for other colors
    valid = (v >= 50) & (s >= 35)
    if not np.any(valid):
        if v_mean < 60:
            return "black"
        if v_mean > 200 and s_mean < 30:
            return "white"
        return "gray"

    hsv_valid = hsv_reshaped[valid]
    s_mean_valid = float(np.mean(hsv_valid[:, 1]))
    v_mean_valid = float(np.mean(hsv_valid[:, 2]))

    if v_mean_valid < 65:  # More lenient black detection for valid pixels
        return "black"
    if v_mean_valid > 210 and s_mean_valid < 35:
        return "white"
    if s_mean_valid < 30:
        return "gray"

    hsv_full = hsv.reshape(-1, 3)
    valid_mask = valid.reshape(-1)
    if np.sum(valid_mask) < 20:
        return "unknown"

    bgr = crop_bgr.reshape(-1, 3)
    bgr_valid = bgr[valid_mask]
    bgr_valid_u8 = np.clip(bgr_valid, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    lab_valid = cv2.cvtColor(bgr_valid_u8, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    lab_med = np.median(lab_valid, axis=0)

    palette_bgr = {
        "red": (0, 0, 255),
        "orange": (0, 165, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "cyan": (255, 255, 0),
        "blue": (255, 0, 0),
        "purple": (255, 0, 255),
        "pink": (203, 192, 255),
        "brown": (42, 42, 165),
    }

    best_name = "unknown"
    best_dist = float("inf")
    for name, bgr_ref in palette_bgr.items():
        ref_u8 = np.array([[bgr_ref]], dtype=np.uint8)
        lab_ref = cv2.cvtColor(ref_u8, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
        dist = float(np.linalg.norm(lab_med - lab_ref))
        
        # Enhanced brown filtering - be more strict about brown detection
        if name == "brown":
            # Brown should have moderate brightness and saturation
            if v_mean_valid > 150 or v_mean_valid < 80:
                dist *= 2.0  # Penalize brown if brightness is wrong
            if s_mean_valid < 20 or s_mean_valid > 60:
                dist *= 1.5  # Penalize brown if saturation is wrong
        
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name


def _detect_car_color_around_plate(image_bgr: np.ndarray, plate_x1: int, plate_y1: int, plate_x2: int, plate_y2: int) -> str:
    """
    Detect the car color by analyzing the area around the license plate.
    
    Args:
        image_bgr: Full image in BGR format
        plate_x1, plate_y1, plate_x2, plate_y2: License plate bounding box
        
    Returns:
        Detected car color
    """
    try:
        h, w = image_bgr.shape[:2]
        plate_width = plate_x2 - plate_x1
        plate_height = plate_y2 - plate_y1
        
        # Define sampling areas around the license plate
        sampling_areas = []
        
        # Area above the plate (most likely car body)
        above_y1 = max(0, plate_y1 - plate_height)
        above_y2 = plate_y1
        above_x1 = max(0, plate_x1 - plate_width // 4)
        above_x2 = min(w, plate_x2 + plate_width // 4)
        if above_y2 > above_y1 and above_x2 > above_x1:
            sampling_areas.append((above_x1, above_y1, above_x2, above_y2))
        
        # Area below the plate (car body)
        below_y1 = plate_y2
        below_y2 = min(h, plate_y2 + plate_height)
        below_x1 = max(0, plate_x1 - plate_width // 4)
        below_x2 = min(w, plate_x2 + plate_width // 4)
        if below_y2 > below_y1 and below_x2 > below_x1:
            sampling_areas.append((below_x1, below_y1, below_x2, below_y2))
        
        # Area to the left of the plate
        left_x1 = max(0, plate_x1 - plate_width // 2)
        left_x2 = plate_x1
        left_y1 = max(0, plate_y1 - plate_height // 4)
        left_y2 = min(h, plate_y2 + plate_height // 4)
        if left_x2 > left_x1 and left_y2 > left_y1:
            sampling_areas.append((left_x1, left_y1, left_x2, left_y2))
        
        # Area to the right of the plate
        right_x1 = plate_x2
        right_x2 = min(w, plate_x2 + plate_width // 2)
        right_y1 = max(0, plate_y1 - plate_height // 4)
        right_y2 = min(h, plate_y2 + plate_height // 4)
        if right_x2 > right_x1 and right_y2 > right_y1:
            sampling_areas.append((right_x1, right_y1, right_x2, right_y2))
        
        # Analyze each sampling area
        color_votes = {}
        
        for area_x1, area_y1, area_x2, area_y2 in sampling_areas:
            # Extract the area
            area_crop = image_bgr[area_y1:area_y2, area_x1:area_x2]
            
            if area_crop.size == 0:
                continue
            
            # Skip if the area is too white (might be plate background)
            mean_bgr = np.mean(area_crop, axis=(0, 1))
            if mean_bgr[0] > 200 and mean_bgr[1] > 200 and mean_bgr[2] > 200:
                continue  # Skip very white areas
            
            # Detect color in this area
            area_color = _classify_color_bgr(area_crop)
            
            # Vote for this color
            if area_color in color_votes:
                color_votes[area_color] += 1
            else:
                color_votes[area_color] = 1
        
        # Select the color with the most votes
        if color_votes:
            best_color = max(color_votes, key=color_votes.get)
            print(f"[DEBUG] Car color detected around license plate: {best_color} (votes: {color_votes})")
            return best_color
        
        # Fallback: analyze a larger area around the plate
        expand_x = plate_width // 2
        expand_y = plate_height // 2
        large_x1 = max(0, plate_x1 - expand_x)
        large_y1 = max(0, plate_y1 - expand_y)
        large_x2 = min(w, plate_x2 + expand_x)
        large_y2 = min(h, plate_y2 + expand_y)
        
        large_area = image_bgr[large_y1:large_y2, large_x1:large_x2]
        if large_area.size > 0:
            fallback_color = _classify_color_bgr(large_area)
            print(f"[DEBUG] Fallback car color: {fallback_color}")
            return fallback_color
    
    except Exception as e:
        print(f"[DEBUG] Error detecting car color around plate: {e}")
    
    return "unknown"


def detect_vehicles_in_image(image_bgr: np.ndarray) -> list:
    """
    Detect vehicles (cars, trucks, buses, motorcycles) in the image.
    Returns list of vehicle bounding boxes with class information.
    
    Args:
        image_bgr: Input image in BGR format
        
    Returns:
        List of tuples: (x1, y1, x2, y2, class_name, confidence)
    """
    vehicle_detections = []
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
    
    try:
        model = get_model("yolo26n.pt")
        detection_results = model(image_bgr)
        
        if detection_results and len(detection_results) > 0:
            detection = detection_results[0]
            if hasattr(detection, 'boxes') and detection.boxes is not None:
                boxes = detection.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                names = detection.names
                
                for i in range(len(xyxy)):
                    if conf[i] > 0.5:  # Confidence threshold
                        class_id = int(cls[i])
                        class_name = names.get(class_id, f"class_{class_id}")
                        
                        # Check if this is a vehicle
                        if class_name.lower() in vehicle_classes:
                            x1, y1, x2, y2 = map(int, xyxy[i])
                            confidence = float(conf[i])
                            vehicle_detections.append((x1, y1, x2, y2, class_name, confidence))
                            print(f"[DEBUG] Vehicle detected: {class_name} at ({x1}, {y1}, {x2}, {y2}) with conf {confidence:.3f}")
        
        print(f"[DEBUG] Total vehicles found: {len(vehicle_detections)}")
        
    except Exception as e:
        print(f"[DEBUG] Error in vehicle detection: {e}")
    
    return vehicle_detections


def detect_license_plates_as_objects(image_bgr: np.ndarray) -> list:
    """
    Enhanced license plate detection with fallback for challenging images.
    NOW ONLY DETECTS LICENSE PLATES WHEN VEHICLES ARE PRESENT.
    """
    license_plate_regions = []
    
    # STEP 1: First detect vehicles in the image
    print(f"[DEBUG] Step 1: Detecting vehicles before license plate detection...")
    vehicles = detect_vehicles_in_image(image_bgr)
    
    if not vehicles:
        print(f"[DEBUG] No vehicles detected in image. Skipping license plate detection.")
        return license_plate_regions
    
    print(f"[DEBUG] Found {len(vehicles)} vehicles. Proceeding with license plate detection...")
    
    try:
        # Method 1: Standard YOLO detection
        model = get_model("yolo26n.pt")
        detection_results = model(image_bgr)
        
        if detection_results and len(detection_results) > 0:
            detection = detection_results[0]
            if hasattr(detection, 'boxes') and detection.boxes is not None:
                boxes = detection.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                
                for i in range(len(xyxy)):
                    if conf[i] > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = map(int, xyxy[i])
                        
                        # Check if license plate is near any vehicle
                        plate_near_vehicle = False
                        for vx1, vy1, vx2, vy2, vclass, vconf in vehicles:
                            # Calculate distance between plate and vehicle
                            plate_center_x = (x1 + x2) / 2
                            plate_center_y = (y1 + y2) / 2
                            vehicle_center_x = (vx1 + vx2) / 2
                            vehicle_center_y = (vy1 + vy2) / 2
                            
                            # Check if plate is within or near vehicle bounds (with some margin)
                            margin = 50  # 50 pixel margin
                            if (vx1 - margin <= plate_center_x <= vx2 + margin and 
                                vy1 - margin <= plate_center_y <= vy2 + margin):
                                plate_near_vehicle = True
                                print(f"[DEBUG] License plate near {vclass}: plate ({x1},{y1},{x2},{y2}) near vehicle ({vx1},{vy1},{vx2},{vy2})")
                                break
                        
                        if plate_near_vehicle:
                            license_plate_regions.append((x1, y1, x2, y2))
                        else:
                            print(f"[DEBUG] License plate at ({x1},{y1},{x2},{y2}) ignored - not near any vehicle")
        
        # Method 2: Enhanced detection for challenging images (if no plates found near vehicles)
        if not license_plate_regions and ENHANCED_DETECTION_AVAILABLE:
            print("[DEBUG] No plates found near vehicles with YOLO, trying enhanced detection...")
            enhanced_result = enhanced_license_plate_detection(image_bgr)
            
            if enhanced_result["plate_detected"] and enhanced_result["plate_bbox"]:
                x1, y1, x2, y2 = enhanced_result["plate_bbox"]
                
                # Check if enhanced detection is near any vehicle
                plate_near_vehicle = False
                for vx1, vy1, vx2, vy2, vclass, vconf in vehicles:
                    plate_center_x = (x1 + x2) / 2
                    plate_center_y = (y1 + y2) / 2
                    
                    margin = 50
                    if (vx1 - margin <= plate_center_x <= vx2 + margin and 
                        vy1 - margin <= plate_center_y <= vy2 + margin):
                        plate_near_vehicle = True
                        print(f"[DEBUG] Enhanced license plate detection found plate near {vclass}")
                        break
                
                if plate_near_vehicle:
                    license_plate_regions.append((x1, y1, x2, y2))
                    print(f"[DEBUG] Enhanced detection found plate: {enhanced_result}")
                else:
                    print(f"[DEBUG] Enhanced detection found plate but not near any vehicle - ignoring")
                
    except Exception as e:
        print(f"[DEBUG] Error in license plate detection: {e}")
    
    print(f"[DEBUG] Final license plate regions found: {len(license_plate_regions)}")
    return license_plate_regions


def _detect_license_plate_in_car(car_bgr: np.ndarray) -> np.ndarray:
    """
    Enhanced license plate detection within a car bounding box using multiple computer vision techniques.
    Specifically optimized for Indian license plates like "MH 20 EE 7602".
    
    Args:
        car_bgr: Car image in BGR format
        
    Returns:
        License plate crop in BGR format, or None if not found
    """
    if car_bgr is None or not isinstance(car_bgr, np.ndarray) or car_bgr.size == 0:
        return None
    
    h, w = car_bgr.shape[:2]
    if h < 20 or w < 20:
        return None
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(car_bgr, cv2.COLOR_BGR2GRAY)
        
        # Store all potential plate candidates
        plate_candidates = []
        
        # Method 1: Enhanced edge-based detection with multiple thresholds
        edges_low = cv2.Canny(gray, 30, 100)
        edges_med = cv2.Canny(gray, 50, 150)
        edges_high = cv2.Canny(gray, 70, 200)
        
        for edge_name, edges in [("low", edges_low), ("med", edges_med), ("high", edges_high)]:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect_ratio = cw / ch
                area = cw * ch
                
                # Enhanced criteria for Indian license plates
                # Indian plates are typically rectangular with aspect ratio 2.0-5.0
                if (1.8 <= aspect_ratio <= 5.5 and 
                    cw >= 60 and ch >= 15 and  # Minimum size for Indian plates
                    area > 800 and 
                    area < (h * w) * 0.4):  # Not too large
                    
                    # Additional validation: check if it's in typical plate location
                    confidence = _calculate_plate_confidence(x, y, cw, ch, w, h, aspect_ratio, area)
                    plate_candidates.append((x, y, cw, ch, confidence, edge_name))
        
        # Method 2: MSER for text-like regions (enhanced)
        mser = cv2.MSER_create()
        # Set parameters using set methods instead of constructor
        mser.setDelta(5)
        mser.setMinArea(100)
        mser.setMaxArea(int(w*h*0.3))
        regions, _ = mser.detectRegions(gray)
        
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            x, y, cw, ch = cv2.boundingRect(hull)
            aspect_ratio = cw / ch
            area = cw * ch
            
            if (1.8 <= aspect_ratio <= 5.5 and 
                cw >= 50 and ch >= 12 and 
                area > 500 and 
                area < (h * w) * 0.25):
                
                confidence = _calculate_plate_confidence(x, y, cw, ch, w, h, aspect_ratio, area)
                plate_candidates.append((x, y, cw, ch, confidence, "mser"))
        
        # Method 3: Color-based detection (white/light plates on dark backgrounds)
        # Convert to LAB color space for better color separation
        lab = cv2.cvtColor(car_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Threshold for light regions (potential white plates)
        _, l_thresh = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours in light regions
        contours_light, _ = cv2.findContours(l_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_light:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = cw / ch
            area = cw * ch
            
            if (1.8 <= aspect_ratio <= 5.5 and 
                cw >= 60 and ch >= 15 and 
                area > 800 and 
                area < (h * w) * 0.3):
                
                confidence = _calculate_plate_confidence(x, y, cw, ch, w, h, aspect_ratio, area)
                plate_candidates.append((x, y, cw, ch, confidence, "color"))
        
        # Method 4: Search in typical license plate locations (bottom portion of car)
        # Indian plates are usually in the bottom 1/3 to 1/4 of the car
        search_regions = [
            (int(h * 0.6), h),  # Bottom 40%
            (int(h * 0.7), h),  # Bottom 30%
            (int(h * 0.75), h), # Bottom 25%
        ]
        
        for start_y, end_y in search_regions:
            region = gray[start_y:end_y, :]
            if region.size > 0:
                # Apply adaptive threshold
                thresh_region = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
                
                contours_region, _ = cv2.findContours(thresh_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours_region:
                    x, y_local, cw, ch = cv2.boundingRect(contour)
                    y = start_y + y_local
                    aspect_ratio = cw / ch
                    area = cw * ch
                    
                    if (2.0 <= aspect_ratio <= 4.5 and 
                        cw >= 70 and ch >= 18 and 
                        area > 1000):
                        
                        confidence = _calculate_plate_confidence(x, y, cw, ch, w, h, aspect_ratio, area)
                        plate_candidates.append((x, y, cw, ch, confidence, f"location_{start_y}"))
        
        # Select best candidates based on confidence
        if plate_candidates:
            # Sort by confidence (descending)
            plate_candidates.sort(key=lambda x: x[4], reverse=True)
            
            print(f"[DEBUG] Found {len(plate_candidates)} plate candidates, testing top 5...")
            
            # Try top candidates
            for i, (x, y, cw, ch, confidence, method) in enumerate(plate_candidates[:5]):
                # Add margin around the detected plate
                margin = 8
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + cw + margin)
                y2 = min(h, y + ch + margin)
                
                plate_crop = car_bgr[y1:y2, x1:x2]
                if plate_crop.size > 0:
                    # Validate the plate region
                    if _validate_plate_region(plate_crop):
                        print(f"[DEBUG] Valid plate found using {method} method (confidence: {confidence:.2f})")
                        return plate_crop
                    else:
                        print(f"[DEBUG] Plate candidate {i+1} ({method}) failed validation")
        
        # Method 5: Last resort - try multiple regions in bottom area
        print("[DEBUG] Trying fallback detection in bottom regions...")
        bottom_regions = [
            (int(h * 0.65), int(h * 0.85)),  # Middle-bottom
            (int(h * 0.75), h),              # Bottom
        ]
        
        for start_y, end_y in bottom_regions:
            if start_y < h:
                region = car_bgr[start_y:end_y, :]
                if region.size > 0:
                    # Try OCR directly on this region
                    ocr_result = _extract_text_ocr(region)
                    if ocr_result and _is_valid_license_plate(ocr_result):
                        print(f"[DEBUG] Found plate text in fallback region: {ocr_result}")
                        return region
        
    except Exception as e:
        print(f"[DEBUG] License plate detection error: {e}")
    
    return None


def _calculate_plate_confidence(x, y, w, h, car_w, car_h, aspect_ratio, area):
    """Calculate confidence score for a potential license plate region."""
    confidence = 0.0
    
    # Position confidence (plates are usually in bottom half)
    position_score = 1.0 - (y / car_h)  # Higher score for lower position
    confidence += position_score * 0.3
    
    # Aspect ratio confidence (ideal around 2.5-3.5 for Indian plates)
    if 2.0 <= aspect_ratio <= 4.0:
        confidence += 0.3
    elif 1.8 <= aspect_ratio <= 5.0:
        confidence += 0.2
    
    # Size confidence (not too small, not too large)
    size_ratio = area / (car_w * car_h)
    if 0.02 <= size_ratio <= 0.15:
        confidence += 0.2
    elif 0.01 <= size_ratio <= 0.2:
        confidence += 0.1
    
    # Center alignment (plates are usually somewhat centered horizontally)
    center_x = x + w / 2
    center_score = 1.0 - abs(center_x - car_w / 2) / (car_w / 2)
    confidence += center_score * 0.2
    
    return confidence


def _validate_plate_region(plate_bgr: np.ndarray) -> bool:
    """
    Validate if a region is likely to be a license plate based on text characteristics.
    
    Args:
        plate_bgr: Plate region in BGR format
        
    Returns:
        True if likely a license plate, False otherwise
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return False
    
    try:
        h, w = plate_bgr.shape[:2]
        if h < 8 or w < 20:
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        
        # Check for high contrast (typical of license plates)
        contrast = gray.std()
        if contrast < 30:  # Low contrast suggests not a license plate
            return False
        
        # Quick OCR check - see if we can extract some alphanumeric characters
        try:
            if pytesseract is not None:
                # Simple threshold and OCR
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text = pytesseract.image_to_string(thresh, config=r'--oem 3 --psm 7', lang='eng')
                
                # Check if we found alphanumeric characters
                alnum_chars = sum(c.isalnum() for c in text)
                if alnum_chars >= 3:  # At least 3 alphanumeric characters
                    return True
        except:
            pass
        
        # If OCR fails, use heuristics
        # License plates typically have regular patterns
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Moderate edge density suggests text/characters
        if 0.05 <= edge_density <= 0.4:
            return True
            
    except Exception:
        pass
    
    return False


def _is_valid_license_plate(text: str) -> bool:
    """Check if text looks like a valid license plate."""
    if not text or len(text) < 4:
        return False
    
    # Remove any non-alphanumeric characters
    alnum_text = ''.join(c for c in text if c.isalnum())
    
    if len(alnum_text) < 4:
        return False
    
    # Check for minimum alphanumeric content
    alnum_ratio = sum(c.isalnum() for c in text) / len(text)
    if alnum_ratio < 0.7:
        return False
    
    # Common license plate patterns
    # Pattern 1: 2 letters + 2 numbers + 2 letters (e.g., MH 20 EE)
    # Pattern 2: Numbers and letters mixed
    # Pattern 3: All numbers
    # Pattern 4: All letters
    
    has_letter = any(c.isalpha() for c in alnum_text)
    has_number = any(c.isdigit() for c in alnum_text)
    
    # Valid if it has both letters and numbers, or is sufficiently long
    if (has_letter and has_number) or len(alnum_text) >= 6:
        return True
    
    return False


@torch.inference_mode()
def _classify_object_resnet18(crop_bgr: np.ndarray) -> str:
    if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
        return "unknown"
    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 10:
        return "unknown"

    cache = _get_resnet18_classifier()
    model = cache["model"]
    device = cache["device"]
    tfm = cache["transforms"]
    categories = cache["categories"]

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    x = tfm(pil_img).unsqueeze(0)
    if device != "cpu":
        x = x.to(device)

    logits = model(x)
    idx = int(torch.argmax(logits, dim=1).item())
    if categories and 0 <= idx < len(categories):
        return str(categories[idx])
    return str(idx)


def _generate_detection_summary(result, enable_resnet=False, enable_ocr=False):
    """Generate a summary of all detected objects with their details."""
    if result is None or not hasattr(result, "boxes") or result.boxes is None:
        return "No objects detected"
    
    boxes = result.boxes
    if len(boxes) == 0:
        return "No objects detected"
    
    names = getattr(result, "names", None)
    if names is None and hasattr(result, "model") and hasattr(result.model, "names"):
        names = result.model.names
    
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
    cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls).astype(int)
    conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
    
    summary_lines = []
    summary_lines.append(f"📊 **Detected {len(boxes)} objects:**")
    summary_lines.append("")
    
    for i in range(len(boxes)):
        class_id = int(cls[i]) if i < len(cls) else -1
        class_name = str(class_id)
        if isinstance(names, dict):
            class_name = names.get(class_id, class_name)
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            class_name = names[class_id]
        
        confidence = f"{float(conf[i]):.2f}" if i < len(conf) else "N/A"
        
        line = f"🔹 **{class_name}** (conf: {confidence})"
        
        # Add color and text info
        try:
            x1, y1, x2, y2 = xyxy[i]
            x1 = int(max(0, round(x1)))
            y1 = int(max(0, round(y1)))
            x2 = int(round(x2))
            y2 = int(round(y2))
            
            # We need the original frame for color extraction, but we don't have it here
            # So we'll just note that color detection is enabled
            if hasattr(result, 'orig_img') and result.orig_img is not None:
                crop = result.orig_img[y1:y2, x1:x2]
                if crop.size > 0:
                    color = _classify_color_bgr(crop)
                    line += f" - **Color:** {color}"
                    
                    # Try to detect text on any object if OCR is enabled
                    if enable_ocr:
                        object_text = None
                        
                        # Special handling for cars: try license plate first
                        if str(class_name).strip().lower() == "car":
                            plate_crop = _detect_license_plate_in_car(crop)
                            if plate_crop is not None:
                                plate_text = _extract_text_ocr(plate_crop)
                                if plate_text and plate_text.strip():
                                    line += f" - **License Plate:** {plate_text}"
                                    object_text = plate_text
                        
                        # Try to detect general text on the object using LightOnOCR
                        general_text = None
                        try:
                            from .text_extraction import LIGHTON_AVAILABLE, extract_text_with_lighton
                            if LIGHTON_AVAILABLE:
                                lighton_result = extract_text_with_lighton(crop, confidence_threshold=0.3)
                                if lighton_result and lighton_result.strip():
                                    cleaned_general = _clean_general_text(lighton_result)
                                    if cleaned_general and len(cleaned_general) >= 2:
                                        general_text = cleaned_general
                        except:
                            pass
                        
                        # Fallback to regular OCR for general text
                        if general_text is None:
                            general_text = _extract_text_ocr(crop)
                            if general_text and general_text.strip():
                                general_text = _clean_general_text(general_text)
                        
                        # Add general text if found and it's different from license plate
                        if general_text and general_text.strip():
                            if object_text is None or general_text.lower() != object_text.lower():
                                if str(class_name).strip().lower() == "car" and object_text:
                                    line += f" - **Text:** {general_text}"
                                else:
                                    line += f" - **Text:** {general_text}"
        except:
            pass
        
        summary_lines.append(line)
    
    return "\n".join(summary_lines)


def _annotate_with_color(
    frame_bgr: np.ndarray,
    result,
    show_labels: bool,
    show_conf: bool,
    enable_resnet: bool = False,
    max_boxes: int = 10,
    resnet_every_n: int = 1,
    stream_key_prefix: str | None = None,
    enable_ocr: bool = False,
    ocr_every_n: int = 1,
):
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
        return frame_bgr
    if result is None or not hasattr(result, "boxes") or result.boxes is None:
        return frame_bgr

    boxes = result.boxes
    if len(boxes) == 0:
        return frame_bgr

    names = getattr(result, "names", None)
    if names is None and hasattr(result, "model") and hasattr(result.model, "names"):
        names = result.model.names

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
    cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls).astype(int)
    conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)

    annotated = frame_bgr.copy()
    ih, iw = annotated.shape[:2]

    total = len(xyxy)
    max_boxes = int(max(1, max_boxes))
    take = min(total, max_boxes)

    for i in range(take):
        x1, y1, x2, y2 = xyxy[i]
        x1 = int(max(0, min(iw - 1, round(x1))))
        y1 = int(max(0, min(ih - 1, round(y1))))
        x2 = int(max(0, min(iw - 1, round(x2))))
        y2 = int(max(0, min(ih - 1, round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = annotated[y1:y2, x1:x2]
        color_name = _classify_color_bgr(crop)

        # Get class name early to avoid UnboundLocalError
        class_id = int(cls[i]) if i < len(cls) else -1
        class_name = str(class_id)
        if isinstance(names, dict):
            class_name = names.get(class_id, class_name)
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            class_name = names[class_id]

        resnet_label = None
        if enable_resnet:
            resnet_every_n = int(max(1, resnet_every_n))
            key = None
            if stream_key_prefix:
                key = f"{stream_key_prefix}:{i}"

            do_classify = True
            if stream_key_prefix:
                _resnet_stream_state["frame_idx"] += 1
                do_classify = (_resnet_stream_state["frame_idx"] % resnet_every_n) == 0
                if not do_classify and key in _resnet_stream_state["labels"]:
                    resnet_label = _resnet_stream_state["labels"][key]

            if resnet_label is None and do_classify:
                resnet_label = _classify_object_resnet18(crop)
                if key is not None:
                    _resnet_stream_state["labels"][key] = resnet_label

        ocr_text = None
        plate_text = None
        if enable_ocr:
            ocr_every_n = int(max(1, ocr_every_n))
            key = None
            if stream_key_prefix:
                key = f"{stream_key_prefix}:ocr:{i}"

            do_ocr = True
            plate_text = None  # Initialize plate_text
            if stream_key_prefix:
                _ocr_stream_state["frame_idx"] += 1
                do_ocr = (_ocr_stream_state["frame_idx"] % ocr_every_n) == 0
                if not do_ocr and key in _ocr_stream_state["texts"]:
                    ocr_text = _ocr_stream_state["texts"][key]

            if ocr_text is None and do_ocr:
                # Special handling for vehicles: try to detect license plate first
                vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
                if str(class_name).strip().lower() in vehicle_classes:
                    plate_crop = _detect_license_plate_in_car(crop)  # This function works for all vehicles
                    if plate_crop is not None:
                        print(f"[DEBUG] License plate detected in {class_name}, extracting text...")
                        plate_text = _extract_text_ocr(plate_crop)
                        if plate_text and plate_text.strip():
                            print(f"[DEBUG] License plate text: '{plate_text}'")
                            ocr_text = f"Plate: {plate_text}"
                        else:
                            print("[DEBUG] No text extracted from license plate")
                    else:
                        print(f"[DEBUG] No license plate detected in {class_name}")
                
                # For all objects, try to detect any text using LightOnOCR
                general_text = None
                try:
                    from .text_extraction import LIGHTON_AVAILABLE, extract_text_with_lighton
                    if LIGHTON_AVAILABLE:
                        print(f"[DEBUG] Using LightOnOCR to detect text on {class_name}...")
                        # Use LightOnOCR for general text detection on the object
                        lighton_result = extract_text_with_lighton(crop, confidence_threshold=0.3)  # Lower threshold for general text
                        
                        if lighton_result and lighton_result.strip():
                            # Clean the text but keep it more flexible for general text
                            cleaned_general = _clean_general_text(lighton_result)
                            if cleaned_general and len(cleaned_general) >= 2:
                                print(f"[DEBUG] LightOnOCR found text on {class_name}: '{cleaned_general}'")
                                general_text = cleaned_general
                except Exception as e:
                    print(f"[DEBUG] LightOnOCR general text detection failed: {e}")
                
                # If no general text found with LightOnOCR, try regular OCR
                if general_text is None:
                    if OPTIMIZED_PADDLEOCR_AVAILABLE and extract_text_optimized is not None:
                        try:
                            ocr_out = extract_text_optimized(
                                crop,
                                confidence_threshold=0.2,
                                lang='en',
                                use_gpu=None,
                                use_cache=True,
                                preprocess=True,
                            )
                            general_text = (ocr_out.get('text') or '').strip()
                        except Exception:
                            general_text = None

                    if general_text is None:
                        general_text = _extract_text_ocr(crop)
                    if general_text and general_text.strip():
                        general_text = _clean_general_text(general_text)
                
                # Combine results: prioritize license plate for vehicles, otherwise use general text
                if ocr_text and general_text:
                    # For vehicles, show both plate and general text if different
                    if str(class_name).strip().lower() in vehicle_classes and plate_text:
                        if general_text.lower() != plate_text.lower():
                            ocr_text = f"Plate: {plate_text} | Text: {general_text}"
                    else:
                        ocr_text = general_text
                elif general_text:
                    ocr_text = general_text
                    
                if key is not None:
                    _ocr_stream_state["texts"][key] = ocr_text

        boy_girl = None
        boy_girl = None

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if show_labels:
            parts = [str(class_name)]
            if show_conf and i < len(conf):
                parts.append(f"{float(conf[i]):.2f}")
            parts.append(str(color_name))
            if enable_resnet and resnet_label:
                parts.append("|")
                parts.append(str(resnet_label))
            # Add license plate text at TOP with car label
            if plate_text:
                parts.append("|")
                parts.append(f"PLATE:{plate_text}")
            text = " ".join(parts)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            ty = y1 - 8
            if ty - th - baseline < 0:
                ty = y1 + th + baseline + 8

            bg_x1 = x1
            bg_y1 = ty - th - baseline
            bg_x2 = x1 + tw + 6
            bg_y2 = ty + 4
            bg_x2 = min(iw - 1, bg_x2)
            bg_y1 = max(0, bg_y1)
            bg_y2 = min(ih - 1, bg_y2)
            cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), -1)
            cv2.putText(annotated, text, (x1 + 3, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            
            if enable_ocr and ocr_text:
                ocr_disp = str(ocr_text).strip()
                if ocr_disp:
                    ocr_disp = ocr_disp[:42]
                    (otw, oth), obase = cv2.getTextSize(ocr_disp, font, 0.55, 2)
                    oy = y2 + oth + 6
                    if oy + obase + 6 > ih:
                        oy = max(20, y1 - 10)
                    ox1 = x1
                    ox2 = min(iw - 1, x1 + otw + 8)
                    oy1 = max(0, oy - oth - obase - 6)
                    oy2 = min(ih - 1, oy + 6)
                    cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (0, 0, 0), -1)
                    cv2.putText(annotated, ocr_disp, (x1 + 4, oy), font, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    return annotated


def _annotate_from_json_results(frame_bgr: np.ndarray, json_results: dict, show_labels: bool) -> np.ndarray:
    """
    Annotate the image with text extraction results from JSON data.
    
    Args:
        frame_bgr: Input image in BGR format
        json_results: JSON text extraction results
        show_labels: Whether to show labels
        
    Returns:
        Annotated image in BGR format
    """
    if not json_results or "text_extraction" not in json_results:
        return frame_bgr
    
    extraction = json_results["text_extraction"]
    annotated = frame_bgr.copy()
    
    # Annotate license plates (including those found in full image text)
    for plate_info in extraction["license_plates"]:
        object_id = plate_info["object_id"]
        plate_text = plate_info["plate_text"]
        confidence = plate_info["confidence"]
        
        if object_id == "full_image":
            # License plate found in full image text - show at top of image
            h, w = annotated.shape[:2]
            
            # Create a banner at the top for license plate info
            plate_label = f"🚗 License Plate: {plate_text} (confidence: {confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (tw, th), baseline = cv2.getTextSize(plate_label, font, font_scale, thickness)
            
            # Draw background banner
            banner_y = 30
            cv2.rectangle(annotated, (10, banner_y - th - 10), (tw + 20, banner_y + 5), (0, 255, 255), -1)
            cv2.putText(annotated, plate_label, (15, banner_y), font, font_scale, (0, 0, 0), thickness)
            
            # Also draw a yellow border around the entire image to indicate license plate found
            cv2.rectangle(annotated, (5, 5), (w-5, h-5), (0, 255, 255), 3)
            
        else:
            # Use the plate's own bounding box if available, otherwise find the object
            plate_bbox = plate_info.get("bounding_box")
            
            if plate_bbox:
                # Use the plate's own bounding box directly
                x1, y1, x2, y2 = plate_bbox["x1"], plate_bbox["y1"], plate_bbox["x2"], plate_bbox["y2"]
                
                # Draw license plate bounding box in yellow
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Yellow box for plates
                
                # Add license plate text
                plate_label = f"Plate: {plate_text}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                (tw, th), baseline = cv2.getTextSize(plate_label, font, font_scale, thickness)
                ty = y1 - 10
                if ty - th - baseline < 0:
                    ty = y1 + th + baseline + 10
                
                # Background for text
                cv2.rectangle(annotated, (x1, ty - th - baseline), (x1 + tw + 4, ty + 4), (0, 255, 255), -1)
                cv2.putText(annotated, plate_label, (x1 + 2, ty), font, font_scale, (0, 0, 0), thickness)
            else:
                # Find the corresponding object for regular license plates
                for obj in extraction["all_objects"]:
                    if obj["object_id"] == object_id:
                        bbox = obj["bounding_box"]
                        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                        
                        # Draw license plate bounding box in different color
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Yellow box for plates
                        
                        # Add license plate text
                        plate_label = f"Plate: {plate_text}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        
                        (tw, th), baseline = cv2.getTextSize(plate_label, font, font_scale, thickness)
                        ty = y1 - 10
                        if ty - th - baseline < 0:
                            ty = y1 + th + baseline + 10
                        
                        # Background for text
                        cv2.rectangle(annotated, (x1, ty - th - baseline), (x1 + tw + 4, ty + 4), (0, 255, 255), -1)
                        cv2.putText(annotated, plate_label, (x1 + 2, ty), font, font_scale, (0, 0, 0), thickness)
                        break
    
    # Annotate general text (excluding full image text to avoid clutter)
    for text_info in extraction["general_text"]:
        object_id = text_info["object_id"]
        text = text_info["text"]
        confidence = text_info["confidence"]
        
        # Skip full image general text to avoid clutter
        if object_id == "full_image":
            continue
        
        # Find the corresponding object
        for obj in extraction["all_objects"]:
            if obj["object_id"] == object_id:
                bbox = obj["bounding_box"]
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                
                # Add general text annotation
                text_label = f"Text: {text}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                
                (tw, th), baseline = cv2.getTextSize(text_label, font, font_scale, thickness)
                ty = y2 + 20  # Place text below the object
                
                # Background for text
                cv2.rectangle(annotated, (x1, ty - th - baseline), (x1 + tw + 4, ty + 4), (255, 255, 0), -1)
                cv2.putText(annotated, text_label, (x1 + 2, ty), font, font_scale, (0, 0, 0), thickness)
                break
    
    return annotated


# Initialize OCR stream state
_ocr_stream_state = {"frame_idx": 0, "texts": {}}
