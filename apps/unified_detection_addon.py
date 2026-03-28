"""
Unified Detection Functions for app.py
Add these functions to the main app.py file for unified detection support
"""

import cv2
import numpy as np
import json
import time
import os
from datetime import datetime
from typing import Tuple, Dict, Any, List
from PIL import Image
import gradio as gr

# Import unified detection from src
try:
    from src.unified_detection.unified_detector import UnifiedDetector, get_unified_detector
    from src.unified_detection.result_formatter import ResultFormatter
    UNIFIED_AVAILABLE = True
except ImportError:
    UNIFIED_AVAILABLE = False
    print("[WARNING] Unified detection module not available")

# Global instances for unified detection
_unified_detector = None
_unified_formatter = None


def get_unified_detector_instance():
    """Get or create unified detector instance"""
    global _unified_detector
    if _unified_detector is None and UNIFIED_AVAILABLE:
        _unified_detector = get_unified_detector(model_path="yolo26n.pt", use_gpu=torch.cuda.is_available())
    return _unified_detector


def get_unified_formatter_instance():
    """Get or create unified formatter instance"""
    global _unified_formatter
    if _unified_formatter is None and UNIFIED_AVAILABLE:
        _unified_formatter = ResultFormatter()
    return _unified_formatter


# Vehicle color ranges for detection (HSV)
COLOR_RANGES = {
    'white': ((0, 0, 200), (180, 30, 255)),
    'black': ((0, 0, 0), (180, 255, 50)),
    'gray': ((0, 0, 50), (180, 30, 200)),
    'red': ((0, 100, 100), (10, 255, 255)),
    'red2': ((160, 100, 100), (180, 255, 255)),
    'blue': ((100, 100, 100), (140, 255, 255)),
    'green': ((40, 100, 100), (80, 255, 255)),
    'yellow': ((20, 100, 100), (40, 255, 255)),
    'orange': ((10, 100, 100), (20, 255, 255)),
    'purple': ((140, 100, 100), (160, 255, 255)),
    'brown': ((10, 50, 50), (30, 150, 150)),
    'silver': ((0, 0, 150), (180, 20, 220)),
}


def detect_vehicle_color(vehicle_region: np.ndarray) -> str:
    """Detect the dominant color of a vehicle"""
    if vehicle_region.size == 0:
        return 'unknown'
    
    try:
        hsv = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2HSV)
        color_scores = {}
        
        for color_name, (lower, upper) in COLOR_RANGES.items():
            if color_name == 'red2':
                continue
            
            lower = np.array(lower)
            upper = np.array(upper)
            
            if color_name == 'red':
                lower2 = np.array(COLOR_RANGES['red2'][0])
                upper2 = np.array(COLOR_RANGES['red2'][1])
                mask1 = cv2.inRange(hsv, lower, upper)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, lower, upper)
            
            color_score = np.sum(mask > 0)
            base_color = color_name.replace('2', '')
            color_scores[base_color] = color_scores.get(base_color, 0) + color_score
        
        if color_scores:
            dominant_color = max(color_scores, key=color_scores.get)
            total_score = sum(color_scores.values())
            if total_score > 0 and color_scores[dominant_color] / total_score > 0.1:
                return dominant_color
        
        return 'unknown'
    except:
        return 'unknown'


def process_unified_detection(image: np.ndarray, conf_threshold: float = 0.5) -> Tuple[np.ndarray, str, str]:
    """
    Unified detection: Object + Vehicle + Plate + PPE + Parking
    
    Returns:
        Tuple of (annotated_image, json_output, summary_text)
    """
    if image is None:
        return None, "{}", "Please upload an image"
    
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Ensure we have BGR format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if RGB and convert to BGR
            if isinstance(image, Image.Image):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        start_time = time.time()
        
        # Initialize YOLO model
        model = get_model("yolo26n")
        device = 0 if torch.cuda.is_available() else "cpu"
        
        # Run detection
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=0.5,
            device=device,
            verbose=False
        )
        
        # Initialize detection containers
        detections = {
            "objects": [],
            "vehicles": [],
            "number_plates": [],
            "ppe": [],
            "parking": []
        }
        
        # Process detections
        annotated = image.copy()
        detection = results[0] if results else None
        
        if detection and hasattr(detection, 'boxes') and detection.boxes is not None:
            boxes = detection.boxes
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = detection.names.get(class_id, f"class_{class_id}").lower()
                
                bbox = [float(x1), float(y1), float(x2), float(y2)]
                
                # Check if vehicle
                vehicle_types = ['car', 'truck', 'bus', 'motorcycle', 'bike', 'van', 'suv']
                is_vehicle = any(vtype in class_name for vtype in vehicle_types)
                
                if is_vehicle:
                    # Detect color
                    vehicle_region = image[int(y1):int(y2), int(x1):int(x2)]
                    color = detect_vehicle_color(vehicle_region)
                    
                    # Determine vehicle type
                    vtype = 'car'
                    if 'truck' in class_name:
                        vtype = 'truck'
                    elif 'bus' in class_name:
                        vtype = 'bus'
                    elif 'bike' in class_name or 'motorcycle' in class_name:
                        vtype = 'bike'
                    
                    vehicle_info = {
                        "id": f"VEH_{i+1:04d}",
                        "type": vtype,
                        "color": color,
                        "confidence": round(confidence, 2),
                        "bbox": bbox
                    }
                    detections["vehicles"].append(vehicle_info)
                    
                    # Draw vehicle box
                    color_map = {
                        'bike': (0, 255, 255),
                        'car': (0, 255, 0),
                        'truck': (255, 0, 0),
                        'bus': (255, 255, 0)
                    }
                    box_color = color_map.get(vtype, (128, 128, 128))
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                    label = f"{vtype.upper()} | {color} | {confidence:.2f}"
                    cv2.putText(annotated, label, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    
                    # Try to detect license plate for 4-wheelers
                    if vtype in ['car', 'truck', 'bus'] and TESSERACT_AVAILABLE:
                        try:
                            plate_region = vehicle_region
                            if plate_region.size > 0:
                                gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                
                                import pytesseract
                                config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                                plate_text = pytesseract.image_to_string(binary, config=config)
                                plate_text = plate_text.strip().upper()
                                
                                if plate_text and len(plate_text) >= 4:
                                    plate_info = {
                                        "text": plate_text,
                                        "confidence": 0.7,
                                        "bbox": bbox
                                    }
                                    detections["number_plates"].append(plate_info)
                                    
                                    # Draw plate text
                                    cv2.putText(annotated, f"PLATE: {plate_text}", 
                                               (int(x1), int(y2) + 20),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        except:
                            pass
                
                # Check if person for PPE
                elif 'person' in class_name:
                    person_info = {
                        "person_id": f"PER_{i+1:04d}",
                        "helmet": False,  # Would need specific PPE detection
                        "seatbelt": False,
                        "vest": False,
                        "confidence": round(confidence, 2),
                        "bbox": bbox,
                        "vehicle_type": "unknown"
                    }
                    detections["ppe"].append(person_info)
                    
                    # Draw person box
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
                    cv2.putText(annotated, f"PERSON {confidence:.2f}", (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                
                # Add to objects list
                obj_info = {
                    "id": f"OBJ_{i+1:04d}",
                    "label": class_name,
                    "confidence": round(confidence, 2),
                    "bbox": bbox
                }
                detections["objects"].append(obj_info)
        
        # Add parking detection (placeholder)
        if PARKING_DETECTION_AVAILABLE:
            try:
                from modules.parking_detection import ParkingDetector
                parking_detector = ParkingDetector()
                parking_result = parking_detector.detect(image)
                
                if hasattr(parking_result, 'slots'):
                    for idx, slot in enumerate(parking_result.slots):
                        slot_info = {
                            "slot_id": idx + 1,
                            "occupied": slot.get('occupied', False),
                            "confidence": slot.get('confidence', 0.5),
                            "bbox": slot.get('bbox', [0, 0, 0, 0])
                        }
                        detections["parking"].append(slot_info)
                        
                        # Draw parking slot
                        bbox = slot.get('bbox', [0, 0, 0, 0])
                        if slot.get('occupied', False):
                            color = (0, 0, 255)
                            status = "OCCUPIED"
                        else:
                            color = (0, 255, 0)
                            status = "EMPTY"
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f"Slot {idx+1}: {status}", (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            except:
                pass
        
        # Create JSON output
        output = {
            "source_type": "IMAGE",
            "timestamp": datetime.now().isoformat(),
            "frame_id": "0",
            "detections": detections,
            "metadata": {
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "total_detections": sum(len(v) for v in detections.values())
            }
        }
        
        json_output = json.dumps(output, indent=2, ensure_ascii=False)
        
        # Create summary
        summary_lines = [
            "🎯 UNIFIED DETECTION RESULTS",
            "=" * 40,
            f"\n🚗 Vehicles: {len(detections['vehicles'])}",
            f"👥 Persons: {len(detections['ppe'])}",
            f"📋 Plates: {len(detections['number_plates'])}",
            f"🅿️ Parking: {len(detections['parking'])} slots",
            f"\n⚡ Processing: {output['metadata']['processing_time_ms']:.1f}ms"
        ]
        
        # Add vehicle details
        if detections['vehicles']:
            summary_lines.append("\nVehicle Details:")
            for v in detections['vehicles']:
                summary_lines.append(f"  • {v['type'].upper()} ({v['color']}) - {v['confidence']:.2f}")
        
        # Add plate details
        if detections['number_plates']:
            summary_lines.append("\nLicense Plates:")
            for p in detections['number_plates']:
                summary_lines.append(f"  • {p['text']}")
        
        summary = "\n".join(summary_lines)
        
        # Convert to RGB for display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        return annotated_rgb, json_output, summary
        
    except Exception as e:
        error_msg = f"Error in unified detection: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return None, "{}", error_msg


def create_unified_detection_tab():
    """Create the unified detection tab for Gradio interface"""
    with gr.Tab("🎯 Unified Detection (All-in-One)"):
        gr.Markdown("## Unified Multi-Detection System")
        gr.Markdown("Detects: Objects + Vehicles + License Plates + PPE + Parking in one pass")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input
                unified_image_input = gr.Image(
                    label="Upload Image",
                    type="numpy"
                )
                
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold"
                )
                
                detect_btn = gr.Button(
                    "🔍 Run Unified Detection",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                # Outputs
                with gr.Row():
                    unified_image_output = gr.Image(
                        label="Detection Result",
                        type="numpy"
                    )
                
                with gr.Row():
                    unified_json_output = gr.Code(
                        label="JSON Output",
                        language="json",
                        lines=20
                    )
                
                unified_summary_output = gr.Textbox(
                    label="Detection Summary",
                    lines=10,
                    interactive=False
                )
        
        # Wire up the button
        detect_btn.click(
            fn=process_unified_detection,
            inputs=[unified_image_input, confidence_slider],
            outputs=[unified_image_output, unified_json_output, unified_summary_output]
        )


# Export the function and tab creator
__all__ = [
    'process_unified_detection',
    'create_unified_detection_tab'
]
