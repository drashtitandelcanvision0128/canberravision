"""
Professional Image Processing Module for YOLO26
Uses professional annotation system to prevent overlapping labels
"""

import cv2
import numpy as np
import time
import json
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any

# Import our professional annotator
try:
    from src.processors.professional_annotator import professional_annotator
    PROFESSIONAL_ANNOTATOR_AVAILABLE = True
    print("[INFO] Professional annotator loaded successfully")
except ImportError as e:
    PROFESSIONAL_ANNOTATOR_AVAILABLE = False
    print(f"[WARNING] Professional annotator not available: {e}")
    professional_annotator = None


def predict_image_professional(
    img,
    conf_threshold,
    iou_threshold,
    model_name,
    show_labels,
    show_conf,
    imgsz,
    enable_resnet,
    max_boxes,
    enable_ocr,
):
    """
    Professional image prediction with non-overlapping labels and enhanced design.
    
    Args:
        img: Input image (PIL or numpy array)
        conf_threshold: Confidence threshold for detection
        iou_threshold: IoU threshold for detection
        model_name: YOLO model name
        show_labels: Whether to show labels
        show_conf: Whether to show confidence scores
        imgsz: Image size for inference
        enable_resnet: Whether to enable ResNet features
        max_boxes: Maximum number of boxes to detect
        enable_ocr: Whether to enable OCR
        
    Returns:
        Tuple of (annotated_image, summary_text)
    """
    if img is None:
        return None, "Please upload an image first"

    try:
        # Import required modules
        from ultralytics import YOLO
        import torch
        
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        model = YOLO(f"{model_name}.pt")
        
        # Convert image to proper format
        if hasattr(img, 'convert'):  # PIL Image
            frame_rgb = np.array(img.convert('RGB'))
        elif isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            if len(img.shape) == 2:  # grayscale
                frame_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                frame_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif img.shape[2] == 3:  # BGR or RGB
                frame_rgb = img
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        # Convert RGB to BGR for YOLO
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = model(
            frame_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False
        )
        
        if not results:
            return img, "No objects detected"
        
        result = results[0]
        
        # Extract detections
        detections = []
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            if hasattr(boxes, 'xyxy') and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
                conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
                cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
                names = result.names
                
                for i in range(len(boxes)):
                    if conf[i] > conf_threshold:
                        x1, y1, x2, y2 = map(int, xyxy[i])
                        confidence = float(conf[i])
                        class_id = int(cls[i])
                        class_name = names.get(class_id, f"class_{class_id}")
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_name': class_name,
                            'class_id': class_id
                        }
                        
                        # Add color detection if available
                        try:
                            # Simple color detection from the crop
                            crop = frame_bgr[y1:y2, x1:x2]
                            if crop.size > 0:
                                # Calculate dominant color
                                avg_color_per_row = np.average(crop, axis=0)
                                avg_color = np.average(avg_color_per_row, axis=0)
                                dominant_color = tuple(map(int, avg_color))
                                
                                # Simple color classification
                                color_name = _classify_color(dominant_color)
                                detection['color'] = color_name
                        except Exception as e:
                            print(f"[DEBUG] Color detection failed: {e}")
                            detection['color'] = 'unknown'
                        
                        detections.append(detection)
        
        if not detections:
            return img, "No objects detected"
        
        # Perform OCR if enabled
        license_plates = {}
        if enable_ocr:
            try:
                license_plates = _extract_license_plates_simple(frame_bgr, detections)
                # Add license plate info to detections
                for detection in detections:
                    bbox_key = tuple(detection['bbox'])
                    if bbox_key in license_plates:
                        detection['license_plate'] = license_plates[bbox_key]
            except Exception as e:
                print(f"[DEBUG] OCR failed: {e}")
        
        # Use professional annotator if available, otherwise fallback
        if PROFESSIONAL_ANNOTATOR_AVAILABLE and professional_annotator:
            print("[INFO] Using professional annotator for enhanced display")
            annotated_bgr = professional_annotator.annotate_detections(
                frame_bgr,
                detections,
                show_confidence=show_conf,
                show_info_panel=True
            )
        else:
            print("[WARNING] Using fallback annotator")
            annotated_bgr = _fallback_annotate(frame_bgr, detections, show_labels, show_conf)
        
        # Generate summary
        summary = _generate_professional_summary(detections, enable_ocr)
        
        # Convert back to RGB for PIL
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_rgb), summary
        
    except Exception as e:
        print(f"[ERROR] Professional prediction failed: {e}")
        return img, f"Error: {str(e)}"


def _classify_color(bgr_color: Tuple[int, int, int]) -> str:
    """
    Simple color classification based on BGR values.
    
    Args:
        bgr_color: BGR color tuple
        
    Returns:
        Color name string
    """
    b, g, r = bgr_color
    
    # Simple color classification logic
    if r > 200 and g > 200 and b > 200:
        return "white"
    elif r < 50 and g < 50 and b < 50:
        return "black"
    elif r > g and r > b:
        if r > 150:
            return "red"
        else:
            return "brown"
    elif g > r and g > b:
        if g > 150:
            return "green"
        else:
            return "olive"
    elif b > r and b > g:
        if b > 150:
            return "blue"
        else:
            return "navy"
    elif r > 150 and g > 150:
        return "yellow"
    elif r > 150 and b > 150:
        return "magenta"
    elif g > 150 and b > 150:
        return "cyan"
    else:
        return "gray"


def _extract_license_plates_simple(frame_bgr: np.ndarray, detections: List[Dict]) -> Dict:
    """
    Simple license plate extraction for vehicles.
    
    Args:
        frame_bgr: Input frame in BGR format
        detections: List of detection dictionaries
        
    Returns:
        Dictionary mapping bbox to license plate text
    """
    license_plates = {}
    vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}
    
    try:
        import pytesseract
        
        for detection in detections:
            if detection['class_name'].lower() in vehicle_classes:
                x1, y1, x2, y2 = detection['bbox']
                
                # Expand search area around vehicle (likely license plate location)
                plate_y1 = int(y2 - (y2 - y1) * 0.3)  # Bottom 30% of vehicle
                plate_y2 = y2
                plate_x1 = int(x1 + (x2 - x1) * 0.2)  # Middle 60% width
                plate_x2 = int(x2 - (x2 - x1) * 0.2)
                
                # Ensure bounds
                plate_y1 = max(0, plate_y1)
                plate_x1 = max(0, plate_x1)
                plate_x2 = min(frame_bgr.shape[1], plate_x2)
                plate_y2 = min(frame_bgr.shape[0], plate_y2)
                
                if plate_x2 > plate_x1 and plate_y2 > plate_y1:
                    plate_crop = frame_bgr[plate_y1:plate_y2, plate_x1:plate_x2]
                    
                    # Preprocess for OCR
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    # Increase contrast
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    gray = clahe.apply(gray)
                    
                    # OCR
                    text = pytesseract.image_to_string(gray, config='--psm 7 --oem 3').strip()
                    
                    # Clean text (keep only alphanumeric)
                    cleaned = ''.join(c for c in text if c.isalnum()).upper()
                    
                    if len(cleaned) >= 5:  # Valid plate format
                        license_plates[tuple(detection['bbox'])] = cleaned
                        
    except ImportError:
        print("[DEBUG] Tesseract not available for OCR")
    except Exception as e:
        print(f"[DEBUG] License plate extraction failed: {e}")
    
    return license_plates


def _fallback_annotate(frame_bgr: np.ndarray, 
                      detections: List[Dict],
                      show_labels: bool,
                      show_conf: bool) -> np.ndarray:
    """
    Fallback annotation method when professional annotator is not available.
    
    Args:
        frame_bgr: Input frame in BGR format
        detections: List of detection dictionaries
        show_labels: Whether to show labels
        show_conf: Whether to show confidence
        
    Returns:
        Annotated frame in BGR format
    """
    annotated = frame_bgr.copy()
    
    # Simple colors for different classes
    colors = {
        'person': (0, 255, 0),
        'car': (255, 0, 0),
        'truck': (0, 0, 255),
        'motorcycle': (255, 255, 0),
        'bicycle': (255, 0, 255),
    }
    
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        color = colors.get(class_name.lower(), (128, 128, 128))
        
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if show_labels:
            label = class_name
            if show_conf:
                label += f" {confidence:.2f}"
            
            # Add color if available
            if 'color' in detection:
                label += f" ({detection['color']})"
            
            # Add license plate if available
            if 'license_plate' in detection:
                label += f" [{detection['license_plate']}]"
            
            # Calculate text position
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x1
            text_y = y1 - 10 if y1 > 30 else y1 + text_size[1] + 10
            
            # Draw background for text
            cv2.rectangle(annotated, 
                        (text_x, text_y - text_size[1] - 5),
                        (text_x + text_size[0], text_y + 5),
                        color, -1)
            
            # Draw text
            cv2.putText(annotated, label, (text_x, text_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return annotated


def _generate_professional_summary(detections: List[Dict], enable_ocr: bool) -> str:
    """
    Generate a professional summary of detections.
    
    Args:
        detections: List of detection dictionaries
        enable_ocr: Whether OCR was enabled
        
    Returns:
        Formatted summary string
    """
    if not detections:
        return "🎯 **No objects detected**"
    
    # Count objects by class
    class_counts = {}
    colors_found = set()
    license_plates_found = []
    
    for detection in detections:
        class_name = detection['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if 'color' in detection:
            colors_found.add(detection['color'])
        
        if 'license_plate' in detection:
            license_plates_found.append(f"{class_name}: {detection['license_plate']}")
    
    # Build summary
    summary_lines = [
        f"🎯 **Detection Complete**",
        f"📊 **Total Objects:** {len(detections)}",
        ""
    ]
    
    # Add object counts
    if class_counts:
        summary_lines.append("🏷️ **Objects Found:**")
        for class_name, count in sorted(class_counts.items()):
            summary_lines.append(f"  • {class_name}: {count}")
        summary_lines.append("")
    
    # Add colors
    if colors_found:
        summary_lines.append("🎨 **Colors Detected:**")
        for color in sorted(colors_found):
            summary_lines.append(f"  • {color}")
        summary_lines.append("")
    
    # Add license plates
    if license_plates_found and enable_ocr:
        summary_lines.append("🚗 **License Plates:**")
        for plate_info in license_plates_found:
            summary_lines.append(f"  • {plate_info}")
        summary_lines.append("")
    
    # Add professional footer
    summary_lines.extend([
        "✨ **Processed with Professional Annotation System**",
        "🔥 Labels are positioned to prevent overlapping",
        "⚡ Powered by YOLO26 AI Engine"
    ])
    
    return "\n".join(summary_lines)
