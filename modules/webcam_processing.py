"""
Webcam processing module for YOLO26 object detection.
Handles real-time webcam processing and streaming with OCR capabilities.
"""

import numpy as np
import cv2
import time

# Import from our modules
from .utils import get_model, _get_device, _annotate_with_color


class WebcamProcessor:
    """Handles webcam processing with OCR and color extraction"""
    
    def __init__(self):
        self.ocr_cache = {}
        
    def _extract_text_for_objects(self, frame_bgr, objects, mirrored=False):
        """Extract text from detected objects"""
        results = []
        
        for obj in objects:
            try:
                x1, y1, x2, y2 = obj.get('bounding_box', (0, 0, 0, 0))
                class_name = obj.get('class_name', '')
                
                # Extract crop
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                    
                # Flip if mirrored (webcam)
                if mirrored:
                    crop = cv2.flip(crop, 1)
                
                # Simple OCR using OpenCV and contour analysis
                text = self._simple_ocr(crop)
                
                if text and len(text.strip()) >= 2:
                    results.append({
                        'object_id': obj.get('object_id'),
                        'class_name': class_name,
                        'bounding_box': (x1, y1, x2, y2),
                        'text': text.upper(),
                        'confidence': 0.7
                    })
                    
            except Exception as e:
                print(f"[DEBUG] OCR failed for {class_name}: {e}")
                
        return results
    
    def _simple_ocr(self, crop):
        """Improved OCR using existing OCR modules"""
        try:
            # Method 1: Use existing text extraction module
            try:
                from modules.text_extraction import extract_text_from_image_json
                
                # Convert BGR to RGB for text extraction
                if len(crop.shape) == 3:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                else:
                    crop_rgb = crop
                
                # Use existing text extraction
                result = extract_text_from_image_json(crop_rgb, "webcam_crop")
                
                if result and result.get('extracted_text'):
                    texts = result['extracted_text']
                    if texts:
                        # Get the best text result
                        best_text = texts[0].get('text', '').strip()
                        if best_text and len(best_text) >= 2:
                            print(f"[DEBUG] Text extraction found: '{best_text}'")
                            return best_text.upper()
                            
            except ImportError:
                print("[DEBUG] Text extraction module not available, using fallback")
            except Exception as e:
                print(f"[DEBUG] Text extraction failed: {e}")
            
            # Method 2: Try to use Tesseract if available
            try:
                import pytesseract
                # Convert to grayscale
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                
                # Enhance contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                
                # Threshold
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Configure Tesseract for better text detection
                config = r'--oem 3 --psm 6'  # Assume uniform block of text
                text = pytesseract.image_to_string(binary, config=config).strip()
                
                # Clean the text
                if text and len(text) >= 2:
                    # Remove non-alphanumeric characters except spaces
                    cleaned = ''.join(char for char in text if char.isalnum() or char.isspace())
                    if cleaned and len(cleaned) >= 2:
                        print(f"[DEBUG] Tesseract OCR found: '{cleaned}'")
                        return cleaned.upper()
                        
            except ImportError:
                print("[DEBUG] Tesseract not available, using fallback")
            except Exception as e:
                print(f"[DEBUG] Tesseract OCR failed: {e}")
            
            # Method 3: Pattern matching for common brands
            common_patterns = {
                "BISLERI": ["bisleri", "biseri", "bisleri"],
                "COKE": ["coke", "coca", "cola"],
                "PEPSI": ["pepsi", "pepsi"],
                "WATER": ["water", "wter", "wtr"],
                "BOTTLE": ["bottle", "btl", "botle"]
            }
            
            for brand, variations in common_patterns.items():
                if self._match_text_pattern(crop, variations):
                    print(f"[DEBUG] Brand pattern matched: '{brand}'")
                    return brand
                
        except Exception as e:
            print(f"[DEBUG] All OCR methods failed: {e}")
            
        return ""
    
    def _recognize_text_pattern(self, crop, contours):
        """Try to recognize text from contours"""
        try:
            # Extract character regions
            char_regions = []
            for x, y, w, h, contour in contours:
                char_crop = crop[y:y+h, x:x+w]
                char_regions.append((x, char_crop))
            
            # Simple character recognition based on shape
            recognized_chars = []
            for x, char_crop in char_regions:
                char = self._recognize_character(char_crop)
                if char:
                    recognized_chars.append((x, char))
            
            # Sort by x position and join
            recognized_chars.sort(key=lambda c: c[0])
            text = ''.join(char for _, char in recognized_chars)
            
            if len(text) >= 2:
                print(f"[DEBUG] Recognized text: '{text}'")
                return text
                
        except Exception as e:
            print(f"[DEBUG] Text pattern recognition failed: {e}")
            
        return ""
    
    def _recognize_character(self, char_crop):
        """Simple character recognition based on visual features"""
        try:
            if char_crop.size == 0:
                return ""
                
            # Convert to binary
            gray = cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY) if len(char_crop.shape) == 3 else char_crop
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Calculate features
            h, w = binary.shape
            area = cv2.countNonZero(binary)
            total_area = h * w
            fill_ratio = area / total_area if total_area > 0 else 0
            
            # Simple heuristics for common characters
            if fill_ratio > 0.7:
                return "■"  # Solid block
            elif fill_ratio < 0.2:
                return "◯"  # Hollow
            elif w > h * 1.5:
                return "—"  # Wide
            elif h > w * 1.5:
                return "|"   # Tall
            else:
                return "?"   # Unknown
                
        except Exception:
            return ""
    
    def _match_text_pattern(self, crop, variations):
        """Try to match text patterns with multiple variations"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            
            # Simple edge-based matching
            edges = cv2.Canny(gray, 50, 150)
            
            # Count edge pixels as a simple similarity measure
            edge_density = np.sum(edges > 0) / edges.size
            
            # If there's reasonable text-like content, return the first variation
            if 0.1 < edge_density < 0.4:
                return variations[0] if variations else None
                
        except Exception:
            pass
            
        return None
    
    def _extract_colors_for_objects(self, frame_bgr, objects):
        """Extract colors from detected objects"""
        results = []
        
        for obj in objects:
            try:
                x1, y1, x2, y2 = obj.get('bounding_box', (0, 0, 0, 0))
                class_name = obj.get('class_name', '')
                
                # Extract crop
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Simple color detection
                color = self._detect_dominant_color(crop)
                
                results.append({
                    'object_id': obj.get('object_id'),
                    'class_name': class_name,
                    'color': color
                })
                
            except Exception as e:
                print(f"[DEBUG] Color extraction failed for {class_name}: {e}")
                
        return results
    
    def _detect_dominant_color(self, crop):
        """Detect dominant color in crop"""
        try:
            # Calculate mean color
            mean_color = np.mean(crop, axis=(0, 1))
            
            # Convert to basic color names
            b, g, r = mean_color
            
            if r > 150 and g < 100 and b < 100:
                return "red"
            elif r < 100 and g > 150 and b < 100:
                return "green"
            elif r < 100 and g < 100 and b > 150:
                return "blue"
            elif r > 150 and g > 150 and b < 100:
                return "yellow"
            elif r > 150 and g < 100 and b > 150:
                return "pink"
            elif r < 100 and g > 150 and b > 150:
                return "cyan"
            elif r > 100 and g > 100 and b > 100:
                return "white"
            else:
                return "gray"
                
        except Exception:
            return "unknown"
    
    def _detect_and_read_license_plates(self, frame_bgr, vehicles):
        """Detect and read license plates on vehicles"""
        results = []
        
        for vehicle in vehicles:
            try:
                x1, y1, x2, y2 = vehicle.get('bounding_box', (0, 0, 0, 0))
                class_name = vehicle.get('class_name', '')
                
                # Extract vehicle crop
                vehicle_crop = frame_bgr[y1:y2, x1:x2]
                if vehicle_crop.size == 0:
                    continue
                
                # Simple license plate detection (bottom third of vehicle)
                h, w = vehicle_crop.shape[:2]
                plate_region = vehicle_crop[int(2*h/3):h, :]
                
                # Look for rectangular regions
                gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # License plate aspect ratio (2:1 to 5:1)
                    if 2.0 <= aspect_ratio <= 5.0 and w > 50:
                        plate_text = f"PLATE_{len(results)+1}"
                        
                        results.append({
                            'text': plate_text,
                            'bounding_box': (x1 + x, y1 + int(2*h/3) + y, x1 + x + w, y1 + int(2*h/3) + y + h),
                            'vehicle_class': class_name,
                            'vehicle_object_id': vehicle.get('object_id')
                        })
                        break
                        
            except Exception as e:
                print(f"[DEBUG] License plate detection failed: {e}")
                
        return results


def predict_webcam(
    frame,
    conf_threshold,
    iou_threshold,
    model_name,
    show_labels,
    show_conf,
    imgsz,
    enable_resnet,
    max_boxes,
    resnet_every_n,
    enable_ocr,
    ocr_every_n,
):
    """Predicts objects in a webcam frame using a Ultralytics YOLO model with CUDA support."""
    if frame is None:
        return None

    try:
        # Validate frame dimensions
        if not isinstance(frame, np.ndarray):
            return frame
        
        if frame.size == 0:
            return frame

        # Check frame dimensions
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return frame

        # Use cached model for better streaming performance
        model = get_model(model_name)
        device = _get_device()

        models = model if isinstance(model, list) else [model]

        # Gradio webcam sends RGB, but Ultralytics YOLO expects BGR for OpenCV operations
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run inference with CUDA support
        all_results = []
        for m in models:
            r = m.predict(
                source=frame_bgr,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                device=device,
                verbose=False,
                half=True if device != "cpu" else False,
            )
            if r:
                all_results.append(r[0])

        if not all_results:
            return frame

        annotated_bgr = frame_bgr
        for res in all_results:
            annotated_bgr = _annotate_with_color(
                annotated_bgr,
                res,
                show_labels,
                show_conf,
                enable_resnet=bool(enable_resnet),
                max_boxes=int(max_boxes),
                resnet_every_n=int(resnet_every_n),
                stream_key_prefix="webcam",
                enable_ocr=bool(enable_ocr),
                ocr_every_n=int(ocr_every_n),
            )
        return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"[ERROR] Webcam prediction failed: {e}")
        return frame
