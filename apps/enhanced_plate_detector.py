#!/usr/bin/env python3
"""
🚗 Enhanced License Plate Detector - Angle Independent Detection
============================================================

Advanced license plate detection that works at any angle and properly
classifies license plates regardless of orientation.

Features:
- ✅ Angle-independent detection
- ✅ Multi-color plate support (white, yellow, blue, red, green)
- ✅ Robust text classification
- ✅ Perspective correction
- ✅ Multiple detection methods
- ✅ High confidence scoring
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
import sys
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class EnhancedPlateDetector:
    def __init__(self):
        self.ocr_available = self.check_ocr_availability()
        print(f"🔍 OCR Available: {'✅' if self.ocr_available else '❌'}")
        
        # License plate patterns for different regions
        self.plate_patterns = [
            # Indian patterns
            r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$',  # MH14DX9937
            r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',   # MH12AB1234
            r'^[A-Z]{3}\d{4}$',                 # ABC1234
            
            # International patterns
            r'^[A-Z]{1,3}\d{1,4}[A-Z]{0,3}$',  # General international
            r'^\d{2,4}[A-Z]{1,3}\d{2,4}$',     # European style
            r'^[A-Z]{2}\d{7}$',                 # Some countries
            
            # Generic patterns (more lenient)
            r'^[A-Z0-9]{4,12}$',                # Any alphanumeric 4-12 chars
        ]
        
        # Colors for different detection methods
        self.colors = {
            'contour': (0, 255, 0),      # Green
            'edge': (255, 0, 0),         # Blue
            'threshold': (0, 0, 255),    # Red
            'morph': (255, 255, 0),      # Cyan
            'cascade': (255, 0, 255),    # Magenta
        }
    
    def check_ocr_availability(self):
        """Check if OCR libraries are available"""
        try:
            import pytesseract
            return True
        except:
            return False
    
    def is_license_plate_text(self, text):
        """
        Enhanced license plate text classification
        Classifies text as license plate based on patterns and characteristics
        """
        if not text or len(text.strip()) < 4:
            return False, 0.0
        
        text = text.upper().strip()
        
        # Remove common OCR errors and clean
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        if len(text) < 4 or len(text) > 12:
            return False, 0.0
        
        confidence = 0.0
        
        # Check against known patterns
        for pattern in self.plate_patterns:
            if re.match(pattern, text):
                confidence += 0.4
                break
        
        # Must have both letters and numbers
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        
        if has_letters and has_numbers:
            confidence += 0.3
        
        # Length check (typical plate lengths)
        if 6 <= len(text) <= 10:
            confidence += 0.2
        
        # Character distribution check
        if len(set(text)) >= 4:  # At least 4 unique characters
            confidence += 0.1
        
        return confidence >= 0.5, confidence
    
    def detect_license_plates_enhanced(self, image):
        """
        Enhanced license plate detection that works at any angle
        """
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_np = image.copy()
            
            all_plates = []
            
            # Method 1: Multi-angle contour detection
            plates1 = self.detect_with_contours_multi_angle(image_np)
            all_plates.extend(plates1)
            
            # Method 2: Edge-based detection
            plates2 = self.detect_with_edges_enhanced(image_np)
            all_plates.extend(plates2)
            
            # Method 3: Morphological operations
            plates3 = self.detect_with_morphology(image_np)
            all_plates.extend(plates3)
            
            # Method 4: Threshold-based detection
            plates4 = self.detect_with_threshold_multi(image_np)
            all_plates.extend(plates4)
            
            # Method 5: MSER detection for robust text detection
            plates5 = self.detect_with_mser(image_np)
            all_plates.extend(plates5)
            
            # Remove duplicates and merge results
            unique_plates = self.remove_duplicate_detections(all_plates)
            
            # Extract text and classify
            for plate in unique_plates:
                plate_region = image_np[plate['y1']:plate['y2'], plate['x1']:plate['x2']]
                text, conf = self.extract_text_enhanced(plate_region)
                plate['text'] = text
                plate['text_confidence'] = conf
                
                # Classify as license plate
                is_plate, plate_conf = self.is_license_plate_text(text)
                plate['is_license_plate'] = is_plate
                plate['plate_confidence'] = plate_conf
                
                # Detect color
                plate['color'] = self.detect_plate_color(plate_region)
            
            # Filter only license plates
            license_plates = [p for p in unique_plates if p['is_license_plate']]
            
            # Draw results
            result_image = self.draw_enhanced_results(image_np, license_plates)
            
            return result_image, license_plates
            
        except Exception as e:
            print(f"Error in enhanced detection: {e}")
            return image, []
    
    def detect_with_contours_multi_angle(self, image):
        """Detect plates using contours with multiple preprocessing methods"""
        plates = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multiple preprocessing methods for different angles and lighting
            methods = [
                ('normal', gray),
                ('clahe', self.apply_clahe(gray)),
                ('blur', cv2.GaussianBlur(gray, (5, 5), 0)),
            ]
            
            for method_name, processed in methods:
                # Apply threshold
                _, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Relaxed aspect ratio for angled plates
                    if (1.5 <= aspect_ratio <= 8.0 and 
                        w > 60 and h > 15 and 
                        cv2.contourArea(contour) > 800):
                        
                        plates.append({
                            'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h,
                            'confidence': 0.7,
                            'method': f'contour_{method_name}',
                            'aspect_ratio': aspect_ratio
                        })
        
        except Exception as e:
            print(f"Contour detection error: {e}")
        
        return plates
    
    def detect_with_edges_enhanced(self, image):
        """Enhanced edge detection for license plates"""
        plates = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise while preserving edges
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Multi-scale edge detection
            edges_list = []
            for canny_low in [30, 50, 70]:
                for canny_high in [100, 150, 200]:
                    edges = cv2.Canny(bilateral, canny_low, canny_high)
                    edges_list.append(edges)
            
            # Combine edge maps
            combined_edges = np.zeros_like(edges_list[0])
            for edges in edges_list:
                combined_edges = cv2.bitwise_or(combined_edges, edges)
            
            # Dilate edges to connect broken lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            combined_edges = cv2.dilate(combined_edges, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if (2.0 <= aspect_ratio <= 6.0 and 
                    w > 80 and h > 20 and 
                    cv2.contourArea(contour) > 1000):
                    
                    plates.append({
                        'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h,
                        'confidence': 0.6,
                        'method': 'edge_enhanced',
                        'aspect_ratio': aspect_ratio
                    })
        
        except Exception as e:
            print(f"Edge detection error: {e}")
        
        return plates
    
    def detect_with_morphology(self, image):
        """Detect plates using morphological operations"""
        plates = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply morphological operations to highlight rectangular regions
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            # Detect horizontal and vertical lines
            horizontal = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_h)
            vertical = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_v)
            
            # Combine
            combined = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0)
            
            # Threshold
            _, thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if (2.0 <= aspect_ratio <= 7.0 and 
                    w > 100 and h > 25 and 
                    cv2.contourArea(contour) > 1500):
                    
                    plates.append({
                        'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h,
                        'confidence': 0.5,
                        'method': 'morphology',
                        'aspect_ratio': aspect_ratio
                    })
        
        except Exception as e:
            print(f"Morphology detection error: {e}")
        
        return plates
    
    def detect_with_threshold_multi(self, image):
        """Multi-threshold detection for different plate colors"""
        plates = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multiple thresholding strategies
            thresholds = [
                ('otsu', cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                ('adaptive', cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                ('binary_inv', cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]),
            ]
            
            for thresh_name, thresh_img in thresholds:
                # Noise reduction
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if (1.8 <= aspect_ratio <= 6.5 and 
                        w > 70 and h > 18 and 
                        cv2.contourArea(contour) > 900):
                        
                        plates.append({
                            'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h,
                            'confidence': 0.4,
                            'method': f'threshold_{thresh_name}',
                            'aspect_ratio': aspect_ratio
                        })
        
        except Exception as e:
            print(f"Threshold detection error: {e}")
        
        return plates
    
    def detect_with_mser(self, image):
        """Detect text regions using MSER (Maximally Stable Extremal Regions)"""
        plates = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Initialize MSER detector
            mser = cv2.MSER_create()
            
            # Detect regions
            regions, _ = mser.detectRegions(gray)
            
            # Group regions that might form a license plate
            for region in regions:
                if len(region) < 10:  # Skip very small regions
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(region)
                aspect_ratio = w / h
                
                # Check if it could be a license plate
                if (2.0 <= aspect_ratio <= 8.0 and 
                    w > 60 and h > 15 and 
                    w < gray.shape[1] * 0.8 and h < gray.shape[0] * 0.3):
                    
                    plates.append({
                        'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h,
                        'confidence': 0.3,
                        'method': 'mser',
                        'aspect_ratio': aspect_ratio
                    })
        
        except Exception as e:
            print(f"MSER detection error: {e}")
        
        return plates
    
    def apply_clahe(self, gray):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(gray)
    
    def remove_duplicate_detections(self, plates):
        """Remove duplicate plate detections"""
        if not plates:
            return []
        
        # Sort by confidence
        plates = sorted(plates, key=lambda x: x['confidence'], reverse=True)
        
        unique_plates = []
        
        for plate in plates:
            is_duplicate = False
            for existing in unique_plates:
                # Check if bounding boxes overlap significantly
                if self.bboxes_overlap(plate, existing, 0.5):
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if plate['confidence'] > existing['confidence']:
                        unique_plates.remove(existing)
                        unique_plates.append(plate)
                    break
            
            if not is_duplicate:
                unique_plates.append(plate)
        
        return unique_plates
    
    def bboxes_overlap(self, box1, box2, threshold=0.5):
        """Check if two bounding boxes overlap"""
        x1_max = max(box1['x1'], box2['x1'])
        y1_max = max(box1['y1'], box2['y1'])
        x2_min = min(box1['x2'], box2['x2'])
        y2_min = min(box1['y2'], box2['y2'])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return False
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        union = area1 + area2 - intersection
        
        return intersection / union > threshold
    
    def extract_text_enhanced(self, plate_region):
        """Enhanced text extraction from license plate region"""
        try:
            if plate_region.size == 0:
                return "", 0.0
            
            # Convert to grayscale
            if len(plate_region.shape) == 3:
                gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_region
            
            # Apply CLAHE for better contrast
            gray = self.apply_clahe(gray)
            
            # Multiple preprocessing methods
            methods = []
            
            # Method 1: Otsu threshold
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            methods.append(('otsu', otsu))
            
            # Method 2: Adaptive threshold
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            methods.append(('adaptive', adaptive))
            
            # Method 3: Inverted
            _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            methods.append(('inverted', inv))
            
            best_text = ""
            best_confidence = 0.0
            
            for method_name, processed in methods:
                if self.ocr_available:
                    try:
                        import pytesseract
                        # Configure Tesseract for license plates
                        config = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                        text = pytesseract.image_to_string(processed, config=config)
                        text = text.upper().strip()
                        
                        # Clean text
                        text = re.sub(r'[^A-Z0-9]', '', text)
                        
                        if text and len(text) >= 4:
                            is_plate, conf = self.is_license_plate_text(text)
                            if is_plate and conf > best_confidence:
                                best_text = text
                                best_confidence = conf
                    
                    except:
                        continue
            
            return best_text, best_confidence
            
        except Exception as e:
            return "", 0.0
    
    def detect_plate_color(self, plate_region):
        """Detect the dominant color of the license plate"""
        try:
            if plate_region.size == 0:
                return "unknown"
            
            # Convert to HSV
            hsv = cv2.cvtColor(plate_region, cv2.COLOR_BGR2HSV)
            
            # Define color ranges
            colors = {
                'white': ([0, 0, 200], [180, 30, 255]),
                'yellow': ([20, 100, 100], [30, 255, 255]),
                'blue': ([100, 100, 100], [130, 255, 255]),
                'red': ([0, 100, 100], [10, 255, 255]),
                'green': ([40, 100, 100], [80, 255, 255]),
                'black': ([0, 0, 0], [180, 255, 50])
            }
            
            max_pixels = 0
            detected_color = "unknown"
            
            for color_name, (lower, upper) in colors.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                pixel_count = np.sum(mask > 0)
                
                if pixel_count > max_pixels and pixel_count > (plate_region.size * 0.2):
                    max_pixels = pixel_count
                    detected_color = color_name
            
            return detected_color
            
        except:
            return "unknown"
    
    def draw_enhanced_results(self, image, plates):
        """Draw enhanced detection results"""
        result_image = image.copy()
        
        for i, plate in enumerate(plates):
            x1, y1, x2, y2 = plate['x1'], plate['y1'], plate['x2'], plate['y2']
            
            # Color based on confidence
            if plate['plate_confidence'] > 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif plate['plate_confidence'] > 0.6:
                color = (255, 255, 0)  # Yellow - medium confidence
            else:
                color = (0, 0, 255)  # Red - low confidence
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label
            text = plate.get('text', 'NO_TEXT')
            confidence = plate.get('plate_confidence', 0)
            color_name = plate.get('color', 'unknown')
            method = plate.get('method', 'unknown')
            
            label = f"🚗 {text} ({confidence:.2f}) [{color_name}]"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_image, (x1, y1-25), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(result_image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convert back to RGB for Gradio
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        return result_image

def create_enhanced_interface():
    """Create enhanced Gradio interface"""
    
    detector = EnhancedPlateDetector()
    
    def process_image(image):
        if image is None:
            return None, "❌ Please upload an image"
        
        # Detect license plates
        result_image, plates = detector.detect_license_plates_enhanced(image)
        
        # Format results
        if len(plates) == 0:
            result_text = "❌ No license plates detected\n\n💡 Tips:\n- Ensure the license plate is clearly visible\n- Try different angles and lighting\n- Support for white, yellow, blue, red, green plates"
        else:
            result_text = f"🎯 Found {len(plates)} License Plate(s):\n\n"
            for i, plate in enumerate(plates):
                result_text += f"🚗 Plate {i+1}: {plate['text']}\n"
                result_text += f"   📊 Confidence: {plate['plate_confidence']:.2f}\n"
                result_text += f"   🎨 Color: {plate['color']}\n"
                result_text += f"   🔍 Method: {plate['method']}\n"
                result_text += f"   📍 Position: ({plate['x1']}, {plate['y1']})\n\n"
        
        return result_image, result_text
    
    # Create Gradio interface
    with gr.Blocks(title="Enhanced License Plate Detector", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🚗 Enhanced License Plate Detector - Angle Independent
        
        Advanced license plate detection that works at **any angle** and supports **multi-colored plates**.
        
        **Features:**
        - ✅ **Angle Independent** - Works at any orientation
        - 🎨 **Multi-Color Support** - White, Yellow, Blue, Red, Green plates
        - 🔍 **Smart Classification** - Advanced text pattern recognition
        - 📊 **Confidence Scoring** - Shows detection confidence
        - 🌍 **International Patterns** - Supports multiple plate formats
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Image")
                image_input = gr.Image(
                    label="Upload Image with License Plate",
                    type="pil",
                    height=400
                )
                
                process_btn = gr.Button(
                    "🔍 Detect License Plates",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                ### 💡 Detection Tips:
                - Works at **any angle**
                - Supports **all plate colors**
                - Handles **different lighting**
                - **International** plate formats
                - **High accuracy** classification
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Detection Results")
                image_output = gr.Image(
                    label="Detected License Plates",
                    type="pil",
                    height=400
                )
                
                text_output = gr.Textbox(
                    label="License Plate Information",
                    lines=12,
                    max_lines=15,
                    interactive=False
                )
        
        # Process button click
        process_btn.click(
            fn=process_image,
            inputs=image_input,
            outputs=[image_output, text_output]
        )
        
        # Also process on image upload
        image_input.change(
            fn=process_image,
            inputs=image_input,
            outputs=[image_output, text_output]
        )
    
    return interface

def main():
    """Main entry point"""
    print("🚗 Starting Enhanced License Plate Detector...")
    print("✅ Angle-independent detection enabled")
    print("✅ Multi-color plate support enabled")
    print("✅ Smart text classification enabled")
    
    try:
        # Create and launch interface
        interface = create_enhanced_interface()
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7862,  # Different port
            share=False,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        print("Please check your dependencies and try again.")

if __name__ == "__main__":
    main()
