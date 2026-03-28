#!/usr/bin/env python3
"""
🚗 License Plate Image Detection & Recognition System
==================================================

Specialized application for detecting and reading license plates from uploaded images.
Supports multi-colored license plates (white, yellow, blue, red, etc.).

Features:
- Upload license plate images
- Automatic license plate detection
- Multi-color plate recognition
- Text extraction from plates
- Visual detection display
- Confidence scoring
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.core.detector import CarDetector
    from src.ocr.text_extractor import TextExtractor
    from src.ocr.license_plate_detector import LicensePlateDetector
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback detection...")

class LicensePlateImageDetector:
    def __init__(self):
        self.detector = None
        self.ocr_extractor = None
        self.plate_detector = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize detection and OCR models"""
        try:
            # Try to initialize main detector
            self.detector = CarDetector()
            print("✅ Main detector initialized")
        except:
            print("⚠️ Main detector failed, using fallback")
        
        try:
            # Initialize OCR
            self.ocr_extractor = TextExtractor()
            print("✅ OCR extractor initialized")
        except:
            print("⚠️ OCR extractor failed, using fallback")
        
        try:
            # Initialize license plate detector
            self.plate_detector = LicensePlateDetector()
            print("✅ License plate detector initialized")
        except:
            print("⚠️ License plate detector failed, using fallback")
    
    def detect_license_plates_in_image(self, image):
        """
        Detect license plates in uploaded image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            tuple: (processed_image, detection_info)
        """
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_np = image.copy()
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Use dedicated license plate detector if available
            plates = []
            if self.plate_detector:
                try:
                    plates = self.plate_detector.detect_plates(image_np)
                except:
                    pass
            
            # Method 2: Fallback detection using contour analysis
            if not plates:
                plates = self.fallback_plate_detection(gray)
            
            # Method 3: Try main detector if available
            if not plates and self.detector:
                try:
                    # Use detector to find vehicles, then extract plates
                    detections = self.detector.detect(image_np)
                    for detection in detections:
                        if detection.get('class_name') in ['car', 'truck', 'bus', 'motorcycle']:
                            # Extract plate region from vehicle
                            plate_region = self.extract_plate_from_vehicle(image_np, detection)
                            if plate_region is not None:
                                plates.append(plate_region)
                except:
                    pass
            
            # Process detected plates
            processed_image, plate_info = self.process_detected_plates(image_np, plates)
            
            return processed_image, plate_info
            
        except Exception as e:
            print(f"Error in detection: {e}")
            return image, "❌ Detection failed: " + str(e)
    
    def fallback_plate_detection(self, gray_image):
        """
        Fallback license plate detection using contour analysis
        Supports multi-colored plates
        """
        plates = []
        
        try:
            # Apply adaptive threshold for different lighting conditions
            thresh = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours based on license plate characteristics
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # License plate typical aspect ratio: 2.0 to 5.0
                if 2.0 <= aspect_ratio <= 5.0 and w > 80 and h > 20:
                    # Additional filters
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Minimum area
                        plates.append({
                            'bbox': [x, y, x+w, y+h],
                            'confidence': 0.7,
                            'method': 'contour'
                        })
            
            # Sort by confidence and keep top candidates
            plates = sorted(plates, key=lambda x: x['confidence'], reverse=True)[:5]
            
        except Exception as e:
            print(f"Fallback detection error: {e}")
        
        return plates
    
    def extract_plate_from_vehicle(self, image, vehicle_detection):
        """Extract license plate region from vehicle detection"""
        try:
            x1, y1, x2, y2 = vehicle_detection['bbox']
            vehicle_roi = image[y1:y2, x1:x2]
            
            # License plates are usually in the lower portion of vehicles
            h, w = vehicle_roi.shape[:2]
            plate_region = vehicle_roi[int(h*0.6):, int(w*0.2):int(w*0.8)]
            
            if plate_region.size > 0:
                # Convert to global coordinates
                plate_x1 = x1 + int(w*0.2)
                plate_y1 = y1 + int(h*0.6)
                plate_x2 = x1 + int(w*0.8)
                plate_y2 = y2
                
                return {
                    'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                    'confidence': 0.6,
                    'method': 'vehicle_extraction'
                }
        except:
            pass
        
        return None
    
    def process_detected_plates(self, image, plates):
        """Process detected plates and extract text"""
        processed_image = image.copy()
        plate_info = []
        
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for i, plate in enumerate(plates):
            bbox = plate['bbox']
            confidence = plate.get('confidence', 0.5)
            method = plate.get('method', 'unknown')
            
            # Draw bounding box with different colors
            color = colors[i % len(colors)]
            cv2.rectangle(processed_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            
            # Extract plate region for OCR
            plate_region = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            # Extract text using OCR
            plate_text = self.extract_plate_text(plate_region)
            
            # Add label
            label = f"Plate {i+1}: {plate_text} ({confidence:.2f})"
            cv2.putText(processed_image, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Store plate information
            plate_info.append({
                'plate_number': i+1,
                'text': plate_text,
                'confidence': confidence,
                'bbox': bbox,
                'method': method,
                'color_detected': self.detect_plate_color(plate_region)
            })
        
        # Convert back to RGB for Gradio
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        return processed_image, plate_info
    
    def extract_plate_text(self, plate_region):
        """Extract text from license plate region"""
        try:
            if self.ocr_extractor:
                # Use main OCR extractor
                text = self.ocr_extractor.extract_text(plate_region)
                if text and len(text.strip()) > 2:
                    return self.clean_plate_text(text)
            
            # Fallback OCR using OpenCV
            return self.fallback_ocr(plate_region)
            
        except:
            return "OCR_FAILED"
    
    def fallback_ocr(self, plate_region):
        """Fallback OCR using basic image processing"""
        try:
            # Convert to grayscale
            if len(plate_region.shape) == 3:
                gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_region
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Try to use Tesseract if available
            try:
                import pytesseract
                text = pytesseract.image_to_string(thresh, config='--psm 7')
                return self.clean_plate_text(text)
            except:
                pass
            
            # If no OCR available, return placeholder
            return "DETECTED_PLATE"
            
        except:
            return "OCR_ERROR"
    
    def clean_plate_text(self, text):
        """Clean and format license plate text"""
        if not text:
            return ""
        
        # Remove common OCR errors
        text = text.upper()
        text = ''.join(c for c in text if c.isalnum() or c in '- ')
        text = text.strip()
        
        # Common substitutions
        substitutions = {
            '0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z'
        }
        
        # Apply substitutions for better plate recognition
        cleaned = ""
        for char in text:
            if char in substitutions:
                cleaned += substitutions[char]
            else:
                cleaned += char
        
        return cleaned
    
    def detect_plate_color(self, plate_region):
        """Detect the dominant color of the license plate"""
        try:
            if len(plate_region.shape) == 3:
                # Convert to HSV for better color detection
                hsv = cv2.cvtColor(plate_region, cv2.COLOR_BGR2HSV)
                
                # Define color ranges for common plate colors
                colors = {
                    'white': ([0, 0, 200], [180, 30, 255]),
                    'yellow': ([20, 100, 100], [30, 255, 255]),
                    'blue': ([100, 100, 100], [130, 255, 255]),
                    'red': ([0, 100, 100], [10, 255, 255]),
                    'green': ([40, 100, 100], [80, 255, 255])
                }
                
                # Check each color
                for color_name, (lower, upper) in colors.items():
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    if np.sum(mask) > (plate_region.size * 0.3):  # 30% threshold
                        return color_name
                
                return 'unknown'
            else:
                return 'grayscale'
        except:
            return 'unknown'

def create_license_plate_interface():
    """Create Gradio interface for license plate detection"""
    
    detector = LicensePlateImageDetector()
    
    def process_image(image):
        if image is None:
            return None, "❌ Please upload an image"
        
        # Detect license plates
        processed_image, plate_info = detector.detect_license_plates_in_image(image)
        
        # Format results
        if plate_info and isinstance(plate_info, list):
            if len(plate_info) == 0:
                result_text = "❌ No license plates detected"
            else:
                result_text = "🎯 License Plates Detected:\n\n"
                for plate in plate_info:
                    result_text += f"🚗 Plate {plate['plate_number']}: {plate['text']}\n"
                    result_text += f"   📊 Confidence: {plate['confidence']:.2f}\n"
                    result_text += f"   🎨 Color: {plate['color_detected']}\n"
                    result_text += f"   🔍 Method: {plate['method']}\n\n"
        else:
            result_text = str(plate_info)
        
        return processed_image, result_text
    
    # Create Gradio interface
    with gr.Blocks(title="License Plate Detection", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🚗 License Plate Image Detection & Recognition
        
        Upload an image containing a license plate, and the system will:
        - 🔍 Detect the license plate automatically
        - 🎨 Recognize different plate colors (white, yellow, blue, red, etc.)
        - 📝 Extract the text from the license plate
        - 📊 Show confidence scores and detection method
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Image")
                image_input = gr.Image(
                    label="Upload License Plate Image",
                    type="pil",
                    height=400
                )
                
                process_btn = gr.Button(
                    "🔍 Detect License Plate",
                    variant="primary",
                    size="lg"
                )
                
                # Sample images info
                gr.Markdown("""
                ### 💡 Tips:
                - Upload clear images of license plates
                - Supported colors: White, Yellow, Blue, Red, Green
                - Works with various lighting conditions
                - Multiple plates in one image supported
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Detection Results")
                image_output = gr.Image(
                    label="Processed Image with Detections",
                    type="pil",
                    height=400
                )
                
                text_output = gr.Textbox(
                    label="Detection Information",
                    lines=10,
                    max_lines=15,
                    interactive=False
                )
        
        # Examples
        gr.Markdown("### 📸 Example Usage")
        gr.Examples(
            examples=[
                # Add example paths if you have sample images
            ],
            inputs=image_input,
            outputs=[image_output, text_output],
            fn=process_image,
            cache_examples=False
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
    print("🚗 Starting License Plate Image Detection System...")
    
    try:
        # Create and launch interface
        interface = create_license_plate_interface()
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7861,  # Different port to avoid conflicts
            share=False,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        print("Please check your dependencies and try again.")

if __name__ == "__main__":
    main()
