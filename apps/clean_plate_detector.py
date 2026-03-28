#!/usr/bin/env python3
"""
🚗 Clean License Plate Detector - No Errors, Clean Display
=========================================================

Simple, clean license plate detector with proper display layout.
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
import re

class CleanPlateDetector:
    def __init__(self):
        self.ocr_available = self.check_ocr()
    
    def check_ocr(self):
        try:
            import pytesseract
            return True
        except:
            return False
    
    def detect_plates(self, image):
        try:
            if isinstance(image, Image.Image):
                image_np = np.array(image)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_np = image.copy()
            
            # Simple detection
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            
            # Find potential plate regions
            plates = []
            
            # Method 1: Contour detection
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if (2.0 <= aspect_ratio <= 6.0 and w > 80 and h > 25):
                    plate_region = image_np[y:y+h, x:x+w]
                    text = self.extract_text(plate_region)
                    
                    if self.is_license_plate(text):
                        plates.append({
                            'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h,
                            'text': text,
                            'confidence': 0.8
                        })
            
            # Draw results cleanly
            result_image = self.draw_clean_results(image_np, plates)
            
            return result_image, plates
            
        except Exception as e:
            print(f"Error: {e}")
            return image, []
    
    def extract_text(self, region):
        try:
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            if self.ocr_available:
                try:
                    import pytesseract
                    text = pytesseract.image_to_string(thresh, config='--psm 7')
                    text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    return text
                except:
                    pass
            
            return "DETECTED"
        except:
            return ""
    
    def is_license_plate(self, text):
        if not text or len(text) < 4:
            return False
        
        # Simple pattern check
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        
        return has_letters and has_numbers and 4 <= len(text) <= 10
    
    def draw_clean_results(self, image, plates):
        result = image.copy()
        
        # Draw license plates at top with clean layout
        y_offset = 30
        
        for i, plate in enumerate(plates):
            x1, y1, x2, y2 = plate['x1'], plate['y1'], plate['x2'], plate['y2']
            
            # Green box for license plate
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Clean label at top of image
            text = f"🚗 {plate['text']}"
            label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Background for label at top
            cv2.rectangle(result, (10, y_offset - 25), (10 + label_size[0] + 10, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(result, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            y_offset += 35
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

def create_interface():
    detector = CleanPlateDetector()
    
    def process_image(image):
        if image is None:
            return None, "Please upload an image"
        
        result_image, plates = detector.detect_plates(image)
        
        if plates:
            text = f"🎉 Found {len(plates)} License Plate(s):\n\n"
            for i, plate in enumerate(plates):
                text += f"🚗 Plate {i+1}: {plate['text']}\n"
        else:
            text = "❌ No license plates detected\n\n💡 Try uploading a clearer image of a license plate"
        
        return result_image, text
    
    with gr.Blocks(title="Clean Plate Detector") as interface:
        gr.markdown("# 🚗 Clean License Plate Detector\n\nSimple, clean detection with proper display layout")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="pil", height=400)
                btn = gr.Button("🔍 Detect Plates", variant="primary", size="lg")
            
            with gr.Column():
                image_output = gr.Image(label="Results", type="pil", height=400)
                text_output = gr.Textbox(label="Detection Info", lines=8)
        
        btn.click(process_image, image_input, [image_output, text_output])
        image_input.change(process_image, image_input, [image_output, text_output])
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_port=7865)
