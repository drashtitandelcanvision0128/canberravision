#!/usr/bin/env python3
"""
🚗 Simple License Plate Detection Tool
=====================================

Easy-to-use tool for detecting license plates in images.
Supports multi-colored plates (white, yellow, blue, red, etc.).

Usage:
    python tools/simple_plate_detector.py
    python tools/simple_plate_detector.py --image path/to/image.jpg
    python tools/simple_plate_detector.py --folder path/to/images/
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class SimplePlateDetector:
    def __init__(self):
        self.ocr_available = self.check_ocr_availability()
        print(f"🔍 OCR Available: {'✅' if self.ocr_available else '❌'}")
    
    def check_ocr_availability(self):
        """Check if OCR libraries are available"""
        try:
            import pytesseract
            return True
        except:
            try:
                from paddleocr import PaddleOCR
                return True
            except:
                return False
    
    def detect_license_plates(self, image_path):
        """
        Detect license plates in an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Detection results
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Cannot read image: {image_path}"}
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply multiple preprocessing techniques for different colored plates
            plates = []
            
            # Method 1: Standard threshold for white/light plates
            plates.extend(self.detect_with_threshold(gray, image, "white"))
            
            # Method 2: Inverted threshold for dark plates
            plates.extend(self.detect_with_threshold(gray, image, "dark"))
            
            # Method 3: Adaptive threshold for varying lighting
            plates.extend(self.detect_with_adaptive_threshold(gray, image))
            
            # Method 4: Edge detection for plate boundaries
            plates.extend(self.detect_with_edges(gray, image))
            
            # Remove duplicates and sort by confidence
            plates = self.remove_duplicate_plates(plates)
            plates = sorted(plates, key=lambda x: x['confidence'], reverse=True)
            
            # Extract text from detected plates
            for plate in plates:
                plate['text'] = self.extract_plate_text(image, plate['bbox'])
                plate['color'] = self.detect_plate_color(image, plate['bbox'])
            
            # Draw results on image
            result_image = self.draw_results(image.copy(), plates)
            
            return {
                "success": True,
                "plates": plates,
                "result_image": result_image,
                "total_plates": len(plates)
            }
            
        except Exception as e:
            return {"error": f"Detection failed: {str(e)}"}
    
    def detect_with_threshold(self, gray, original, plate_type="white"):
        """Detect plates using thresholding"""
        plates = []
        
        try:
            if plate_type == "white":
                # For white/light plates
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            else:
                # For dark plates
                _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            
            # Noise reduction
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # License plate characteristics
                if (2.0 <= aspect_ratio <= 6.0 and 
                    w > 60 and h > 15 and 
                    w < gray.shape[1] * 0.8 and h < gray.shape[0] * 0.3):
                    
                    area = cv2.contourArea(contour)
                    if area > 500:
                        plates.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': 0.7,
                            'method': f'threshold_{plate_type}'
                        })
        
        except Exception as e:
            print(f"Threshold detection error: {e}")
        
        return plates
    
    def detect_with_adaptive_threshold(self, gray, original):
        """Detect plates using adaptive thresholding"""
        plates = []
        
        try:
            # Adaptive threshold for varying lighting
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if (2.0 <= aspect_ratio <= 6.0 and 
                    w > 80 and h > 20 and 
                    cv2.contourArea(contour) > 1000):
                    
                    plates.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 0.6,
                        'method': 'adaptive_threshold'
                    })
        
        except Exception as e:
            print(f"Adaptive threshold error: {e}")
        
        return plates
    
    def detect_with_edges(self, gray, original):
        """Detect plates using edge detection"""
        plates = []
        
        try:
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to connect broken lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            edges = cv2.dilate(edges, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if (2.0 <= aspect_ratio <= 5.5 and 
                    w > 100 and h > 25 and 
                    cv2.contourArea(contour) > 1500):
                    
                    plates.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 0.5,
                        'method': 'edge_detection'
                    })
        
        except Exception as e:
            print(f"Edge detection error: {e}")
        
        return plates
    
    def remove_duplicate_plates(self, plates):
        """Remove duplicate plate detections"""
        if not plates:
            return []
        
        unique_plates = []
        
        for plate in plates:
            is_duplicate = False
            for existing in unique_plates:
                # Check if bounding boxes overlap significantly
                if self.bboxes_overlap(plate['bbox'], existing['bbox'], 0.5):
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if plate['confidence'] > existing['confidence']:
                        unique_plates.remove(existing)
                        unique_plates.append(plate)
                    break
            
            if not is_duplicate:
                unique_plates.append(plate)
        
        return unique_plates
    
    def bboxes_overlap(self, bbox1, bbox2, threshold=0.5):
        """Check if two bounding boxes overlap"""
        x1_max = max(bbox1[0], bbox2[0])
        y1_max = max(bbox1[1], bbox2[1])
        x2_min = min(bbox1[2], bbox2[2])
        y2_min = min(bbox1[3], bbox2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return False
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union > threshold
    
    def extract_plate_text(self, image, bbox):
        """Extract text from license plate region"""
        try:
            x1, y1, x2, y2 = bbox
            plate_region = image[y1:y2, x1:x2]
            
            if plate_region.size == 0:
                return "NO_TEXT"
            
            # Preprocess for OCR
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply threshold
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Try OCR if available
            if self.ocr_available:
                try:
                    import pytesseract
                    text = pytesseract.image_to_string(
                        thresh, 
                        config='--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                    )
                    return self.clean_text(text)
                except:
                    pass
                
                try:
                    from paddleocr import PaddleOCR
                    ocr = PaddleOCR(use_angle_cls=True, lang='en')
                    result = ocr.ocr(thresh, cls=True)
                    if result and result[0]:
                        text = result[0][0][1][0]
                        return self.clean_text(text)
                except:
                    pass
            
            return "OCR_UNAVAILABLE"
            
        except Exception as e:
            return f"TEXT_ERROR: {str(e)[:20]}"
    
    def clean_text(self, text):
        """Clean OCR text"""
        if not text:
            return ""
        
        # Remove non-alphanumeric characters except spaces and hyphens
        text = ''.join(c for c in text if c.isalnum() or c in ' -')
        text = text.upper().strip()
        
        # Common OCR corrections
        corrections = {
            '0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z'
        }
        
        # Only apply corrections if it makes sense for license plates
        cleaned = ""
        for char in text:
            if char in corrections and len(text) > 4:  # Only for longer text
                cleaned += corrections[char]
            else:
                cleaned += char
        
        return cleaned
    
    def detect_plate_color(self, image, bbox):
        """Detect the dominant color of the license plate"""
        try:
            x1, y1, x2, y2 = bbox
            plate_region = image[y1:y2, x1:x2]
            
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
    
    def draw_results(self, image, plates):
        """Draw detection results on image"""
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
        ]
        
        for i, plate in enumerate(plates):
            bbox = plate['bbox']
            color = colors[i % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Prepare label
            text = plate.get('text', 'NO_TEXT')
            confidence = plate.get('confidence', 0)
            color_name = plate.get('color', 'unknown')
            
            label = f"{text} ({confidence:.2f}) [{color_name}]"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (bbox[0], bbox[1]-25), 
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (bbox[0], bbox[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def process_single_image(self, image_path):
        """Process a single image"""
        print(f"\n🔍 Processing: {image_path}")
        
        result = self.detect_license_plates(image_path)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"✅ Found {result['total_plates']} license plate(s):")
        
        for i, plate in enumerate(result['plates']):
            print(f"\n🚗 Plate {i+1}:")
            print(f"   📝 Text: {plate['text']}")
            print(f"   🎨 Color: {plate['color']}")
            print(f"   📊 Confidence: {plate['confidence']:.2f}")
            print(f"   🔍 Method: {plate['method']}")
            print(f"   📍 Location: {plate['bbox']}")
        
        # Save result image
        output_path = image_path.replace('.', '_result.')
        cv2.imwrite(output_path, result['result_image'])
        print(f"\n💾 Result saved: {output_path}")
    
    def process_folder(self, folder_path):
        """Process all images in a folder"""
        print(f"\n📁 Processing folder: {folder_path}")
        
        # Supported image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        if not image_files:
            print("❌ No image files found")
            return
        
        print(f"📊 Found {len(image_files)} image(s)")
        
        for image_path in image_files:
            self.process_single_image(image_path)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Simple License Plate Detection Tool')
    parser.add_argument('--image', help='Path to single image file')
    parser.add_argument('--folder', help='Path to folder with images')
    
    args = parser.parse_args()
    
    print("🚗 Simple License Plate Detection Tool")
    print("=" * 50)
    
    detector = SimplePlateDetector()
    
    if args.image:
        if os.path.exists(args.image):
            detector.process_single_image(args.image)
        else:
            print(f"❌ Image not found: {args.image}")
    elif args.folder:
        if os.path.exists(args.folder):
            detector.process_folder(args.folder)
        else:
            print(f"❌ Folder not found: {args.folder}")
    else:
        print("\n📋 Usage:")
        print("   python tools/simple_plate_detector.py --image path/to/image.jpg")
        print("   python tools/simple_plate_detector.py --folder path/to/images/")
        print("\n💡 Or place images in 'inputs/' folder and run:")
        print("   python tools/simple_plate_detector.py --folder inputs/")

if __name__ == "__main__":
    main()
