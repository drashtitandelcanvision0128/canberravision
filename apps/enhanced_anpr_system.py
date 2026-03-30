#!/usr/bin/env python3
"""
Enhanced ANPR (Automatic Number Plate Recognition) System
Similar to the reference image with comprehensive vehicle detection and analysis.
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Set working directory to project root
script_dir = Path(__file__).parent
project_root = script_dir.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import gradio as gr
import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO
import PIL.Image as Image
from sklearn.cluster import KMeans

# Import existing modules
try:
    from src.ocr.text_extractor import TextExtractor
    from src.ocr.license_plate_detector import LicensePlateDetector
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[WARNING] OCR modules not available")

# Import new modules
try:
    from modules.vehicle_classification import VehicleClassifier, VehicleColorDetector
    from modules.vehicle_database import VehicleDatabase, AlertSystem, VehicleMatcher
    CLASSIFICATION_AVAILABLE = True
except ImportError:
    CLASSIFICATION_AVAILABLE = False
    print("[WARNING] Classification modules not available")

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

# Vehicle color mapping
COLOR_NAMES = {
    0: 'Black', 1: 'White', 2: 'Gray', 3: 'Silver', 4: 'Blue',
    5: 'Red', 6: 'Green', 7: 'Yellow', 8: 'Brown', 9: 'Purple'
}

# Vehicle make/model mapping (simplified)
VEHICLE_MAKES = {
    'car': 'Sedan',
    'truck': 'Truck',
    'bus': 'Bus',
    'motorcycle': 'Motorcycle',
    'bicycle': 'Bicycle'
}

class EnhancedANPRSystem:
    """Enhanced ANPR System with comprehensive vehicle analysis."""
    
    def __init__(self):
        self.yolo_model = None
        self.text_extractor = None
        self.license_detector = None
        
        # New modules
        self.vehicle_classifier = None
        self.color_detector = None
        self.vehicle_database = None
        self.alert_system = None
        self.vehicle_matcher = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models."""
        try:
            # Load YOLO model for object detection
            self.yolo_model = YOLO('yolov8n.pt')
            self.yolo_model.to(DEVICE)
            print("[INFO] YOLO model loaded successfully")
            
            # Initialize OCR components
            if OCR_AVAILABLE:
                self.text_extractor = TextExtractor()
                self.license_detector = LicensePlateDetector()
                print("[INFO] OCR components initialized")
            
            # Initialize new classification modules
            if CLASSIFICATION_AVAILABLE:
                self.vehicle_classifier = VehicleClassifier()
                self.color_detector = VehicleColorDetector()
                print("[INFO] Vehicle classification modules initialized")
                
                # Initialize database and alert system
                self.vehicle_database = VehicleDatabase()
                self.alert_system = AlertSystem(self.vehicle_database)
                self.vehicle_matcher = VehicleMatcher(self.vehicle_database)
                print("[INFO] Database and alert system initialized")
            
        except Exception as e:
            print(f"[ERROR] Model initialization failed: {e}")
    
    def _load_vehicle_database(self) -> Dict:
        """Load or create a mock vehicle database."""
        # Mock database for demonstration
        return {
            'LD62 WRC': {'make': 'BMW', 'model': '3 Series', 'color': 'Silver', 'alert': False},
            'YY15 FUD': {'make': 'Audi', 'model': 'A4', 'color': 'Black', 'alert': False},
            'XX12 GHD': {'make': 'Mercedes', 'model': 'C-Class', 'color': 'White', 'alert': False},
            'AB12 GHT': {'make': 'Toyota', 'model': 'Camry', 'color': 'Blue', 'alert': True},
            'KP09 ZXE': {'make': 'Honda', 'model': 'Civic', 'color': 'Red', 'alert': False}
        }
    
    def detect_vehicles(self, image: np.ndarray) -> List[Dict]:
        """Detect vehicles in the image."""
        results = self.yolo_model(image)
        vehicles = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Filter for vehicle classes
                    if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        vehicles.append({
                            'bbox': (x1, y1, x2, y2),
                            'class': cls,
                            'confidence': conf,
                            'label': self.yolo_model.names[cls]
                        })
        
        return vehicles
    
    def classify_vehicle_type(self, vehicle_crop: np.ndarray) -> str:
        """Classify vehicle type using the enhanced classifier."""
        if self.vehicle_classifier is None:
            return 'Unknown'
        
        try:
            result = self.vehicle_classifier.classify_vehicle(vehicle_crop)
            return f"{result['make']} {result['model']}"
        except Exception as e:
            print(f"[ERROR] Vehicle classification failed: {e}")
            return 'Unknown'
    
    def detect_vehicle_color(self, vehicle_crop: np.ndarray) -> str:
        """Detect vehicle color using the enhanced color detector."""
        if self.color_detector is None:
            return 'Unknown'
        
        try:
            # Try HSV-based detection first
            result = self.color_detector.detect_color(vehicle_crop)
            if result['confidence'] > 0.1:
                return result['color']
            
            # Fallback to K-means if HSV confidence is low
            result = self.color_detector.detect_color_kmeans(vehicle_crop)
            return result['color']
            
        except Exception as e:
            print(f"[ERROR] Color detection failed: {e}")
            return 'Unknown'
    
    def extract_license_plate(self, image: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """Extract license plate from vehicle region."""
        try:
            if not OCR_AVAILABLE:
                return None
                
            x1, y1, x2, y2 = vehicle_bbox
            vehicle_crop = image[y1:y2, x1:x2]
            
            # Use text extractor to find license plates
            result = self.text_extractor.extract_text_comprehensive(vehicle_crop)
            
            # Look for license plates in results
            for plate_info in result.get('license_plates', []):
                if plate_info.get('confidence', 0) > 0.5:
                    return plate_info['text']
            
            return None
            
        except Exception as e:
            print(f"[ERROR] License plate extraction failed: {e}")
            return None
    
    def estimate_speed(self, current_position: Tuple[int, int], previous_position: Optional[Tuple[int, int]], 
                      time_diff: float) -> float:
        """Estimate vehicle speed (simplified)."""
        if previous_position is None or time_diff <= 0:
            return 0.0
        
        # Calculate pixel distance
        pixel_distance = np.sqrt((current_position[0] - previous_position[0])**2 + 
                                (current_position[1] - previous_position[1])**2)
        
        # Convert to km/h (simplified conversion)
        # Assuming 1 pixel = 0.1 meters and the conversion factor
        speed_ms = (pixel_distance * 0.1) / time_diff
        speed_kmh = speed_ms * 3.6
        
        return min(speed_kmh, 200)  # Cap at 200 km/h
    
    def query_database(self, license_plate: str) -> Dict:
        """Query vehicle database for information."""
        if self.vehicle_database is None:
            return {
                'make': 'Unknown',
                'model': 'Unknown',
                'color': 'Unknown',
                'alert': False
            }
        
        vehicle = self.vehicle_database.get_vehicle(license_plate)
        if vehicle:
            return {
                'make': vehicle.get('make', 'Unknown'),
                'model': vehicle.get('model', 'Unknown'),
                'color': vehicle.get('color', 'Unknown'),
                'alert': bool(vehicle.get('is_stolen', False)),
                'owner': vehicle.get('owner_name', 'Unknown'),
                'alert_reason': vehicle.get('alert_reason', None)
            }
        
        return {
            'make': 'Unknown',
            'model': 'Unknown',
            'color': 'Unknown',
            'alert': False
        }
    
    def check_alerts(self, license_plate: str, vehicle_data: Dict) -> List[Dict]:
        """Check for alerts related to the vehicle."""
        if self.alert_system is None:
            return []
        
        return self.alert_system.check_alerts(license_plate, vehicle_data)
    
    def find_similar_vehicles(self, license_plate: str) -> List[Dict]:
        """Find similar vehicles in the database."""
        if self.vehicle_matcher is None:
            return []
        
        return self.vehicle_matcher.find_similar_vehicles(license_plate)
    
    def process_image(self, image: np.ndarray) -> Dict:
        """Process image and extract comprehensive vehicle information."""
        start_time = time.time()
        
        # Detect vehicles
        vehicles = self.detect_vehicles(image)
        
        results = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'date': datetime.now().strftime('%d/%m/%Y'),
            'vehicles': [],
            'alerts': [],
            'similar_vehicles': [],
            'processing_time': 0
        }
        
        for i, vehicle in enumerate(vehicles):
            x1, y1, x2, y2 = vehicle['bbox']
            vehicle_crop = image[y1:y2, x1:x2]
            
            # Extract information
            license_plate = self.extract_license_plate(image, vehicle['bbox'])
            vehicle_type = self.classify_vehicle_type(vehicle_crop)
            color = self.detect_vehicle_color(vehicle_crop)
            speed = self.estimate_speed((x1 + x2) // 2, (y1 + y2) // 2, None, 1.0)
            
            # Query database
            db_info = self.query_database(license_plate) if license_plate else {}
            
            # Create vehicle data for alert checking
            vehicle_data = {
                'speed': speed,
                'color': color,
                'type': vehicle_type
            }
            
            # Check for alerts
            alerts = self.check_alerts(license_plate, vehicle_data) if license_plate else []
            
            # Find similar vehicles
            similar_vehicles = self.find_similar_vehicles(license_plate) if license_plate else []
            
            vehicle_info = {
                'id': i + 1,
                'bbox': vehicle['bbox'],
                'license_plate': license_plate or 'Not Detected',
                'make': db_info.get('make', vehicle_type.split()[0] if ' ' in vehicle_type else vehicle_type),
                'model': db_info.get('model', vehicle_type.split()[1] if ' ' in vehicle_type else 'Unknown'),
                'color': db_info.get('color', color),
                'confidence': vehicle['confidence'],
                'speed': round(speed, 1),
                'alert': db_info.get('alert', False),
                'label': vehicle['label'],
                'owner': db_info.get('owner', 'Unknown'),
                'alerts': alerts,
                'similar_vehicles': similar_vehicles
            }
            
            results['vehicles'].append(vehicle_info)
            results['alerts'].extend(alerts)
            
            # Add unique similar vehicles to results
            for similar in similar_vehicles:
                if similar not in results['similar_vehicles']:
                    results['similar_vehicles'].append(similar)
        
        results['processing_time'] = round(time.time() - start_time, 2)
        return results
    
    def create_anpr_display(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """Create ANPR display similar to the reference image."""
        # Create a copy for annotation
        display_image = image.copy()
        
        # Draw bounding boxes and information for each vehicle
        for vehicle in results['vehicles']:
            x1, y1, x2, y2 = vehicle['bbox']
            
            # Draw bounding box with alert color
            if vehicle['alert'] or vehicle['alerts']:
                color = (0, 0, 255)  # Red for alerts
                label_prefix = "ALERT: "
            else:
                color = (0, 255, 0)  # Green for normal
                label_prefix = ""
            
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            
            # Add license plate text
            plate_text = vehicle['license_plate']
            if plate_text != 'Not Detected':
                label = f"{label_prefix}{plate_text}"
                cv2.putText(display_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Add vehicle info
            info_text = f"{vehicle['make']} {vehicle['model']}"
            cv2.putText(display_image, info_text, (x1, y2 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            # Add speed
            speed_text = f"Speed: {vehicle['speed']} km/h"
            cv2.putText(display_image, speed_text, (x1, y2 + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Add timestamp and date
        cv2.putText(display_image, f"Time: {results['timestamp']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_image, f"Date: {results['date']}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add alert summary if any alerts exist
        if results['alerts']:
            alert_text = f"ALERTS: {len(results['alerts'])} active"
            cv2.putText(display_image, alert_text, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return display_image

# Initialize the ANPR system
anpr_system = EnhancedANPRSystem()

def process_anpr_image(input_image):
    """Process image for ANPR detection."""
    if input_image is None:
        return None, "No image provided", "", ""
    
    try:
        # Convert PIL to numpy
        image = np.array(input_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Process image
        results = anpr_system.process_image(image)
        
        # Create display image
        display_image = anpr_system.create_anpr_display(image, results)
        
        # Generate detailed information text
        info_text = f"🔍 ANPR Detection Results\n"
        info_text += f"⏱️ Processing Time: {results['processing_time']}s\n"
        info_text += f"📅 Date: {results['date']}\n"
        info_text += f"🕐 Time: {results['timestamp']}\n"
        info_text += f"🚗 Vehicles Detected: {len(results['vehicles'])}\n"
        info_text += f"🚨 Active Alerts: {len(results['alerts'])}\n\n"
        
        for vehicle in results['vehicles']:
            info_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            info_text += f"🚗 Vehicle {vehicle['id']}:\n"
            info_text += f"📋 License Plate: {vehicle['license_plate']}\n"
            info_text += f"🏭 Make/Model: {vehicle['make']} {vehicle['model']}\n"
            info_text += f"🎨 Color: {vehicle['color']}\n"
            info_text += f"⚡ Speed: {vehicle['speed']} km/h\n"
            info_text += f"👤 Owner: {vehicle['owner']}\n"
            info_text += f"📊 Confidence: {vehicle['confidence']:.2f}\n"
            info_text += f"🚨 Status: {'ALERT' if vehicle['alert'] or vehicle['alerts'] else 'Clear'}\n"
            
            # Add vehicle-specific alerts
            if vehicle['alerts']:
                info_text += "⚠️ Alerts:\n"
                for alert in vehicle['alerts']:
                    info_text += f"   • {alert['message']}\n"
            
            info_text += "\n"
        
        # Generate database matches with similarity scores
        db_matches = f"🗄️ Database Matches & Similar Vehicles\n"
        db_matches += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        if results['similar_vehicles']:
            for similar in results['similar_vehicles'][:10]:  # Top 10 matches
                vehicle = similar['vehicle']
                similarity = similar['similarity']
                alert_status = "🚨 ALERT" if vehicle.get('is_stolen') else "✅ Clear"
                
                db_matches += f"📋 {vehicle['license_plate']}\n"
                db_matches += f"   🏭 Vehicle: {vehicle['make']} {vehicle['model']}\n"
                db_matches += f"   🎨 Color: {vehicle['color']}\n"
                db_matches += f"   📊 Match: {similarity:.1%}\n"
                db_matches += f"   🚨 Status: {alert_status}\n"
                if vehicle.get('alert_reason'):
                    db_matches += f"   ⚠️ Reason: {vehicle['alert_reason']}\n"
                db_matches += "\n"
        else:
            db_matches += "❌ No similar vehicles found in database\n\n"
        
        # Generate active alerts summary
        alerts_summary = f"🚨 Active Alerts Summary\n"
        alerts_summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        if results['alerts']:
            for i, alert in enumerate(results['alerts'], 1):
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(alert.get('severity', 'medium'), "⚪")
                alerts_summary += f"{i}. {severity_icon} {alert['type'].upper()}\n"
                alerts_summary += f"   📋 Plate: {alert['license_plate']}\n"
                alerts_summary += f"   📝 Message: {alert['message']}\n\n"
        else:
            alerts_summary += "✅ No active alerts\n\n"
        
        # Convert back to RGB for display
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        return display_image, info_text, db_matches, alerts_summary
        
    except Exception as e:
        error_msg = f"❌ Error processing image: {str(e)}"
        return None, error_msg, "", ""

# Create Gradio interface
def create_anpr_interface():
    """Create the ANPR system interface."""
    with gr.Blocks(title="Enhanced ANPR System") as interface:
        gr.Markdown("# 🚗 Enhanced ANPR System")
        gr.Markdown("Automatic Number Plate Recognition with Vehicle Analysis")
        gr.Markdown("🔍 Real-time vehicle detection, license plate recognition, and database matching")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="📷 Input Image", type="pil")
                process_btn = gr.Button("🔍 Process Image", variant="primary", size="lg")
                
                gr.Markdown("### 📝 System Features")
                gr.Markdown("""
                - 🚗 **Vehicle Detection**: Advanced YOLO-based detection
                - 📋 **License Plate Recognition**: OCR with multiple methods
                - 🏭 **Make/Model Classification**: Deep learning identification
                - 🎨 **Color Detection**: HSV and K-means analysis
                - ⚡ **Speed Estimation**: Real-time calculation
                - 🗄️ **Database Matching**: Similarity scoring
                - 🚨 **Alert System**: Stolen vehicle detection
                """)
                
            with gr.Column(scale=2):
                output_image = gr.Image(label="🎯 ANPR Detection Result", type="pil")
        
        with gr.Row():
            with gr.Column():
                info_output = gr.Textbox(
                    label="🔍 Detection Information", 
                    lines=20, 
                    max_lines=25
                )
            
            with gr.Column():
                db_output = gr.Textbox(
                    label="🗄️ Database Matches", 
                    lines=20, 
                    max_lines=25
                )
                
        with gr.Row():
            alerts_output = gr.Textbox(
                label="🚨 Active Alerts", 
                lines=10, 
                max_lines=15
            )
        
        gr.Markdown("## 📸 Example Images")
        with gr.Row():
            with gr.Column():
                example1_btn = gr.Button("📸 Load Example 1", size="sm")
            with gr.Column():
                example2_btn = gr.Button("📸 Load Example 2", size="sm")
            with gr.Column():
                example3_btn = gr.Button("📸 Load Example 3", size="sm")
        
        with gr.Row():
            with gr.Column():
                gr.Image("inputs/input_image_1772014809.jpg", label="Example 1 Preview")
            with gr.Column():
                gr.Image("inputs/input_image_1772017300.jpg", label="Example 2 Preview")
            with gr.Column():
                gr.Image("inputs/input_image_1772017353.jpg", label="Example 3 Preview")
        
        # Event handlers
        process_btn.click(
            fn=process_anpr_image,
            inputs=[input_image],
            outputs=[output_image, info_output, db_output, alerts_output]
        )
        
        # Auto-process when image is uploaded
        input_image.change(
            fn=process_anpr_image,
            inputs=[input_image],
            outputs=[output_image, info_output, db_output, alerts_output]
        )
        
        # Example image handlers
        def load_example1():
            img = Image.open("inputs/input_image_1772014809.jpg")
            return process_anpr_image(img)
        
        def load_example2():
            img = Image.open("inputs/input_image_1772017300.jpg")
            return process_anpr_image(img)
        
        def load_example3():
            img = Image.open("inputs/input_image_1772017353.jpg")
            return process_anpr_image(img)
        
        example1_btn.click(
            fn=load_example1,
            outputs=[output_image, info_output, db_output, alerts_output]
        )
        
        example2_btn.click(
            fn=load_example2,
            outputs=[output_image, info_output, db_output, alerts_output]
        )
        
        example3_btn.click(
            fn=load_example3,
            outputs=[output_image, info_output, db_output, alerts_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        🔧 **System Information**: This ANPR system uses advanced computer vision techniques 
        including YOLO object detection, OCR text extraction, and deep learning-based 
        vehicle classification to provide comprehensive vehicle analysis.
        """)
    
    return interface

if __name__ == "__main__":
    print("[INFO] Starting Enhanced ANPR System...")
    
    # Create and launch interface
    interface = create_anpr_interface()
    
    interface.launch(
        server_name="localhost",
        server_port=7865,
        share=False,
        debug=True
    )
