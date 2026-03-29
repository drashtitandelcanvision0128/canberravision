"""
Unified Detection Web Application - Gradio Interface
Canberra Vision - All-in-One Detection System

Features:
- Image Upload Detection
- Video Upload Detection
- Live Webcam Detection
- Unified Output (Object, Vehicle, Plate, PPE, Parking)
- JSON Result Display
- Database Storage
"""

import gradio as gr
import cv2
import numpy as np
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import threading
import tempfile
from PIL import Image

# Set working directory to project root
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Set environment variables for OpenCV
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '0'

# Import unified detection components
sys.path.insert(0, str(project_root / "src" / "unified_detection"))
from unified_detector import UnifiedDetector, get_unified_detector
from frame_extractor import FrameExtractor, InputSourceType
from result_formatter import ResultFormatter
from database_service import DatabaseService, get_database_service

# Global detector instance
_detector = None
_formatter = None
_database = None
_webcam_running = False

def get_detector():
    """Get or create global detector instance"""
    global _detector
    if _detector is None:
        _detector = get_unified_detector(model_path="yolo26n.pt", use_gpu=True)
    return _detector

def get_formatter():
    """Get or create global formatter instance"""
    global _formatter
    if _formatter is None:
        _formatter = ResultFormatter()
    return _formatter

def get_database():
    """Get or create global database instance"""
    global _database
    if _database is None:
        _database = get_database_service()
    return _database


def process_image_unified(image: np.ndarray) -> Tuple[np.ndarray, str, str]:
    """
    Process image with unified detection pipeline
    
    Returns:
        Tuple of (annotated_image, json_output, summary_text)
    """
    if image is None:
        return None, "{}", "Please upload an image"
    
    try:
        # Get components
        detector = get_detector()
        formatter = get_formatter()
        database = get_database()
        
        # Reset trackers for new image
        detector.reset_trackers()
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Process image
        result = detector.detect_frame(image, frame_number=0, source="IMAGE")
        
        # Save to database
        if database and database.enabled:
            database.save_detection(result)
        
        # Create annotated image
        annotated = create_annotated_image(image, result)
        
        # Convert to RGB for display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Format JSON output
        json_output = formatter.to_json(result, indent=2)
        
        # Create summary
        summary = create_summary_text(result)
        
        return annotated_rgb, json_output, summary
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return None, "{}", error_msg


def process_video_unified(video_path: str, progress=gr.Progress()) -> Tuple[str, str, str]:
    """
    Process video with unified detection pipeline
    
    Returns:
        Tuple of (output_video_path, json_output, summary_text)
    """
    if not video_path or not os.path.exists(video_path):
        return None, "{}", "Please upload a valid video file"
    
    try:
        # Get components
        detector = get_detector()
        formatter = get_formatter()
        database = get_database()
        
        # Reset trackers
        detector.reset_trackers()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "{}", "Failed to open video"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video path
        timestamp = int(time.time())
        output_path = f"outputs/unified_video_{timestamp}.mp4"
        os.makedirs("outputs", exist_ok=True)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        all_results = []
        
        progress(0, desc="Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 3rd frame for speed
            if frame_count % 3 == 0:
                result = detector.detect_frame(frame, frame_number=frame_count, source="VIDEO")
                all_results.append(result)
                
                # Save to database (every 30 frames)
                if database and database.enabled and frame_count % 30 == 0:
                    database.save_detection(result, source_id=video_path)
                
                # Create annotated frame
                annotated = create_annotated_image(frame, result)
            else:
                annotated = frame
            
            out.write(annotated)
            
            # Update progress
            if total_frames > 0:
                progress(min(frame_count / total_frames, 0.99), desc=f"Processing frame {frame_count}/{total_frames}")
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        progress(1.0, desc="Complete!")
        
        # Format batch results
        if all_results:
            batch_output = formatter.format_batch(all_results)
            json_output = json.dumps(batch_output, indent=2, ensure_ascii=False)
            
            # Create summary from last frame
            summary = create_summary_text(all_results[-1])
            summary += f"\n\n📹 Video: {frame_count} frames processed"
        else:
            json_output = "{}"
            summary = "No detections found"
        
        return output_path, json_output, summary
        
    except Exception as e:
        error_msg = f"Error processing video: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return None, "{}", error_msg


def process_webcam_frame(frame: np.ndarray) -> Tuple[np.ndarray, str, str]:
    """
    Process single webcam frame
    
    Returns:
        Tuple of (annotated_frame, json_output, summary_text)
    """
    if frame is None:
        return None, "{}", "No frame captured"
    
    try:
        # Get components
        detector = get_detector()
        formatter = get_formatter()
        
        # Process frame
        result = detector.detect_frame(frame, source="WEBCAM")
        
        # Create annotated frame
        annotated = create_annotated_image(frame, result)
        
        # Convert to RGB for display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Format JSON
        json_output = formatter.to_json(result, indent=2)
        
        # Create summary
        summary = create_summary_text(result)
        
        return annotated_rgb, json_output, summary
        
    except Exception as e:
        return frame, "{}", f"Error: {str(e)}"


def create_annotated_image(frame: np.ndarray, result) -> np.ndarray:
    """Create annotated image with all detections"""
    annotated = frame.copy()
    
    # Draw vehicles
    for vehicle in result.vehicle_detections:
        x1, y1, x2, y2 = map(int, vehicle.bbox)
        
        # Color based on vehicle type
        if vehicle.vehicle_type == 'bike':
            color = (0, 255, 255)  # Cyan
        elif vehicle.vehicle_type == 'car':
            color = (0, 255, 0)  # Green
        elif vehicle.vehicle_type == 'truck':
            color = (255, 0, 0)  # Blue
        elif vehicle.vehicle_type == 'bus':
            color = (255, 255, 0)  # Yellow
        else:
            color = (128, 128, 128)  # Gray
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{vehicle.vehicle_type.upper()} | {vehicle.color} | {vehicle.confidence:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw PPE detections
    for ppe in result.ppe_detections:
        x1, y1, x2, y2 = map(int, ppe.bbox)
        
        # Determine status color
        if ppe.vehicle_type == 'bike' and not ppe.helmet:
            color = (0, 0, 255)  # Red - violation
        elif ppe.vehicle_type in ['car', 'truck', 'bus'] and not ppe.seatbelt:
            color = (0, 0, 255)  # Red - violation
        else:
            color = (0, 255, 0)  # Green - compliant
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # PPE status
        status_parts = []
        if ppe.helmet:
            status_parts.append("H")
        if ppe.seatbelt:
            status_parts.append("S")
        if ppe.vest:
            status_parts.append("V")
        
        status = f"[{ppe.vehicle_type[:3].upper()}] " + "/".join(status_parts) if status_parts else "[NO PPE]"
        cv2.putText(annotated, status, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw number plates
    for plate in result.plate_detections:
        x1, y1, x2, y2 = map(int, plate.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
        label = f"PLATE: {plate.text}"
        cv2.putText(annotated, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Draw parking slots
    for slot in result.parking_detections:
        x1, y1, x2, y2 = map(int, slot.bbox)
        if slot.occupied:
            color = (0, 0, 255)  # Red
            status = "OCCUPIED"
        else:
            color = (0, 255, 0)  # Green
            status = "EMPTY"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"Slot {slot.slot_id}: {status}"
        cv2.putText(annotated, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw summary panel
    draw_summary_panel(annotated, result)
    
    return annotated


def draw_summary_panel(frame: np.ndarray, result):
    """Draw detection summary panel"""
    panel_height = 140
    panel_width = 400
    overlay = frame.copy()
    
    cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Count violations
    violations = sum(
        1 for p in result.ppe_detections
        if (p.vehicle_type == 'bike' and not p.helmet) or
           (p.vehicle_type in ['car', 'truck', 'bus'] and not p.seatbelt)
    )
    
    lines = [
        "UNIFIED DETECTION SYSTEM",
        f"Vehicles: {len(result.vehicle_detections)}",
        f"Persons: {len(result.ppe_detections)} | Violations: {violations}",
        f"Plates: {len(result.plate_detections)} | Parking: {len(result.parking_detections)}",
        f"Processing: {result.processing_time_ms:.1f}ms"
    ]
    
    y_offset = 30
    for i, line in enumerate(lines):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        size = 0.6 if i == 0 else 0.5
        cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)
        y_offset += 22


def create_summary_text(result) -> str:
    """Create human-readable summary text"""
    lines = []
    
    # Header
    lines.append("🎯 UNIFIED DETECTION RESULTS")
    lines.append("=" * 40)
    
    # Vehicles
    if result.vehicle_detections:
        lines.append(f"\n🚗 VEHICLES: {len(result.vehicle_detections)}")
        for v in result.vehicle_detections:
            lines.append(f"  • {v.vehicle_id}: {v.vehicle_type.upper()} ({v.color}) - {v.confidence:.2f}")
    
    # PPE
    if result.ppe_detections:
        lines.append(f"\n👥 PERSONS: {len(result.ppe_detections)}")
        for p in result.ppe_detections:
            helmet = "✅" if p.helmet else "❌"
            seatbelt = "✅" if p.seatbelt else "❌"
            vest = "✅" if p.vest else "❌"
            lines.append(f"  • {p.person_id}: H{helmet} S{seatbelt} V{vest} ({p.vehicle_type})")
    
    # Plates
    if result.plate_detections:
        lines.append(f"\n📋 LICENSE PLATES: {len(result.plate_detections)}")
        for plate in result.plate_detections:
            lines.append(f"  • {plate.text} (conf: {plate.confidence:.2f})")
    
    # Parking
    if result.parking_detections:
        occupied = sum(1 for s in result.parking_detections if s.occupied)
        empty = len(result.parking_detections) - occupied
        lines.append(f"\n🅿️ PARKING: {len(result.parking_detections)} slots")
        lines.append(f"  • Occupied: {occupied} | Empty: {empty}")
    
    # Performance
    lines.append(f"\n⚡ Processing Time: {result.processing_time_ms:.1f}ms")
    
    return "\n".join(lines)


def create_interface():
    """Create Gradio interface"""
    
    css = """
    .json-output {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        background-color: var(--color-background-secondary);
        color: var(--color-text-primary);
        padding: 15px;
        border-radius: 8px;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid var(--color-border-primary);
    }
    .detection-panel {
        border: 2px solid var(--color-accent-soft);
        border-radius: 10px;
        padding: 20px;
        background: var(--color-background-primary);
    }
    .title-text {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        background: linear-gradient(45deg, var(--color-accent-primary), var(--color-accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    """
    
    with gr.Blocks(css=css, title="Canberra Vision - Unified Detection") as interface:
        
        # Title
        gr.HTML("<div class='title-text'>🚀 CANBERRA VISION</div>")
        gr.Markdown("## Unified Multi-Detection System (Object + Vehicle + Plate + PPE + Parking)", elem_classes="title-text")
        
        # Main detection panel
        with gr.Tab("🎯 Unified Detection Panel"):
            
            with gr.Row():
                # Left column - Inputs
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Input Source")
                    
                    # Input type selection
                    input_type = gr.Radio(
                        choices=["Image", "Video", "Webcam"],
                        value="Image",
                        label="Select Input Type"
                    )
                    
                    # Image input
                    image_input = gr.Image(
                        label="Upload Image",
                        type="numpy",
                        visible=True
                    )
                    
                    # Video input
                    video_input = gr.Video(
                        label="Upload Video",
                        visible=False
                    )
                    
                    # Webcam input
                    webcam_input = gr.Image(
                        label="Live Webcam",
                        source="webcam",
                        streaming=True,
                        visible=False
                    )
                    
                    # Process buttons
                    with gr.Row():
                        process_image_btn = gr.Button(
                            "🔍 Detect All (Image)",
                            variant="primary",
                            size="lg",
                            visible=True
                        )
                        process_video_btn = gr.Button(
                            "🔍 Detect All (Video)",
                            variant="primary",
                            size="lg",
                            visible=False
                        )
                
                # Right column - Results
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Detection Results")
                    
                    with gr.Row():
                        # Annotated output
                        with gr.Column(scale=1):
                            output_image = gr.Image(
                                label="Detected Objects",
                                type="numpy"
                            )
                            output_video = gr.Video(
                                label="Processed Video",
                                visible=False
                            )
                        
                        # JSON output
                        with gr.Column(scale=1):
                            json_output = gr.Code(
                                label="JSON Output",
                                language="json",
                                elem_classes=["json-output"]
                            )
                    
                    # Summary text
                    summary_text = gr.Textbox(
                        label="Detection Summary",
                        lines=10,
                        interactive=False
                    )
        
        # Tab for detection history
        with gr.Tab("📜 Detection History"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Recent Detections")
                    history_btn = gr.Button("🔄 Refresh History")
                    history_output = gr.Dataframe(
                        headers=["Time", "Source", "Vehicles", "Persons", "Plates", "Parking"],
                        datatype=["str", "str", "number", "number", "number", "number"],
                        interactive=False
                    )
        
        # Tab for statistics
        with gr.Tab("📈 Statistics"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Detection Statistics")
                    stats_btn = gr.Button("🔄 Refresh Stats")
                    stats_output = gr.JSON(label="Statistics")
        
        # Tab for about
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## Canberra Vision - Unified Detection System
            
            ### Features
            - **Object Detection**: Detect persons, vehicles, and objects
            - **Vehicle Detection**: Classify type (bike/car/truck/bus) and color
            - **Number Plate Recognition (ANPR)**: Extract license plate text
            - **PPE Detection**: Helmet, seatbelt, and safety vest detection
            - **Parking Detection**: Identify occupied and empty parking slots
            
            ### Detection Rules
            1. **Helmet vs Seatbelt**: If helmet is true, seatbelt MUST be false (and vice versa)
            2. **2-wheeler (bike)**: Only helmet detection allowed
            3. **4-wheeler (car/truck/bus)**: Only seatbelt detection allowed
            4. **Association**: Each person linked to nearest vehicle
            5. **No Duplicates**: Avoid duplicate detections
            
            ### Output Format
            Strict JSON with detections for: objects, vehicles, number_plates, ppe, parking
            
            ### Database
            All detections are stored in PostgreSQL for analytics and history
            """)
        
        # Event handlers
        
        # Input type change
        def on_input_type_change(choice):
            if choice == "Image":
                return {
                    image_input: gr.Image(visible=True),
                    video_input: gr.Video(visible=False),
                    webcam_input: gr.Image(visible=False),
                    process_image_btn: gr.Button(visible=True),
                    process_video_btn: gr.Button(visible=False),
                    output_image: gr.Image(visible=True),
                    output_video: gr.Video(visible=False)
                }
            elif choice == "Video":
                return {
                    image_input: gr.Image(visible=False),
                    video_input: gr.Video(visible=True),
                    webcam_input: gr.Image(visible=False),
                    process_image_btn: gr.Button(visible=False),
                    process_video_btn: gr.Button(visible=True),
                    output_image: gr.Image(visible=False),
                    output_video: gr.Video(visible=True)
                }
            else:  # Webcam
                return {
                    image_input: gr.Image(visible=False),
                    video_input: gr.Video(visible=False),
                    webcam_input: gr.Image(visible=True, streaming=True),
                    process_image_btn: gr.Button(visible=True),
                    process_video_btn: gr.Button(visible=False),
                    output_image: gr.Image(visible=True),
                    output_video: gr.Video(visible=False)
                }
        
        input_type.change(
            fn=on_input_type_change,
            inputs=[input_type],
            outputs=[image_input, video_input, webcam_input, 
                    process_image_btn, process_video_btn,
                    output_image, output_video]
        )
        
        # Process image
        process_image_btn.click(
            fn=process_image_unified,
            inputs=[image_input],
            outputs=[output_image, json_output, summary_text]
        )
        
        # Process video
        process_video_btn.click(
            fn=process_video_unified,
            inputs=[video_input],
            outputs=[output_video, json_output, summary_text]
        )
        
        # Webcam stream
        webcam_input.stream(
            fn=process_webcam_frame,
            inputs=[webcam_input],
            outputs=[output_image, json_output, summary_text]
        )
    
    return interface


def main():
    """Main entry point"""
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Create interface
    interface = create_interface()
    
    # Launch
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
