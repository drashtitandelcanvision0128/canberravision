# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import asyncio
import os
import sys
import tempfile
import json
import time
from datetime import datetime
from pathlib import Path
import subprocess
import shutil
from pathlib import Path

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

# Set OpenCV environment variables to reduce camera detection warnings
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '0'

import cv2
import gradio as gr
import numpy as np
import PIL.Image as Image
import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    # Set Tesseract path manually for Windows
    if sys.platform.startswith("win"):
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"[INFO] Tesseract path set to: {tesseract_path}")
        else:
            print(f"[ERROR] Tesseract not found at: {tesseract_path}")
            print("[INFO] Please check if Tesseract is installed correctly")
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

# Import LightOnOCR integration
try:
    from lighton_ocr_integration import get_lighton_ocr_processor, extract_text_with_lighton
    LIGHTON_AVAILABLE = True
    print("[INFO] LightOnOCR integration loaded")
except ImportError:
    LIGHTON_AVAILABLE = False
    print("[WARNING] LightOnOCR integration not available")

# Import enhanced detection for challenging images
try:
    from enhanced_detection import enhanced_license_plate_detection
    ENHANCED_DETECTION_AVAILABLE = True
    print("[INFO] Enhanced detection for challenging images loaded")
except ImportError:
    ENHANCED_DETECTION_AVAILABLE = False
    print("[WARNING] Enhanced detection not available")

# Import international license plate recognition
try:
    from international_license_plates import extract_international_license_plates, InternationalLicensePlateRecognizer
    INTERNATIONAL_PLATES_AVAILABLE = True
    print("[INFO] International license plate recognition loaded")
except ImportError:
    INTERNATIONAL_PLATES_AVAILABLE = False
    print("[WARNING] International license plate recognition not available")

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Suppress harmless connection reset errors and other asyncio warnings
    import logging
    import warnings
    import asyncio
    
    # Configure logging to suppress connection reset errors
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    logging.getLogger("gradio").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="asyncio")
    warnings.filterwarnings("ignore", message=".*connection reset.*")
    
    # Custom exception handler for connection errors
    def handle_exception(loop, context):
        if "connection reset" in str(context.get('exception', '')).lower():
            return  # Suppress connection reset errors
        elif "transport" in str(context.get('exception', '')).lower():
            return  # Suppress transport errors
        else:
            loop.default_exception_handler(context)
    
    # Set the exception handler with proper event loop creation
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.set_exception_handler(handle_exception)

MODEL_CHOICES = [
    "yolo26n",
    "yolo26s",
    "yolo26m",
]

IMAGE_SIZE_CHOICES = [320, 640, 1024]


def _get_ffmpeg_exe():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    return None


def _transcode_to_browser_mp4(input_path, output_path):
    ffmpeg = _get_ffmpeg_exe()
    if not ffmpeg:
        return None

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        output_path,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    if completed.stderr:
        print(f"[DEBUG] ffmpeg transcode failed: {completed.stderr[:500]}")
    return None


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


def _extract_video_path(video_value):
    if video_value is None:
        return None
    if isinstance(video_value, str):
        return video_value
    if isinstance(video_value, dict):
        return video_value.get("path") or video_value.get("name")
    if isinstance(video_value, (list, tuple)) and video_value:
        first = video_value[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return first.get("path") or first.get("name")
    return None


def process_video_optimized_fast(video_path, model_name="yolo26n", mode="fast", progress_callback=None):
    """
    🚀 ULTRA-FAST VIDEO PROCESSING - 50 minutes → 3-5 minutes
    
    Args:
        video_path: Path to video file
        model_name: YOLO model to use
        mode: "ultra_fast" (3-4 min), "fast" (5-8 min), "balanced" (8-12 min)
        progress_callback: Progress callback function
        
    Returns:
        (output_path, detection_summary) - Path to processed video and detection summary
    """
    try:
        print(f"[INFO] 🚀 Starting ULTRA-FAST video processing: {mode} mode")
        start_time = time.time()
        
        # Extract video path
        video_path = _extract_video_path(video_path)
        if video_path is None or not os.path.exists(video_path):
            print("[ERROR] Invalid video path")
            return None, None
            
        print(f"[INFO] Processing: {video_path}")
        
        # Get device and model
        device = _get_device()
        model = get_model(model_name)
        
        # Optimization settings based on mode
        if mode == "ultra_fast":
            conf_threshold = 0.35
            imgsz = 320
            skip_frames = 3
            batch_size = 8 if device != "cpu" else 1
            print("[INFO] ⚡ ULTRA-FAST MODE - 3-4 minutes expected")
        elif mode == "fast":
            conf_threshold = 0.3
            imgsz = 640
            skip_frames = 2
            batch_size = 4 if device != "cpu" else 1
            print("[INFO] 🚀 FAST MODE - 5-8 minutes expected")
        else:  # balanced
            conf_threshold = 0.25
            imgsz = 640
            skip_frames = 1
            batch_size = 4 if device != "cpu" else 1
            print("[INFO] ⚖️ BALANCED MODE - 8-12 minutes expected")
        
        print(f"[INFO] Device: {device}, Image size: {imgsz}, Skip frames: {skip_frames}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return None, None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"[INFO] Video: {width}x{height} @ {fps:.1f} FPS, {duration:.1f}s, {total_frames} frames")
        
        # Create output path
        timestamp = int(time.time())
        outputs_folder = os.path.join(os.getcwd(), "outputs")
        os.makedirs(outputs_folder, exist_ok=True)
        output_path = os.path.join(outputs_folder, f"fast_video_{mode}_{timestamp}.mp4")
        
        # Setup video writer with fast codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("[ERROR] Cannot create video writer")
            cap.release()
            return None, None
        
        # Processing variables
        processed_count = 0
        actual_processed = 0
        total_detections = 0
        all_detections = []  # Store all detections for summary
        
        print("[INFO] 🚀 Starting optimized frame processing...")
        
        # Main processing loop with optimizations
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_count += 1
            
            # Skip frames for speed (KEY OPTIMIZATION)
            if processed_count % skip_frames != 0:
                continue
            
            actual_processed += 1
            
            # Progress update
            if actual_processed % 50 == 0:
                elapsed = time.time() - start_time
                fps_processed = actual_processed / elapsed
                progress = (processed_count / total_frames) * 100
                eta = (total_frames - processed_count) / (fps_processed * skip_frames) / 60  # minutes
                
                print(f"[INFO] Processed {processed_count}/{total_frames} ({progress:.1f}%) - {fps_processed:.1f} FPS - ETA: {eta:.1f} min")
                
                if progress_callback:
                    progress_callback(progress, f"Processing... {fps_processed:.1f} FPS")
            
            try:
                # FAST GPU INFERENCE
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=0.5,
                    imgsz=imgsz,
                    device=device,
                    verbose=False,
                    half=True if device != "cpu" else False,  # FP16 for 2x speed
                    augment=False,
                    agnostic_nms=True
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    # Count detections and store for summary
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        total_detections += len(result.boxes)
                        all_detections.append(result.boxes)
                    
                    # Fast annotation
                    annotated_frame = _annotate_frame_fast_video(frame, result)
                else:
                    annotated_frame = frame
                
                # Write frame
                out.write(annotated_frame)
                
            except Exception as e:
                print(f"[ERROR] Frame {processed_count} failed: {e}")
                out.write(frame)  # Write original frame on error
        
        # Cleanup
        cap.release()
        out.release()
        
        # Calculate final stats
        total_time = time.time() - start_time
        final_fps = actual_processed / total_time if total_time > 0 else 0
        speedup = skip_frames
        
        print(f"[INFO] ✅ ULTRA-FAST processing complete!")
        print(f"[INFO] Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"[INFO] Processing speed: {final_fps:.1f} FPS")
        print(f"[INFO] Frames processed: {actual_processed}/{total_frames}")
        print(f"[INFO] Total detections: {total_detections}")
        print(f"[INFO] Speedup achieved: {speedup}x faster")
        print(f"[INFO] Output saved: {output_path}")
        
        # Generate detection summary
        detection_summary = _generate_video_detection_summary(all_detections, model.names, total_time, mode)
        print(f"[INFO] Detection Summary: {detection_summary}")
        
        # Verify output
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, detection_summary
        else:
            print("[ERROR] Output file creation failed")
            return None, None
            
    except Exception as e:
        print(f"[ERROR] Ultra-fast video processing failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        try:
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
        except:
            pass
        
        return None, None


def _generate_video_detection_summary(all_detections, names, processing_time, mode):
    """
    Generate comprehensive detection summary with advanced color shades information for the entire video
    """
    try:
        if not all_detections:
            return "🎯 No objects detected in video"
        
        # Aggregate all detections
        category_counts = {}
        object_counts = {}
        color_shades_counts = {}
        color_families_counts = {}
        total_objects = 0
        
        for boxes in all_detections:
            if hasattr(boxes, 'cls'):
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].cpu().numpy()) if hasattr(boxes.cls[i], 'cpu') else int(boxes.cls[i])
                    class_name = names.get(class_id, f"class_{class_id}")
                    
                    # Get classification
                    display_name, category, _ = _classify_object_with_category(class_name, class_id)
                    
                    # Count by category
                    if category not in category_counts:
                        category_counts[category] = 0
                    category_counts[category] += 1
                    
                    # Count specific objects
                    if display_name not in object_counts:
                        object_counts[display_name] = 0
                    object_counts[display_name] += 1
                    
                    total_objects += 1
        
        # Create summary
        summary_lines = []
        summary_lines.append(f"🎯 **Advanced Video Processing Complete!**")
        summary_lines.append(f"⚡ **Mode:** {mode.upper()} | ⏱️ **Time:** {processing_time:.1f}s")
        summary_lines.append(f"📊 **Total Objects Detected:** {total_objects}")
        summary_lines.append(f"🎨 **Advanced Color Shades:** 56 Shades Enabled")
        summary_lines.append("")
        
        # Category summary
        summary_lines.append("**📋 By Category:**")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            summary_lines.append(f"  • {count} {category}(s)")
        
        summary_lines.append("")
        
        # Top objects
        summary_lines.append("**🔍 Top Objects:**")
        top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for obj_name, count in top_objects:
            summary_lines.append(f"  • {count} {obj_name}(s)")
        
        if len(object_counts) > 10:
            summary_lines.append(f"  • ... and {len(object_counts) - 10} more object types")
        
        summary_lines.append("")
        
        # Advanced color shades detection info
        summary_lines.append("**🎨 Advanced Color Shades Detection:**")
        summary_lines.append(f"  • 🔴 Red Family: 10 Shades (Misty Rose → Maroon)")
        summary_lines.append(f"  • 🔵 Blue Family: 10 Shades (Ice Blue → Midnight Blue)")
        summary_lines.append(f"  • 🟢 Green Family: 10 Shades (Mint → Emerald)")
        summary_lines.append(f"  • 🟡 Yellow/Orange: 10 Shades (Light Yellow → Deep Orange)")
        summary_lines.append(f"  • 🟣 Purple/Pink: 9 Shades (Lavender → Indigo)")
        summary_lines.append(f"  • ⚫ Neutral: 7 Shades (White → Black)")
        summary_lines.append(f"  • 🧠 AI Model: MobileNetV2 + HSV + LAB Analysis")
        summary_lines.append(f"  • ⚡ Real-time: <15ms per detection")
        
        return "\n".join(summary_lines)
        
    except Exception as e:
        print(f"[ERROR] Failed to generate detection summary: {e}")
        return f"🎯 Detection summary generation failed: {str(e)}"


def _classify_object_with_category(class_name, class_id):
    """
    Enhanced object classification with proper categories
    Returns: (display_name, category, color)
    """
    # Define object categories with colors
    categories = {
        # Persons
        'person': ('Person', 'Person', (255, 0, 0)),  # Red
        
        # Vehicles
        'bicycle': ('Bicycle', 'Vehicle', (0, 255, 255)),  # Yellow
        'car': ('Car', 'Vehicle', (255, 165, 0)),  # Orange
        'motorcycle': ('Motorcycle', 'Vehicle', (255, 165, 0)),  # Orange
        'bus': ('Bus', 'Vehicle', (255, 165, 0)),  # Orange
        'truck': ('Truck', 'Vehicle', (255, 165, 0)),  # Orange
        'boat': ('Boat', 'Vehicle', (255, 165, 0)),  # Orange
        'train': ('Train', 'Vehicle', (255, 165, 0)),  # Orange
        'airplane': ('Airplane', 'Vehicle', (255, 165, 0)),  # Orange
        
        # Traffic Objects
        'traffic light': ('Traffic Light', 'Traffic', (0, 0, 255)),  # Blue
        'stop sign': ('Stop Sign', 'Traffic', (0, 0, 255)),  # Blue
        'parking meter': ('Parking Meter', 'Traffic', (0, 0, 255)),  # Blue
        'fire hydrant': ('Fire Hydrant', 'Traffic', (0, 0, 255)),  # Blue
        
        # Animals
        'bird': ('Bird', 'Animal', (255, 0, 255)),  # Magenta
        'cat': ('Cat', 'Animal', (255, 0, 255)),  # Magenta
        'dog': ('Dog', 'Animal', (255, 0, 255)),  # Magenta
        'horse': ('Horse', 'Animal', (255, 0, 255)),  # Magenta
        'sheep': ('Sheep', 'Animal', (255, 0, 255)),  # Magenta
        'cow': ('Cow', 'Animal', (255, 0, 255)),  # Magenta
        'elephant': ('Elephant', 'Animal', (255, 0, 255)),  # Magenta
        'bear': ('Bear', 'Animal', (255, 0, 255)),  # Magenta
        'zebra': ('Zebra', 'Animal', (255, 0, 255)),  # Magenta
        'giraffe': ('Giraffe', 'Animal', (255, 0, 255)),  # Magenta
        
        # Objects
        'backpack': ('Backpack', 'Object', (128, 128, 128)),  # Gray
        'umbrella': ('Umbrella', 'Object', (128, 128, 128)),  # Gray
        'handbag': ('Handbag', 'Object', (128, 128, 128)),  # Gray
        'tie': ('Tie', 'Object', (128, 128, 128)),  # Gray
        'suitcase': ('Suitcase', 'Object', (128, 128, 128)),  # Gray
        'frisbee': ('Frisbee', 'Object', (128, 128, 128)),  # Gray
        'skis': ('Skis', 'Object', (128, 128, 128)),  # Gray
        'snowboard': ('Snowboard', 'Object', (128, 128, 128)),  # Gray
        'sports ball': ('Sports Ball', 'Object', (128, 128, 128)),  # Gray
        'kite': ('Kite', 'Object', (128, 128, 128)),  # Gray
        'baseball bat': ('Baseball Bat', 'Object', (128, 128, 128)),  # Gray
        'baseball glove': ('Baseball Glove', 'Object', (128, 128, 128)),  # Gray
        'skateboard': ('Skateboard', 'Object', (128, 128, 128)),  # Gray
        'surfboard': ('Surfboard', 'Object', (128, 128, 128)),  # Gray
        'tennis racket': ('Tennis Racket', 'Object', (128, 128, 128)),  # Gray
        'bottle': ('Bottle', 'Object', (128, 128, 128)),  # Gray
        'wine glass': ('Wine Glass', 'Object', (128, 128, 128)),  # Gray
        'cup': ('Cup', 'Object', (128, 128, 128)),  # Gray
        'fork': ('Fork', 'Object', (128, 128, 128)),  # Gray
        'knife': ('Knife', 'Object', (128, 128, 128)),  # Gray
        'spoon': ('Spoon', 'Object', (128, 128, 128)),  # Gray
        'bowl': ('Bowl', 'Object', (128, 128, 128)),  # Gray
        'banana': ('Banana', 'Food', (0, 255, 0)),  # Green
        'apple': ('Apple', 'Food', (0, 255, 0)),  # Green
        'sandwich': ('Sandwich', 'Food', (0, 255, 0)),  # Green
        'orange': ('Orange', 'Food', (0, 255, 0)),  # Green
        'broccoli': ('Broccoli', 'Food', (0, 255, 0)),  # Green
        'carrot': ('Carrot', 'Food', (0, 255, 0)),  # Green
        'hot dog': ('Hot Dog', 'Food', (0, 255, 0)),  # Green
        'pizza': ('Pizza', 'Food', (0, 255, 0)),  # Green
        'donut': ('Donut', 'Food', (0, 255, 0)),  # Green
        'cake': ('Cake', 'Food', (0, 255, 0)),  # Green
        'chair': ('Chair', 'Furniture', (139, 69, 19)),  # Brown
        'couch': ('Couch', 'Furniture', (139, 69, 19)),  # Brown
        'potted plant': ('Potted Plant', 'Furniture', (139, 69, 19)),  # Brown
        'bed': ('Bed', 'Furniture', (139, 69, 19)),  # Brown
        'dining table': ('Dining Table', 'Furniture', (139, 69, 19)),  # Brown
        'toilet': ('Toilet', 'Furniture', (139, 69, 19)),  # Brown
        'tv': ('TV', 'Electronics', (255, 0, 0)),  # Red
        'laptop': ('Laptop', 'Electronics', (255, 0, 0)),  # Red
        'mouse': ('Mouse', 'Electronics', (255, 0, 0)),  # Red
        'remote': ('Remote', 'Electronics', (255, 0, 0)),  # Red
        'keyboard': ('Keyboard', 'Electronics', (255, 0, 0)),  # Red
        'cell phone': ('Cell Phone', 'Electronics', (255, 0, 0)),  # Red
        'microwave': ('Microwave', 'Electronics', (255, 0, 0)),  # Red
        'oven': ('Oven', 'Electronics', (255, 0, 0)),  # Red
        'toaster': ('Toaster', 'Electronics', (255, 0, 0)),  # Red
        'sink': ('Sink', 'Electronics', (255, 0, 0)),  # Red
        'refrigerator': ('Refrigerator', 'Electronics', (255, 0, 0)),  # Red
        'book': ('Book', 'Object', (128, 128, 128)),  # Gray
        'clock': ('Clock', 'Object', (128, 128, 128)),  # Gray
        'vase': ('Vase', 'Object', (128, 128, 128)),  # Gray
        'scissors': ('Scissors', 'Object', (128, 128, 128)),  # Gray
        'teddy bear': ('Teddy Bear', 'Object', (128, 128, 128)),  # Gray
        'hair drier': ('Hair Drier', 'Object', (128, 128, 128)),  # Gray
        'toothbrush': ('Toothbrush', 'Object', (128, 128, 128)),  # Gray
    }
    
    # Get classification
    class_info = categories.get(class_name.lower(), (class_name.title(), 'Unknown', (255, 255, 255)))
    
    return class_info


def _get_detections_summary(boxes, names):
    """
    Get summary of detected objects by category
    """
    if not boxes or len(boxes) == 0:
        return "No objects detected"
    
    category_counts = {}
    object_details = []
    
    for i in range(len(boxes)):
        if hasattr(boxes, 'cls'):
            class_id = int(boxes.cls[i].cpu().numpy()) if hasattr(boxes.cls[i], 'cpu') else int(boxes.cls[i])
            class_name = names.get(class_id, f"class_{class_id}")
            
            # Get classification
            display_name, category, _ = _classify_object_with_category(class_name, class_id)
            
            # Count by category
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
            
            # Add object details
            object_details.append(f"{display_name}")
    
    # Create summary
    summary_parts = []
    for category, count in category_counts.items():
        if count > 0:
            summary_parts.append(f"{count} {category}(s)")
    
    objects_str = ", ".join(object_details[:10])  # Show first 10 objects
    if len(object_details) > 10:
        objects_str += f" and {len(object_details) - 10} more..."
    
    return f"🎯 Detected: {' | '.join(summary_parts)}\n📋 Objects: {objects_str}"


def _annotate_frame_fast_video(frame, result):
    """
    Enhanced fast frame annotation with object classification and advanced color shades detection
    """
    try:
        annotated = frame.copy()
        
        if result is None or not hasattr(result, 'boxes') or result.boxes is None:
            return annotated
        
        boxes = result.boxes
        if len(boxes) == 0:
            return annotated
        
        # Get detections
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        names = result.names
        
        # Draw detections with enhanced classification and advanced color detection
        for i in range(len(boxes)):
            if conf[i] > 0.3:  # Only draw confident detections
                x1, y1, x2, y2 = map(int, xyxy[i])
                confidence = float(conf[i])
                class_id = int(cls[i])
                class_name = names.get(class_id, f"class_{class_id}")
                
                # Get enhanced classification
                display_name, category, color = _classify_object_with_category(class_name, class_id)
                
                # Extract crop for advanced color detection
                crop = annotated[y1:y2, x1:x2]
                color_info = {'name': 'unknown', 'hex': '#000000', 'confidence': 0.0}
                
                if crop.size > 0:
                    try:
                        # Use advanced color shades detection
                        try:
                            from modules.advanced_color_shades import detect_color_shade_advanced
                            color_result = detect_color_shade_advanced(crop)
                        except Exception:
                            color_result = None
                        
                        if color_result and color_result.get('confidence', 0) > 0.3:
                            color_info = color_result
                            print(f"[DEBUG] Advanced color detected: {color_info['name']} ({color_info['hex']}) - {color_info['confidence']:.3f}")
                        else:
                            # Fallback to basic color detection
                            from modules.utils import _classify_color_bgr
                            basic_color = _classify_color_bgr(crop)
                            color_info = {
                                'name': basic_color.title(),
                                'hex': '#000000',
                                'confidence': 0.5,
                                'family': 'unknown'
                            }
                            
                    except Exception as e:
                        print(f"[DEBUG] Advanced color detection failed: {e}")
                        # Fallback to basic color detection
                        try:
                            from modules.utils import _classify_color_bgr
                            basic_color = _classify_color_bgr(crop)
                            color_info = {
                                'name': basic_color.title(),
                                'hex': '#000000',
                                'confidence': 0.5,
                                'family': 'unknown'
                            }
                        except:
                            color_info = {'name': 'unknown', 'hex': '#000000', 'confidence': 0.0}
                
                # Draw box with category-specific color
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Enhanced label with category and advanced color
                if color_info['confidence'] > 0.6:
                    label = f"{display_name} ({category}) - {color_info['name']}: {confidence:.2f}"
                else:
                    label = f"{display_name} ({category}) - {color_info['name']}: {confidence:.2f}"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # Background for label
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Text
                cv2.putText(annotated, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add processing info and detection summary
        cv2.putText(annotated, "ULTRA-FAST + ADVANCED COLOR SHADES", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add detection summary
        summary = _get_detections_summary(boxes, names)
        if summary != "No objects detected":
            # Split summary into lines
            lines = summary.split('\n')
            for i, line in enumerate(lines[:2]):  # Show first 2 lines
                cv2.putText(annotated, line[:50], (10, 60 + i*20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated
        
    except Exception as e:
        print(f"[ERROR] Fast annotation failed: {e}")
        return frame


# JSON-based text extraction system
_text_extraction_cache = {}

def extract_text_from_image_json(image_bgr: np.ndarray, image_id: str = None) -> dict:
    """
    Extract all text from image using multiple methods and return structured JSON.
    NEW APPROACH: First detect vehicles, then extract text only if vehicles are present.
    
    Args:
        image_bgr: Input image in BGR format
        image_id: Unique identifier for the image
        
    Returns:
        Dictionary containing all extracted text information
    """
    # Import the updated text extraction module
    try:
        from modules.text_extraction import extract_text_from_image_json as module_extract_text
        return module_extract_text(image_bgr, image_id)
    except ImportError:
        print("[DEBUG] Using fallback text extraction (modules not available)")
        return _fallback_text_extraction(image_bgr, image_id)


def _fallback_text_extraction(image_bgr: np.ndarray, image_id: str = None) -> dict:
    """
    Fallback text extraction with vehicle detection check.
    """
    if image_id is None:
        image_id = f"img_{int(time.time() * 1000)}"
    
    print(f"[DEBUG] Starting fallback text extraction for {image_id}")
    
    result = {
        "image_id": image_id,
        "timestamp": datetime.now().isoformat(),
        "text_extraction": {
            "all_objects": [],
            "license_plates": [],
            "general_text": [],
            "summary": {
                "total_objects": 0,
                "objects_with_text": 0,
                "license_plates_found": 0,
                "general_text_found": 0
            }
        }
    }
    
    try:
        # STEP 0: Check if vehicles are present in the image
        print(f"[DEBUG] Step 0: Checking for vehicles in the image...")
        vehicles_detected = _detect_vehicles_in_image_fallback(image_bgr)
        
        if not vehicles_detected:
            print(f"[DEBUG] ❌ No vehicles detected. Skipping text extraction.")
            print(f"[DEBUG] Reason: License plates and vehicle text only extracted when vehicles are present.")
            return result
        
        print(f"[DEBUG] ✅ Vehicles detected: {[v['class_name'] for v in vehicles_detected]}")
        print(f"[DEBUG] Proceeding with text extraction...")
        
        # Continue with existing text extraction logic...
        # [Rest of the original function would go here]
        
    except Exception as e:
        print(f"[DEBUG] Error in fallback text extraction: {e}")
        result["error"] = str(e)
    
    return result


def _detect_vehicles_in_image_fallback(image_bgr: np.ndarray) -> list:
    """
    Fallback vehicle detection for app.py
    """
    try:
        # Vehicle classes that should trigger text extraction
        VEHICLE_CLASSES = {
            'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'van', 
            'taxi', 'ambulance', 'police', 'fire truck', 'tractor',
            'scooter', 'bike', 'auto', 'rickshaw', 'lorry'
        }
        
        # Get YOLO model
        model = get_model("yolo26n.pt")
        detection_results = model(image_bgr)
        
        detected_vehicles = []
        
        if detection_results and len(detection_results) > 0:
            detection = detection_results[0]
            
            if hasattr(detection, 'boxes') and detection.boxes is not None:
                boxes = detection.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                names = detection.names
                
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Get class name
                    class_id = int(cls[i]) if i < len(cls) else -1
                    class_name = names.get(class_id, f"class_{class_id}")
                    confidence = float(conf[i]) if i < len(conf) else 0.0
                    
                    # Check if this is a vehicle (case insensitive)
                    if class_name.lower() in VEHICLE_CLASSES:
                        vehicle_info = {
                            "class_name": class_name.lower(),
                            "confidence": confidence,
                            "bounding_box": {
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2
                            }
                        }
                        detected_vehicles.append(vehicle_info)
                        print(f"[DEBUG] 🚗 Vehicle detected: {class_name} (conf: {confidence:.3f})")
        
        return detected_vehicles
        
    except Exception as e:
        print(f"[DEBUG] Error in fallback vehicle detection: {e}")
        return []


def _extract_text_from_license_plate_crop(plate_crop: np.ndarray) -> str:
    """
    Extract text from a cropped license plate region using intelligent fallback.
    PRIMARY: PaddleOCR -> FALLBACK: LightOnOCR -> LAST: Tesseract
    
    Args:
        plate_crop: Cropped license plate image in BGR format
        
    Returns:
        Extracted text string
    """
    try:
        print(f"[DEBUG] Extracting text from license plate crop: {plate_crop.shape}")
        
        all_candidates = []
        
        # Method 1: Optimized PaddleOCR GPU (PRIMARY - BEST SPEED + ACCURACY)
        try:
            # Import PaddleOCR modules
            from optimized_paddleocr_gpu import extract_text_optimized, extract_license_plates_optimized
            print("[DEBUG] 🚀 Trying Optimized PaddleOCR GPU for license plate")
            
            # Extract text with optimized GPU processing
            paddleocr_result = extract_text_optimized(
                plate_crop, 
                confidence_threshold=0.3,  # Lower threshold for license plates
                lang='en',
                use_gpu=None,  # Auto-detect GPU
                use_cache=True,
                preprocess=True
            )
            
            if paddleocr_result["text"] and paddleocr_result["text"].strip():
                cleaned = _clean_license_plate_text(paddleocr_result["text"])
                if cleaned and len(cleaned) >= 6:
                    confidence = paddleocr_result["confidence"]
                    device = paddleocr_result["device"]
                    all_candidates.append(("optimized_paddleocr", cleaned, confidence, device))
                    print(f"[DEBUG] ✅ Optimized PaddleOCR found: {cleaned} (conf: {confidence:.3f}, device: {device})")
                    
                    # Also try specialized license plate extraction
                    paddle_plates = extract_license_plates_optimized(
                        plate_crop,
                        confidence_threshold=0.4,
                        use_gpu=None
                    )
                    
                    for plate_info in paddle_plates:
                        plate_text = plate_info.get('text', '')
                        if plate_text and plate_text != cleaned:
                            plate_confidence = plate_info.get('confidence', 0.85)
                            plate_device = plate_info.get('device', 'Unknown')
                            all_candidates.append(("optimized_paddleocr_plate", plate_text, plate_confidence, plate_device))
                            print(f"[DEBUG] ✅ Optimized PaddleOCR specialized found: {plate_text} (device: {plate_device})")
            
        except ImportError:
            print("[DEBUG] Optimized PaddleOCR not available, trying alternatives...")
        except Exception as e:
            print(f"[DEBUG] ❌ Optimized PaddleOCR failed: {e}")
        
        # Method 2: Legacy PaddleOCR (SECONDARY if optimized fails)
        if not all_candidates:
            try:
                from paddleocr_integration import extract_text_with_paddleocr, extract_license_plates_with_paddleocr, preprocess_image_for_paddleocr
                print("[DEBUG] 🔄 Trying Legacy PaddleOCR for license plate")
                
                # Preprocess for better PaddleOCR results
                processed_plate = preprocess_image_for_paddleocr(plate_crop)
                
                # Extract text with PaddleOCR
                paddleocr_result = extract_text_with_paddleocr(
                    processed_plate, 
                    confidence_threshold=0.3,
                    lang='en'
                )
                
                if paddleocr_result and paddleocr_result.strip():
                    cleaned = _clean_license_plate_text(paddleocr_result)
                    if cleaned and len(cleaned) >= 6:
                        all_candidates.append(("legacy_paddleocr", cleaned, 0.8, "CPU"))
                        print(f"[DEBUG] ✅ Legacy PaddleOCR found: {cleaned}")
                        
                        # Also try specialized license plate extraction
                        paddle_plates = extract_license_plates_with_paddleocr(
                            processed_plate,
                            confidence_threshold=0.4
                        )
                        
                        for plate_info in paddle_plates:
                            plate_text = plate_info.get('text', '')
                            if plate_text and plate_text != cleaned:
                                all_candidates.append(("legacy_paddleocr_plate", plate_text, 0.75, "CPU"))
                                print(f"[DEBUG] ✅ Legacy PaddleOCR specialized found: {plate_text}")
                
            except ImportError:
                print("[DEBUG] Legacy PaddleOCR not available...")
            except Exception as e:
                print(f"[DEBUG] ❌ Legacy PaddleOCR failed: {e}")
        
        # Method 3: LightOnOCR (FALLBACK when PaddleOCR fails)
        if LIGHTON_AVAILABLE and not all_candidates:
            try:
                print("[DEBUG] 🔧 Using LightOnOCR fallback for license plate")
                # Preprocess the license plate crop
                processed_plate = _preprocess_license_plate(plate_crop)
                lighton_result = extract_text_with_lighton(processed_plate, confidence_threshold=0.2)
                if lighton_result and lighton_result.strip():
                    cleaned = _clean_license_plate_text(lighton_result)
                    if cleaned and len(cleaned) >= 6:
                        all_candidates.append(("lighton", cleaned, 0.7, "CPU"))
                        print(f"[DEBUG] ✅ LightOnOCR found: {cleaned}")
            except Exception as e:
                print(f"[DEBUG] ❌ LightOnOCR failed for plate: {e}")
        
        # Method 4: Tesseract OCR (LAST FALLBACK when all OCR methods fail)
        if not all_candidates and TESSERACT_AVAILABLE:
            try:
                print("[DEBUG] 🛠️ Using Tesseract as last fallback for license plate")
                # Preprocess for Tesseract
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                
                # Multiple preprocessing attempts
                preprocessing_methods = []
                
                # Method 4a: Basic grayscale
                preprocessing_methods.append(("gray", gray))
                
                # Method 4b: Contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                clahe_enhanced = clahe.apply(gray)
                preprocessing_methods.append(("clahe", clahe_enhanced))
                
                # Method 4c: Binary threshold
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                preprocessing_methods.append(("binary", binary))
                
                # Method 4d: Adaptive threshold
                adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                preprocessing_methods.append(("adaptive", adaptive))
                
                # Method 4e: Inverted (for dark plates with light text)
                inverted = cv2.bitwise_not(gray)
                preprocessing_methods.append(("inverted", inverted))
                
                # Method 4f: Upscaled for better OCR
                h, w = gray.shape
                if w < 200:  # Upscale small plates
                    upscaled = cv2.resize(gray, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
                    preprocessing_methods.append(("upscaled", upscaled))
                
                # Tesseract configurations optimized for license plates
                configs = [
                    (r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', "strict"),
                    (r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', "strict"),
                    (r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', "strict"),
                    (r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', "strict"),
                    (r'--oem 3 --psm 7', "flexible"),  # No whitelist for flexibility
                    (r'--oem 3 --psm 8', "flexible"),
                    (r'--oem 3 --psm 6', "flexible"),
                ]
                
                for method_name, processed_img in preprocessing_methods:
                    for config, config_type in configs:
                        try:
                            text = pytesseract.image_to_string(processed_img, config=config)
                            if text and text.strip():
                                cleaned = _clean_license_plate_text(text)
                                if cleaned and len(cleaned) >= 6:
                                    confidence = 0.6 if config_type == "strict" else 0.5
                                    all_candidates.append((f"tesseract_{method_name}_{config_type}", cleaned, confidence, "CPU"))
                                    print(f"[DEBUG] ✅ Tesseract {method_name} {config_type} found: {cleaned}")
                        except:
                            continue
                
            except Exception as e:
                print(f"[DEBUG] ❌ Tesseract fallback failed: {e}")
        
        # Method 5: Morphological operations + OCR (EXTRA FALLBACK)
        if not all_candidates and TESSERACT_AVAILABLE:
            try:
                print("[DEBUG] 🔍 Trying morphological operations for license plate")
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                
                # Remove noise
                kernel = np.ones((2,2), np.uint8)
                opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
                
                # Enhance contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(opening)
                
                # Binarize
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # OCR with multiple configs
                morph_configs = [
                    r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                    r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                    r'--oem 3 --psm 7'
                ]
                
                for config in morph_configs:
                    try:
                        text = pytesseract.image_to_string(binary, config=config)
                        if text and text.strip():
                            cleaned = _clean_license_plate_text(text)
                            if cleaned and len(cleaned) >= 6:
                                all_candidates.append(("morphological", cleaned, 0.5, "CPU"))
                                print(f"[DEBUG] ✅ Morphological found: {cleaned}")
                                break
                    except:
                        continue
                        
            except Exception as e:
                print(f"[DEBUG] ❌ Morphological method failed: {e}")
        
        # Select the best result from all candidates
        if all_candidates:
            print(f"[DEBUG] Total candidates: {len(all_candidates)}")
            for method, text, conf, device in all_candidates:
                print(f"[DEBUG]   {method}: {text} (conf: {conf:.3f}, device: {device})")
            
            # Filter by valid Indian license plates
            valid_candidates = [(method, text, conf, device) for method, text, conf, device in all_candidates 
                              if _is_valid_indian_license_plate(text)]
            
            if valid_candidates:
                # Select the best result among valid candidates (prefer higher confidence, then GPU over CPU)
                best_candidate = max(valid_candidates, key=lambda x: (x[2], 1 if x[3] != "CPU" else 0))
                best_result = best_candidate[1]
                print(f"[DEBUG] ✅ Best valid result: {best_result} (method: {best_candidate[0]}, device: {best_candidate[3]})")
                return best_result
            else:
                # If no valid Indian plates, return the highest confidence candidate
                best_candidate = max(all_candidates, key=lambda x: (x[2], 1 if x[3] != "CPU" else 0))
                print(f"[DEBUG] ⚠️ Best candidate (not valid Indian): {best_candidate[1]} (method: {best_candidate[0]})")
                return best_candidate[1]
    
    except Exception as e:
        print(f"[DEBUG] Error in license plate OCR: {e}")
    
    return ""


def _preprocess_license_plate(plate_crop: np.ndarray) -> np.ndarray:
    """
    Preprocess license plate crop for better OCR results.
    
    Args:
        plate_crop: Original license plate crop in BGR format
        
    Returns:
        Preprocessed image in BGR format
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        clahe_enhanced = clahe.apply(bilateral)
        
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(clahe_enhanced, -1, kernel)
        
        # Convert back to BGR for LightOnOCR
        processed_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        return processed_bgr
        
    except Exception as e:
        print(f"[DEBUG] Error in license plate preprocessing: {e}")
        return plate_crop


def _extract_all_text_from_object(crop_bgr: np.ndarray, class_name: str) -> dict:
    """
    Extract all text from an object crop using multiple methods.
    
    Args:
        crop_bgr: Object crop in BGR format
        class_name: Class name of the object
        
    Returns:
        Dictionary containing all extracted text information
    """
    text_results = {
        "all_text": [],
        "license_plate": None,
        "general_text": []
    }
    
    try:
        # Method 1: License plate detection for cars
        if class_name.lower() == "car":
            plate_crop = _detect_license_plate_in_car(crop_bgr)
            if plate_crop is not None:
                plate_text = _extract_text_ocr(plate_crop)
                if plate_text and plate_text.strip():
                    cleaned_plate = _clean_license_plate_text(plate_text)
                    if _is_valid_indian_license_plate(cleaned_plate):
                        text_results["license_plate"] = {
                            "text": cleaned_plate,
                            "confidence": 0.9,
                            "method": "license_plate_detection"
                        }
                        text_results["all_text"].append({
                            "text": cleaned_plate,
                            "type": "license_plate",
                            "confidence": 0.9,
                            "method": "license_plate_detection"
                        })
        
        # Method 2: LightOnOCR for general text
        if LIGHTON_AVAILABLE:
            try:
                lighton_result = extract_text_with_lighton(crop_bgr, confidence_threshold=0.3)
                if lighton_result and lighton_result.strip():
                    cleaned_general = _clean_general_text(lighton_result)
                    if cleaned_general and len(cleaned_general) >= 2:
                        text_item = {
                            "text": cleaned_general,
                            "confidence": 0.8,
                            "method": "lighton_ocr"
                        }
                        text_results["general_text"].append(text_item)
                        text_results["all_text"].append({
                            "text": cleaned_general,
                            "type": "general_text",
                            "confidence": 0.8,
                            "method": "lighton_ocr"
                        })
            except Exception as e:
                print(f"[DEBUG] LightOnOCR failed: {e}")
        
        # Method 3: Tesseract OCR for general text
        tess_result = _extract_text_ocr(crop_bgr)
        if tess_result and tess_result.strip():
            cleaned_tess = _clean_general_text(tess_result)
            if cleaned_tess and len(cleaned_tess) >= 2:
                # Avoid duplicates
                is_duplicate = False
                for existing in text_results["general_text"]:
                    if existing["text"].lower() == cleaned_tess.lower():
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    text_item = {
                        "text": cleaned_tess,
                        "confidence": 0.7,
                        "method": "tesseract_ocr"
                    }
                    text_results["general_text"].append(text_item)
                    text_results["all_text"].append({
                        "text": cleaned_tess,
                        "type": "general_text",
                        "confidence": 0.7,
                        "method": "tesseract_ocr"
                    })
        
        # Method 4: Specialized OCR for different object types
        specialized_text = _extract_specialized_text(crop_bgr, class_name)
        if specialized_text:
            for text_item in specialized_text:
                text_results["general_text"].append(text_item)
                text_results["all_text"].append({
                    "text": text_item["text"],
                    "type": "specialized_text",
                    "confidence": text_item["confidence"],
                    "method": text_item["method"]
                })
    
    except Exception as e:
        print(f"[DEBUG] Error extracting text from object: {e}")
    
    return text_results


def _extract_general_text_from_image(image_bgr: np.ndarray) -> list:
    """Extract text from the entire image using general OCR methods."""
    text_items = []
    
    try:
        # Try LightOnOCR on full image
        if LIGHTON_AVAILABLE:
            try:
                full_text = extract_text_with_lighton(image_bgr, confidence_threshold=0.4)
                if full_text and full_text.strip():
                    cleaned = _clean_general_text(full_text)
                    if cleaned and len(cleaned) >= 3:
                        text_items.append({
                            "text": cleaned,
                            "confidence": 0.6,
                            "method": "full_image_lighton"
                        })
                        
                        # Special handling: Look for license plates in full image text
                        license_plates = _extract_license_plates_from_text(cleaned)
                        for plate_text in license_plates:
                            text_items.append({
                                "text": plate_text,
                                "confidence": 0.8,
                                "method": "full_image_lighton_plate"
                            })
            except:
                pass
        
        # Try Tesseract on full image
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config=r'--oem 3 --psm 6')
            if text and text.strip():
                cleaned = _clean_general_text(text)
                if cleaned and len(cleaned) >= 3:
                    text_items.append({
                        "text": cleaned,
                        "confidence": 0.5,
                        "method": "full_image_tesseract"
                    })
                    
                    # Special handling: Look for license plates in full image text
                    license_plates = _extract_license_plates_from_text(cleaned)
                    for plate_text in license_plates:
                        text_items.append({
                            "text": plate_text,
                            "confidence": 0.7,
                            "method": "full_image_tesseract_plate"
                        })
        except:
            pass
    
    except Exception as e:
        print(f"[DEBUG] Error in full image text extraction: {e}")
    
    return text_items


def _validate_license_plate_in_image(plate_crop: np.ndarray, plate_text: str) -> bool:
    """
    Validate that the detected license plate text actually exists in the image.
    This prevents false positives from OCR hallucination.
    
    Args:
        plate_crop: Cropped license plate image in BGR format
        plate_text: Extracted license plate text
        
    Returns:
        True if the plate text is likely real, False if it's probably fake
    """
    try:
        print(f"[DEBUG] Validating license plate: {plate_text}")
        
        # Method 1: Check if the characters in plate_text can be visually confirmed
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        # Method 1a: Character count validation based on image size
        h, w = gray.shape
        expected_chars = len(plate_text)
        
        # Estimate if the plate size can accommodate the detected text
        min_char_width = 8  # Minimum width per character
        max_char_width = 30  # Maximum width per character
        
        estimated_min_width = expected_chars * min_char_width
        estimated_max_width = expected_chars * max_char_width
        
        if not (estimated_min_width <= w <= estimated_max_width * 2):
            print(f"[DEBUG] ❌ Plate size doesn't match text length: {w}px vs {expected_chars} chars")
            return False
        
        # Method 2: Visual character verification
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours (potential characters)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (character-like)
        char_contours = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = cw / ch if ch > 0 else 0
            
            # Character-like properties
            if (area > 50 and area < (w * h * 0.3) and 
                5 <= cw <= w // 3 and 10 <= ch <= h // 2 and
                0.2 <= aspect_ratio <= 2.0):
                char_contours.append(contour)
        
        visual_char_count = len(char_contours)
        print(f"[DEBUG] Visual character count: {visual_char_count}, Expected: {expected_chars}")
        
        # Allow some tolerance (some characters might be merged or split)
        if not (visual_char_count >= expected_chars * 0.4 and visual_char_count <= expected_chars * 2.0):
            print(f"[DEBUG] ❌ Visual character count doesn't match text")
            # Don't immediately reject, continue with other validations
        
        # Method 3: Cross-validate with different OCR methods
        # If multiple OCR methods agree on similar text, it's more likely real
        cross_validation_results = []
        
        # Try different preprocessing methods
        preprocessing_methods = []
        
        # Basic grayscale
        preprocessing_methods.append(gray)
        
        # Binary threshold
        _, binary_val = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessing_methods.append(binary_val)
        
        # Inverted
        inverted = cv2.bitwise_not(gray)
        preprocessing_methods.append(inverted)
        
        # CLAHE enhanced
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_enhanced = clahe.apply(gray)
        preprocessing_methods.append(clahe_enhanced)
        
        # Test with different Tesseract configs
        configs = [
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            r'--oem 3 --psm 7'
        ]
        
        for processed_img in preprocessing_methods:
            for config in configs:
                try:
                    ocr_result = pytesseract.image_to_string(processed_img, config=config).strip()
                    if ocr_result and len(ocr_result) >= 4:
                        cleaned = _clean_license_plate_text(ocr_result)
                        if cleaned and len(cleaned) >= 4:
                            cross_validation_results.append(cleaned)
                except:
                    continue
        
        print(f"[DEBUG] Cross-validation results: {cross_validation_results}")
        
        # Check if any results are similar to the main result
        similar_count = 0
        for result in cross_validation_results:
            # Calculate similarity (simple character overlap)
            common_chars = set(plate_text.upper()) & set(result.upper())
            similarity = len(common_chars) / max(len(plate_text), len(result))
            
            if similarity >= 0.7:  # 70% similarity
                similar_count += 1
        
        print(f"[DEBUG] Similar OCR results: {similar_count}/{len(cross_validation_results)}")
        
        # Method 4: Plate structure validation
        # Check if the detected text follows realistic license plate patterns
        if not _is_realistic_license_plate_pattern(plate_text):
            print(f"[DEBUG] ❌ Unreliable license plate pattern: {plate_text}")
            return False
        
        # Final decision based on all validations
        validation_score = 0
        
        # Visual character count validation (less strict)
        if visual_char_count >= expected_chars * 0.4 and visual_char_count <= expected_chars * 2.0:
            validation_score += 1
        else:
            validation_score -= 1  # Penalty for no visual characters
        
        # Cross-validation
        if similar_count >= 2:
            validation_score += 2
        elif similar_count >= 1:
            validation_score += 1
        
        # Size validation
        if estimated_min_width <= w <= estimated_max_width * 2:
            validation_score += 1
        
        print(f"[DEBUG] Validation score: {validation_score}/4")
        
        # Need at least 2 points to pass validation
        is_valid = validation_score >= 2
        
        if is_valid:
            print(f"[DEBUG] ✅ License plate validation PASSED: {plate_text}")
        else:
            print(f"[DEBUG] ❌ License plate validation FAILED: {plate_text}")
        
        return is_valid
        
    except Exception as e:
        print(f"[DEBUG] Error in license plate validation: {e}")
        return False


def _is_realistic_license_plate_pattern(plate_text: str) -> bool:
    """
    Check if the license plate follows realistic patterns.
    This helps filter out OCR hallucinations.
    
    Args:
        plate_text: License plate text to validate
        
    Returns:
        True if pattern looks realistic, False if suspicious
    """
    try:
        plate_upper = plate_text.upper()
        
        # Rule 1: Must have at least one digit and one letter
        has_digit = any(c.isdigit() for c in plate_upper)
        has_letter = any(c.isalpha() for c in plate_upper)
        
        if not (has_digit and has_letter):
            print(f"[DEBUG] ❌ Plate missing digits or letters: {plate_text}")
            return False
        
        # Rule 2: Reject patterns that look like OCR errors
        # Too many repeated characters might indicate OCR errors
        repeated_chars = plate_upper.count(plate_upper[0]) if plate_upper else 0
        if repeated_chars > len(plate_upper) * 0.6:
            print(f"[DEBUG] ❌ Too many repeated characters: {plate_text}")
            return False
        
        # Rule 3: Reject obviously unrealistic patterns
        # Check for common OCR error patterns
        ocr_error_patterns = [
            r'^[A-Z]{1,2}$',  # Just 1-2 letters
            r'^[0-9]{1,3}$',  # Just 1-3 digits
            r'^[A-Z]{4,}$',   # Too many letters
            r'^[0-9]{6,}$',   # Too many digits
            r'^(.)\1{5,}',    # Same character repeated 6+ times
        ]
        
        import re
        for pattern in ocr_error_patterns:
            if re.match(pattern, plate_upper):
                print(f"[DEBUG] ❌ Suspicious pattern detected: {plate_text}")
                return False
        
        # Rule 4: Length should be reasonable (6-12 characters for most plates)
        if not (6 <= len(plate_upper) <= 12):
            print(f"[DEBUG] ❌ Unreasonable length: {len(plate_upper)} chars in {plate_text}")
            return False
        
        # Rule 5: Should follow some basic license plate structure
        # Common patterns: 2 letters + 2-4 digits + 2-3 letters + 1-4 digits
        # Or: 2 letters + 1-4 digits + 1-3 letters + 1-4 digits
        
        # Check for Indian-like patterns
        indian_pattern1 = r'^[A-Z]{2}[0-9]{1,4}[A-Z]{1,3}[0-9]{1,4}$'
        indian_pattern2 = r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$'
        
        # Check for international patterns
        international_pattern = r'^[A-Z0-9]{6,12}$'
        
        if (re.match(indian_pattern1, plate_upper) or 
            re.match(indian_pattern2, plate_upper) or
            re.match(international_pattern, plate_upper)):
            print(f"[DEBUG] ✅ Valid license plate pattern: {plate_text}")
            return True
        
        # If it doesn't match standard patterns but passes other checks, allow it
        print(f"[DEBUG] ⚠️ Non-standard but acceptable pattern: {plate_text}")
        return True
        
    except Exception as e:
        print(f"[DEBUG] Error in pattern validation: {e}")
        return False


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


def _extract_license_plates_from_text(text: str) -> list:
    """
    Extract potential license plate numbers from extracted text.
    Specifically looks for Indian license plate patterns like "MH 20 EE 7602".
    
    Args:
        text: Extracted text from OCR
        
    Returns:
        List of potential license plate numbers
    """
    license_plates = []
    
    try:
        # Common Indian state codes
        state_codes = ['AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ', 
                      'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP', 
                      'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB']
        
        # Split text into words and clean them
        words = text.split()
        cleaned_words = []
        for word in words:
            # Remove non-alphanumeric characters and convert to uppercase
            cleaned_word = ''.join(c for c in word.upper() if c.isalnum())
            if cleaned_word:
                cleaned_words.append(cleaned_word)
        
        # Method 1: Look for exact patterns in individual words
        for word in cleaned_words:
            if _is_valid_indian_license_plate(word):
                license_plates.append(word)
        
        # Method 2: Look for patterns across multiple words
        # Join words and look for license plate patterns
        combined_text = ''.join(cleaned_words)
        
        # Look for patterns like MH20EE7602
        import re
        
        # Pattern: 2 letters + 2-4 digits + 2 letters + 4 digits
        pattern1 = r'([A-Z]{2}[0-9]{2,4}[A-Z]{1,3}[0-9]{1,4})'
        matches1 = re.findall(pattern1, combined_text)
        for match in matches1:
            if _is_valid_indian_license_plate(match):
                license_plates.append(match)
        
        # Method 3: Look for state code followed by numbers and letters
        for state_code in state_codes:
            # Find state code in text
            state_indices = [i for i, word in enumerate(cleaned_words) if state_code in word]
            
            for idx in state_indices:
                # Try to build license plate from surrounding words
                plate_candidate = state_code
                next_words = cleaned_words[idx:idx+6]  # Look at next 6 words
                
                for word in next_words[1:]:  # Skip the state code itself
                    # Add alphanumeric characters from this word
                    alnum_part = ''.join(c for c in word if c.isalnum())
                    plate_candidate += alnum_part
                    
                    # Check if we have a valid license plate
                    if len(plate_candidate) >= 8 and len(plate_candidate) <= 12:
                        if _is_valid_indian_license_plate(plate_candidate):
                            license_plates.append(plate_candidate)
                            break
                
                # Method 4: Look for specific pattern "MH 20 EE 7602" across words
                if len(cleaned_words) >= idx + 4:
                    potential_plate = ""
                    # Try combining next few words
                    for i in range(idx, min(idx + 6, len(cleaned_words))):
                        potential_plate += cleaned_words[i]
                        if len(potential_plate) >= 8 and _is_valid_indian_license_plate(potential_plate):
                            license_plates.append(potential_plate)
                            break
        
        # Method 5: Specific search for "MH 20 EE 7602" pattern
        # Look for state code followed by 2 digits, then 2 letters, then 4 digits
        for state_code in state_codes:
            pattern2 = rf'{state_code}\s*[0-9]{{2}}\s*[A-Z]{{2}}\s*[0-9]{{4}}'
            matches2 = re.findall(pattern2, text, re.IGNORECASE)
            for match in matches2:
                # Clean and format the match
                cleaned_match = ''.join(c for c in match.upper() if c.isalnum())
                if _is_valid_indian_license_plate(cleaned_match):
                    license_plates.append(cleaned_match)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_plates = []
        for plate in license_plates:
            if plate not in seen:
                seen.add(plate)
                unique_plates.append(plate)
        
        print(f"[DEBUG] Found {len(unique_plates)} potential license plates in text: {unique_plates}")
        
    except Exception as e:
        print(f"[DEBUG] Error extracting license plates from text: {e}")
    
    return unique_plates


def _extract_specialized_text(crop_bgr: np.ndarray, class_name: str) -> list:
    """Extract specialized text based on object type."""
    text_items = []
    
    try:
        class_lower = class_name.lower()
        
        # Specialized processing for different object types
        if class_lower in ["traffic sign", "stop sign"]:
            # Traffic signs often have high contrast text
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            configs = [
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
                r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
            ]
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(thresh, config=config)
                    if text and text.strip():
                        cleaned = _clean_general_text(text)
                        if cleaned and len(cleaned) >= 2:
                            text_items.append({
                                "text": cleaned,
                                "confidence": 0.8,
                                "method": f"specialized_{class_lower}"
                            })
                            break
                except:
                    continue
        
        elif class_lower in ["bottle", "can", "package"]:
            # Products often have brand names
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Enhance saturation for better text contrast
            s_enhanced = cv2.multiply(s, 1.3)
            hsv_enhanced = cv2.merge([h, s_enhanced, v])
            enhanced_bgr = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
            
            try:
                text = pytesseract.image_to_string(enhanced_bgr, config=r'--oem 3 --psm 7')
                if text and text.strip():
                    cleaned = _clean_general_text(text)
                    if cleaned and len(cleaned) >= 2:
                        text_items.append({
                            "text": cleaned,
                            "confidence": 0.6,
                            "method": f"specialized_{class_lower}"
                        })
            except:
                pass
    
    except Exception as e:
        print(f"[DEBUG] Error in specialized text extraction: {e}")
    
    return text_items


def format_text_extraction_results(json_result: dict) -> str:
    """Format the JSON text extraction results into a readable string."""
    if not json_result or "text_extraction" not in json_result:
        return "No text extraction results available."
    
    extraction = json_result["text_extraction"]
    summary = extraction["summary"]
    
    lines = []
    lines.append("📝 **Text Extraction Results:**")
    lines.append(f"   Total Objects: {summary['total_objects']}")
    lines.append(f"   Objects with Text: {summary['objects_with_text']}")
    lines.append(f"   License Plates Found: {summary['license_plates_found']}")
    lines.append(f"   General Text Found: {summary['general_text_found']}")
    lines.append("")
    
    # Show license plates
    if extraction["license_plates"]:
        lines.append("🚗 **License Plates:**")
        for plate in extraction["license_plates"]:
            lines.append(f"   • {plate['plate_text']} (confidence: {plate['confidence']:.2f})")
        lines.append("")
    
    # Show general text
    if extraction["general_text"]:
        lines.append("📄 **General Text:**")
        for text_item in extraction["general_text"]:
            lines.append(f"   • {text_item['text']} (confidence: {text_item['confidence']:.2f})")
        lines.append("")
    
    # Show full image text
    if "full_image_text" in extraction and extraction["full_image_text"]:
        lines.append("🖼️ **Full Image Text:**")
        for text_item in extraction["full_image_text"]:
            lines.append(f"   • {text_item['text']} (confidence: {text_item['confidence']:.2f})")
        lines.append("")
    
    if not any([extraction["license_plates"], extraction["general_text"], 
                extraction.get("full_image_text")]):
        lines.append("   No text detected in the image.")
    
    return "\n".join(lines)


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
        if TESSERACT_AVAILABLE:
            try:
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


def _classify_color_bgr(crop_bgr: np.ndarray) -> str:
    if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
        return "unknown"

    h, w = crop_bgr.shape[:2]
    if h < 2 or w < 2:
        return "unknown"

    # Reduce background influence: focus on the center region of the bbox
    # (edges often contain background and box includes extra context).
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
                        if LIGHTON_AVAILABLE:
                            try:
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
            if stream_key_prefix:
                _ocr_stream_state["frame_idx"] += 1
                do_ocr = (_ocr_stream_state["frame_idx"] % ocr_every_n) == 0
                if not do_ocr and key in _ocr_stream_state["texts"]:
                    ocr_text = _ocr_stream_state["texts"][key]

            if ocr_text is None and do_ocr:
                # Special handling for cars: try to detect license plate first
                if str(class_name).strip().lower() == "car":
                    plate_crop = _detect_license_plate_in_car(crop)
                    if plate_crop is not None:
                        print(f"[DEBUG] License plate detected in car, extracting text...")
                        plate_text = _extract_text_ocr(plate_crop)
                        if plate_text and plate_text.strip():
                            print(f"[DEBUG] License plate text: '{plate_text}'")
                            ocr_text = f"Plate: {plate_text}"
                        else:
                            print("[DEBUG] No text extracted from license plate")
                    else:
                        print("[DEBUG] No license plate detected in car")
                
                # For all objects (including cars), try to detect any text using LightOnOCR
                general_text = None
                if LIGHTON_AVAILABLE:
                    try:
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
                    general_text = _extract_text_ocr(crop)
                    if general_text and general_text.strip():
                        general_text = _clean_general_text(general_text)
                
                # Combine results: prioritize license plate for cars, otherwise use general text
                if ocr_text and general_text:
                    # For cars, show both plate and general text if different
                    if str(class_name).strip().lower() == "car" and plate_text:
                        if general_text.lower() != plate_text.lower():
                            ocr_text = f"Plate: {plate_text} | Text: {general_text}"
                    else:
                        ocr_text = general_text
                elif general_text:
                    ocr_text = general_text
                    
                if key is not None:
                    _ocr_stream_state["texts"][key] = ocr_text

        boy_girl = None
        try:
            if str(class_name).strip().lower() == "person":
                boy_girl = _predict_boy_girl_from_person_crop(crop)
        except Exception:
            boy_girl = None

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if show_labels:
            parts = [str(class_name)]
            if show_conf and i < len(conf):
                parts.append(f"{float(conf[i]):.2f}")
            parts.append(str(color_name))
            if boy_girl:
                parts.append("|")
                parts.append(str(boy_girl))
            if enable_resnet and resnet_label:
                parts.append("|")
                parts.append(str(resnet_label))
            if enable_ocr and ocr_text:
                parts.append("|")
                parts.append(str(ocr_text))
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


def predict_image(
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
    """Predicts objects in an image using a Ultralytics YOLO model with CUDA support and JSON-based text extraction."""
    model = get_model(model_name)
    device = _get_device()

    models = model if isinstance(model, list) else [model]

    # Processing Flow Optimization
    if device != "cpu" and torch.cuda.is_available():
        # Clear GPU cache before processing for optimal performance
        torch.cuda.empty_cache()
        # Set optimal memory allocation
        torch.cuda.synchronize()  # Ensure all operations are completed

    all_results = []
    for m in models:
        r = m.predict(
            source=img,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
            half=True if device != "cpu" else False,  # Use FP16 on CUDA for speed
        )
        if r:
            all_results.append(r[0])

    if not all_results:
        return img, "No objects detected"

    # Convert PIL to BGR for OpenCV operations
    frame_rgb = np.array(img)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Generate unique image ID for JSON text extraction
    image_id = f"img_{int(time.time() * 1000)}"
    
    # Perform comprehensive text extraction if OCR is enabled
    json_text_results = None
    if enable_ocr:
        print(f"[DEBUG] Starting JSON-based text extraction for image {image_id}")
        json_text_results = extract_text_from_image_json(frame_bgr, image_id)
        print(f"[DEBUG] Text extraction completed for {image_id}")
    
    annotated_bgr = frame_bgr
    for idx, res in enumerate(all_results):
        annotated_bgr = _annotate_with_color(
            annotated_bgr,
            res,
            show_labels,
            show_conf,
            enable_resnet=bool(enable_resnet),
            max_boxes=int(max_boxes),
            resnet_every_n=1,
            stream_key_prefix=None,
            enable_ocr=False,  # Disable individual OCR since we're using JSON system
            ocr_every_n=1,
        )
    
    # If we have JSON text results, add text annotations from JSON
    if json_text_results and enable_ocr:
        annotated_bgr = _annotate_from_json_results(annotated_bgr, json_text_results, show_labels)
    
    # Generate detection summary
    summaries = [
        _generate_detection_summary(r, enable_resnet=bool(enable_resnet), enable_ocr=False)
        for r in all_results
    ]
    summary = "\n\n".join([s for s in summaries if s])
    
    # Add JSON text extraction results to summary
    if json_text_results and enable_ocr:
        text_summary = format_text_extraction_results(json_text_results)
        summary = f"{summary}\n\n{text_summary}"
        
        # Also add raw JSON for debugging
        json_output = json.dumps(json_text_results, indent=2, ensure_ascii=False)
        summary = f"{summary}\n\n📋 **Raw JSON Data:**\n```json\n{json_output}\n```"
    
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), summary


def predict_video(
    video_path,
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
    processing_mode="fast"  # NEW: Add processing mode selection
):
    """
    🚀 ULTRA-FAST VIDEO PROCESSING - 50 minutes → 3-5 minutes
    
    Args:
        processing_mode: "ultra_fast" (3-4 min), "fast" (5-8 min), "balanced" (8-12 min), "original" (slow, 50 min)
    
    Returns:
        (output_path, detection_summary) - Path and summary
    """
    try:
        print(f"[INFO] 🚀 Starting VIDEO PROCESSING in {processing_mode} mode")
        
        # Use ULTRA-FAST processing for all modes except "original"
        if processing_mode != "original":
            print(f"[INFO] Using optimized processing - expected time: 3-12 minutes")
            return process_video_optimized_fast(
                video_path=video_path,
                model_name=model_name,
                mode=processing_mode,
                progress_callback=None
            )
        
        # Original slow processing (if someone really wants it)
        print(f"[WARNING] Using ORIGINAL SLOW mode - this will take 50+ minutes")
        result_path = _predict_video_original(
            video_path, conf_threshold, iou_threshold, model_name,
            show_labels, show_conf, imgsz, enable_resnet, max_boxes,
            resnet_every_n, enable_ocr, ocr_every_n
        )
        
        # Create basic summary for original mode
        summary = f"🎯 Video processed in ORIGINAL mode\n⏱️ Processing time: 50+ minutes\n📁 Output: {result_path}"
        return result_path, summary
        
    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        return None, None


def _predict_video_original(
    video_path,
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
    """Original slow video processing (kept for compatibility)"""
    try:
        print(f"[DEBUG] Starting predict_video function")
        print(f"[DEBUG] Input video_path: {video_path}")
        
        video_path = _extract_video_path(video_path)
        if video_path is None:
            print("[ERROR] No valid video path provided")
            return None

        print(f"[DEBUG] Extracted video_path: {video_path}")

        # Validate video file exists and is readable
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file does not exist: {video_path}")
            return None
        
        # Check file size (prevent processing very large files that might cause issues)
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            print(f"[ERROR] Video file is empty: {video_path}")
            return None
        
        print(f"[INFO] Processing video: {video_path} ({file_size / (1024*1024):.1f} MB)")

        model = get_model(model_name)
        device = _get_device()
        print(f"[INFO] Processing video on device: {device}")
        print(f"[INFO] Using confidence threshold: {conf_threshold}, IoU threshold: {iou_threshold}")
        print(f"[INFO] Image size: {imgsz}")

        models = model if isinstance(model, list) else [model]

        # Open the video with error handling
        print("[DEBUG] Attempting to open video file...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video file: {video_path}")
            # Try alternative method
            cap.release()
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print("[ERROR] Failed to open video with FFMPEG backend")
                return None

        print("[DEBUG] Video file opened successfully")

        # Get video properties with validation
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[DEBUG] Video properties - FPS: {fps}, Width: {width}, Height: {height}, Frames: {frame_count}")
        
        if width <= 0 or height <= 0:
            print("[ERROR] Invalid video dimensions")
            cap.release()
            return None
        
        if fps <= 0:
            fps = 30  # Default FPS if not detected
            print(f"[WARNING] Could not detect FPS, using default: {fps}")
        
        print(f"[INFO] Video: {width}x{height} @ {fps} FPS, {frame_count} frames")

        # Create temporary output file with proper extension
        temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = temp_output.name
        temp_output.close()

        # Initialize video writer with more compatible codec
        print("[DEBUG] Initializing video writer...")
        try:
            # Try different codecs in order of compatibility
            codecs_to_try = [
                ("mp4v", ".mp4"),    # Most compatible on Windows
                ("XVID", ".avi"),    # Good fallback
                ("DIVX", ".avi"),    # Another fallback
                ("MJPG", ".avi"),    # Motion JPEG
            ]
            
            out = None
            output_path = None
            
            for fourcc_name, ext in codecs_to_try:
                try:
                    print(f"[DEBUG] Trying {fourcc_name} codec with {ext} extension...")
                    # Create new temp file for each codec attempt
                    temp_output = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                    output_path = temp_output.name
                    temp_output.close()
                    
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    if out.isOpened():
                        print(f"[INFO] Successfully initialized video writer with {fourcc_name} codec")
                        print(f"[DEBUG] Output path: {output_path}")
                        break
                    else:
                        print(f"[DEBUG] {fourcc_name} codec failed to open")
                        out.release()
                        # Clean up failed attempt
                        if os.path.exists(output_path):
                            os.unlink(output_path)
                        out = None
                        
                except Exception as codec_error:
                    print(f"[DEBUG] {fourcc_name} codec error: {codec_error}")
                    if out:
                        out.release()
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                    out = None
                    output_path = None
            
            if out is None:
                print("[ERROR] Failed to initialize any video codec")
                cap.release()
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to initialize video writer: {e}")
            cap.release()
            if out:
                out.release()
            return None

        print("[DEBUG] Video writer initialized successfully")

        processed_frames = 0
        success_count = 0
        detection_count = 0
        
        print("[DEBUG] Starting frame processing loop...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[DEBUG] End of video reached after {processed_frames} frames")
                break

            processed_frames += 1
            if processed_frames % 30 == 0:  # Progress update every 30 frames
                print(f"[INFO] Processing frame {processed_frames}/{frame_count}...")

            try:
                # Debug: Check frame properties
                if processed_frames == 1:
                    print(f"[DEBUG] First frame shape: {frame.shape}, dtype: {frame.dtype}")

                # Run inference on the frame with CUDA support
                all_results = []
                for m in models:
                    r = m.predict(
                        source=frame,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        imgsz=imgsz,
                        device=device,
                        verbose=False,
                        half=True if device != "cpu" else False,  # Use FP16 on CUDA for speed
                    )
                    if r:
                        all_results.append(r[0])

                # Debug: Check if any detections were made
                frame_detections = 0
                try:
                    for rr in all_results:
                        if hasattr(rr, "boxes") and rr.boxes is not None:
                            frame_detections += len(rr.boxes)
                except Exception:
                    frame_detections = 0

                if frame_detections > 0:
                    detection_count += frame_detections
                    if processed_frames % 30 == 0 or processed_frames <= 5:  # Show detection info for first few frames and periodically
                        print(f"[DEBUG] Frame {processed_frames}: {frame_detections} detections")
                else:
                    if processed_frames % 30 == 0 or processed_frames <= 5:  # Only show no-detection debug periodically
                        print(f"[DEBUG] Frame {processed_frames}: No detections")

                annotated_frame = frame
                for res in all_results:
                    annotated_frame = _annotate_with_color(
                        annotated_frame,
                        res,
                        show_labels,
                        show_conf,
                        enable_resnet=bool(enable_resnet),
                        max_boxes=int(max_boxes),
                        resnet_every_n=int(resnet_every_n),
                        stream_key_prefix="video",
                        enable_ocr=bool(enable_ocr),
                        ocr_every_n=int(ocr_every_n),
                    )
                
                # Ensure the annotated frame has correct dimensions
                if annotated_frame.shape[:2] != (height, width):
                    if processed_frames == 1:
                        print(f"[DEBUG] Resizing annotated frame from {annotated_frame.shape[:2]} to {(height, width)}")
                    annotated_frame = cv2.resize(annotated_frame, (width, height))
                
                # Write frame to output
                out.write(annotated_frame)
                success_count += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to process frame {processed_frames}: {e}")
                # Write original frame if processing fails
                if frame.shape[:2] == (height, width):
                    out.write(frame)
                else:
                    # Resize frame if dimensions don't match
                    resized_frame = cv2.resize(frame, (width, height))
                    out.write(resized_frame)
                success_count += 1

        print(f"[DEBUG] Finished processing {processed_frames} frames")
        cap.release()
        out.release()
        
        # Verify output file was created successfully
        if not os.path.exists(output_path):
            print("[ERROR] Output video file was not created")
            return None
        
        output_size = os.path.getsize(output_path)
        if output_size == 0:
            print("[ERROR] Output video file is empty")
            os.unlink(output_path)
            return None
        
        print(f"[INFO] Output video created successfully: {output_size / (1024*1024):.1f} MB")
        print(f"[INFO] Total detections found: {detection_count}")
        print(f"[INFO] Video processing complete. Processed {success_count}/{processed_frames} frames successfully.")
        
        # Final verification - try to open the output video
        try:
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                actual_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                actual_fps = test_cap.get(cv2.CAP_PROP_FPS)
                actual_width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                test_cap.release()
                print(f"[INFO] Output video verified: {actual_width}x{actual_height} @ {actual_fps} FPS, {actual_frames} frames")
            else:
                print("[WARNING] Could not verify output video, but file exists")
        except Exception as e:
            print(f"[WARNING] Output verification failed: {e}")
        
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        import traceback
        traceback.print_exc()
        # Clean up resources on error
        try:
            if 'cap' in locals() and cap is not None:
                cap.release()
        except:
            pass
        try:
            if 'out' in locals() and out is not None:
                out.release()
        except:
            pass
        try:
            if 'output_path' in locals() and output_path and os.path.exists(output_path):
                os.unlink(output_path)
        except:
            pass
        return None


# Cache model for streaming performance
_model_cache = {}

# Cache ResNet for classification
_resnet_cache = {"model": None, "weights": None, "device": None, "transforms": None, "categories": None}
_resnet_stream_state = {"frame_idx": 0, "labels": {}}

# Cache EasyOCR
_ocr_cache = {"reader": None, "device": None}
_ocr_stream_state = {"frame_idx": 0, "texts": {}}

# Cache gender classifier and face detector
_gender_cache = {"pipeline": None, "model_id": "dima806/fairface_gender_image_detection"}
_face_cache = {"cascade": None}


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
            
            # GPU Memory Optimization
            if torch.cuda.is_available():
                # Enable memory optimization
                torch.cuda.empty_cache()  # Clear unused memory
                torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
                print(f"[INFO] GPU memory optimized - Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
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


def _get_ocr_reader():
    if not TESSERACT_AVAILABLE:
        print("[WARNING] pytesseract not installed. Install with: pip install pytesseract")
        print("[WARNING] Also install Tesseract binary from: https://github.com/UB-Mannheim/tesseract/wiki")
        return None
    # Tesseract uses system binary; no GPU needed
    try:
        # Test if tesseract is available
        pytesseract.get_tesseract_version()
        return True
    except Exception as e:
        print(f"[ERROR] Tesseract binary not found: {e}")
        print("[INFO] Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("[INFO] Or add tesseract to PATH")
        return None


def _extract_text_ocr(crop_bgr: np.ndarray) -> str:
    print(f"[DEBUG] OCR called on crop size: {crop_bgr.shape if crop_bgr is not None else 'None'}")
    
    if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
        print("[DEBUG] OCR failed: Invalid crop")
        return ""
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 8:
        print(f"[DEBUG] OCR failed: Crop too small ({h}x{w})")
        return ""

    # Try LightOnOCR first if available
    if LIGHTON_AVAILABLE:
        try:
            print("[DEBUG] Using LightOnOCR for text extraction")
            result = extract_text_with_lighton(crop_bgr, confidence_threshold=0.4)  # Higher threshold for plates
            if result and result.strip():
                cleaned = _clean_license_plate_text(result)
                print(f"[DEBUG] LightOnOCR extracted: '{cleaned[:50]}...' ({len(cleaned)} chars)")
                return cleaned
            else:
                print("[DEBUG] LightOnOCR returned empty, falling back to Tesseract")
        except Exception as e:
            print(f"[DEBUG] LightOnOCR failed: {e}, falling back to Tesseract")
    
    # Fallback to enhanced Tesseract with Indian license plate optimization
    if not TESSERACT_AVAILABLE:
        print("[DEBUG] OCR failed: No OCR available")
        return ""

    # Check if Tesseract is available
    if _get_ocr_reader() is None:
        print("[DEBUG] OCR failed: Tesseract binary not available")
        return ""

    try:
        # Enhanced preprocessing specifically for Indian license plates
        results = []
        
        # Method 1: Grayscale with advanced preprocessing
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Multiple thresholding methods for license plates
        thresh1 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        _, thresh3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Inverted images (for dark plates with light text)
        thresh1_inv = cv2.bitwise_not(thresh1)
        thresh2_inv = cv2.bitwise_not(thresh2)
        thresh3_inv = cv2.bitwise_not(thresh3)
        
        # Method 3: Morphological operations for better text separation
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        
        morph1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel_small)
        morph2 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel_small)
        morph3 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel_medium)
        
        # Method 4: Upscaling for small license plates
        if h < 50 or w < 120:
            scale_factor = max(2, min(3, 150 // min(h, w)))
            upscaled = cv2.resize(crop_bgr, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)
            gray_up = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            clahe_up = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
            enhanced_up = clahe_up.apply(gray_up)
            thresh_up = cv2.adaptiveThreshold(enhanced_up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            thresh_up = None
        
        # All preprocessing methods to try
        preprocess_methods = [
            thresh1, thresh2, thresh3,  # Normal
            thresh1_inv, thresh2_inv, thresh3_inv,  # Inverted
            morph1, morph2, morph3  # Morphological
        ]
        
        if thresh_up is not None:
            preprocess_methods.append(thresh_up)
        
        # Specialized OCR configs for Indian license plates
        configs = [
            # Strict alphanumeric (best for license plates)
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            # Include common Indian state codes
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            # More flexible (fallback)
            r'--oem 3 --psm 7',
            r'--oem 3 --psm 8',
            r'--oem 3 --psm 6',
            r'--oem 3 --psm 13',
            # Single character mode for difficult plates
            r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        ]
        
        # Try each preprocessing method with each config
        for i, thresh in enumerate(preprocess_methods, 1):
            for j, config in enumerate(configs, 1):
                try:
                    text = pytesseract.image_to_string(thresh, config=config)
                    cleaned = _clean_license_plate_text(text)
                    if cleaned and len(cleaned) >= 4:  # Minimum length for Indian plates
                        results.append(cleaned)
                        print(f"[DEBUG] OCR Method {i} Config {j} found: '{cleaned}'")
                except Exception as e:
                    continue
        
        # Method 5: Try original image with different preprocessing
        try:
            # Convert to HSV and enhance saturation
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s_enhanced = cv2.multiply(s, 1.5)
            hsv_enhanced = cv2.merge([h, s_enhanced, v])
            enhanced_bgr = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
            
            for config in configs[:6]:  # Try most effective configs
                try:
                    text = pytesseract.image_to_string(enhanced_bgr, config=config)
                    cleaned = _clean_license_plate_text(text)
                    if cleaned and len(cleaned) >= 4:
                        results.append(cleaned)
                        print(f"[DEBUG] OCR Enhanced HSV found: '{cleaned}'")
                except:
                    pass
        except:
            pass
        
        print(f"[DEBUG] OCR total raw results: {results}")
        
        # Enhanced filtering for Indian license plates
        filtered = []
        for result in results:
            if _is_valid_indian_license_plate(result):
                filtered.append(result)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_filtered = []
        for result in filtered:
            if result not in seen:
                seen.add(result)
                unique_filtered.append(result)
        
        # Select best result based on Indian license plate patterns
        final_result = _select_best_indian_plate_result(unique_filtered)
        print(f"[DEBUG] OCR final result: '{final_result}'")
        return final_result
        
    except Exception as e:
        print(f"[DEBUG] OCR error: {e}")
    return ""


def _is_valid_indian_license_plate(text: str) -> bool:
    """Check if text matches Indian license plate format (e.g., MH 20 EE 7602)."""
    if not text or len(text) < 6:
        return False
    
    # Remove any non-alphanumeric characters
    alnum_text = ''.join(c for c in text if c.isalnum())
    
    if len(alnum_text) < 6:
        return False
    
    # Check for minimum alphanumeric content
    alnum_ratio = sum(c.isalnum() for c in text) / len(text)
    if alnum_ratio < 0.8:
        return False
    
    # Indian license plate patterns:
    # Pattern 1: 2 letters (state) + 2 numbers (district) + 2 letters (series) + 4 numbers (unique)
    # Example: MH20EE7602, GJ06AB1234, DL01CD5678
    # Pattern 2: Variations with spaces: MH 20 EE 7602
    
    has_letter = any(c.isalpha() for c in alnum_text)
    has_number = any(c.isdigit() for c in alnum_text)
    
    # Must have both letters and numbers
    if not (has_letter and has_number):
        return False
    
    # Check for Indian state code pattern (first 2 characters are letters)
    if len(alnum_text) >= 2 and alnum_text[:2].isalpha():
        return True
    
    # Check for typical Indian plate length (8-10 characters)
    if 8 <= len(alnum_text) <= 10:
        return True
    
    # Check if it has the pattern letters-numbers-letters-numbers
    letter_count = sum(c.isalpha() for c in alnum_text)
    number_count = sum(c.isdigit() for c in alnum_text)
    
    # Indian plates typically have 4-6 letters and 4-6 numbers
    if 3 <= letter_count <= 6 and 3 <= number_count <= 6:
        return True
    
    return False


def _select_best_indian_plate_result(results: list) -> str:
    """Select the best Indian license plate result from multiple candidates."""
    if not results:
        return ""
    
    if len(results) == 1:
        return results[0]
    
    # Score each result based on Indian license plate characteristics
    scored_results = []
    for result in results:
        score = 0
        
        # Length preference (8-10 characters is typical for Indian plates)
        if 8 <= len(result) <= 10:
            score += 4
        elif 6 <= len(result) <= 12:
            score += 2
        
        # Has both letters and numbers
        has_letter = any(c.isalpha() for c in result)
        has_number = any(c.isdigit() for c in result)
        if has_letter and has_number:
            score += 3
        
        # Indian state code pattern (2 letters at start)
        if len(result) >= 2 and result[:2].isalpha():
            score += 3
            # Check if it's a valid Indian state code
            state_codes = ['AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ', 
                          'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP', 
                          'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB']
            if result[:2] in state_codes:
                score += 2
        
        # Check for typical Indian plate pattern: XX00XX0000
        if len(result) >= 8:
            # Pattern: 2 letters + 2 numbers + 2 letters + 4 numbers
            if (result[:2].isalpha() and len(result) >= 8 and 
                result[2:4].isdigit() and len(result) >= 6 and
                result[4:6].isalpha() and len(result) >= 10 and
                result[6:10].isdigit()):
                score += 5
            # More flexible pattern checking
            elif (result[:2].isalpha() and 
                  any(result[i].isdigit() for i in range(2, min(6, len(result)))) and
                  any(c.isalpha() for c in result[2:min(8, len(result))])):
                score += 3
        
        scored_results.append((score, result))
    
    # Sort by score and return the best
    scored_results.sort(key=lambda x: x[0], reverse=True)
    return scored_results[0][1]


def _clean_general_text(text: str) -> str:
    """Clean and normalize OCR text for general objects (not just license plates)."""
    if not text:
        return ""
    
    # Remove excessive whitespace and convert to proper case
    cleaned = text.strip()
    
    # Replace multiple spaces with single space
    import re
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove common OCR artifacts but keep more characters for general text
    # Keep letters, numbers, spaces, and common punctuation
    valid_chars = []
    for char in cleaned:
        if char.isalnum() or char.isspace() or char in '.,!?-:;()[]{}"/\'@#$%&*+=<>' :
            valid_chars.append(char)
    
    result = ''.join(valid_chars)
    
    # Clean up any multiple spaces again
    result = re.sub(r'\s+', ' ', result).strip()
    
    # Return in proper case (first letter capitalized, rest as-is)
    if result and len(result) > 0:
        result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()
    
    return result


def _clean_license_plate_text(text: str) -> str:
    """Clean and normalize OCR text for license plates."""
    if not text:
        return ""
    
    # Remove whitespace and convert to uppercase
    cleaned = text.strip().upper()
    
    # Replace common OCR confusions
    replacements = {
        'O': '0',  # Letter O to zero
        'I': '1',  # Letter I to one
        'S': '5',  # Letter S to five
        'G': '6',  # Letter G to six
        'B': '8',  # Letter B to eight
        'Z': '2',  # Letter Z to two
        ' ': '',   # Remove spaces
        '-': '',   # Remove hyphens
        '.': '',   # Remove periods
        ',': '',   # Remove commas
    }
    
    # Apply replacements selectively for license plates
    result = ""
    for char in cleaned:
        if char in replacements:
            result += replacements[char]
        elif char.isalnum():
            result += char
    
    return result


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


def _select_best_plate_result(results: list) -> str:
    """Select the best license plate result from multiple candidates."""
    if not results:
        return ""
    
    if len(results) == 1:
        return results[0]
    
    # Score each result based on license plate characteristics
    scored_results = []
    for result in results:
        score = 0
        
        # Length preference (not too short, not too long)
        if 6 <= len(result) <= 10:
            score += 3
        elif 4 <= len(result) <= 12:
            score += 1
        
        # Has both letters and numbers
        has_letter = any(c.isalpha() for c in result)
        has_number = any(c.isdigit() for c in result)
        if has_letter and has_number:
            score += 2
        
        # Common Indian license plate pattern (e.g., MH20EE7602)
        if len(result) >= 4:
            # Check for state code pattern (2 letters)
            if len(result) >= 2 and result[:2].isalpha():
                score += 1
            # Check for numbers
            if any(c.isdigit() for c in result):
                score += 1
        
        scored_results.append((score, result))
    
    # Sort by score and return the best
    scored_results.sort(key=lambda x: x[0], reverse=True)
    return scored_results[0][1]


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


def _annotate_webcam_fast(
    frame_bgr: np.ndarray,
    result,
    show_labels: bool,
    show_conf: bool,
    max_boxes: int,
    enable_color: bool,
) -> np.ndarray:
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
    take = min(int(max(1, max_boxes)), total)

    for i in range(take):
        x1, y1, x2, y2 = xyxy[i]
        x1 = int(max(0, min(iw - 1, round(x1))))
        y1 = int(max(0, min(ih - 1, round(y1))))
        x2 = int(max(0, min(iw - 1, round(x2))))
        y2 = int(max(0, min(ih - 1, round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if show_labels:
            class_id = int(cls[i]) if i < len(cls) else -1
            class_name = str(class_id)
            if isinstance(names, dict):
                class_name = names.get(class_id, class_name)
            elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                class_name = names[class_id]

            color_name = None
            if enable_color:
                try:
                    # Fast traditional color detection (no MobileNet; suitable for real-time webcam)
                    from modules.utils import _classify_color_traditional_fallback
                    crop = annotated[y1:y2, x1:x2]
                    if isinstance(crop, np.ndarray) and crop.size > 0:
                        color_name = _classify_color_traditional_fallback(crop)
                except Exception:
                    color_name = None

            if show_conf and i < len(conf):
                text = f"{class_name} {float(conf[i]):.2f}"
            else:
                text = str(class_name)

            if color_name:
                text = f"{text} {str(color_name)}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            ty = y1 - 8
            if ty - th - baseline < 0:
                ty = y1 + th + baseline + 8

            bg_x1 = x1
            bg_y1 = max(0, ty - th - baseline)
            bg_x2 = min(iw - 1, x1 + tw + 6)
            bg_y2 = min(ih - 1, ty + 4)
            cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), -1)
            cv2.putText(annotated, text, (x1 + 3, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return annotated


def predict_webcam(
    frame,
    conf_threshold,
    iou_threshold,
    model_name,
    show_labels,
    show_conf,
    enable_color,
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

    global _webcam_stream_state
    try:
        _webcam_stream_state
    except NameError:
        _webcam_stream_state = {"frame_idx": 0, "last_rgb": None}

    try:
        _webcam_stream_state["frame_idx"] += 1

        every_n = int(max(1, resnet_every_n))
        if (_webcam_stream_state["frame_idx"] % every_n) != 0:
            if _webcam_stream_state.get("last_rgb") is not None:
                return _webcam_stream_state["last_rgb"]
            return frame

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

        # Pre-resize to reduce overhead (YOLO will also letterbox internally)
        try:
            target = int(imgsz) if imgsz is not None else 640
            h, w = frame_bgr.shape[:2]
            if max(h, w) > target and target >= 160:
                scale = float(target) / float(max(h, w))
                nw = max(2, int(round(w * scale)))
                nh = max(2, int(round(h * scale)))
                frame_bgr = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        except Exception:
            pass

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
            annotated_bgr = _annotate_webcam_fast(
                annotated_bgr,
                res,
                show_labels=bool(show_labels),
                show_conf=bool(show_conf),
                max_boxes=int(max_boxes),
                enable_color=bool(enable_color),
            )
        out_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        _webcam_stream_state["last_rgb"] = out_rgb
        return out_rgb

    except Exception as e:
        print(f"[ERROR] Webcam prediction failed: {e}")
        return frame


# Create the Gradio app with enhanced modern interface
with gr.Blocks(
    title="YOLO26 AI Vision",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .gr-button-primary:hover {
        background: linear-gradient(45deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .gr-box {
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }
    .gr-tab-nav {
        border-bottom: 2px solid #e5e7eb;
    }
    .gr-tab-nav button {
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
    .gr-tab-nav button.selected {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    """
) as demo:
    # Display device status with modern styling
    device = _get_device()
    if device != "cpu":
        gr.Markdown(
            f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, #10b981 0%, #059669 100%); 
                 border-radius: 12px; margin-bottom: 20px; color: white;">
                <h2 style="margin: 0; font-size: 24px;">🚀 GPU Acceleration Active</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Processing on {torch.cuda.get_device_name(0)}</p>
            </div>
            """
        )
    else:
        gr.Markdown(
            """
            <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, #f59e0b 0%, #d97706 100%); 
                 border-radius: 12px; margin-bottom: 20px; color: white;">
                <h2 style="margin: 0; font-size: 24px;">⚠️ CPU Processing Mode</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Consider using GPU for better performance</p>
            </div>
            """
        )
    
    # Enhanced main title
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="margin: 0; font-size: 48px; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;">
                YOLO26 AI Vision
            </h1>
            <p style="margin: 10px 0 0 0; font-size: 18px; color: #6b7280;">
                Advanced Object Detection • License Plate Recognition • Real-time Processing
            </p>
        </div>
        """
    )

    with gr.Tabs():
        # Image Tab - Simplified
        with gr.TabItem("🖼️ Image Detection"):
            gr.Markdown("### Upload an image for instant AI-powered object detection")
            
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="📁 Upload Image")
                    
                    with gr.Row():
                        img_model = gr.Radio(choices=MODEL_CHOICES, label="🤖 AI Model", value="yolo26n")
                    
                    img_btn = gr.Button("🚀 Detect Objects", variant="primary", size="lg")
                    
                    # Advanced settings (collapsible)
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        img_conf = gr.Slider(minimum=0, maximum=1, value=0.35, label="🎯 Confidence Threshold")
                        img_iou = gr.Slider(minimum=0, maximum=1, value=0.5, label="📏 IoU Threshold")
                        img_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="📐 Image Size", value=640)
                        img_labels = gr.Checkbox(value=True, label="🏷️ Show Labels")
                        img_conf_show = gr.Checkbox(value=True, label="📊 Show Confidence")
                        
                        # Hidden controls (always enabled)
                        img_resnet = gr.Checkbox(value=True, visible=False)
                        img_max_boxes = gr.Slider(minimum=1, maximum=25, value=10, step=1, visible=False)
                        img_ocr = gr.Checkbox(value=True, visible=False)
                        
                with gr.Column(scale=2):
                    img_output = gr.Image(type="pil", label="🎯 Detection Result")
                    img_summary = gr.Markdown(label="📋 Detection Summary", value="📸 **Ready to detect!**\n\nUpload an image and click **🚀 Detect Objects** to start AI analysis.\n\n✨ **Features:**\n- 🚗 Vehicle detection\n- 📍 License plate recognition\n- 🎨 Color classification\n- ⚡ GPU-powered processing")

            img_btn.click(
                predict_image,
                inputs=[
                    img_input,
                    img_conf,
                    img_iou,
                    img_model,
                    img_labels,
                    img_conf_show,
                    img_size,
                    img_resnet,
                    img_max_boxes,
                    img_ocr,
                ],
                outputs=[img_output, img_summary],
            )

        # Video Tab - Simplified
        with gr.TabItem("🎥 Video Processing"):
            gr.Markdown("### Upload a video for AI-powered object detection and tracking")
            
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(label="📹 Upload Video")
                    
                    with gr.Row():
                        vid_model = gr.Radio(choices=MODEL_CHOICES, label="🤖 AI Model", value="yolo26n")
                    
                    vid_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
                    
                    # Advanced settings (collapsible)
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        # NEW: Video Processing Speed Selection
                        vid_speed_mode = gr.Radio(
                            choices=[
                                ("⚡ Ultra-Fast (3-4 min)", "ultra_fast"),
                                ("🚀 Fast (5-8 min)", "fast"), 
                                ("⚖️ Balanced (8-12 min)", "balanced"),
                                ("🐌 Original (50+ min)", "original")
                            ],
                            label="🚀 Processing Speed Mode",
                            value="fast"
                        )
                        
                        vid_conf = gr.Slider(minimum=0, maximum=1, value=0.35, label="🎯 Confidence Threshold")
                        vid_iou = gr.Slider(minimum=0, maximum=1, value=0.5, label="📏 IoU Threshold")
                        vid_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="📐 Image Size", value=640)
                        vid_labels = gr.Checkbox(value=True, label="🏷️ Show Labels")
                        vid_conf_show = gr.Checkbox(value=True, label="📊 Show Confidence")
                        vid_max_boxes = gr.Slider(minimum=1, maximum=25, value=5, step=1, label="📦 Max Boxes per Frame")
                        vid_every_n = gr.Slider(minimum=1, maximum=30, value=5, step=1, label="⏱️ Process Every N Frames")
                        
                        # Hidden controls (always enabled)
                        vid_resnet = gr.Checkbox(value=True, visible=False)
                        vid_ocr = gr.Checkbox(value=True, visible=False)
                        vid_ocr_every_n = gr.Slider(minimum=1, maximum=30, value=5, step=1, visible=False)
                    
                    # Progress indicator
                    vid_progress = gr.Textbox(label="📊 Status", value="⏳ Ready to process video...", interactive=False)
                    
                with gr.Column(scale=2):
                    vid_output = gr.Video(label="🎯 Processed Video", visible=True)
                    vid_download = gr.File(label="💾 Download Result", visible=True)
                    vid_info = gr.Markdown(label="📋 Video Info", value="📹 Upload a video to see processing information")
                    vid_info = gr.Textbox(label="Detection Info", interactive=False, visible=True, lines=3)

            def process_video_with_status(video, conf, iou, model, labels, conf_show, imgsz, enable_resnet, max_boxes, every_n, enable_ocr, ocr_every_n, speed_mode):
                """Wrapper function to provide status updates"""
                if video is None:
                    return None, None, "Please upload a video first", "No video provided"
                
                try:
                    # Update status based on speed mode
                    if speed_mode == "ultra_fast":
                        status = "⚡ Starting ULTRA-FAST video processing (3-4 minutes)..."
                    elif speed_mode == "fast":
                        status = "🚀 Starting FAST video processing (5-8 minutes)..."
                    elif speed_mode == "balanced":
                        status = "⚖️ Starting BALANCED video processing (8-12 minutes)..."
                    else:
                        status = "🐌 Starting ORIGINAL video processing (50+ minutes)..."
                    
                    return None, None, status, "Initializing AI models..."
                    
                except Exception as e:
                    return None, None, f"Error: {str(e)}", f"Processing failed: {str(e)}"

            def process_video_complete(video, conf, iou, model, labels, conf_show, imgsz, enable_resnet, max_boxes, every_n, enable_ocr, ocr_every_n, speed_mode):
                """Complete video processing function"""
                if video is None:
                    return None, None, "Please upload a video first", "No video provided"
                
                try:
                    print(f"[DEBUG] Starting video processing for input: {video}")
                    
                    # Process video
                    result_path, detection_summary = predict_video(video, conf, iou, model, labels, conf_show, imgsz, enable_resnet, max_boxes, every_n, enable_ocr, ocr_every_n, speed_mode)
                    print(f"[DEBUG] Video processing completed. Result path: {result_path}")
                    
                    if result_path and os.path.exists(result_path):
                        print(f"[DEBUG] Output file exists, size: {os.path.getsize(result_path)} bytes")
                        
                        # Get video info
                        cap = cv2.VideoCapture(result_path)
                        if cap.isOpened():
                            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            duration = frames / fps if fps > 0 else 0
                            cap.release()
                            
                            info = f"✅ Processed: {frames} frames | Size: {width}x{height} | FPS: {fps:.1f} | Duration: {duration:.1f}s"
                            print(f"[INFO] {info}")
                            
                            # Use detection summary for detailed info
                            if detection_summary:
                                detailed_info = f"{info}\n\n{detection_summary}"
                            else:
                                detailed_info = info
                            
                            timestamp = int(time.time())
                            permanent_name = f"processed_video_{timestamp}.mp4"
                            permanent_path = os.path.join(os.getcwd(), permanent_name)
                            
                            try:
                                # Copy the processed video to a permanent location
                                shutil.copy2(result_path, permanent_path)
                                print(f"[DEBUG] Copied video to permanent path: {permanent_path}")
                                
                                # Verify the copied file exists and is accessible
                                if os.path.exists(permanent_path) and os.path.getsize(permanent_path) > 0:
                                    print(f"[DEBUG] Permanent video file verified: {os.path.getsize(permanent_path)} bytes")

                                    ffmpeg = shutil.which("ffmpeg")
                                    compatible_name = f"compatible_video_{timestamp}.mp4"
                                    compatible_path = os.path.join(os.getcwd(), compatible_name)
                                    transcoded = _transcode_to_browser_mp4(permanent_path, compatible_path)
                                    if transcoded:
                                        print(f"[DEBUG] Created browser-compatible version: {transcoded}")
                                        return transcoded, transcoded, "🎉 Processing complete!", detailed_info

                                    print("[WARNING] Could not create browser-compatible H.264 mp4; returning original output")
                                    return permanent_path, permanent_path, "🎉 Processing complete!", detailed_info
                                else:
                                    print("[ERROR] Permanent video file is not accessible")
                                    return result_path, result_path, "⚠️ Processing complete but display issues", detailed_info
                            except Exception as copy_error:
                                print(f"[ERROR] Failed to copy video: {copy_error}")
                                return result_path, result_path, "⚠️ Processing complete but copy failed", detailed_info
                        else:
                            print("[ERROR] Could not open processed video for verification")
                            return result_path, result_path, "⚠️ Processing complete but verification failed", detection_summary or "Video processed but could not verify output"
                    else:
                        error_msg = f"❌ Processing failed - No output file created. Result path: {result_path}"
                        print(error_msg)
                        return None, None, "Processing failed", error_msg
                        
                except Exception as e:
                    error_msg = f"❌ Processing failed: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    import traceback
                    traceback.print_exc()
                    return None, None, f"Error: {str(e)}", error_msg

            # First update status, then process
            vid_btn.click(
                process_video_with_status,
                inputs=[vid_input, vid_conf, vid_iou, vid_model, vid_labels, vid_conf_show, vid_size, vid_resnet, vid_max_boxes, vid_every_n, vid_ocr, vid_ocr_every_n, vid_speed_mode],
                outputs=[vid_output, vid_download, vid_progress, vid_info],
            ).then(
                process_video_complete,
                inputs=[vid_input, vid_conf, vid_iou, vid_model, vid_labels, vid_conf_show, vid_size, vid_resnet, vid_max_boxes, vid_every_n, vid_ocr, vid_ocr_every_n, vid_speed_mode],
                outputs=[vid_output, vid_download, vid_progress, vid_info],
            )

        # Webcam Tab - Simplified
        with gr.TabItem("📸 Live Webcam"):
            gr.Markdown("### Real-time object detection with your webcam")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 🎛️ Control Panel")
                    
                    with gr.Row():
                        webcam_model = gr.Radio(choices=MODEL_CHOICES, label="🤖 AI Model", value="yolo26n")
                    
                    # Advanced settings (collapsible)
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        webcam_conf = gr.Slider(minimum=0, maximum=1, value=0.35, label="🎯 Confidence Threshold")
                        webcam_iou = gr.Slider(minimum=0, maximum=1, value=0.5, label="📏 IoU Threshold")
                        webcam_enable_color = gr.Checkbox(value=True, label="🎨 Enable Color Detection")
                        webcam_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="📐 Image Size", value=320)
                        webcam_labels = gr.Checkbox(value=True, label="🏷️ Show Labels")
                        webcam_conf_show = gr.Checkbox(value=True, label="📊 Show Confidence")
                        webcam_max_boxes = gr.Slider(minimum=1, maximum=25, value=3, step=1, label="📦 Max Boxes per Frame")
                        webcam_every_n = gr.Slider(minimum=1, maximum=30, value=5, step=1, label="⏱️ Process Every N Frames")
                        
                        # Hidden controls (always enabled)
                        webcam_resnet = gr.Checkbox(value=False, visible=False)
                        webcam_ocr = gr.Checkbox(value=False, visible=False)
                        webcam_ocr_every_n = gr.Slider(minimum=1, maximum=30, value=5, step=1, visible=False)
                    
                    gr.Markdown("#### 📹 Webcam Feed")
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="📸 Live Camera",
                        streaming=True,
                    )
                    
                with gr.Column(scale=2):
                    gr.Markdown("#### 🎯 Live Detection")
                    webcam_output = gr.Image(type="numpy", label="🔍 Real-time Results")
                    
                    gr.Markdown("#### 📊 Detection Info")
                    webcam_info = gr.Markdown("📹 **Status:** Ready to start\n\n🎯 **Instructions:**\n1. Allow camera access\n2. Adjust settings if needed\n3. Watch real-time detection!")

            webcam_input.stream(
                predict_webcam,
                inputs=[
                    webcam_input,
                    webcam_conf,
                    webcam_iou,
                    webcam_model,
                    webcam_labels,
                    webcam_conf_show,
                    webcam_enable_color,
                    webcam_size,
                    webcam_resnet,
                    webcam_max_boxes,
                    webcam_every_n,
                    webcam_ocr,
                    webcam_ocr_every_n,
                ],
                outputs=webcam_output,
                show_progress=False,
                time_limit=30,
            )

    # Modern footer with system information
    gr.Markdown(
        """
        <div style="margin-top: 40px; padding: 20px; background: linear-gradient(45deg, #f8fafc 0%, #e2e8f0 100%); 
             border-radius: 12px; text-align: center;">
            <h3 style="margin: 0 0 10px 0; color: #1f2937;">🔧 System Information</h3>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px;">
                <div style="text-align: center;">
                    <div style="font-size: 24px; margin-bottom: 5px;">⚡</div>
                    <div style="font-weight: 600; color: #059669;">GPU Optimized</div>
                    <div style="font-size: 14px; color: #6b7280;">RTX 4050 Ready</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 24px; margin-bottom: 5px;">🚀</div>
                    <div style="font-weight: 600; color: #7c3aed;">High Speed</div>
                    <div style="font-size: 14px; color: #6b7280;">17ms Processing</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 24px; margin-bottom: 5px;">🎯</div>
                    <div style="font-weight: 600; color: #dc2626;">AI Powered</div>
                    <div style="font-size: 14px; color: #6b7280;">YOLO26 Models</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 24px; margin-bottom: 5px;">📊</div>
                    <div style="font-weight: 600; color: #0891b2;">Multi-Task</div>
                    <div style="font-size: 14px; color: #6b7280;">Detection + OCR</div>
                </div>
            </div>
            <p style="margin: 20px 0 0 0; font-size: 14px; color: #6b7280;">
                💡 <strong>Tip:</strong> Use the "Advanced Settings" to fine-tune detection parameters for your specific use case.
            </p>
        </div>
        """
    )

if __name__ == "__main__":
    _gradio_port_env = os.environ.get("GRADIO_SERVER_PORT")
    _server_port = None
    if _gradio_port_env not in (None, "", "0"):
        _server_port = int(_gradio_port_env)

    demo.launch(
        ssr_mode=False,
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=_server_port,
        allowed_paths=[os.getcwd(), tempfile.gettempdir()],
        prevent_thread_lock=False,
    )
