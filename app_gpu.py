# GPU-Enhanced YOLO Vision App
# Automatically detects and uses GPU when available

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

# GPU Detection and Configuration
print("="*60)
print("GPU DETECTION AND CONFIGURATION")
print("="*60)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    
    print(f"🎉 GPU DETECTED!")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Enable GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set CUDA memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print("✅ GPU optimizations enabled")
    
else:
    device = torch.device('cpu')
    print("❌ GPU NOT AVAILABLE")
    print("⚠️  Running on CPU (slower performance)")
    print("")
    print("To enable GPU:")
    print("1. Install NVIDIA drivers: https://www.nvidia.com/drivers")
    print("2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
    print("3. Restart this application")
    
print("="*60)
print(f"Using device: {device}")
print("="*60)

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

# Import other modules (same as original app)
try:
    from lighton_ocr_integration import get_lighton_ocr_processor, extract_text_with_lighton
    LIGHTON_AVAILABLE = True
    print("[INFO] LightOnOCR integration loaded")
except ImportError:
    LIGHTON_AVAILABLE = False
    print("[WARNING] LightOnOCR integration not available")

try:
    from gender_detection_model import load_gender_model, predict_gender, get_gender_transform
    GENDER_MODEL_AVAILABLE = True
    gender_model = load_gender_model()
    gender_transform = get_gender_transform()
    if gender_model:
        print("[INFO] Gender detection model loaded successfully")
except ImportError as e:
    GENDER_MODEL_AVAILABLE = False
    gender_model = None
    gender_transform = None
    print(f"[WARNING] Gender detection model not available: {e}")

try:
    from enhanced_detection import enhanced_license_plate_detection
    ENHANCED_DETECTION_AVAILABLE = True
    print("[INFO] Enhanced detection for challenging images loaded")
except ImportError:
    ENHANCED_DETECTION_AVAILABLE = False
    print("[WARNING] Enhanced detection not available")

try:
    from international_license_plates import extract_international_license_plates, InternationalLicensePlateRecognizer
    INTERNATIONAL_PLATES_AVAILABLE = True
    print("[INFO] International license plate recognition loaded")
except ImportError:
    INTERNATIONAL_PLATES_AVAILABLE = False
    print("[WARNING] International license plate recognition not available")

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import logging
    logging.getLogger("asyncio").setLevel(logging.WARNING)

def _get_device():
    """Get the best available device for processing."""
    if torch.cuda.is_available():
        return 0  # Use first GPU
    else:
        return "cpu"

# Load YOLO model with GPU support
def load_model_gpu(model_name='yolov8n.pt'):
    """Load YOLO model with GPU support if available."""
    if not model_name.endswith('.pt'):
        model_path = f"{model_name}.pt"
    else:
        model_path = model_name
        
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model.to('cuda')
        print(f"[✅] Model moved to GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[⚠️] Model using CPU")
        
    return model

# Import all functions from original app
exec(open('app.py').read().split('# Set OpenCV environment variables')[1])

# Override the model loading function
def process_image_gpu(image, model_name='yolov8n.pt', conf_threshold=0.25):
    """Process image with GPU-accelerated YOLO model."""
    try:
        # Load model with GPU support
        model = load_model_gpu(model_name)
        
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
            
        # Run inference
        results = model(image_array, conf=conf_threshold)
        
        # Process results (same as original)
        processed_image = results[0].plot()
        detections = []
        
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
            conf = result.conf[0].cpu().numpy()
            cls = int(result.cls[0].cpu().numpy())
            class_name = model.names[cls]
            
            detections.append({
                'class': class_name,
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
            
        return processed_image, detections
        
    except Exception as e:
        print(f"[ERROR] GPU processing failed: {e}")
        # Fallback to CPU processing
        return process_image(image, model_name, conf_threshold)

# Create Gradio interface
def create_gpu_interface():
    """Create Gradio interface with GPU indicators."""
    
    with gr.Blocks(title="YOLO Vision - GPU Enhanced", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🚀 YOLO Vision - GPU Enhanced Object Detection")
        
        # GPU Status Indicator
        if torch.cuda.is_available():
            gpu_status = f"🎉 **GPU Active**: {torch.cuda.get_device_name(0)}"
            gpu_color = "green"
        else:
            gpu_status = "⚠️ **CPU Only** - Install NVIDIA drivers for GPU acceleration"
            gpu_color = "orange"
            
        gr.Markdown(f"### Status: {gpu_status}")
        
        # Original interface components
        with gr.Tab("Image Detection"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Upload Image")
                    model_choice = gr.Dropdown(
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                        value='yolov8n.pt',
                        label="Model Selection"
                    )
                    conf_slider = gr.Slider(0.0, 1.0, 0.25, label="Confidence Threshold")
                    detect_btn = gr.Button("🔍 Detect Objects", variant="primary")
                    
                with gr.Column():
                    output_image = gr.Image(type="numpy", label="Detection Results")
                    detection_info = gr.JSON(label="Detections")
                    
            detect_btn.click(
                fn=process_image_gpu,
                inputs=[input_image, model_choice, conf_slider],
                outputs=[output_image, detection_info]
            )
            
        # Add other tabs from original interface...
        # (Copy remaining tabs from original app)
        
    return interface

if __name__ == "__main__":
    print("[INFO] Starting GPU-Enhanced YOLO Vision Application...")
    
    # Create and launch interface
    interface = create_gpu_interface()
    
    # Launch with appropriate settings
    interface.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        show_error=True,
        show_tips=True
    )
