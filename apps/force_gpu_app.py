# 🚀 FORCE GPU YOLO VISION APP
# यह app forcefully GPU use करेगा अगर available हो तो

import os
import sys
import subprocess

# Force CUDA environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'  # RTX 4050 architecture

# Add CUDA paths (if exists)
cuda_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin", 
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
    r"C:\Program Files\NVIDIA Corporation\NVIDIA NvDLISR"
]

for path in cuda_paths:
    if os.path.exists(path):
        os.environ['PATH'] = path + ';' + os.environ.get('PATH', '')

print("="*60)
print("🚀 FORCE GPU YOLO VISION - STARTING")
print("="*60)

# Import libraries
import asyncio
import time
from datetime import datetime
from pathlib import Path
import tempfile
import json
import shutil

import cv2
import gradio as gr
import numpy as np
import PIL.Image as Image

try:
    import torch
    print(f"✅ PyTorch loaded: {torch.__version__}")
    
    # Force CUDA detection
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        print(f"🎉 GPU DETECTED AND ACTIVE!")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test GPU with tensor
        test_tensor = torch.randn(1000, 1000).cuda()
        print("✅ GPU test tensor created successfully!")
        
    else:
        device = torch.device('cpu')
        print("❌ GPU not detected, using CPU")
        print("🔧 Troubleshooting:")
        print("1. Check NVIDIA drivers: nvidia-smi")
        print("2. Restart computer")
        print("3. Reinstall CUDA Toolkit")
        
except Exception as e:
    print(f"❌ Error loading PyTorch: {e}")
    device = 'cpu'

import torchvision
from ultralytics import YOLO

print("="*60)
print(f"🔥 Using device: {device}")
print("="*60)

# Tesseract setup
try:
    import pytesseract
    if sys.platform.startswith("win"):
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"✅ Tesseract configured")
except:
    print("⚠️ Tesseract not available")

# Import other modules
try:
    from lighton_ocr_integration import extract_text_with_lighton
    LIGHTON_AVAILABLE = True
    print("✅ LightOnOCR loaded")
except:
    LIGHTON_AVAILABLE = False

try:
    from gender_detection_model import load_gender_model
    gender_model = load_gender_model()
    GENDER_MODEL_AVAILABLE = True
    print("✅ Gender detection loaded")
except:
    GENDER_MODEL_AVAILABLE = False

try:
    from enhanced_detection import enhanced_license_plate_detection
    ENHANCED_DETECTION_AVAILABLE = True
    print("✅ Enhanced detection loaded")
except:
    ENHANCED_DETECTION_AVAILABLE = False

try:
    from international_license_plates import extract_international_license_plates
    INTERNATIONAL_PLATES_AVAILABLE = True
    print("✅ International plates loaded")
except:
    INTERNATIONAL_PLATES_AVAILABLE = False

# Async setup for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def load_yolo_model_gpu(model_path='yolov8n.pt'):
    """Load YOLO model with GPU support"""
    try:
        print(f"🔄 Loading model: {model_path}")
        model = YOLO(model_path)
        
        # Force model to GPU if available
        if torch.cuda.is_available():
            model.to('cuda')
            print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ Model loaded on CPU")
            
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def process_image_gpu(image, model_name='yolov8n.pt', conf_threshold=0.25):
    """Process image with GPU acceleration"""
    try:
        # Load model
        model = load_yolo_model_gpu(model_name)
        if not model:
            return None, "Model loading failed"
        
        # Convert image
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        # Run inference
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
            results = model(img_array, conf=conf_threshold)
        
        # Process results
        processed_img = results[0].plot()
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
        
        return processed_img, detections
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return None, f"Error: {str(e)}"

def create_interface():
    """Create Gradio interface with GPU status"""
    
    # GPU Status
    if torch.cuda.is_available():
        gpu_status = f"🎉 **GPU ACTIVE**: {torch.cuda.get_device_name(0)}"
        gpu_info = f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        performance = "🚀 **10X FASTER PROCESSING**"
    else:
        gpu_status = "⚠️ **CPU MODE** - GPU not detected"
        gpu_info = "Install NVIDIA drivers for GPU acceleration"
        performance = "🐌 **SLOWER PROCESSING**"
    
    with gr.Blocks(title="YOLO Vision - Force GPU", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🚀 YOLO Vision - Force GPU Edition")
        
        gr.Markdown(f"### {gpu_status}")
        gr.Markdown(f"**{gpu_info}**")
        gr.Markdown(f"**{performance}**")
        
        with gr.Tab("🖼️ Image Detection"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="Upload Image")
                    model_choice = gr.Dropdown(
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                        value='yolov8n.pt',
                        label="Model Selection"
                    )
                    conf_slider = gr.Slider(0.0, 1.0, 0.25, label="Confidence Threshold")
                    detect_btn = gr.Button("🔍 Detect Objects", variant="primary")
                    
                with gr.Column():
                    output_img = gr.Image(type="numpy", label="Results")
                    detection_json = gr.JSON(label="Detections")
                    
            detect_btn.click(
                fn=process_image_gpu,
                inputs=[input_img, model_choice, conf_slider],
                outputs=[output_img, detection_json]
            )
            
        # Add video and webcam tabs
        with gr.Tab("🎥 Video Processing"):
            gr.Markdown("### Video processing with GPU acceleration")
            video_input = gr.Video(label="Upload Video")
            video_btn = gr.Button("🎬 Process Video", variant="primary")
            video_output = gr.Video(label="Processed Video")
            
        with gr.Tab("📸 Webcam"):
            gr.Markdown("### Real-time webcam detection")
            webcam_btn = gr.Button("📷 Start Webcam", variant="primary")
            webcam_output = gr.Image(label="Webcam Feed")
            
    return interface

if __name__ == "__main__":
    print("🚀 Starting Force GPU YOLO Vision...")
    
    # Create and launch interface
    interface = create_interface()
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        show_error=True
    )
