"""
Fast Video Processing App for YOLO26
Optimized Gradio interface with CUDA acceleration
"""

import asyncio
import os
import sys
import tempfile
import time
import shutil
from pathlib import Path

# Set OpenCV environment variables
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '0'

import cv2
import gradio as gr
import torch

# Import our modular components
from modules import (
    predict_image,
    predict_video_optimized,
    predict_webcam,
    benchmark_video_processing,
    _get_device
)

# Configure asyncio for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Suppress warnings
    import logging
    import warnings
    import asyncio
    
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    logging.getLogger("gradio").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="asyncio")
    warnings.filterwarnings("ignore", message=".*connection reset.*")

# Model choices
MODEL_CHOICES = [
    "yolo26n",  # Fastest
    "yolo26s",  # Fast
    "yolo26m",  # Medium
]

# Processing modes
PROCESSING_MODES = [
    ("ultra_fast", "⚡ Ultra Fast (Quick Preview)"),
    ("fast", "🚀 Fast (Recommended)"),
    ("balanced", "⚖️ Balanced (High Quality)"),
]

def process_video_optimized_wrapper(video, conf, iou, model, labels, conf_show, imgsz, mode, skip_frames, batch_size):
    """Wrapper for optimized video processing"""
    if video is None:
        return None, None, "Please upload a video first", "No video provided"
    
    try:
        print(f"[INFO] Starting optimized video processing in {mode} mode")
        
        # Process video with optimization
        result_path = predict_video_optimized(
            video_path=video,
            conf_threshold=conf,
            iou_threshold=iou,
            model_name=model,
            show_labels=labels,
            show_conf=conf_show,
            imgsz=imgsz,
            mode=mode,
            skip_frames=int(skip_frames),
            batch_size=int(batch_size),
            enable_resnet=False,  # Disable for speed
            enable_ocr=False       # Disable for speed
        )
        
        if result_path and os.path.exists(result_path):
            # Get video info
            cap = cv2.VideoCapture(result_path)
            if cap.isOpened():
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frames / fps if fps > 0 else 0
                cap.release()
                
                info = f"✅ Processed in {mode} mode | {frames} frames | {width}x{height} | {fps:.1f} FPS | {duration:.1f}s"
                print(f"[INFO] {info}")
                
                return result_path, result_path, "🎉 Processing complete!", info
            else:
                return result_path, result_path, "⚠️ Processing complete", "Video processed but verification failed"
        else:
            return None, None, "❌ Processing failed", "No output file created"
            
    except Exception as e:
        error_msg = f"❌ Processing failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return None, None, f"Error: {str(e)}", error_msg

def run_benchmark(video, model):
    """Run benchmark on different processing modes"""
    if video is None:
        return "Please upload a video first for benchmarking"
    
    try:
        results = benchmark_video_processing(video, model)
        
        output = "## 🏁 Benchmark Results\n\n"
        
        for mode, result in results.items():
            if "error" not in result:
                output += f"### {mode.upper()}\n"
                output += f"- Processing time: {result['time']:.1f} seconds\n"
                output += f"- Output size: {result['file_size'] / (1024*1024):.1f} MB\n\n"
            else:
                output += f"### {mode.upper()}\n"
                output += f"- ❌ Failed to process\n\n"
        
        # Find fastest mode
        successful_modes = {k: v for k, v in results.items() if "error" not in v}
        if successful_modes:
            fastest_mode = min(successful_modes.keys(), key=lambda k: successful_modes[k]['time'])
            output += f"🏆 **Fastest mode: {fastest_mode.upper()}** ({successful_modes[fastest_mode]['time']:.1f}s)\n\n"
        
        output += "### 💡 Recommendations:\n"
        output += "- Use **ultra_fast** for quick previews\n"
        output += "- Use **fast** for normal processing\n"
        output += "- Use **balanced** for high-quality results\n"
        
        return output
        
    except Exception as e:
        return f"❌ Benchmark failed: {str(e)}"

# Create the Gradio app
with gr.Blocks(title="YOLO26 Fast Video Processing") as demo:
    # Display device status
    device = _get_device()
    if device != "cpu":
        gr.Markdown(f"### 🚀 GPU Acceleration Enabled - {torch.cuda.get_device_name(0)}")
        gr.Markdown("**CUDA optimized for ultra-fast video processing**")
    else:
        gr.Markdown("### ⚠️ CPU Processing Mode - Consider using a CUDA GPU for better performance")
    
    gr.Markdown("# YOLO26 Fast Video Processing")
    gr.Markdown("Optimized video processing with CUDA acceleration and performance modes")

    with gr.Tabs():
        # Fast Video Tab
        with gr.TabItem("Fast Video Processing"):
            gr.Markdown("### Upload a video for optimized processing")
            
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload Video")
                    
                    # Basic settings
                    vid_conf = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                    vid_iou = gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold")
                    vid_model = gr.Radio(choices=MODEL_CHOICES, label="Model", value="yolo26n")
                    vid_size = gr.Radio(choices=[320, 640], label="Image Size", value=640)
                    
                    # Optimization settings
                    vid_mode = gr.Radio(
                        choices=PROCESSING_MODES,
                        label="Processing Mode",
                        value="fast"
                    )
                    
                    with gr.Row():
                        vid_skip = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Skip frames (1=process all)")
                        vid_batch = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Batch size (CUDA only)")
                    
                    # Display settings
                    vid_labels = gr.Checkbox(value=True, label="Show Labels")
                    vid_conf_show = gr.Checkbox(value=True, label="Show Confidence")
                    
                    vid_btn = gr.Button("🚀 Process Video Fast", variant="primary")
                    
                    # Status
                    vid_progress = gr.Textbox(label="Status", interactive=False, visible=True)
                    
                with gr.Column():
                    vid_output = gr.Video(
                        label="Processed Video", 
                        visible=True,
                        autoplay=True
                    )
                    vid_download = gr.File(label="Download Processed Video", visible=True)
                    vid_info = gr.Textbox(label="Processing Info", interactive=False, visible=True, lines=3)

            vid_btn.click(
                process_video_optimized_wrapper,
                inputs=[vid_input, vid_conf, vid_iou, vid_model, vid_labels, vid_conf_show, 
                       vid_size, vid_mode, vid_skip, vid_batch],
                outputs=[vid_output, vid_download, vid_progress, vid_info],
            )

        # Benchmark Tab
        with gr.TabItem("Performance Benchmark"):
            gr.Markdown("### Compare performance of different processing modes")
            
            with gr.Row():
                with gr.Column():
                    bench_input = gr.Video(label="Upload Video for Benchmark")
                    bench_model = gr.Radio(choices=MODEL_CHOICES, label="Model", value="yolo26n")
                    bench_btn = gr.Button("🏁 Run Benchmark", variant="primary")
                    
                with gr.Column():
                    bench_output = gr.Markdown(label="Benchmark Results", value="Upload a video and click 'Run Benchmark'")
            
            bench_btn.click(
                run_benchmark,
                inputs=[bench_input, bench_model],
                outputs=[bench_output]
            )

        # Image Tab (keep original)
        with gr.TabItem("Image"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Upload Image")
                    img_conf = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                    img_iou = gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold")
                    img_model = gr.Radio(choices=MODEL_CHOICES, label="Model", value="yolo26n")
                    img_labels = gr.Checkbox(value=True, label="Show Labels")
                    img_conf_show = gr.Checkbox(value=True, label="Show Confidence")
                    img_size = gr.Radio(choices=[320, 640], label="Image Size", value=640)
                    img_resnet = gr.Checkbox(value=True, label="Enable ResNet", visible=False)  # Hidden but required
                    img_max_boxes = gr.Slider(minimum=1, maximum=25, value=10, step=1, label="Max boxes", visible=False)  # Hidden
                    img_ocr = gr.Checkbox(value=True, label="Enable OCR", visible=False)  # Hidden but required
                    img_btn = gr.Button("Detect Objects", variant="primary")
                    
                with gr.Column():
                    img_output = gr.Image(type="pil", label="Result")
                    img_summary = gr.Markdown(label="Detection Summary", value="No detection yet")

            img_btn.click(
                predict_image,
                inputs=[img_input, img_conf, img_iou, img_model, img_labels, img_conf_show, img_size, img_resnet, img_max_boxes, img_ocr],
                outputs=[img_output, img_summary],
            )

        # Performance Tips
        with gr.TabItem("Performance Tips"):
            gr.Markdown("""
            ## 🚀 Performance Optimization Tips
            
            ### CUDA GPU Acceleration
            - ✅ Your system has CUDA GPU: **NVIDIA GeForce RTX 4050 Laptop GPU**
            - ✅ 6.4 GB GPU memory available
            - Processing is **5-10x faster** with CUDA
            
            ### Processing Modes
            - **⚡ Ultra Fast**: Quick previews, lower quality
            - **🚀 Fast**: Recommended for most videos
            - **⚖️ Balanced**: High quality, slower processing
            
            ### Optimization Settings
            - **Image Size**: 320px for speed, 640px for quality
            - **Skip Frames**: Process every Nth frame (2-3 for speed)
            - **Batch Size**: 4-8 frames per batch (CUDA only)
            - **Confidence**: Higher threshold (0.3) for fewer detections
            
            ### Hardware Tips
            - Close other applications to free GPU memory
            - Use SSD for faster video I/O
            - Ensure sufficient RAM (16GB+ recommended)
            
            ### Model Selection
            - **yolo26n**: Fastest, good for real-time
            - **yolo26s**: Good balance of speed and accuracy
            - **yolo26m**: Most accurate, slower processing
            """)

if __name__ == "__main__":
    _gradio_port_env = os.environ.get("GRADIO_SERVER_PORT")
    _server_port = None
    if _gradio_port_env not in (None, "", "0"):
        _server_port = int(_gradio_port_env)

    # Ensure outputs directory exists
    outputs_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    print("🚀 Starting Fast Video Processing App...")
    print(f"🔧 Device: {_get_device()}")
    if torch.cuda.is_available():
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    demo.launch(
        ssr_mode=False,
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=_server_port,
        allowed_paths=[os.getcwd()],
        prevent_thread_lock=False,
        theme=gr.themes.Soft(),
    )
