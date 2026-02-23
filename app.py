# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Set OpenCV environment variables to reduce camera detection warnings
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '0'

import cv2
import gradio as gr
import numpy as np
import PIL.Image as Image
import torch
from ultralytics import YOLO

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

MODEL_CHOICES = [
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26n-seg",
    "yolo26s-seg",
    "yolo26m-seg",
    "yolo26n-pose",
    "yolo26s-pose",
    "yolo26m-pose",
    "yolo26n-obb",
    "yolo26s-obb",
    "yolo26m-obb",
    "yolo26n-cls",
    "yolo26s-cls",
    "yolo26m-cls",
]

IMAGE_SIZE_CHOICES = [320, 640, 1024]


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


def predict_image(img, conf_threshold, iou_threshold, model_name, show_labels, show_conf, imgsz):
    """Predicts objects in an image using a Ultralytics YOLO model with CUDA support."""
    model = get_model(model_name)
    device = _get_device()
    
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        device=device,
        verbose=False,
        half=True if device != "cpu" else False,  # Use FP16 on CUDA for speed
    )

    for r in results:
        im_array = r.plot(labels=show_labels, conf=show_conf)
        im = Image.fromarray(im_array[..., ::-1])

    return im


def predict_video(video_path, conf_threshold, iou_threshold, model_name, show_labels, show_conf, imgsz):
    """Predicts objects in a video using a Ultralytics YOLO model with CUDA support."""
    video_path = _extract_video_path(video_path)
    if video_path is None:
        print("[ERROR] No valid video path provided")
        return None

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

    # Open the video with error handling
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        # Try alternative method
        cap.release()
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("[ERROR] Failed to open video with FFMPEG backend")
            return None

    # Get video properties with validation
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
    try:
        # Try H.264 first (most compatible)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            # Fallback to MP4V
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                # Final fallback to XVID
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    except Exception as e:
        print(f"[ERROR] Failed to initialize video writer: {e}")
        cap.release()
        return None

    processed_frames = 0
    success_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frames += 1
        if processed_frames % 30 == 0:  # Progress update every 30 frames
            print(f"[INFO] Processing frame {processed_frames}/{frame_count}...")

        try:
            # Run inference on the frame with CUDA support
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                device=device,
                verbose=False,
                half=True if device != "cpu" else False,  # Use FP16 on CUDA for speed
            )

            # Get the annotated frame
            annotated_frame = results[0].plot(labels=show_labels, conf=show_conf)
            out.write(annotated_frame)
            success_count += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to process frame {processed_frames}: {e}")
            # Write original frame if processing fails
            out.write(frame)

    cap.release()
    out.release()
    
    # Verify output file was created successfully
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        print("[ERROR] Output video file was not created properly")
        return None
    
    print(f"[INFO] Video processing complete. Processed {success_count}/{processed_frames} frames successfully.")
    return output_path


# Cache model for streaming performance
_model_cache = {}


def get_model(model_name):
    """Get or create a cached model instance with CUDA support."""
    if model_name not in _model_cache:
        print(f"[INFO] Loading model: {model_name}")
        model = YOLO(model_name)
        
        # Move model to CUDA device if available
        device = _get_device()
        if device != "cpu":
            model.to(device)
            print(f"[INFO] Model moved to CUDA device: {device}")
        
        _model_cache[model_name] = model
        print(f"[INFO] Model {model_name} loaded and cached successfully")
    
    return _model_cache[model_name]


def predict_webcam(frame, conf_threshold, iou_threshold, model_name, show_labels, show_conf, imgsz):
    """Predicts objects in a webcam frame using a Ultralytics YOLO model with CUDA support."""
    if frame is None:
        return None

    try:
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

        # Gradio webcam sends RGB, but Ultralytics YOLO expects BGR for OpenCV operations
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run inference with CUDA support
        results = model.predict(
            source=frame_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
            half=True if device != "cpu" else False,
        )

        # YOLO's plot() returns BGR, convert back to RGB for Gradio display
        annotated_frame = results[0].plot(labels=show_labels, conf=show_conf)
        if annotated_frame is not None:
            return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        else:
            return frame

    except Exception as e:
        print(f"[ERROR] Webcam prediction failed: {e}")
        return frame


# Create the Gradio app with simplified interface
with gr.Blocks(title="YOLO26 Object Detection") as demo:
    # Display device status
    device = _get_device()
    if device != "cpu":
        gr.Markdown(f"### 🚀 GPU Acceleration Enabled - Processing on {torch.cuda.get_device_name(0)}")
    else:
        gr.Markdown("### ⚠️ CPU Processing Mode")
    
    gr.Markdown("# YOLO26 Object Detection")

    with gr.Tabs():
        # Image Tab
        with gr.TabItem("Image"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Upload Image")
                    img_conf = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                    img_iou = gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold")
                    img_model = gr.Radio(choices=MODEL_CHOICES, label="Model Name", value="yolo26n")
                    img_labels = gr.Checkbox(value=True, label="Show Labels")
                    img_conf_show = gr.Checkbox(value=True, label="Show Confidence")
                    img_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="Image Size", value=640)
                    img_btn = gr.Button("Detect Objects", variant="primary")
                with gr.Column():
                    img_output = gr.Image(type="pil", label="Result")

            img_btn.click(
                predict_image,
                inputs=[img_input, img_conf, img_iou, img_model, img_labels, img_conf_show, img_size],
                outputs=img_output,
            )

        # Video Tab
        with gr.TabItem("Video"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload Video")
                    vid_conf = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold")
                    vid_iou = gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold")
                    vid_model = gr.Radio(choices=MODEL_CHOICES, label="Model Name", value="yolo26n")
                    vid_labels = gr.Checkbox(value=True, label="Show Labels")
                    vid_conf_show = gr.Checkbox(value=True, label="Show Confidence")
                    vid_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="Image Size", value=640)
                    vid_btn = gr.Button("Process Video", variant="primary")
                with gr.Column():
                    vid_output = gr.Video(label="Result")

            vid_btn.click(
                predict_video,
                inputs=[vid_input, vid_conf, vid_iou, vid_model, vid_labels, vid_conf_show, vid_size],
                outputs=vid_output,
            )

        # Webcam Tab
        with gr.TabItem("Webcam"):
            with gr.Row():
                with gr.Column():
                    webcam_conf = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence")
                    webcam_iou = gr.Slider(minimum=0, maximum=1, value=0.7, label="IoU threshold")
                    webcam_model = gr.Radio(choices=MODEL_CHOICES, label="Model", value="yolo26n")
                    webcam_labels = gr.Checkbox(value=True, label="Show Labels")
                    webcam_conf_show = gr.Checkbox(value=True, label="Show Confidence")
                    webcam_size = gr.Radio(choices=IMAGE_SIZE_CHOICES, label="Size", value=640)
                with gr.Column():
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Webcam",
                        streaming=True,
                    )
                    webcam_output = gr.Image(type="numpy", label="Live Result")

            webcam_input.stream(
                predict_webcam,
                inputs=[webcam_input, webcam_conf, webcam_iou, webcam_model, webcam_labels, webcam_conf_show, webcam_size],
                outputs=webcam_output,
                show_progress=False,
                time_limit=30,
            )

demo.launch(
    ssr_mode=False,
    share=False,
    show_error=True,
    quiet=False,
    inbrowser=True,
    server_name="127.0.0.1",
    server_port=7863,
    prevent_thread_lock=False,
)
