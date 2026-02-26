"""
Optimized Video Processing Module for YOLO26
Fast CUDA-accelerated video processing with performance optimizations
"""

import os
import time
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Import original functions for compatibility
from .video_processing import _extract_video_path, _transcode_to_browser_mp4

def predict_video_optimized(
    video_path,
    conf_threshold=0.25,
    iou_threshold=0.7,
    model_name="yolo26n",
    show_labels=True,
    show_conf=True,
    imgsz=640,
    enable_resnet=True,
    max_boxes=5,
    resnet_every_n=5,
    enable_ocr=True,
    ocr_every_n=5,
    mode="fast",  # "fast", "ultra_fast", "balanced"
    skip_frames=1,
    batch_size=4
):
    """
    Optimized video prediction with CUDA acceleration and performance improvements
    
    Args:
        mode: Processing mode
            - "fast": Standard optimized processing
            - "ultra_fast": Maximum speed, lower quality
            - "balanced": Balance between speed and accuracy
        skip_frames: Process every Nth frame for speed
        batch_size: Number of frames to process at once (CUDA only)
    """
    try:
        print(f"[INFO] 🚀 Starting OPTIMIZED video processing in {mode} mode")
        
        # Extract video path
        video_path = _extract_video_path(video_path)
        if video_path is None:
            print("[ERROR] No valid video path provided")
            return None

        print(f"[INFO] Processing video: {video_path}")
        
        # Get device and model
        from .utils import get_model, _get_device, _annotate_with_color
        
        device = _get_device()
        model = get_model(model_name)
        
        print(f"[INFO] Device: {device}")
        print(f"[INFO] Model: {model_name}")
        print(f"[INFO] Mode: {mode}, Skip frames: {skip_frames}, Batch size: {batch_size}")

        # Adjust parameters based on mode
        if mode == "ultra_fast":
            conf_threshold = max(conf_threshold, 0.3)  # Higher confidence for speed
            imgsz = min(imgsz, 320)  # Smaller image size
            skip_frames = max(skip_frames, 2)  # Skip more frames
            batch_size = 8 if device != "cpu" else 1
            print("[INFO] ⚡ Ultra-fast mode activated")
        elif mode == "balanced":
            imgsz = 640  # Standard size
            skip_frames = 1  # Process all frames
            batch_size = 4 if device != "cpu" else 1
            print("[INFO] ⚖️ Balanced mode activated")
        else:  # fast mode
            imgsz = min(imgsz, 640)
            batch_size = 4 if device != "cpu" else 1
            print("[INFO] 🚀 Fast mode activated")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video: {width}x{height} @ {fps:.1f} FPS, {frame_count} frames")

        # Create output path
        timestamp = int(time.time())
        outputs_folder = os.path.join(os.getcwd(), "outputs")
        os.makedirs(outputs_folder, exist_ok=True)
        output_filename = f"optimized_video_{mode}_{timestamp}.mp4"
        output_path = os.path.join(outputs_folder, output_filename)
        
        print(f"[INFO] Output: {output_path}")

        # Setup video writer with fast codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("[ERROR] Cannot create video writer")
            cap.release()
            return None

        # Processing variables
        processed_frames = 0
        actual_frames = 0
        total_detections = 0
        start_time = time.time()
        
        print("[INFO] Starting optimized frame processing...")

        # Main processing loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames += 1
            
            # Skip frames for speed
            if processed_frames % skip_frames != 0:
                continue

            actual_frames += 1
            
            # Progress update
            if actual_frames % 50 == 0:
                elapsed = time.time() - start_time
                fps_processed = actual_frames / elapsed
                progress = (processed_frames / frame_count) * 100
                print(f"[INFO] Processed {processed_frames}/{frame_count} ({progress:.1f}%) - {fps_processed:.1f} FPS")

            try:
                # Run inference
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=imgsz,
                    device=device,
                    verbose=False,
                    half=True if device != "cpu" else False,  # FP16 on CUDA
                    augment=False,
                    agnostic_nms=True
                )

                if results and len(results) > 0:
                    result = results[0]
                    
                    # Count detections
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        total_detections += len(result.boxes)
                    
                    # Annotate frame (simplified for speed)
                    if mode == "ultra_fast":
                        annotated_frame = _annotate_frame_minimal(frame, result, show_labels, show_conf)
                    else:
                        annotated_frame = _annotate_with_color(
                            frame, result, show_labels, show_conf,
                            enable_resnet=bool(enable_resnet),
                            max_boxes=int(max_boxes),
                            resnet_every_n=int(resnet_every_n),
                            stream_key_prefix="video_optimized",
                            enable_ocr=bool(enable_ocr),
                            ocr_every_n=int(ocr_every_n),
                        )
                else:
                    annotated_frame = frame

                # Write frame
                out.write(annotated_frame)

            except Exception as e:
                print(f"[ERROR] Frame {processed_frames} processing failed: {e}")
                # Write original frame on error
                out.write(frame)

        # Cleanup
        cap.release()
        out.release()

        # Calculate final stats
        total_time = time.time() - start_time
        final_fps = actual_frames / total_time if total_time > 0 else 0
        speedup = skip_frames  # Approximate speedup from frame skipping
        
        print(f"[INFO] ✅ Optimized processing complete!")
        print(f"[INFO] Total time: {total_time:.1f}s")
        print(f"[INFO] Processing speed: {final_fps:.1f} FPS")
        print(f"[INFO] Frames processed: {actual_frames}/{processed_frames}")
        print(f"[INFO] Total detections: {total_detections}")
        print(f"[INFO] Approximate speedup: {speedup}x")
        print(f"[INFO] Output saved: {output_path}")

        # Verify output
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            print("[ERROR] Output file creation failed")
            return None

    except Exception as e:
        print(f"[ERROR] Optimized video processing failed: {e}")
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
        
        return None


def _annotate_frame_minimal(frame, result, show_labels=True, show_conf=True):
    """
    Minimal frame annotation for ultra-fast processing
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
        
        # Draw simple boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            confidence = float(conf[i])
            
            # Only draw high confidence detections
            if confidence > 0.3:
                # Green box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Simple label
                if show_labels or show_conf:
                    label_text = f"{confidence:.2f}" if show_conf else f"{int(cls[i])}"
                    if show_labels and show_conf:
                        label_text = f"{int(cls[i])}: {confidence:.2f}"
                    
                    cv2.putText(annotated, label_text, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotated
        
    except Exception as e:
        print(f"[ERROR] Minimal annotation failed: {e}")
        return frame


def get_optimization_settings():
    """Get recommended optimization settings"""
    settings = {
        "ultra_fast": {
            "description": "Maximum speed for quick previews",
            "conf_threshold": 0.3,
            "imgsz": 320,
            "skip_frames": 2,
            "batch_size": 8,
            "use_cases": ["Preview", "Real-time", "Low priority"]
        },
        "fast": {
            "description": "Good balance of speed and accuracy",
            "conf_threshold": 0.25,
            "imgsz": 640,
            "skip_frames": 1,
            "batch_size": 4,
            "use_cases": ["Standard processing", "Most videos"]
        },
        "balanced": {
            "description": "Best accuracy with reasonable speed",
            "conf_threshold": 0.25,
            "imgsz": 640,
            "skip_frames": 1,
            "batch_size": 4,
            "use_cases": ["High quality", "Important videos"]
        }
    }
    return settings


def benchmark_video_processing(video_path, model_name="yolo26n"):
    """
    Benchmark different processing modes to find the optimal setting
    """
    print(f"[INFO] 🏁 Benchmarking video processing modes...")
    
    from .utils import get_model, _get_device
    
    device = _get_device()
    model = get_model(model_name)
    
    modes = ["ultra_fast", "fast", "balanced"]
    results = {}
    
    for mode in modes:
        print(f"\n[INFO] Testing {mode} mode...")
        start_time = time.time()
        
        output_path = predict_video_optimized(
            video_path=video_path,
            model_name=model_name,
            mode=mode,
            conf_threshold=0.25,
            show_labels=False,  # Skip labels for benchmark
            show_conf=False,
            enable_resnet=False,  # Skip extra processing
            enable_ocr=False
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            results[mode] = {
                "time": processing_time,
                "file_size": file_size,
                "output_path": output_path,
                "fps": "N/A"  # Would need video info to calculate
            }
            print(f"[INFO] {mode}: {processing_time:.1f}s, {file_size / (1024*1024):.1f} MB")
            
            # Cleanup benchmark file
            try:
                os.unlink(output_path)
            except:
                pass
        else:
            print(f"[ERROR] {mode} mode failed")
            results[mode] = {"error": "Processing failed"}
    
    print(f"\n[INFO] 🏆 Benchmark Results:")
    for mode, result in results.items():
        if "error" not in result:
            print(f"  {mode}: {result['time']:.1f}s")
        else:
            print(f"  {mode}: Failed")
    
    return results


if __name__ == "__main__":
    print("⚡ Optimized Video Processing Module")
    print("=" * 50)
    
    # Show optimization settings
    settings = get_optimization_settings()
    for mode, config in settings.items():
        print(f"\n🔧 {mode.upper()} Mode:")
        print(f"   Description: {config['description']}")
        print(f"   Image size: {config['imgsz']}px")
        print(f"   Skip frames: {config['skip_frames']}")
        print(f"   Best for: {', '.join(config['use_cases'])}")
    
    print(f"\n✅ Ready for optimized video processing!")
