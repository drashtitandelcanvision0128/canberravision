"""
Video processing module for YOLO26 object detection.
Handles all video processing, annotation, and output generation.
"""

import os
import tempfile
import subprocess
import shutil
import time
from pathlib import Path

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

import cv2
import numpy as np


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


def _extract_video_path(video_value):
    """Extract video path from Gradio input with better error handling"""
    if video_value is None:
        return None
    
    # Handle different Gradio video input formats
    if isinstance(video_value, str):
        # Direct path string
        if os.path.exists(video_value):
            return video_value
        return None
    
    if isinstance(video_value, dict):
        # Gradio dictionary format
        path = video_value.get("path") or video_value.get("name")
        if path and os.path.exists(path):
            return path
        return None
    
    if isinstance(video_value, (list, tuple)) and video_value:
        # List format - take first element
        first = video_value[0]
        if isinstance(first, str):
            if os.path.exists(first):
                return first
            return None
        if isinstance(first, dict):
            path = first.get("path") or first.get("name")
            if path and os.path.exists(path):
                return path
            return None
    
    return None


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
):
    """Predicts objects in a video using a Ultralytics YOLO model with CUDA support."""
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
        
        # Move input video to inputs folder with better error handling
        import shutil
        timestamp = int(time.time())
        video_filename = f"input_video_{timestamp}.mp4"
        inputs_folder = os.path.join(os.getcwd(), "inputs")
        os.makedirs(inputs_folder, exist_ok=True)
        input_video_path = os.path.join(inputs_folder, video_filename)
        
        # Try to copy the input video to avoid temp file permission issues
        try:
            # If the source is a temp file, copy it to our inputs folder
            if video_path != input_video_path:
                shutil.copy2(video_path, input_video_path)
                print(f"[INFO] Input video copied to: {input_video_path}")
                # Use the copied file for processing
                video_path = input_video_path
        except Exception as e:
            print(f"[WARNING] Could not copy input video to inputs folder: {e}")
            # Continue with original path if copy fails
            pass

        # Import here to avoid circular imports
        from .utils import get_model, _get_device, _annotate_with_color

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

        # Create output file in outputs folder
        timestamp = int(time.time())
        outputs_folder = os.path.join(os.getcwd(), "outputs")
        os.makedirs(outputs_folder, exist_ok=True)
        output_filename = f"processed_video_{timestamp}.mp4"
        output_path = os.path.join(outputs_folder, output_filename)
        
        print(f"[INFO] Output will be saved to: {output_path}")

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
            final_output_path = output_path  # Keep our original output path
            
            for fourcc_name, ext in codecs_to_try:
                try:
                    print(f"[DEBUG] Trying {fourcc_name} codec with {ext} extension...")
                    
                    # Use our predefined output path, but change extension if needed
                    if ext != ".mp4":
                        test_output_path = output_path.replace(".mp4", ext)
                    else:
                        test_output_path = output_path
                    
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                    out = cv2.VideoWriter(test_output_path, fourcc, fps, (width, height))
                    
                    if out.isOpened():
                        print(f"[INFO] Successfully initialized video writer with {fourcc_name} codec")
                        output_path = test_output_path  # Update to the working path
                        print(f"[DEBUG] Output path: {output_path}")
                        break
                    else:
                        print(f"[DEBUG] {fourcc_name} codec failed to open")
                        out.release()
                        # Clean up failed attempt
                        if os.path.exists(test_output_path):
                            os.unlink(test_output_path)
                        out = None
                        
                except Exception as codec_error:
                    print(f"[DEBUG] {fourcc_name} codec error: {codec_error}")
                    if out:
                        out.release()
                    if 'test_output_path' in locals() and os.path.exists(test_output_path):
                        os.unlink(test_output_path)
                    out = None
            
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
