"""
Standalone Video Processing Script for Canberra Vision
Bypasses the Gradio UI to process videos directly
"""

import cv2
import numpy as np
import torch
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def process_ppe_video_stepper(video_path, model_path="models/best_ppe.pt", auto_detect=False, skip=1):
    """Interactive PPE video player: pause/resume and run detection on demand.

    Args:
        video_path: input video path
        model_path: PPE model path (best_ppe.pt)
        auto_detect: if True, run detection automatically while playing
        skip: process every Nth frame when auto_detect is enabled
    """
    if not os.path.exists(video_path):
        print(f" Video file not found: {video_path}")
        return 1

    from modules.ppe_detection import get_ppe_detector

    detector = get_ppe_detector(model_path=model_path, debug=False, auto_recovery=True, force_new=False)
    if hasattr(detector, "reset_video_tracker"):
        detector.reset_video_tracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Cannot open video")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = fps if fps and fps > 0 else 30

    print(f" PPE Stepper: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    print(" Controls:")
    print("   - SPACE: pause/resume")
    print("   - N: next frame (when paused)")
    print("   - D: detect current frame (when paused)")
    print("   - A: auto-detect ON/OFF (while playing)")
    print("   - Q / ESC: quit")

    paused = False
    frame_number = 0
    last_frame = None
    last_annotated = None
    last_result = None
    auto_mode = bool(auto_detect)
    skip = max(int(skip), 1)
    _auto_counter = 0

    window_name = "Canberra Vision - PPE Stepper"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        if not paused or last_frame is None:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            last_frame = frame
            # Default display is raw frame; if auto-mode is enabled we'll replace with annotated
            last_annotated = frame

            # Auto-detect while playing
            if auto_mode:
                _auto_counter += 1
                if _auto_counter % skip == 0:
                    if hasattr(detector, "detect_video"):
                        last_result = detector.detect_video(last_frame, frame_number=frame_number, debug=False)
                    else:
                        last_result = detector.detect(last_frame, debug=False)
                    last_annotated = detector.visualize(last_frame, last_result, show_labels=True)
                else:
                    last_result = None

        display = last_annotated if last_annotated is not None else last_frame
        overlay = display.copy()
        cv2.putText(
            overlay,
            f"Frame: {frame_number}/{total_frames}  paused={paused}  auto={auto_mode}  skip={skip}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        if last_result is not None:
            cv2.putText(
                overlay,
                f"Persons={last_result.total_persons} H={last_result.helmet_detected} V={last_result.vest_detected} M={last_result.mask_detected}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        cv2.imshow(window_name, overlay)

        wait_ms = 1 if not paused else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            paused = not paused
            continue
        if key == ord('a'):
            auto_mode = not auto_mode
            continue
        if paused and key == ord('n'):
            # step one frame
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            last_frame = frame
            last_annotated = frame
            last_result = None
            continue
        if paused and key == ord('d'):
            # detect on current frame
            if last_frame is not None:
                if hasattr(detector, "detect_video"):
                    last_result = detector.detect_video(last_frame, frame_number=frame_number, debug=False)
                else:
                    last_result = detector.detect(last_frame, debug=False)
                last_annotated = detector.visualize(last_frame, last_result, show_labels=True)
            continue

    cap.release()
    cv2.destroyAllWindows()
    return 0

def process_video_direct(video_path, model_name="yolo26n", mode="ultra_fast"):
    """
    Process video directly without Gradio UI
    
    Args:
        video_path: Path to video file
        model_name: YOLO model to use (yolo26n, yolov8n, etc.)
        mode: Processing mode - "ultra_fast", "fast", or "balanced"
    
    Returns:
        output_path: Path to processed video
    """
    try:
        print(f" Starting video processing: {mode} mode")
        print(f" Input: {video_path}")
        
        # Validate video
        if not os.path.exists(video_path):
            print(f" Video file not found: {video_path}")
            return None
        
        # Import YOLO
        from ultralytics import YOLO
        
        # Load model
        model_path = f"models/{model_name}.pt"
        if not os.path.exists(model_path):
            print(f" Model not found: {model_path}")
            return None
        
        print(f" Loading model: {model_path}")
        model = YOLO(model_path)
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f" Using device: {device}")
        
        # Mode settings
        if mode == "ultra_fast":
            conf_threshold = 0.4
            imgsz = 256
            skip_frames = 3
            print(" ULTRA-FAST MODE")
        elif mode == "fast":
            conf_threshold = 0.35
            imgsz = 320
            skip_frames = 2
            print(" FAST MODE")
        else:  # balanced
            conf_threshold = 0.3
            imgsz = 416
            skip_frames = 1
            print(" BALANCED MODE")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(" Cannot open video")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f" Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Create output
        timestamp = int(time.time())
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"processed_{mode}_{timestamp}.mp4")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(" Cannot create output video")
            cap.release()
            return None
        
        # Processing loop
        processed_count = 0
        actual_processed = 0
        total_detections = 0
        start_time = time.time()
        
        print(" Processing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_count += 1
            
            # Skip frames for speed
            if processed_count % skip_frames != 0:
                continue
            
            actual_processed += 1
            
            # Progress update
            if actual_processed % 50 == 0:
                elapsed = time.time() - start_time
                fps_processed = actual_processed / elapsed
                progress = (processed_count / total_frames) * 100
                eta = (total_frames - processed_count) / (fps_processed * skip_frames) / 60
                print(f" Progress: {progress:.1f}% - {fps_processed:.1f} FPS - ETA: {eta:.1f} min")
            
            try:
                # Run detection
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=0.5,
                    imgsz=imgsz,
                    device=device,
                    verbose=False,
                    half=True if device != "cpu" else False
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    # Annotate frame
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        total_detections += len(result.boxes)
                        annotated_frame = result.plot()
                    else:
                        annotated_frame = frame
                else:
                    annotated_frame = frame
                
                # Write frame
                out.write(annotated_frame)
                
            except Exception as e:
                print(f" Frame {processed_count} error: {e}")
                out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Stats
        total_time = time.time() - start_time
        final_fps = actual_processed / total_time
        
        print(f"\n Processing complete!")
        print(f" Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f" Speed: {final_fps:.1f} FPS")
        print(f" Processed: {actual_processed}/{total_frames} frames")
        print(f" Detections: {total_detections}")
        print(f" Output: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process video with YOLO detection')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--model', default='yolo26n', help='Model name (yolo26n, yolov8n, etc.)')
    parser.add_argument('--mode', default='ultra_fast', choices=['ultra_fast', 'fast', 'balanced'],
                        help='Processing mode')
    parser.add_argument('--ppe_stepper', action='store_true', help='Interactive PPE pause/detect/resume player')
    parser.add_argument('--ppe_model', default='models/best_ppe.pt', help='PPE model path for stepper (default: models/best_ppe.pt)')
    parser.add_argument('--ppe_auto', action='store_true', help='Enable auto-detect while playing (stepper mode)')
    parser.add_argument('--ppe_skip', type=int, default=1, help='Process every Nth frame when auto-detect is enabled (default: 1)')
    
    args = parser.parse_args()
    
    if args.ppe_stepper:
        rc = process_ppe_video_stepper(
            args.video_path,
            model_path=args.ppe_model,
            auto_detect=args.ppe_auto,
            skip=args.ppe_skip,
        )
        sys.exit(rc)

    output = process_video_direct(args.video_path, args.model, args.mode)
    
    if output:
        print(f"\n Video processed successfully!")
        print(f"Output saved to: {output}")
    else:
        print("\n Video processing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
