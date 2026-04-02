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
        print(f"🚀 Starting video processing: {mode} mode")
        print(f"📹 Input: {video_path}")
        
        # Validate video
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return None
        
        # Import YOLO
        from ultralytics import YOLO
        
        # Load model
        model_path = f"models/{model_name}.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None
        
        print(f"🤖 Loading model: {model_path}")
        model = YOLO(model_path)
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 Using device: {device}")
        
        # Mode settings
        if mode == "ultra_fast":
            conf_threshold = 0.4
            imgsz = 256
            skip_frames = 3
            print("⚡ ULTRA-FAST MODE")
        elif mode == "fast":
            conf_threshold = 0.35
            imgsz = 320
            skip_frames = 2
            print("🚀 FAST MODE")
        else:  # balanced
            conf_threshold = 0.3
            imgsz = 416
            skip_frames = 1
            print("⚖️ BALANCED MODE")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Cannot open video")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Create output
        timestamp = int(time.time())
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"processed_{mode}_{timestamp}.mp4")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("❌ Cannot create output video")
            cap.release()
            return None
        
        # Processing loop
        processed_count = 0
        actual_processed = 0
        total_detections = 0
        start_time = time.time()
        
        print("🎬 Processing frames...")
        
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
                print(f"📊 Progress: {progress:.1f}% - {fps_processed:.1f} FPS - ETA: {eta:.1f} min")
            
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
                print(f"⚠️ Frame {processed_count} error: {e}")
                out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Stats
        total_time = time.time() - start_time
        final_fps = actual_processed / total_time
        
        print(f"\n✅ Processing complete!")
        print(f"⏱️ Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"🚀 Speed: {final_fps:.1f} FPS")
        print(f"📈 Processed: {actual_processed}/{total_frames} frames")
        print(f"🔍 Detections: {total_detections}")
        print(f"💾 Output: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error: {e}")
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
    
    args = parser.parse_args()
    
    output = process_video_direct(args.video_path, args.model, args.mode)
    
    if output:
        print(f"\n🎥 Video processed successfully!")
        print(f"Output saved to: {output}")
    else:
        print("\n❌ Video processing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
