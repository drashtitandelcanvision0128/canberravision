"""
Simple Video Processing - Direct model loading
Avoids import issues that cause Button errors
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

def process_video_simple(video_path, model_name="yolo26n", mode="ultra_fast"):
    """
    Simple video processing with direct model loading
    No imports from apps.app to avoid Button errors
    """
    try:
        print(f"🚀 Starting ULTRA-FAST video processing: {mode} mode")
        start_time = time.time()
        
        # Validate video
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return None, "Error: Video file not found"
        
        # Import YOLO directly
        from ultralytics import YOLO
        
        # Load model directly
        model_path = f"models/{model_name}.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None, f"Error: Model not found: {model_path}"
        
        print(f"🤖 Loading model: {model_path}")
        model = YOLO(model_path)
        
        # Device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 Device: {device}")
        
        # Optimize settings
        if mode == "ultra_fast":
            conf_threshold = 0.4
            imgsz = 256
            skip_frames = 3
            print("⚡ ULTRA-FAST MODE - Maximum speed")
        elif mode == "fast":
            conf_threshold = 0.35
            imgsz = 320
            skip_frames = 2
            print("🚀 FAST MODE - Balanced")
        else:
            conf_threshold = 0.3
            imgsz = 416
            skip_frames = 1
            print("⚖️ BALANCED MODE - Better quality")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Error: Cannot open video file"
        
        # Properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Output
        timestamp = int(time.time())
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"ultra_fast_{mode}_{timestamp}.mp4")
        
        # Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return None, "Error: Cannot create output video"
        
        # Processing
        processed_count = 0
        actual_processed = 0
        total_detections = 0
        
        print("🎬 Processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_count += 1
            
            if processed_count % skip_frames != 0:
                continue
            
            actual_processed += 1
            
            # Progress
            if actual_processed % 50 == 0:
                elapsed = time.time() - start_time
                fps_proc = actual_processed / elapsed
                progress = (processed_count / total_frames) * 100
                eta = (total_frames - processed_count) / (fps_proc * skip_frames) / 60
                print(f"📊 {progress:.1f}% - {fps_proc:.1f} FPS - ETA: {eta:.1f} min")
            
            try:
                # Predict
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=0.5,
                    imgsz=imgsz,
                    device=device,
                    verbose=False,
                    half=True if device != "cpu" else False
                )
                
                # Annotate
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        total_detections += len(result.boxes)
                        annotated = result.plot()
                    else:
                        annotated = frame
                else:
                    annotated = frame
                
                out.write(annotated)
                
            except Exception as e:
                print(f"⚠️ Frame error: {e}")
                out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Stats
        total_time = time.time() - start_time
        final_fps = actual_processed / total_time
        speedup = skip_frames
        
        print(f"\n✅ Complete!")
        print(f"⏱️ Time: {total_time:.1f}s")
        print(f"🚀 Speed: {final_fps:.1f} FPS")
        print(f"📈 Speedup: ~{speedup}x")
        print(f"📁 Output: {output_path}")
        
        # Summary
        summary = f"""🎥 **Video Processing Complete!**

📊 **Statistics:**
• Mode: {mode.upper()}
• Time: {total_time:.1f}s
• Speed: {final_fps:.1f} FPS
• Speedup: ~{speedup}x
• Frames: {actual_processed}/{total_frames}

🔍 **Detections:**
• Total: {total_detections}
• Per Frame: {total_detections/max(1, actual_processed):.1f}

💾 **Output:**
• {output_path}"""
        
        return output_path, summary
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return None, f"Error: {str(e)}"


# Keep the old function name for compatibility
def process_video_ultra_fast(video_path, model_name="yolo26n", mode="ultra_fast"):
    """Wrapper for backward compatibility"""
    return process_video_simple(video_path, model_name, mode)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='Video path')
    parser.add_argument('--model', default='yolo26n')
    parser.add_argument('--mode', default='ultra_fast', choices=['ultra_fast', 'fast', 'balanced'])
    
    args = parser.parse_args()
    
    output, summary = process_video_simple(args.video, args.model, args.mode)
    
    if output:
        print(f"\n✅ Success! Output: {output}")
    else:
        print(f"\n❌ Failed: {summary}")
