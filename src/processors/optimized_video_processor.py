"""
Optimized Video Processor for YOLO26 with CUDA acceleration
Fast video processing with GPU optimization and performance improvements
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

class OptimizedVideoProcessor:
    """
    High-performance video processor with CUDA optimization
    """
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.num_workers = min(4, mp.cpu_count())  # Limit workers to prevent overload
        print(f"[INFO] Optimized Video Processor initialized on {self.device}")
        print(f"[INFO] Using {self.num_workers} worker threads")
    
    def _get_optimal_device(self):
        """Get the best available device"""
        if torch.cuda.is_available():
            # Check CUDA memory
            device = torch.device("cuda:0")
            try:
                # Test CUDA memory
                torch.cuda.set_device(device)
                print(f"[INFO] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"[INFO] CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                return "cuda:0"
            except Exception as e:
                print(f"[WARNING] CUDA initialization failed: {e}")
                return "cpu"
        return "cpu"
    
    def process_video_fast(self, video_path, model, conf_threshold=0.25, iou_threshold=0.7, 
                          imgsz=640, batch_size=4, skip_frames=1):
        """
        Ultra-fast video processing with batch inference and frame skipping
        
        Args:
            video_path: Path to input video
            model: YOLO model
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold  
            imgsz: Image size for inference
            batch_size: Number of frames to process at once (CUDA only)
            skip_frames: Process every Nth frame for speed
            
        Returns:
            Path to processed video
        """
        try:
            start_time = time.time()
            print(f"[INFO] Starting FAST video processing: {video_path}")
            print(f"[INFO] Device: {self.device}, Batch size: {batch_size}, Skip frames: {skip_frames}")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"[INFO] Video: {width}x{height} @ {fps:.1f} FPS, {frame_count} frames")
            
            # Setup output video
            timestamp = int(time.time())
            outputs_folder = os.path.join(os.getcwd(), "outputs")
            os.makedirs(outputs_folder, exist_ok=True)
            output_path = os.path.join(outputs_folder, f"fast_processed_{timestamp}.mp4")
            
            # Use optimized codec for speed
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise RuntimeError("Cannot create output video writer")
            
            # Process frames in batches for CUDA
            frames_batch = []
            frame_indices = []
            processed_count = 0
            total_detections = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for speed
                if processed_count % skip_frames != 0:
                    processed_count += 1
                    continue
                
                frames_batch.append(frame)
                frame_indices.append(processed_count)
                
                # Process batch when full or at end
                if len(frames_batch) >= batch_size or processed_count >= frame_count - 1:
                    if frames_batch:
                        batch_results = self._process_batch_fast(
                            frames_batch, model, conf_threshold, iou_threshold, imgsz
                        )
                        
                        # Write processed frames
                        for i, (frame, detections) in enumerate(zip(frames_batch, batch_results)):
                            annotated = self._annotate_frame_fast(frame, detections)
                            out.write(annotated)
                            total_detections += len(detections)
                        
                        frames_batch = []
                        frame_indices = []
                
                processed_count += 1
                
                # Progress update
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_processed = processed_count / elapsed
                    progress = (processed_count / frame_count) * 100
                    print(f"[INFO] Processed {processed_count}/{frame_count} frames ({progress:.1f}%) - {fps_processed:.1f} FPS")
            
            # Process remaining frames
            if frames_batch:
                batch_results = self._process_batch_fast(
                    frames_batch, model, conf_threshold, iou_threshold, imgsz
                )
                for frame, detections in zip(frames_batch, batch_results):
                    annotated = self._annotate_frame_fast(frame, detections)
                    out.write(annotated)
                    total_detections += len(detections)
            
            # Cleanup
            cap.release()
            out.release()
            
            # Final stats
            total_time = time.time() - start_time
            final_fps = processed_count / total_time
            print(f"[INFO] ✅ FAST processing complete!")
            print(f"[INFO] Total time: {total_time:.1f}s")
            print(f"[INFO] Processing speed: {final_fps:.1f} FPS")
            print(f"[INFO] Total detections: {total_detections}")
            print(f"[INFO] Output saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"[ERROR] Fast video processing failed: {e}")
            # Cleanup on error
            try:
                if 'cap' in locals():
                    cap.release()
                if 'out' in locals():
                    out.release()
            except:
                pass
            return None
    
    def _process_batch_fast(self, frames, model, conf_threshold, iou_threshold, imgsz):
        """
        Process a batch of frames efficiently
        """
        try:
            if self.device != "cpu":
                # CUDA batch processing
                results = model.predict(
                    source=frames,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=imgsz,
                    device=self.device,
                    verbose=False,
                    half=True,  # FP16 for speed
                    augment=False,
                    agnostic_nms=True
                )
                return results
            else:
                # CPU processing - one by one
                results = []
                for frame in frames:
                    result = model.predict(
                        source=frame,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        imgsz=imgsz,
                        device=self.device,
                        verbose=False,
                        half=False
                    )
                    results.append(result[0])
                return results
                
        except Exception as e:
            print(f"[ERROR] Batch processing failed: {e}")
            return [None] * len(frames)
    
    def _annotate_frame_fast(self, frame, result):
        """
        Fast frame annotation with minimal overhead
        """
        try:
            annotated = frame.copy()
            
            if result is None or not hasattr(result, 'boxes') or result.boxes is None:
                return annotated
            
            # Get detections
            boxes = result.boxes
            if len(boxes) == 0:
                return annotated
            
            # Convert to numpy for speed
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            
            # Draw detections
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                confidence = float(conf[i])
                class_id = int(cls[i])
                
                # Draw bounding box (green for high confidence, red for low)
                color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label (only for high confidence)
                if confidence > 0.3:
                    label = f"{class_id}: {confidence:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return annotated
            
        except Exception as e:
            print(f"[ERROR] Frame annotation failed: {e}")
            return frame
    
    def process_video_ultra_fast(self, video_path, model, conf_threshold=0.3, 
                                imgsz=320, skip_frames=2):
        """
        Ultra-fast mode for quick previews
        - Lower resolution (320px)
        - Higher confidence threshold
        - More frame skipping
        - Minimal annotation
        """
        print("[INFO] 🚀 ULTRA FAST MODE - Quick preview processing")
        return self.process_video_fast(
            video_path=video_path,
            model=model,
            conf_threshold=conf_threshold,
            iou_threshold=0.5,  # Lower IoU for speed
            imgsz=imgsz,
            batch_size=8 if self.device != "cpu" else 1,
            skip_frames=skip_frames
        )
    
    def get_performance_tips(self):
        """Get performance optimization tips"""
        tips = [
            "🚀 Use CUDA GPU for 5-10x speed improvement",
            "📏 Reduce image size (320px for ultra-fast, 640px for balanced)",
            "⏭️ Skip frames (process every 2nd or 3rd frame)",
            "🎯 Increase confidence threshold (0.3-0.4 for fewer detections)",
            "📦 Use batch processing (4-8 frames per batch on CUDA)",
            "💾 Use MP4V codec for fastest encoding",
            "🔧 Close other applications to free GPU memory"
        ]
        return tips


# Global optimized processor instance
optimized_processor = OptimizedVideoProcessor()

def process_video_optimized(video_path, model, mode="fast", **kwargs):
    """
    Convenience function for optimized video processing
    
    Args:
        video_path: Path to input video
        model: YOLO model
        mode: "fast", "ultra_fast", or "balanced"
        **kwargs: Additional parameters
    """
    if mode == "ultra_fast":
        return optimized_processor.process_video_ultra_fast(video_path, model, **kwargs)
    elif mode == "balanced":
        return optimized_processor.process_video_fast(
            video_path, model, 
            conf_threshold=0.25, imgsz=640, skip_frames=1, **kwargs
        )
    else:  # fast mode (default)
        return optimized_processor.process_video_fast(video_path, model, **kwargs)


if __name__ == "__main__":
    print("⚡ Optimized Video Processor")
    print("=" * 40)
    
    # Show performance tips
    tips = optimized_processor.get_performance_tips()
    for tip in tips:
        print(tip)
    
    print(f"\n🔧 Device: {optimized_processor.device}")
    print(f"🧵 Workers: {optimized_processor.num_workers}")
    print("\n✅ Ready for ultra-fast video processing!")
