"""
App Video Integration - Add to your main app.py
Video output saving functionality
"""

# ================================================================
# ADD THESE IMPORTS TO YOUR app.py (at the top)
# ================================================================

"""
# Add these imports after your existing imports
from video_output_handler import (
    save_processed_video, 
    save_detection_frame, 
    cleanup_outputs,
    get_outputs_info
)
"""

# ================================================================
# ADD THIS FUNCTION TO YOUR app.py
# ================================================================

def process_and_save_video(video_path, detection_results, fps=30):
    """
    Process video and save results to outputs folder
    
    Args:
        video_path: Input video path
        detection_results: List of detection results per frame
        fps: Video fps
        
    Returns:
        Path to saved processed video
    """
    try:
        import cv2
        
        print(f"[INFO] Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video info: {total_frames} frames, {original_fps} fps")
        
        processed_frames = []
        frame_count = 0
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get detection for this frame
            frame_detections = detection_results[frame_count] if frame_count < len(detection_results) else None
            
            # Draw detections on frame
            if frame_detections:
                frame = draw_detections_on_frame(frame, frame_detections)
                
                # Save frame snapshot if has detections
                if frame_detections.get('license_plates') or frame_detections.get('objects'):
                    save_detection_frame(frame, frame_count, frame_detections)
            
            processed_frames.append(frame)
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                print(f"[INFO] Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        
        # Save processed video
        output_path = save_processed_video(video_path, processed_frames, fps)
        
        # Cleanup old files (keep last 7 days)
        cleanup_outputs(days_old=7)
        
        # Get output summary
        outputs_info = get_outputs_info()
        print(f"[INFO] Output summary: {outputs_info['total_size_mb']:.2f} MB total")
        
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        return ""

def draw_detections_on_frame(frame, detection_results):
    """
    Draw detection boxes and info on frame
    Add this function to your app.py
    """
    try:
        import cv2
        import numpy as np
        
        annotated_frame = frame.copy()
        
        # Draw license plates
        if detection_results and 'license_plates' in detection_results:
            for plate in detection_results['license_plates']:
                if 'bounding_box' in plate:
                    x1, y1, x2, y2 = plate['bounding_box']
                    
                    # Green box for license plates
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add plate text
                    plate_text = plate.get('plate_text', 'Unknown')
                    cv2.putText(annotated_frame, f"Plate: {plate_text}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw other objects
        if detection_results and 'objects' in detection_results:
            for obj in detection_results['objects']:
                if 'bounding_box' in obj:
                    x1, y1, x2, y2 = obj['bounding_box']
                    
                    # Blue box for objects
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Add object info
                    class_name = obj.get('class_name', 'Object')
                    confidence = obj.get('confidence', 0)
                    color = obj.get('color', 'Unknown')
                    
                    text = f"{class_name}: {confidence:.2f} ({color})"
                    cv2.putText(annotated_frame, text, 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return annotated_frame
        
    except Exception as e:
        print(f"[ERROR] Failed to draw detections: {e}")
        return frame

# ================================================================
# INTEGRATION WITH YOUR EXISTING VIDEO PROCESSING FUNCTION
# ================================================================

"""
In your main video processing function, add this at the end:

# After processing all frames
if video_path and detection_results:
    # Save processed video to outputs folder
    saved_video_path = process_and_save_video(video_path, detection_results)
    
    if saved_video_path:
        print(f"✅ Video saved to: {saved_video_path}")
        
        # Add to your results
        result['saved_video_path'] = saved_video_path
        result['output_info'] = get_outputs_info()
    else:
        print("❌ Failed to save video")
"""

# ================================================================
# GRADIO INTEGRATION EXAMPLE
# ================================================================

def create_gradio_video_interface():
    """
    Example of how to integrate with Gradio interface
    Add this to your Gradio UI functions
    """
    
    def process_video_with_output(video_file):
        """
        Process video and return both display video and saved path
        """
        try:
            # Your existing video processing
            # detection_results = your_existing_processing(video_file)
            
            # For example, create dummy results
            detection_results = []
            
            # Save processed video
            saved_path = process_and_save_video(video_file.name, detection_results)
            
            # Get output info
            output_info = get_outputs_info()
            
            # Return results for Gradio
            return {
                'processed_video': saved_path,  # For Gradio video display
                'saved_path': saved_path,       # For download/info
                'output_info': output_info,     # For display
                'message': f"✅ Video processed and saved! Total outputs: {output_info['total_files']} files ({output_info['total_size_mb']:.1f} MB)"
            }
            
        except Exception as e:
            return {
                'error': f"❌ Processing failed: {str(e)}"
            }
    
    return process_video_with_output

# ================================================================
# CLEANUP AND MAINTENANCE FUNCTIONS
# ================================================================

def setup_automatic_cleanup():
    """
    Setup automatic cleanup of old files
    Call this when your app starts
    """
    try:
        # Clean up files older than 7 days
        cleanup_outputs(days_old=7)
        
        # Get current output info
        info = get_outputs_info()
        
        print(f"[INFO] Output cleanup completed")
        print(f"[INFO] Current outputs: {info['total_files']} files ({info['total_size_mb']:.1f} MB)")
        
        # If too large, clean more aggressively
        if info['total_size_mb'] > 2000:  # 2GB limit
            cleanup_outputs(days_old=3)
            print(f"[INFO] Aggressive cleanup completed (3 days)")
        
    except Exception as e:
        print(f"[ERROR] Cleanup setup failed: {e}")

def get_download_links():
    """
    Get download links for output files
    For Gradio interface
    """
    try:
        from video_output_handler import video_handler
        import os
        
        download_links = []
        
        # Get recent video files
        videos_dir = video_handler.videos_dir
        if videos_dir.exists():
            for video_file in sorted(videos_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                file_size = video_file.stat().st_size / (1024 * 1024)  # MB
                download_links.append({
                    'name': video_file.name,
                    'path': str(video_file),
                    'size_mb': f"{file_size:.1f} MB",
                    'modified': video_file.stat().st_mtime
                })
        
        return download_links
        
    except Exception as e:
        print(f"[ERROR] Failed to get download links: {e}")
        return []

# ================================================================
# USAGE EXAMPLES
# ================================================================

if __name__ == "__main__":
    print("📹 App Video Integration Guide")
    print("=" * 40)
    
    print("📋 Integration Steps:")
    print("1. Add imports to app.py")
    print("2. Add process_and_save_video function")
    print("3. Add draw_detections_on_frame function")
    print("4. Update your video processing function")
    print("5. Add Gradio integration if needed")
    
    print("\n🔧 Key Features:")
    print("✅ Automatic video saving to outputs/ folder")
    print("✅ Frame snapshots for detections")
    print("✅ Thumbnail generation")
    print("✅ Automatic cleanup (7 days)")
    print("✅ Size management (max 2GB)")
    print("✅ GitIgnore ready (heavy files excluded)")
    
    print("\n📁 Output Structure:")
    print("outputs/")
    print("├── videos/        # Processed videos")
    print("├── frames/        # Detection frames")
    print("└── thumbnails/    # Video thumbnails")
    
    print("\n✅ Integration ready! Add to your app.py now!")
