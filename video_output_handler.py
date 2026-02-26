"""
Video Output Handler for YOLO26
Save processed videos to output folder
Light deployment friendly
"""

import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import shutil

class VideoOutputHandler:
    """
    Handle video output saving and management
    """
    
    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.videos_dir = self.output_dir / "videos"
        self.frames_dir = self.output_dir / "frames"
        self.thumbnails_dir = self.output_dir / "thumbnails"
        
        for dir_path in [self.videos_dir, self.frames_dir, self.thumbnails_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"[INFO] Video output handler initialized: {self.output_dir}")
    
    def save_processed_video(self, input_video_path: str, processed_frames: list, 
                           fps: int = 30, include_annotations: bool = True) -> str:
        """
        Save processed video with annotations
        
        Args:
            input_video_path: Original video path
            processed_frames: List of processed frames (numpy arrays)
            fps: Frames per second
            include_annotations: Whether to include timestamp and info
            
        Returns:
            Path to saved video
        """
        try:
            if not processed_frames:
                print("[ERROR] No frames to save")
                return ""
            
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_name = Path(input_video_path).stem
            output_filename = f"{input_name}_processed_{timestamp}.mp4"
            output_path = self.videos_dir / output_filename
            
            # Get video dimensions from first frame
            height, width = processed_frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            print(f"[INFO] Saving processed video: {len(processed_frames)} frames")
            
            for i, frame in enumerate(processed_frames):
                # Add timestamp and info if enabled
                if include_annotations:
                    frame = self._add_frame_info(frame, i, len(processed_frames))
                
                out.write(frame)
            
            out.release()
            
            # Generate thumbnail
            self._generate_thumbnail(str(output_path))
            
            print(f"[INFO] ✅ Video saved: {output_path}")
            print(f"[INFO] 📁 File size: {self._get_file_size_mb(output_path):.2f} MB")
            
            return str(output_path)
            
        except Exception as e:
            print(f"[ERROR] Failed to save video: {e}")
            return ""
    
    def save_frame_snapshot(self, frame: np.ndarray, frame_number: int, 
                          detections: list = None, timestamp: str = None) -> str:
        """
        Save individual frame with detections
        
        Args:
            frame: Frame to save
            frame_number: Frame number
            detections: List of detections (optional)
            timestamp: Timestamp string (optional)
            
        Returns:
            Path to saved frame
        """
        try:
            # Generate filename
            timestamp_str = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{frame_number:06d}_{timestamp_str}.jpg"
            output_path = self.frames_dir / filename
            
            # Add annotations to frame
            if detections:
                frame = self._draw_detections_on_frame(frame, detections)
            
            # Add frame info
            frame = self._add_frame_info(frame, frame_number, info_text=f"Frame {frame_number}")
            
            # Save frame
            cv2.imwrite(str(output_path), frame)
            
            return str(output_path)
            
        except Exception as e:
            print(f"[ERROR] Failed to save frame: {e}")
            return ""
    
    def create_detection_summary_video(self, video_path: str, detection_results: list) -> str:
        """
        Create summary video with only frames that have detections
        
        Args:
            video_path: Original video path
            detection_results: List of detection results per frame
            
        Returns:
            Path to summary video
        """
        try:
            # Filter frames with detections
            detection_frames = []
            
            for i, result in enumerate(detection_results):
                if result and result.get('has_detections', False):
                    # You'll need to load the actual frame
                    # This is a placeholder - implement based on your video loading
                    pass
            
            if detection_frames:
                return self.save_processed_video(
                    video_path, 
                    detection_frames, 
                    include_annotations=True
                )
            else:
                print("[INFO] No frames with detections found for summary")
                return ""
                
        except Exception as e:
            print(f"[ERROR] Failed to create summary video: {e}")
            return ""
    
    def _add_frame_info(self, frame: np.ndarray, frame_num: int, total_frames: int = None, 
                       info_text: str = None) -> np.ndarray:
        """Add timestamp and info to frame"""
        try:
            # Create a copy to avoid modifying original
            annotated_frame = frame.copy()
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Add text at top
            cv2.putText(annotated_frame, f"YOLO26 Detection - {timestamp}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add frame info
            if total_frames:
                frame_info = f"Frame: {frame_num}/{total_frames}"
            else:
                frame_info = f"Frame: {frame_num}"
            
            cv2.putText(annotated_frame, frame_info, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add custom info if provided
            if info_text:
                cv2.putText(annotated_frame, info_text, 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            return annotated_frame
            
        except Exception as e:
            print(f"[ERROR] Failed to add frame info: {e}")
            return frame
    
    def _draw_detections_on_frame(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw detection boxes and labels on frame"""
        try:
            annotated_frame = frame.copy()
            
            for detection in detections:
                if 'bbox' in detection:
                    x1, y1, x2, y2 = detection['bbox']
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = detection.get('label', 'Object')
                    confidence = detection.get('confidence', 0)
                    
                    text = f"{label}: {confidence:.2f}"
                    cv2.putText(annotated_frame, text, 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return annotated_frame
            
        except Exception as e:
            print(f"[ERROR] Failed to draw detections: {e}")
            return frame
    
    def _generate_thumbnail(self, video_path: str) -> str:
        """Generate thumbnail from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get middle frame
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame = frame_count // 2
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            
            if ret:
                # Resize to thumbnail size
                thumbnail = cv2.resize(frame, (320, 240))
                
                # Save thumbnail
                video_name = Path(video_path).stem
                thumbnail_path = self.thumbnails_dir / f"{video_name}_thumb.jpg"
                cv2.imwrite(str(thumbnail_path), thumbnail)
                
                cap.release()
                return str(thumbnail_path)
            
            cap.release()
            return ""
            
        except Exception as e:
            print(f"[ERROR] Failed to generate thumbnail: {e}")
            return ""
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def cleanup_old_files(self, days_old: int = 7, max_size_mb: float = 1000):
        """
        Clean up old output files to save space
        
        Args:
            days_old: Delete files older than this many days
            max_size_mb: Keep total size under this limit
        """
        try:
            import time
            
            current_time = time.time()
            cutoff_time = current_time - (days_old * 24 * 60 * 60)
            
            total_size = 0
            files_to_delete = []
            
            # Check all files in output directory
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    file_age = file_path.stat().st_mtime
                    
                    if file_age < cutoff_time:
                        files_to_delete.append(file_path)
                    else:
                        total_size += file_path.stat().st_size
            
            # If over size limit, delete oldest files
            if total_size > (max_size_mb * 1024 * 1024):
                all_files = []
                for file_path in self.output_dir.rglob("*"):
                    if file_path.is_file() and file_path not in files_to_delete:
                        all_files.append((file_path.stat().st_mtime, file_path))
                
                # Sort by age (oldest first)
                all_files.sort()
                
                for file_age, file_path in all_files:
                    if total_size > (max_size_mb * 1024 * 1024):
                        files_to_delete.append(file_path)
                        total_size -= file_path.stat().st_size
                    else:
                        break
            
            # Delete files
            deleted_count = 0
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except:
                    pass
            
            print(f"[INFO] Cleanup completed: {deleted_count} files deleted")
            
        except Exception as e:
            print(f"[ERROR] Cleanup failed: {e}")
    
    def get_output_summary(self) -> dict:
        """Get summary of output directory"""
        try:
            summary = {
                'total_files': 0,
                'total_size_mb': 0.0,
                'videos': 0,
                'frames': 0,
                'thumbnails': 0
            }
            
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    summary['total_files'] += 1
                    summary['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
                    
                    if file_path.parent.name == 'videos':
                        summary['videos'] += 1
                    elif file_path.parent.name == 'frames':
                        summary['frames'] += 1
                    elif file_path.parent.name == 'thumbnails':
                        summary['thumbnails'] += 1
            
            return summary
            
        except Exception as e:
            print(f"[ERROR] Failed to get summary: {e}")
            return {}


# Global instance for easy access
video_handler = VideoOutputHandler()

# Convenience functions
def save_processed_video(input_path: str, frames: list, fps: int = 30) -> str:
    """Save processed video to outputs folder"""
    return video_handler.save_processed_video(input_path, frames, fps)

def save_detection_frame(frame: np.ndarray, frame_num: int, detections: list = None) -> str:
    """Save frame with detections to outputs folder"""
    return video_handler.save_frame_snapshot(frame, frame_num, detections)

def cleanup_outputs(days_old: int = 7):
    """Clean up old output files"""
    video_handler.cleanup_old_files(days_old)

def get_outputs_info() -> dict:
    """Get information about output files"""
    return video_handler.get_output_summary()


if __name__ == "__main__":
    print("📹 Video Output Handler for YOLO26")
    print("=" * 40)
    
    # Test initialization
    handler = VideoOutputHandler()
    
    # Show output directory structure
    print(f"📁 Output directory: {handler.output_dir}")
    print(f"📹 Videos: {handler.videos_dir}")
    print(f"🖼️  Frames: {handler.frames_dir}")
    print(f"📸 Thumbnails: {handler.thumbnails_dir}")
    
    # Get current summary
    summary = handler.get_output_summary()
    print(f"\n📊 Current Output Summary:")
    print(f"   Total files: {summary['total_files']}")
    print(f"   Total size: {summary['total_size_mb']:.2f} MB")
    print(f"   Videos: {summary['videos']}")
    print(f"   Frames: {summary['frames']}")
    print(f"   Thumbnails: {summary['thumbnails']}")
    
    print("\n✅ Video output handler ready!")
    print("📖 Usage:")
    print("   save_processed_video(input_path, frames)")
    print("   save_detection_frame(frame, frame_num, detections)")
    print("   cleanup_outputs(days_old=7)")
