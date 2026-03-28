"""
Video Processing Module for YOLO26
Handle all video-related operations
Clean, modular, and efficient
"""

import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator

# Import our modules
try:
    from image_processor import ImageProcessor, image_processor
    from video_output_handler import save_processed_video, save_detection_frame, get_outputs_info
    from simple_plate_detection import extract_license_plates_simple
    from kmeans_color_detector import kmeans_detector, detect_image_colors, categorize_detected_object, analyze_scene
    IMAGE_PROCESSOR_AVAILABLE = True
    VIDEO_OUTPUT_AVAILABLE = True
    KMEANS_COLOR_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Some modules not available: {e}")
    IMAGE_PROCESSOR_AVAILABLE = False
    VIDEO_OUTPUT_AVAILABLE = False
    KMEANS_COLOR_AVAILABLE = False

class VideoProcessor:
    """
    Handle all video processing operations
    """
    
    def __init__(self):
        self.processed_count = 0
        self.start_time = time.time()
        self.current_video_info = {}
        self.kmeans_detector = None
        
        # Initialize K-means color detector
        if KMEANS_COLOR_AVAILABLE:
            try:
                self.kmeans_detector = kmeans_detector
                print("[INFO] K-means color detector initialized in video processor")
            except Exception as e:
                print(f"[WARNING] Failed to initialize K-means color detector: {e}")
                self.kmeans_detector = None
        
        print("[INFO] Video Processor initialized")
    
    def process_video(self, video_path: str, save_output: bool = True, 
                     progress_callback: callable = None) -> Dict:
        """
        Main video processing function
        
        Args:
            video_path: Path to input video
            save_output: Whether to save processed video
            progress_callback: Callback function for progress updates
            
        Returns:
            Complete processing results
        """
        try:
            start_time = time.time()
            self.processed_count += 1
            
            print(f"[INFO] Processing video #{self.processed_count}: {video_path}")
            
            # Validate video
            if not Path(video_path).exists():
                return {'error': f'Video file not found: {video_path}'}
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': 'Failed to open video file'}
            
            # Get video properties
            video_info = self._get_video_info(cap)
            self.current_video_info = video_info
            
            print(f"[INFO] Video info: {video_info['frame_count']} frames, {video_info['fps']} fps, {video_info['duration']:.1f}s")
            
            # Initialize results
            results = {
                'video_info': video_info,
                'processing_info': {
                    'start_time': datetime.now().isoformat(),
                    'total_frames': video_info['frame_count'],
                    'processed_frames': 0,
                    'frames_with_detections': 0,
                    'processing_time': 0,
                    'fps_processed': 0
                },
                'detections': {
                    'all_frames': [],
                    'summary': {
                        'total_license_plates': 0,
                        'unique_plates': set(),
                        'frame_numbers_with_plates': []
                    }
                },
                'output_files': {
                    'saved_video': None,
                    'saved_frames': [],
                    'thumbnails': []
                }
            }
            
            # Process frames
            processed_frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame using image processor
                frame_result = self._process_frame(frame, frame_count)
                
                # Store results
                results['detections']['all_frames'].append(frame_result)
                results['processing_info']['processed_frames'] += 1
                
                # Check for detections
                if frame_result.get('has_detections', False):
                    results['processing_info']['frames_with_detections'] += 1
                    results['detections']['summary']['frame_numbers_with_plates'].append(frame_count)
                    
                    # Save frame with detections
                    if VIDEO_OUTPUT_AVAILABLE:
                        saved_frame = save_detection_frame(
                            frame, 
                            frame_count, 
                            frame_result.get('license_plates', [])
                        )
                        if saved_frame:
                            results['output_files']['saved_frames'].append(saved_frame)
                
                # Collect unique license plates
                for plate in frame_result.get('license_plates', []):
                    plate_text = plate.get('plate_text', '')
                    if plate_text:
                        results['detections']['summary']['unique_plates'].add(plate_text)
                        results['detections']['summary']['total_license_plates'] += 1
                
                # Add frame to processed frames (for video output)
                if save_output:
                    annotated_frame = self._create_annotated_frame(frame, frame_result)
                    processed_frames.append(annotated_frame)
                
                frame_count += 1
                
                # Progress callback
                if progress_callback and frame_count % 30 == 0:  # Every 30 frames
                    progress = (frame_count / video_info['frame_count']) * 100
                    progress_callback(progress, frame_count, video_info['frame_count'])
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / video_info['frame_count']) * 100
                    print(f"[INFO] Processed {frame_count}/{video_info['frame_count']} frames ({progress:.1f}%)")
            
            cap.release()
            
            # Convert set to list for JSON serialization
            results['detections']['summary']['unique_plates'] = list(results['detections']['summary']['unique_plates'])
            
            # Save processed video
            if save_output and processed_frames and VIDEO_OUTPUT_AVAILABLE:
                print(f"[INFO] Saving processed video...")
                saved_video_path = save_processed_video(
                    video_path, 
                    processed_frames, 
                    video_info['fps']
                )
                
                if saved_video_path:
                    results['output_files']['saved_video'] = saved_video_path
                    print(f"[INFO] ✅ Video saved: {saved_video_path}")
            
            # Calculate final stats
            processing_time = time.time() - start_time
            results['processing_info']['processing_time'] = processing_time
            results['processing_info']['fps_processed'] = frame_count / processing_time if processing_time > 0 else 0
            
            print(f"[INFO] Video processing completed in {processing_time:.1f}s")
            print(f"[INFO] Found {len(results['detections']['summary']['unique_plates'])} unique license plates")
            print(f"[INFO] Detections in {results['processing_info']['frames_with_detections']} frames")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Video processing failed: {e}")
            return {'error': str(e)}
    
    def process_video_streaming(self, video_path: str, chunk_size: int = 30) -> Generator[Dict, None, None]:
        """
        Process video in streaming mode (yield results for each chunk)
        
        Args:
            video_path: Path to input video
            chunk_size: Number of frames per chunk
            
        Yields:
            Results for each chunk
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                yield {'error': 'Failed to open video file'}
                return
            
            video_info = self._get_video_info(cap)
            frame_count = 0
            chunk_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_result = self._process_frame(frame, frame_count)
                chunk_frames.append(frame_result)
                frame_count += 1
                
                # Yield chunk when ready
                if len(chunk_frames) >= chunk_size:
                    chunk_result = {
                        'chunk_info': {
                            'start_frame': frame_count - chunk_size,
                            'end_frame': frame_count - 1,
                            'frame_count': len(chunk_frames)
                        },
                        'video_info': video_info,
                        'frames': chunk_frames,
                        'summary': self._summarize_chunk(chunk_frames)
                    }
                    
                    yield chunk_result
                    chunk_frames = []
            
            # Yield remaining frames
            if chunk_frames:
                chunk_result = {
                    'chunk_info': {
                        'start_frame': frame_count - len(chunk_frames),
                        'end_frame': frame_count - 1,
                        'frame_count': len(chunk_frames)
                    },
                    'video_info': video_info,
                    'frames': chunk_frames,
                    'summary': self._summarize_chunk(chunk_frames)
                }
                
                yield chunk_result
            
            cap.release()
            
        except Exception as e:
            yield {'error': str(e)}
    
    def _get_video_info(self, cap: cv2.VideoCapture) -> Dict:
        """Get video information"""
        return {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(1, cap.get(cv2.CAP_PROP_FPS))
        }
    
    def _process_frame(self, frame: np.ndarray, frame_number: int) -> Dict:
        """Process single frame with K-means color detection"""
        try:
            if IMAGE_PROCESSOR_AVAILABLE:
                # Use image processor for frame processing
                result = image_processor.process_image(frame, use_enhanced=False, use_international=False)
                
                # Add frame-specific info
                result['frame_number'] = frame_number
                result['has_detections'] = len(result['detections']['license_plates']) > 0
                
                # Add K-means color analysis
                if self.kmeans_detector:
                    try:
                        # Get objects from image processor result
                        objects = result.get('objects', [])
                        
                        if objects:
                            # Analyze scene colors
                            scene_analysis = self.kmeans_detector.analyze_scene_colors(frame, objects)
                            result['scene_analysis'] = scene_analysis
                            
                            # Process each object with K-means colors
                            kmeans_colors = []
                            for obj in objects:
                                if 'bounding_box' in obj:
                                    x1, y1, x2, y2 = obj['bounding_box']
                                    color_result = self.kmeans_detector.detect_colors_kmeans(frame, (x1, y1, x2, y2))
                                    
                                    # Categorize object with color
                                    categorized_obj = self.kmeans_detector.categorize_object_with_color(
                                        obj.get('class_name', ''), color_result, obj.get('confidence', 0.8)
                                    )
                                    
                                    # Update object with enhanced info
                                    obj.update({
                                        'kmeans_color_info': color_result,
                                        'enhanced_category': categorized_obj,
                                        'color_family': color_result.get('primary_color', {}).get('family', 'Unknown'),
                                        'color_shade': color_result.get('primary_color', {}).get('shade', 'Unknown')
                                    })
                                    
                                    kmeans_colors.append(color_result)
                            
                            result['kmeans_colors'] = kmeans_colors
                            result['has_color_analysis'] = True
                        else:
                            result['kmeans_colors'] = []
                            result['has_color_analysis'] = False
                            
                    except Exception as e:
                        print(f"[ERROR] K-means color processing failed for frame {frame_number}: {e}")
                        result['kmeans_colors'] = []
                        result['has_color_analysis'] = False
                
                return result
            else:
                # Simple fallback processing
                return {
                    'frame_number': frame_number,
                    'has_detections': False,
                    'license_plates': [],
                    'objects': [],
                    'kmeans_colors': [],
                    'has_color_analysis': False,
                    'error': 'Image processor not available'
                }
                
        except Exception as e:
            print(f"[ERROR] Frame {frame_number} processing failed: {e}")
            return {
                'frame_number': frame_number,
                'has_detections': False,
                'kmeans_colors': [],
                'has_color_analysis': False,
                'error': str(e)
            }
    
    def _create_annotated_frame(self, frame: np.ndarray, frame_result: Dict) -> np.ndarray:
        """Create annotated frame with detections and K-means colors"""
        try:
            annotated = frame.copy()
            
            # Draw license plates - IMPROVED: Better formatting with yellow box
            for plate in frame_result.get('license_plates', []):
                if 'bounding_box' in plate and plate['bounding_box']:
                    x1, y1, x2, y2 = plate['bounding_box']
                    
                    # Get plate text with validation
                    plate_text = plate.get('plate_text', plate.get('text', 'Unknown')).strip()
                    if not plate_text or len(plate_text) < 4:
                        continue
                    
                    # Yellow box for license plates (matching image style)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 4)
                    
                    # Add plate text with "Plate: " prefix - BIGGER FONT
                    label = f"Plate: {plate_text[:20]}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    thickness = 3
                    
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Position label above the plate
                    ly = y1 - 15
                    if ly - text_height - baseline < 0:
                        ly = y2 + text_height + 15
                    
                    # Draw yellow background for label - BIGGER
                    cv2.rectangle(annotated, 
                                 (x1, ly - text_height - baseline), 
                                 (x1 + text_width + 12, ly + 6), 
                                 (0, 255, 255), -1)
                    
                    # Draw label text in black - BIGGER
                    cv2.putText(annotated, label, (x1 + 6, ly), 
                               font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            
            # Draw objects with K-means color information
            objects = frame_result.get('objects', [])
            for i, obj in enumerate(objects[:10]):  # Limit to 10 objects
                if 'bounding_box' in obj:
                    x1, y1, x2, y2 = obj['bounding_box']
                    
                    # Get enhanced display name with color
                    if 'enhanced_category' in obj:
                        display_name = obj['enhanced_category'].get('enhanced_label', obj.get('class_name', 'Unknown'))
                        color_family = obj.get('color_family', 'Unknown')
                        color_shade = obj.get('color_shade', 'Unknown')
                    else:
                        display_name = obj.get('class_name', 'Unknown')
                        color_family = 'Unknown'
                        color_shade = 'Unknown'
                    
                    confidence = obj.get('confidence', 0.0)
                    
                    # Draw bounding box
                    box_color = (0, 255, 255)  # Yellow for objects
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Create enhanced label - BIGGER FONT
                    if color_family != 'Unknown':
                        label = f"{display_name[:12]} {color_shade[:6]}"
                    else:
                        label = f"{display_name[:15]} {confidence:.2f}"
                    
                    # Draw label background - BIGGER
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]
                    cv2.rectangle(annotated, (x1, y1 - 35), (x1 + label_size[0] + 10, y1), box_color, -1)
                    
                    # Draw label text - BIGGER
                    cv2.putText(annotated, label, 
                               (x1 + 5, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
            
            # Add frame number - BIGGER
            frame_text = f"Frame: {frame_result.get('frame_number', 0)}"
            cv2.putText(annotated, frame_text, 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Add detection status - BIGGER
            if frame_result.get('has_detections', False):
                status_text = f"Plates: {len(frame_result.get('license_plates', []))}"
                cv2.putText(annotated, status_text, 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Add object count - BIGGER
            obj_text = f"Objects: {len(objects)}"
            cv2.putText(annotated, obj_text, 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # Add color analysis status - BIGGER
            if frame_result.get('has_color_analysis', False):
                kmeans_colors = frame_result.get('kmeans_colors', [])
                color_count = len([c for c in kmeans_colors if c.get('success')])
                color_text = f"Colors: {color_count}"
                cv2.putText(annotated, color_text, 
                           (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
                
                # Add scene theme if available - BIGGER
                scene_analysis = frame_result.get('scene_analysis', {})
                if scene_analysis.get('dominant_theme'):
                    theme = scene_analysis['dominant_theme'][:25]
                    theme_text = f"Theme: {theme}"
                    cv2.putText(annotated, theme_text, 
                               (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            
            # Add processing indicator - BIGGER
            cv2.putText(annotated, "K-MEANS VIDEO PROCESSING", 
                       (10, annotated.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            print(f"[ERROR] Failed to create annotated frame: {e}")
            return frame
    
    def _summarize_chunk(self, frames: List[Dict]) -> Dict:
        """Summarize chunk results"""
        total_frames = len(frames)
        frames_with_detections = sum(1 for f in frames if f.get('has_detections', False))
        all_plates = []
        
        for frame in frames:
            all_plates.extend([p.get('plate_text', '') for p in frame.get('license_plates', [])])
        
        unique_plates = list(set(filter(None, all_plates)))
        
        return {
            'total_frames': total_frames,
            'frames_with_detections': frames_with_detections,
            'detection_rate': (frames_with_detections / total_frames) * 100 if total_frames > 0 else 0,
            'unique_plates': unique_plates,
            'total_plates_found': len(all_plates)
        }
    
    def create_detection_summary(self, video_results: Dict) -> Dict:
        """Create comprehensive summary of video results"""
        try:
            summary = {
                'video_file': video_results.get('video_info', {}).get('filename', 'Unknown'),
                'processing_summary': {
                    'total_frames': video_results.get('processing_info', {}).get('processed_frames', 0),
                    'frames_with_detections': video_results.get('processing_info', {}).get('frames_with_detections', 0),
                    'detection_rate': 0,
                    'processing_time': video_results.get('processing_info', {}).get('processing_time', 0),
                    'fps_processed': video_results.get('processing_info', {}).get('fps_processed', 0)
                },
                'license_plate_summary': {
                    'total_plates_found': video_results.get('detections', {}).get('summary', {}).get('total_license_plates', 0),
                    'unique_plates': video_results.get('detections', {}).get('summary', {}).get('unique_plates', []),
                    'plates_per_frame': 0,
                    'most_common_plates': []
                },
                'output_files': video_results.get('output_files', {}),
                'recommendations': []
            }
            
            # Calculate detection rate
            total_frames = summary['processing_summary']['total_frames']
            frames_with_detections = summary['processing_summary']['frames_with_detections']
            
            if total_frames > 0:
                summary['processing_summary']['detection_rate'] = (frames_with_detections / total_frames) * 100
            
            # Calculate plates per frame
            unique_plates = summary['license_plate_summary']['unique_plates']
            if frames_with_detections > 0:
                summary['license_plate_summary']['plates_per_frame'] = len(unique_plates) / frames_with_detections
            
            # Find most common plates (would need frequency analysis)
            summary['license_plate_summary']['most_common_plates'] = unique_plates[:5]  # Top 5
            
            # Add recommendations
            if summary['processing_summary']['detection_rate'] < 10:
                summary['recommendations'].append("Low detection rate - consider using enhanced detection")
            
            if summary['processing_summary']['fps_processed'] < 10:
                summary['recommendations'].append("Slow processing - consider reducing video resolution")
            
            if len(unique_plates) == 0:
                summary['recommendations'].append("No license plates detected - check video quality")
            
            return summary
            
        except Exception as e:
            print(f"[ERROR] Failed to create summary: {e}")
            return {'error': str(e)}
    
    def get_processing_stats(self) -> Dict:
        """Get video processing statistics"""
        runtime = time.time() - self.start_time
        
        return {
            'total_videos_processed': self.processed_count,
            'runtime_seconds': runtime,
            'avg_time_per_video': runtime / max(1, self.processed_count),
            'current_video_info': self.current_video_info
        }


# Global instance for easy access
video_processor = VideoProcessor()

# Convenience functions
def process_single_video(video_path: str, **kwargs) -> Dict:
    """Process single video with default settings"""
    return video_processor.process_video(video_path, **kwargs)

def process_video_in_chunks(video_path: str, chunk_size: int = 30):
    """Process video in chunks for memory efficiency"""
    return video_processor.process_video_streaming(video_path, chunk_size)

def get_video_processing_stats() -> Dict:
    """Get video processing statistics"""
    return video_processor.get_processing_stats()


if __name__ == "__main__":
    print("📹 Video Processor Module")
    print("=" * 30)
    
    print("🧪 Testing video processor...")
    
    # Test video info extraction
    test_video_path = "test_video.mp4"  # This would be a real video file
    
    print("📖 Usage:")
    print("   from video_processor import process_single_video")
    print("   result = process_single_video('video.mp4')")
    print("   print(result['detections']['summary'])")
    
    print("\n✅ Video processor ready!")
    print("   Features:")
    print("   - Frame-by-frame processing")
    print("   - Memory-efficient streaming")
    print("   - Automatic video output saving")
    print("   - Progress tracking")
    print("   - Comprehensive summaries")
