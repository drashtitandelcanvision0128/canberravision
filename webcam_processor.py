"""
Webcam Processing Module for YOLO26
Handle all webcam/live camera operations
Real-time processing with performance optimization
"""

import cv2
import numpy as np
import time
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from queue import Queue
import json

# Import our modules
try:
    from image_processor import ImageProcessor, image_processor
    from simple_plate_detection import extract_license_plates_simple
    IMAGE_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Some modules not available: {e}")
    IMAGE_PROCESSOR_AVAILABLE = False

class WebcamProcessor:
    """
    Handle all webcam/live camera processing operations
    """
    
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.processing_thread = None
        self.frame_queue = Queue(maxsize=30)  # Buffer for 30 frames
        self.result_queue = Queue(maxsize=30)
        self.current_frame = None
        self.latest_result = None
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'start_time': None,
            'fps': 0,
            'total_detections': 0,
            'unique_plates': set(),
            'processing_times': []
        }
        
        print("[INFO] Webcam Processor initialized")
    
    def list_available_cameras(self) -> List[Dict]:
        """List all available cameras"""
        cameras = []
        
        # Try camera indices 0-9
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                cameras.append({
                    'index': i,
                    'name': f"Camera {i}",
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'available': True
                })
                cap.release()
            else:
                cameras.append({
                    'index': i,
                    'name': f"Camera {i}",
                    'available': False
                })
        
        return cameras
    
    def start_camera(self, camera_index: int = 0, resolution: Tuple[int, int] = (640, 480), 
                    fps: int = 30) -> Dict:
        """
        Start webcam capture
        
        Args:
            camera_index: Camera device index
            resolution: Camera resolution (width, height)
            fps: Target FPS
            
        Returns:
            Status dictionary
        """
        try:
            if self.is_running:
                return {'error': 'Camera already running'}
            
            # Open camera
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                return {'error': f'Failed to open camera {camera_index}'}
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            
            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            print(f"[INFO] Camera started: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Initialize statistics
            self.stats['start_time'] = time.time()
            self.stats['frames_processed'] = 0
            self.stats['unique_plates'] = set()
            self.stats['processing_times'] = []
            
            # Start processing thread
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            return {
                'success': True,
                'camera_index': camera_index,
                'resolution': f"{actual_width}x{actual_height}",
                'fps': actual_fps,
                'message': 'Camera started successfully'
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to start camera: {e}")
            return {'error': str(e)}
    
    def stop_camera(self) -> Dict:
        """Stop webcam capture"""
        try:
            if not self.is_running:
                return {'error': 'Camera not running'}
            
            # Stop processing
            self.is_running = False
            
            # Wait for thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2)
            
            # Release camera
            if self.camera:
                self.camera.release()
                self.camera = None
            
            # Clear queues
            while not self.frame_queue.empty():
                self.frame_queue.get()
            
            while not self.result_queue.empty():
                self.result_queue.get()
            
            print("[INFO] Camera stopped")
            
            return {
                'success': True,
                'message': 'Camera stopped successfully',
                'final_stats': self.get_current_stats()
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to stop camera: {e}")
            return {'error': str(e)}
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from camera"""
        try:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    self.current_frame = frame
                    return frame
            return None
        except Exception as e:
            print(f"[ERROR] Failed to get frame: {e}")
            return None
    
    def get_processed_frame(self) -> Optional[np.ndarray]:
        """Get latest processed frame with annotations"""
        try:
            if not self.result_queue.empty():
                result = self.result_queue.get()
                self.latest_result = result
                return result.get('annotated_frame')
            return None
        except Exception as e:
            print(f"[ERROR] Failed to get processed frame: {e}")
            return None
    
    def get_latest_result(self) -> Optional[Dict]:
        """Get latest processing result"""
        return self.latest_result
    
    def _processing_loop(self):
        """Main processing loop (runs in separate thread)"""
        last_fps_time = time.time()
        frame_count = 0
        
        while self.is_running:
            try:
                # Get frame
                frame = self.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                start_time = time.time()
                result = self._process_frame_realtime(frame)
                processing_time = time.time() - start_time
                
                # Update statistics
                self.stats['frames_processed'] += 1
                self.stats['processing_times'].append(processing_time)
                
                # Keep only last 100 processing times for average
                if len(self.stats['processing_times']) > 100:
                    self.stats['processing_times'].pop(0)
                
                # Add unique plates
                for plate in result.get('license_plates', []):
                    plate_text = plate.get('plate_text', '')
                    if plate_text:
                        self.stats['unique_plates'].add(plate_text)
                
                # Create annotated frame
                annotated_frame = self._create_annotated_frame(frame, result)
                result['annotated_frame'] = annotated_frame
                
                # Update FPS calculation
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:  # Update FPS every second
                    self.stats['fps'] = frame_count / (current_time - last_fps_time)
                    frame_count = 0
                    last_fps_time = current_time
                
                # Add to result queue (remove old if full)
                if self.result_queue.full():
                    self.result_queue.get()
                
                self.result_queue.put(result)
                
            except Exception as e:
                print(f"[ERROR] Processing loop error: {e}")
                time.sleep(0.01)
    
    def _process_frame_realtime(self, frame: np.ndarray) -> Dict:
        """Process frame for real-time display (optimized for speed)"""
        try:
            if IMAGE_PROCESSOR_AVAILABLE:
                # Use simplified processing for real-time
                result = image_processor.process_image(
                    frame, 
                    use_enhanced=False,  # Skip enhanced for speed
                    use_international=False  # Skip international for speed
                )
                
                # Add real-time specific info
                result['timestamp'] = datetime.now().isoformat()
                result['processing_mode'] = 'realtime'
                
                return result
            else:
                # Simple fallback
                return {
                    'timestamp': datetime.now().isoformat(),
                    'license_plates': [],
                    'objects': [],
                    'processing_mode': 'fallback'
                }
                
        except Exception as e:
            print(f"[ERROR] Real-time frame processing failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'license_plates': [],
                'objects': []
            }
    
    def _create_annotated_frame(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Create annotated frame for real-time display"""
        try:
            annotated = frame.copy()
            
            # Add timestamp
            timestamp = result.get('timestamp', datetime.now().isoformat())
            time_str = timestamp.split('T')[1][:8]  # HH:MM:SS
            cv2.putText(annotated, f"YOLO26 Live - {time_str}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add FPS
            fps_text = f"FPS: {self.stats['fps']:.1f}"
            cv2.putText(annotated, fps_text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw license plates
            plates = result.get('license_plates', [])
            for i, plate in enumerate(plates):
                if 'bounding_box' in plate and plate['bounding_box']:
                    x1, y1, x2, y2 = plate['bounding_box']
                    
                    # Green box for license plates
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add plate text
                    plate_text = plate.get('plate_text', 'Unknown')
                    cv2.putText(annotated, f"Plate {i+1}: {plate_text}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add detection count
            detection_text = f"Plates: {len(plates)}"
            cv2.putText(annotated, detection_text, 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add unique plates count
            unique_count = len(self.stats['unique_plates'])
            unique_text = f"Unique: {unique_count}"
            cv2.putText(annotated, unique_text, 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add processing mode
            mode = result.get('processing_mode', 'unknown')
            mode_text = f"Mode: {mode}"
            cv2.putText(annotated, mode_text, 
                       (10, annotated.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return annotated
            
        except Exception as e:
            print(f"[ERROR] Failed to create annotated frame: {e}")
            return frame
    
    def capture_snapshot(self, save_path: Optional[str] = None) -> Dict:
        """Capture current frame snapshot"""
        try:
            if self.current_frame is None:
                return {'error': 'No frame available'}
            
            # Generate filename if not provided
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"outputs/webcam_snapshot_{timestamp}.jpg"
            
            # Save frame
            success = cv2.imwrite(save_path, self.current_frame)
            
            if success:
                return {
                    'success': True,
                    'path': save_path,
                    'timestamp': datetime.now().isoformat(),
                    'message': 'Snapshot saved successfully'
                }
            else:
                return {'error': 'Failed to save snapshot'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def get_current_stats(self) -> Dict:
        """Get current processing statistics"""
        try:
            runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
            
            # Calculate average processing time
            if self.stats['processing_times']:
                avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            else:
                avg_processing_time = 0
            
            return {
                'is_running': self.is_running,
                'runtime_seconds': runtime,
                'frames_processed': self.stats['frames_processed'],
                'current_fps': self.stats['fps'],
                'avg_processing_time_ms': avg_processing_time * 1000,
                'total_detections': self.stats['total_detections'],
                'unique_plates_found': list(self.stats['unique_plates']),
                'unique_plates_count': len(self.stats['unique_plates']),
                'processing_queue_size': self.result_queue.qsize()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def export_session_data(self, filename: Optional[str] = None) -> Dict:
        """Export session data to JSON file"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"outputs/webcam_session_{timestamp}.json"
            
            session_data = {
                'session_info': {
                    'start_time': self.stats['start_time'],
                    'export_time': datetime.now().isoformat(),
                    'total_runtime': time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
                },
                'statistics': self.get_current_stats(),
                'detected_plates': list(self.stats['unique_plates']),
                'performance_metrics': {
                    'avg_processing_times': self.stats['processing_times'],
                    'fps_history': []  # Would need to implement FPS history
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            return {
                'success': True,
                'filename': filename,
                'message': 'Session data exported successfully'
            }
            
        except Exception as e:
            return {'error': str(e)}


# Global instance for easy access
webcam_processor = WebcamProcessor()

# Convenience functions
def start_webcam(camera_index: int = 0, **kwargs) -> Dict:
    """Start webcam with default settings"""
    return webcam_processor.start_camera(camera_index, **kwargs)

def stop_webcam() -> Dict:
    """Stop webcam"""
    return webcam_processor.stop_camera()

def get_webcam_frame() -> Optional[np.ndarray]:
    """Get current webcam frame"""
    return webcam_processor.get_processed_frame()

def get_webcam_stats() -> Dict:
    """Get webcam statistics"""
    return webcam_processor.get_current_stats()


if __name__ == "__main__":
    print("📷 Webcam Processor Module")
    print("=" * 30)
    
    # Test camera listing
    print("🔍 Scanning for cameras...")
    cameras = webcam_processor.list_available_cameras()
    
    available_cameras = [c for c in cameras if c.get('available', False)]
    print(f"Found {len(available_cameras)} available cameras")
    
    for camera in available_cameras:
        print(f"  - {camera['name']}: {camera['resolution']} @ {camera['fps']}fps")
    
    print("\n📖 Usage:")
    print("   from webcam_processor import start_webcam, stop_webcam, get_webcam_frame")
    print("   start_webcam(0)")
    print("   while True:")
    print("       frame = get_webcam_frame()")
    print("       if frame is not None:")
    print("           cv2.imshow('YOLO26 Live', frame)")
    print("           if cv2.waitKey(1) & 0xFF == ord('q'):")
    print("               break")
    print("   stop_webcam()")
    
    print("\n✅ Webcam processor ready!")
    print("   Features:")
    print("   - Multi-threaded processing")
    print("   - Real-time performance optimization")
    print("   - Live statistics")
    print("   - Session data export")
    print("   - Snapshot capture")
