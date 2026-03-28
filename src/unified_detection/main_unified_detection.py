"""
Main Unified Detection System - Entry Point
Canberra Vision - Multi-Detection Engine

This module provides the main entry point for unified multi-detection processing
supporting webcam, video files, and static images.

Detection Pipeline:
Input (Image / Video / Webcam)
        ↓
Frame Extractor
        ↓
Unified Detection Engine
   ├── PPE Detection
   ├── Vehicle Detection
   ├── Number Plate (ANPR)
   ├── Parking Detection
        ↓
Result Formatter (JSON)
        ↓
Database Service (PostgreSQL)
"""

import cv2
import numpy as np
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Union

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.unified_detection.frame_extractor import (
    FrameExtractor, InputSourceType,
    webcam, video_file, image_file
)
from src.unified_detection.unified_detector import (
    UnifiedDetector, get_unified_detector
)
from src.unified_detection.result_formatter import (
    ResultFormatter, format_single_result
)
from src.unified_detection.database_service import (
    DatabaseService, get_database_service
)


class UnifiedDetectionSystem:
    """
    Main Unified Detection System
    Coordinates all components for multi-detection processing
    """
    
    def __init__(self,
                 model_path: str = "yolo26n.pt",
                 use_gpu: bool = True,
                 enable_display: bool = True,
                 enable_database: bool = True,
                 output_dir: str = "outputs/unified"):
        """
        Initialize Unified Detection System
        
        Args:
            model_path: Path to YOLO model
            use_gpu: Whether to use GPU acceleration
            enable_display: Whether to show real-time display
            enable_database: Whether to save results to database
            output_dir: Directory for output files
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.enable_display = enable_display
        self.enable_database = enable_database
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.detector = None
        self.formatter = None
        self.database = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        print("=" * 60)
        print("CANBERRA VISION - UNIFIED DETECTION SYSTEM")
        print("=" * 60)
        
        # Initialize detector
        print("\n[1/3] Initializing Unified Detection Engine...")
        self.detector = get_unified_detector(
            model_path=self.model_path,
            use_gpu=self.use_gpu
        )
        
        # Initialize formatter
        print("\n[2/3] Initializing Result Formatter...")
        self.formatter = ResultFormatter()
        
        # Initialize database if enabled
        if self.enable_database:
            print("\n[3/3] Initializing Database Service...")
            self.database = get_database_service()
        else:
            print("\n[3/3] Database Service disabled")
        
        print("\n" + "=" * 60)
        print("SYSTEM READY")
        print("=" * 60)
    
    def process_webcam(self, 
                       camera_index: int = 0,
                       display_size: tuple = (1280, 720)):
        """
        Process live webcam feed
        
        Args:
            camera_index: Camera device index
            display_size: Size for display window
        """
        print(f"\n[INFO] Starting webcam processing (Camera {camera_index})")
        print("[INFO] Press 'q' to quit, 's' to save current frame\n")
        
        # Reset trackers for new source
        self.detector.reset_trackers()
        
        # Create frame extractor
        with FrameExtractor(camera_index, InputSourceType.WEBCAM) as extractor:
            if not extractor.is_opened:
                print("[ERROR] Failed to open webcam")
                return
            
            frame_count = 0
            
            while True:
                ret, frame = extractor.get_frame()
                
                if not ret:
                    print("[WARNING] Failed to get frame from webcam")
                    break
                
                # Process frame
                result = self.detector.detect_frame(
                    frame, 
                    frame_number=frame_count,
                    source="WEBCAM"
                )
                
                # Save to database
                if self.enable_database and self.database:
                    self.database.save_detection(result, source_id=f"webcam_{camera_index}")
                
                # Create annotated frame
                annotated = self._create_annotated_frame(frame, result)
                
                # Resize for display
                if display_size:
                    annotated = cv2.resize(annotated, display_size)
                
                # Show display
                if self.enable_display:
                    cv2.imshow('Canberra Vision - Unified Detection', annotated)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] Stopped by user")
                        break
                    elif key == ord('s'):
                        self._save_frame(annotated, result, frame_count)
                
                # Print detection summary every 30 frames
                if frame_count % 30 == 0:
                    self._print_frame_summary(result, frame_count)
                
                frame_count += 1
        
        cv2.destroyAllWindows()
        print(f"\n[INFO] Webcam processing completed. Total frames: {frame_count}")
    
    def process_video(self,
                      video_path: str,
                      output_video: Optional[str] = None,
                      skip_frames: int = 0,
                      show_progress: bool = True):
        """
        Process video file
        
        Args:
            video_path: Path to input video
            output_video: Path for output video (auto-generated if None)
            skip_frames: Number of frames to skip between processing
            show_progress: Whether to show progress bar
        """
        print(f"\n[INFO] Processing video: {video_path}")
        
        # Reset trackers for new source
        self.detector.reset_trackers()
        
        # Create frame extractor
        with FrameExtractor(video_path, InputSourceType.VIDEO) as extractor:
            if not extractor.is_opened:
                print("[ERROR] Failed to open video")
                return
            
            props = extractor.get_properties()
            print(f"[INFO] Video properties: {props['width']}x{props['height']} @ {props['fps']:.1f} FPS")
            
            # Setup output video writer
            if output_video is None:
                timestamp = int(time.time())
                output_video = self.output_dir / f"unified_output_{timestamp}.mp4"
            else:
                output_video = Path(output_video)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_video),
                fourcc,
                props['fps'] / (skip_frames + 1),
                (props['width'], props['height'])
            )
            
            # Process frames
            frame_count = 0
            processed_count = 0
            
            for frame_num, frame in extractor.get_frames(skip_frames=skip_frames):
                # Process frame
                result = self.detector.detect_frame(
                    frame,
                    frame_number=frame_num,
                    source="VIDEO"
                )
                
                # Save to database
                if self.enable_database and self.database:
                    self.database.save_detection(
                        result, 
                        source_id=video_path
                    )
                
                # Create annotated frame
                annotated = self._create_annotated_frame(frame, result)
                
                # Write to output
                out.write(annotated)
                
                # Show progress
                if show_progress and frame_num % 30 == 0:
                    progress = (frame_num / props['total_frames']) * 100 if props['total_frames'] > 0 else 0
                    print(f"[INFO] Progress: {progress:.1f}% (Frame {frame_num})")
                
                processed_count += 1
            
            out.release()
            
            print(f"\n[INFO] Video processing completed")
            print(f"[INFO] Output saved: {output_video}")
            print(f"[INFO] Total frames processed: {processed_count}")
    
    def process_image(self,
                      image_path: str,
                      save_output: bool = True,
                      display_result: bool = True):
        """
        Process single image
        
        Args:
            image_path: Path to input image
            save_output: Whether to save annotated image
            display_result: Whether to display result
        """
        print(f"\n[INFO] Processing image: {image_path}")
        
        # Reset trackers for new source
        self.detector.reset_trackers()
        
        # Create frame extractor
        with FrameExtractor(image_path, InputSourceType.IMAGE) as extractor:
            if not extractor.is_opened:
                print("[ERROR] Failed to open image")
                return
            
            ret, frame = extractor.get_frame()
            
            if not ret:
                print("[ERROR] Failed to read image")
                return
            
            # Process frame
            result = self.detector.detect_frame(
                frame,
                frame_number=0,
                source="IMAGE"
            )
            
            # Save to database
            if self.enable_database and self.database:
                self.database.save_detection(result, source_id=image_path)
            
            # Create annotated frame
            annotated = self._create_annotated_frame(frame, result)
            
            # Save output
            if save_output:
                timestamp = int(time.time())
                output_path = self.output_dir / f"unified_image_{timestamp}.jpg"
                cv2.imwrite(str(output_path), annotated)
                print(f"[INFO] Output saved: {output_path}")
            
            # Display result
            if display_result and self.enable_display:
                # Resize for display if too large
                display_frame = annotated.copy()
                if display_frame.shape[1] > 1280:
                    scale = 1280 / display_frame.shape[1]
                    new_width = int(display_frame.shape[1] * scale)
                    new_height = int(display_frame.shape[0] * scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))
                
                cv2.imshow('Canberra Vision - Unified Detection', display_frame)
                print("[INFO] Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Print and save JSON result
            json_output = self.formatter.to_json(result)
            print("\n[DETECTION RESULT]")
            print(json_output)
            
            # Save JSON
            json_path = self.output_dir / f"unified_result_{int(time.time())}.json"
            with open(json_path, 'w') as f:
                f.write(json_output)
            print(f"\n[INFO] JSON saved: {json_path}")
    
    def _create_annotated_frame(self, frame: np.ndarray, 
                                result) -> np.ndarray:
        """Create annotated frame with all detections visualized"""
        annotated = frame.copy()
        
        # Draw vehicles
        for vehicle in result.vehicle_detections:
            x1, y1, x2, y2 = map(int, vehicle.bbox)
            
            # Color based on vehicle type
            if vehicle.vehicle_type == 'bike':
                color = (0, 255, 255)  # Cyan for 2-wheelers
            elif vehicle.vehicle_type == 'car':
                color = (0, 255, 0)    # Green for cars
            elif vehicle.vehicle_type == 'truck':
                color = (255, 0, 0)   # Blue for trucks
            elif vehicle.vehicle_type == 'bus':
                color = (255, 255, 0) # Yellow for buses
            else:
                color = (128, 128, 128)  # Gray for unknown
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{vehicle.vehicle_type} | {vehicle.color} | {vehicle.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw vehicle ID
            cv2.putText(annotated, vehicle.vehicle_id, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw PPE detections
        for ppe in result.ppe_detections:
            x1, y1, x2, y2 = map(int, ppe.bbox)
            
            # Determine status color
            if ppe.vehicle_type == 'bike' and not ppe.helmet:
                color = (0, 0, 255)  # Red - violation
            elif ppe.vehicle_type in ['car', 'truck', 'bus'] and not ppe.seatbelt:
                color = (0, 0, 255)  # Red - violation
            else:
                color = (0, 255, 0)  # Green - compliant
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw PPE status
            status_parts = []
            if ppe.helmet:
                status_parts.append("H")
            if ppe.seatbelt:
                status_parts.append("S")
            if ppe.vest:
                status_parts.append("V")
            
            status = f"[{ppe.vehicle_type[:3].upper()}] " + "/".join(status_parts) if status_parts else "[NO PPE]"
            
            cv2.putText(annotated, status, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw person ID
            cv2.putText(annotated, ppe.person_id, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw number plates
        for plate in result.plate_detections:
            x1, y1, x2, y2 = map(int, plate.bbox)
            
            # Draw plate bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            # Draw plate text
            label = f"PLATE: {plate.text}"
            cv2.rectangle(annotated, (x1, y1 - 25), (x1 + 200, y1), (255, 0, 255), -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw parking slots
        for slot in result.parking_detections:
            x1, y1, x2, y2 = map(int, slot.bbox)
            
            # Color based on occupancy
            if slot.occupied:
                color = (0, 0, 255)  # Red - occupied
                status = "OCCUPIED"
            else:
                color = (0, 255, 0)  # Green - empty
                status = "EMPTY"
            
            # Draw slot boundary
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw slot info
            label = f"Slot {slot.slot_id}: {status}"
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw summary panel
        self._draw_summary_panel(annotated, result)
        
        return annotated
    
    def _draw_summary_panel(self, frame: np.ndarray, result):
        """Draw detection summary panel on frame"""
        # Create overlay for panel
        panel_height = 120
        panel_width = 350
        overlay = frame.copy()
        
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw summary text
        y_offset = 35
        line_height = 25
        
        # Title
        cv2.putText(frame, "CANBERRA VISION - UNIFIED DETECTION", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += line_height
        
        # Vehicle count
        vehicle_text = f"Vehicles: {len(result.vehicle_detections)}"
        cv2.putText(frame, vehicle_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # PPE count
        violations = sum(
            1 for p in result.ppe_detections
            if (p.vehicle_type == 'bike' and not p.helmet) or
               (p.vehicle_type in ['car', 'truck', 'bus'] and not p.seatbelt)
        )
        ppe_text = f"Persons: {len(result.ppe_detections)} | Violations: {violations}"
        cv2.putText(frame, ppe_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Plate and parking count
        plate_text = f"Plates: {len(result.plate_detections)} | Parking: {len(result.parking_detections)}"
        cv2.putText(frame, plate_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Processing time
        time_text = f"Processing: {result.processing_time_ms:.1f}ms"
        cv2.putText(frame, time_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _print_frame_summary(self, result, frame_count: int):
        """Print summary of frame detections"""
        print(f"\n[FRAME {frame_count} SUMMARY]")
        print(f"  Vehicles: {len(result.vehicle_detections)}")
        print(f"  Persons: {len(result.ppe_detections)}")
        print(f"  Plates: {len(result.plate_detections)}")
        print(f"  Parking Slots: {len(result.parking_detections)}")
        print(f"  Processing Time: {result.processing_time_ms:.1f}ms")
        
        # Print violations
        violations = [
            p for p in result.ppe_detections
            if (p.vehicle_type == 'bike' and not p.helmet) or
               (p.vehicle_type in ['car', 'truck', 'bus'] and not p.seatbelt)
        ]
        if violations:
            print(f"  ⚠️  VIOLATIONS: {len(violations)}")
    
    def _save_frame(self, frame: np.ndarray, result, frame_number: int):
        """Save current frame with detections"""
        timestamp = int(time.time())
        frame_path = self.output_dir / f"frame_{frame_number}_{timestamp}.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        # Also save JSON
        json_path = self.output_dir / f"frame_{frame_number}_{timestamp}.json"
        with open(json_path, 'w') as f:
            f.write(self.formatter.to_json(result))
        
        print(f"[INFO] Frame saved: {frame_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Canberra Vision - Unified Multi-Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process webcam
  python main_unified_detection.py --webcam
  
  # Process video file
  python main_unified_detection.py --video path/to/video.mp4
  
  # Process image
  python main_unified_detection.py --image path/to/image.jpg
  
  # Process without database
  python main_unified_detection.py --webcam --no-database
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--webcam', action='store_true',
                            help='Process live webcam feed')
    input_group.add_argument('--video', type=str, metavar='PATH',
                            help='Process video file')
    input_group.add_argument('--image', type=str, metavar='PATH',
                            help='Process static image')
    
    # Model options
    parser.add_argument('--model', type=str, default='yolo26n.pt',
                       help='Path to YOLO model (default: yolo26n.pt)')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    
    # Display options
    parser.add_argument('--no-display', action='store_true',
                       help='Disable real-time display')
    parser.add_argument('--no-database', action='store_true',
                       help='Disable database storage')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='outputs/unified',
                       help='Output directory for results')
    
    # Webcam options
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index for webcam (default: 0)')
    
    # Video options
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Skip N frames between processing (default: 0)')
    parser.add_argument('--output-video', type=str,
                       help='Output video path (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Initialize system
    system = UnifiedDetectionSystem(
        model_path=args.model,
        use_gpu=not args.cpu,
        enable_display=not args.no_display,
        enable_database=not args.no_database,
        output_dir=args.output_dir
    )
    
    # Process based on input type
    if args.webcam:
        system.process_webcam(camera_index=args.camera)
    elif args.video:
        system.process_video(
            video_path=args.video,
            output_video=args.output_video,
            skip_frames=args.skip_frames
        )
    elif args.image:
        system.process_image(
            image_path=args.image,
            save_output=True,
            display_result=not args.no_display
        )


if __name__ == "__main__":
    main()
