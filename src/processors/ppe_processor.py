"""
PPE (Personal Protective Equipment) Processor Module
Handles image and video processing with PPE detection
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
import time
import os
from pathlib import Path
import json

try:
    from modules.ppe_detection import PPEDetector, PPEResult, get_ppe_detector
    PPE_AVAILABLE = True
except ImportError:
    PPE_AVAILABLE = False
    print("[WARNING] PPE detection module not available")


class PPEProcessor:
    """
    PPE Processor for images and videos
    Similar structure to parking detection processor
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", output_dir: str = "ppe_outputs"):
        """
        Initialize PPE Processor
        
        Args:
            model_path: Path to YOLO model
            output_dir: Directory for output files
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.detector = None
        if PPE_AVAILABLE:
            self.detector = get_ppe_detector(model_path)
        
        print(f"[INFO] PPE Processor initialized")
        print(f"[INFO] Output directory: {self.output_dir}")
        print(f"[INFO] PPE Detection available: {PPE_AVAILABLE}")
    
    def process_image(self, 
                     image: np.ndarray,
                     confidence_threshold: float = 0.3,
                     show_labels: bool = True,
                     show_confidence: bool = True,
                     save_output: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Process a single image for PPE detection
        
        Args:
            image: Input image (BGR or RGB)
            confidence_threshold: Detection confidence threshold
            show_labels: Whether to show text labels
            show_confidence: Whether to show confidence scores
            save_output: Whether to save processed image
            
        Returns:
            Tuple of (annotated_image, results_dict)
        """
        if not PPE_AVAILABLE or self.detector is None:
            raise RuntimeError("PPE detection module not available")
        
        # Ensure image is numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
            if image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        start_time = time.time()
        timestamp = int(time.time())
        
        print(f"[INFO] Processing image for PPE detection...")
        
        # Perform PPE detection
        result = self.detector.detect(
            image,
            confidence_threshold=confidence_threshold
        )
        
        # Create annotated image
        annotated_image = self.detector.visualize(
            image,
            result,
            show_labels=show_labels,
            show_confidence=show_confidence
        )
        
        # Convert results to dictionary
        results_dict = self.detector.to_dict(result)
        results_dict['total_time'] = time.time() - start_time
        
        # Save output if requested
        if save_output:
            output_path = self._save_image(annotated_image, timestamp)
            results_dict['output_path'] = output_path
        
        print(f"[INFO] PPE detection completed: {result.compliant} compliant, {result.non_compliant} non-compliant")
        
        return annotated_image, results_dict
    
    def process_video(self,
                     video_path: str,
                     confidence_threshold: float = 0.3,
                     show_labels: bool = True,
                     show_confidence: bool = True,
                     every_n_frames: int = 5,
                     save_output: bool = True) -> Tuple[str, Dict]:
        """
        Process a video file for PPE detection
        
        Args:
            video_path: Path to input video file
            confidence_threshold: Detection confidence threshold
            show_labels: Whether to show text labels
            show_confidence: Whether to show confidence scores
            every_n_frames: Process every N frames
            save_output: Whether to save processed video
            
        Returns:
            Tuple of (output_video_path, results_summary)
        """
        if not PPE_AVAILABLE or self.detector is None:
            raise RuntimeError("PPE detection module not available")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Processing video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
        
        # Setup output video writer
        output_path = None
        out = None
        if save_output:
            timestamp = int(time.time())
            output_path = self.output_dir / f"ppe_video_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps / every_n_frames, (width, height))
        
        frame_count = 0
        processed_count = 0
        all_results = []
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every N frames
            if frame_count % every_n_frames != 0:
                continue
            
            # Perform PPE detection
            result = self.detector.detect(
                frame,
                confidence_threshold=confidence_threshold
            )
            
            # Create annotated frame
            annotated_frame = self.detector.visualize(
                frame,
                result,
                show_labels=show_labels,
                show_confidence=show_confidence
            )
            
            # Save to output video
            if out:
                out.write(annotated_frame)
            
            # Store results
            all_results.append(self.detector.to_dict(result))
            processed_count += 1
            
            # Print progress
            if processed_count % 10 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"[INFO] Processed {processed_count} frames ({progress:.1f}%)")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = self._generate_video_summary(all_results, total_time, processed_count)
        summary['output_path'] = str(output_path) if output_path else None
        summary['input_fps'] = fps
        summary['total_frames'] = total_frames
        summary['processed_frames'] = processed_count
        
        print(f"[INFO] Video processing completed in {total_time:.2f}s")
        print(f"[INFO] Output saved to: {output_path}")
        
        return str(output_path) if output_path else None, summary
    
    def process_webcam_frame(self,
                            frame: np.ndarray,
                            confidence_threshold: float = 0.3,
                            show_labels: bool = True,
                            show_confidence: bool = True) -> Tuple[np.ndarray, str]:
        """
        Process a single webcam frame for PPE detection
        
        Args:
            frame: Input frame from webcam
            confidence_threshold: Detection confidence threshold
            show_labels: Whether to show text labels
            show_confidence: Whether to show confidence scores
            
        Returns:
            Tuple of (annotated_frame, info_text)
        """
        if not PPE_AVAILABLE or self.detector is None:
            # Return original frame with warning text
            warning_frame = frame.copy()
            cv2.putText(warning_frame, "PPE Detection Not Available", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return warning_frame, "PPE detection module not available"
        
        # Perform PPE detection
        result = self.detector.detect(
            frame,
            confidence_threshold=confidence_threshold
        )
        
        # Create annotated frame
        annotated_frame = self.detector.visualize(
            frame,
            result,
            show_labels=show_labels,
            show_confidence=show_confidence
        )
        
        # Generate comprehensive info text with both helmet and seatbelt information
        info_lines = [
            f"📹 PPE Detection - Live",
            f"👥 Persons: {result.total_persons}",
            f"🪖 Helmets: {result.helmet_detected}",
            f"🚗 Seatbelts: {result.seatbelt_detected}",
            f"✅ Compliant: {sum(1 for p in result.persons if p.status == 'compliant')}",
            f"❌ Violations: {sum(1 for p in result.persons if p.status == 'violation')}",
            f"⏱️ Processing: {result.processing_time*1000:.1f}ms",
        ]
        
        # Add detailed person information with both helmet and seatbelt status
        for person in result.persons:
            status = "✅" if person.status == 'compliant' else "❌"
            helmet_status = "H" if person.helmet.present else "-"
            belt_status = "S" if person.seatbelt.present else "-"
            vehicle = person.vehicle_type.upper()[:3] if person.vehicle_type != "unknown" else "???"
            
            # Show both helmet and seatbelt information for each person
            if person.vehicle_type == "2-wheeler":
                info_lines.append(f"  P{person.person_id[1:]} [{vehicle}]: {status} [H:{helmet_status} S:{belt_status}] - Helmet: {'YES' if person.helmet.present else 'NO'}")
            elif person.vehicle_type == "4-wheeler":
                info_lines.append(f"  P{person.person_id[1:]} [{vehicle}]: {status} [H:{helmet_status} S:{belt_status}] - Seatbelt: {'YES' if person.seatbelt.present else 'NO'}")
            else:
                info_lines.append(f"  P{person.person_id[1:]} [{vehicle}]: {status} [H:{helmet_status} S:{belt_status}]")
        
        info_text = "\n".join(info_lines)
        
        return annotated_frame, info_text
    
    def _save_image(self, image: np.ndarray, timestamp: int) -> str:
        """Save processed image to output directory"""
        filename = f"ppe_detection_{timestamp}.jpg"
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), image)
        print(f"[INFO] Saved: {output_path}")
        return str(output_path)
    
    def _generate_video_summary(self, all_results: List[Dict], total_time: float, frame_count: int) -> Dict:
        """Generate summary statistics for video processing"""
        if not all_results:
            return {
                'total_persons': 0,
                'helmet_detected': 0,
                'seatbelt_detected': 0,
                'compliant': 0,
                'violations': 0,
                'processing_time': total_time,
                'frame_count': frame_count
            }
        
        total_persons = sum(r['totalPersons'] for r in all_results)
        total_helmets = sum(r.get('helmetDetected', 0) for r in all_results)
        total_seatbelts = sum(r.get('seatbeltDetected', 0) for r in all_results)
        total_compliant = sum(1 for r in all_results for p in r.get('persons', []) if p.get('status') == 'compliant')
        total_violations = sum(1 for r in all_results for p in r.get('persons', []) if p.get('status') == 'violation')
        
        frames_with_persons = sum(1 for r in all_results if r['totalPersons'] > 0)
        
        return {
            'total_persons_detected': total_persons,
            'avg_persons_per_frame': total_persons / frame_count if frame_count > 0 else 0,
            'total_helmet_detections': total_helmets,
            'total_seatbelt_detections': total_seatbelts,
            'total_compliant_detections': total_compliant,
            'total_violation_detections': total_violations,
            'avg_compliant_per_frame': total_compliant / frame_count if frame_count > 0 else 0,
            'avg_violations_per_frame': total_violations / frame_count if frame_count > 0 else 0,
            'frames_with_persons': frames_with_persons,
            'frames_without_persons': frame_count - frames_with_persons,
            'processing_time': total_time,
            'avg_time_per_frame': total_time / frame_count if frame_count > 0 else 0,
            'frame_count': frame_count
        }
    
    def get_detector_info(self) -> Dict:
        """Get information about the PPE detector"""
        if self.detector:
            return self.detector.get_stats()
        return {'available': False}


# Global processor instance
_ppe_processor = None


def get_ppe_processor(model_path: str = "yolov8n.pt") -> PPEProcessor:
    """Get or create global PPE processor instance"""
    global _ppe_processor
    if _ppe_processor is None:
        _ppe_processor = PPEProcessor(model_path)
    return _ppe_processor


if __name__ == "__main__":
    print("[INFO] PPE Processor module ready")
    print(f"[INFO] PPE_AVAILABLE: {PPE_AVAILABLE}")
