"""
Model Calibration and Confidence Threshold Tuning Module
Optimizes detection thresholds for different camera setups and conditions
"""

import cv2
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CalibrationResult:
    """Results from model calibration"""
    camera_id: str
    zone_id: str
    optimal_threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float
    false_positive_rate: float
    false_negative_rate: float
    calibration_curve: List[Dict[str, float]]
    test_images_count: int
    timestamp: str

@dataclass
class CameraCalibration:
    """Camera-specific calibration settings"""
    camera_id: str
    zone_id: str
    confidence_threshold: float
    iou_threshold: float
    brightness_adjustment: float
    contrast_adjustment: float
    last_calibrated: str
    performance_metrics: Dict[str, float]

class ModelCalibrator:
    """Handles model calibration and threshold optimization"""
    
    def __init__(self, config_path: str = "parking_dataset/config/parking_zones.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.calibration_path = Path("parking_dataset/models/calibration")
        self.calibration_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing calibrations
        self.calibrations = self._load_calibrations()
        
    def _load_config(self) -> Dict:
        """Load parking configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
            
    def _load_calibrations(self) -> Dict[str, CameraCalibration]:
        """Load existing camera calibrations"""
        calibrations = {}
        calibration_file = self.calibration_path / "camera_calibrations.json"
        
        if calibration_file.exists():
            try:
                with open(calibration_file, 'r') as f:
                    data = json.load(f)
                    for cam_id, cal_data in data.items():
                        calibrations[cam_id] = CameraCalibration(**cal_data)
                logger.info(f"Loaded {len(calibrations)} camera calibrations")
            except Exception as e:
                logger.error(f"Failed to load calibrations: {e}")
                
        return calibrations
        
    def _save_calibrations(self):
        """Save camera calibrations to file"""
        calibration_file = self.calibration_path / "camera_calibrations.json"
        
        try:
            data = {cam_id: asdict(cal) for cam_id, cal in self.calibrations.items()}
            with open(calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Camera calibrations saved successfully")
        except Exception as e:
            logger.error(f"Failed to save calibrations: {e}")
            
    def calibrate_camera(self, camera_id: str, zone_id: str, 
                        test_images: List[str], ground_truth: List[Dict],
                        model_path: str = "yolov8n.pt") -> CalibrationResult:
        """
        Calibrate confidence threshold for a specific camera
        
        Args:
            camera_id: Camera identifier
            zone_id: Zone identifier  
            test_images: List of test image paths
            ground_truth: List of ground truth annotations for each image
            model_path: Path to YOLO model
        """
        logger.info(f"Calibrating camera {camera_id} in zone {zone_id}")
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Collect predictions and ground truth
            all_predictions = []
            all_ground_truths = []
            
            for img_path, gt_data in zip(test_images, ground_truth):
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"Cannot load image: {img_path}")
                    continue
                    
                # Get parking spot coordinates for this camera
                spot_coords = self._get_spot_coordinates(zone_id, camera_id)
                if not spot_coords:
                    logger.warning(f"No spot coordinates for {zone_id}_{camera_id}")
                    continue
                    
                # Run inference with different confidence thresholds
                results = model(image, conf=0.1, verbose=False)  # Low threshold to get all predictions
                
                # Process predictions
                predictions = self._process_predictions(results, spot_coords, image.shape)
                ground_truths = self._process_ground_truth(gt_data, spot_coords)
                
                all_predictions.extend(predictions)
                all_ground_truths.extend(ground_truths)
                
            if not all_predictions:
                raise ValueError("No valid predictions generated")
                
            # Find optimal threshold
            optimal_result = self._find_optimal_threshold(all_predictions, all_ground_truths)
            
            # Create calibration result
            calibration_result = CalibrationResult(
                camera_id=camera_id,
                zone_id=zone_id,
                optimal_threshold=optimal_result['threshold'],
                precision_at_threshold=optimal_result['precision'],
                recall_at_threshold=optimal_result['recall'],
                f1_at_threshold=optimal_result['f1'],
                false_positive_rate=optimal_result['fpr'],
                false_negative_rate=optimal_result['fnr'],
                calibration_curve=optimal_result['curve'],
                test_images_count=len(test_images),
                timestamp=datetime.now().isoformat()
            )
            
            # Save calibration
            self._save_calibration_result(calibration_result)
            
            # Update camera calibration
            self._update_camera_calibration(camera_id, zone_id, calibration_result)
            
            logger.info(f"Camera {camera_id} calibrated with optimal threshold: {optimal_result['threshold']:.3f}")
            return calibration_result
            
        except Exception as e:
            logger.error(f"Camera calibration failed: {e}")
            raise
            
    def _get_spot_coordinates(self, zone_id: str, camera_id: str) -> Dict[str, Tuple[int, int, int, int]]:
        """Get parking spot coordinates for specific camera"""
        try:
            if zone_id in self.config.get('zones', {}):
                zone_config = self.config['zones'][zone_id]
                if 'coordinates' in zone_config and camera_id in zone_config['coordinates']:
                    return zone_config['coordinates'][camera_id]['spots']
            return {}
        except Exception as e:
            logger.error(f"Failed to get spot coordinates: {e}")
            return {}
            
    def _process_predictions(self, results, spot_coords: Dict, image_shape: Tuple) -> List[Dict]:
        """Process YOLO predictions into parking spot format"""
        predictions = []
        
        try:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = result.names[cls]
                        
                        # Check which parking spot this detection belongs to
                        spot_id = self._find_overlapping_spot((x1, y1, x2, y2), spot_coords)
                        
                        if spot_id:
                            predictions.append({
                                'spot_id': spot_id,
                                'confidence': conf,
                                'predicted_class': 'OCCUPIED' if class_name in ['car', 'truck', 'bus', 'motorcycle'] else 'EMPTY',
                                'bbox': (int(x1), int(y1), int(x2), int(y2))
                            })
                            
        except Exception as e:
            logger.error(f"Prediction processing error: {e}")
            
        return predictions
        
    def _process_ground_truth(self, gt_data: Dict, spot_coords: Dict) -> List[Dict]:
        """Process ground truth annotations"""
        ground_truths = []
        
        try:
            for spot_id, spot_info in gt_data.items():
                if spot_id in spot_coords:
                    ground_truths.append({
                        'spot_id': spot_id,
                        'true_class': spot_info.get('status', 'EMPTY'),
                        'confidence': 1.0  # Ground truth has full confidence
                    })
        except Exception as e:
            logger.error(f"Ground truth processing error: {e}")
            
        return ground_truths
        
    def _find_overlapping_spot(self, bbox: Tuple, spot_coords: Dict) -> Optional[str]:
        """Find which parking spot overlaps with detection bbox"""
        max_overlap = 0
        best_spot = None
        
        for spot_id, spot_bbox in spot_coords.items():
            overlap = self._calculate_overlap_ratio(bbox, spot_bbox)
            if overlap > max_overlap and overlap > 0.3:  # Minimum 30% overlap
                max_overlap = overlap
                best_spot = spot_id
                
        return best_spot
        
    def _calculate_overlap_ratio(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_intersect = max(x1_1, x1_2)
        y1_intersect = max(y1_1, y1_2)
        x2_intersect = min(x2_1, x2_2)
        y2_intersect = min(y2_1, y2_2)
        
        if x2_intersect <= x1_intersect or y2_intersect <= y1_intersect:
            return 0.0
            
        intersection_area = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        return intersection_area / bbox2_area if bbox2_area > 0 else 0.0
        
    def _find_optimal_threshold(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """Find optimal confidence threshold using precision-recall curve"""
        try:
            # Match predictions with ground truth
            matched_data = self._match_predictions_with_gt(predictions, ground_truths)
            
            if not matched_data:
                raise ValueError("No matched predictions and ground truth")
                
            # Extract confidence scores and true labels
            confidences = np.array([item['confidence'] for item in matched_data])
            true_labels = np.array([1 if item['true_class'] == 'OCCUPIED' else 0 for item in matched_data])
            pred_labels = np.array([1 if item['predicted_class'] == 'OCCUPIED' else 0 for item in matched_data])
            
            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(true_labels, confidences)
            
            # Calculate F1 score for each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Find optimal threshold (max F1 score)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            # Calculate metrics at optimal threshold
            optimal_predictions = (confidences >= optimal_threshold).astype(int)
            
            tp = np.sum((optimal_predictions == 1) & (true_labels == 1))
            fp = np.sum((optimal_predictions == 1) & (true_labels == 0))
            fn = np.sum((optimal_predictions == 0) & (true_labels == 1))
            tn = np.sum((optimal_predictions == 0) & (true_labels == 0))
            
            precision_opt = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_opt = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_opt = 2 * (precision_opt * recall_opt) / (precision_opt + recall_opt + 1e-8)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Create calibration curve data
            curve_data = []
            for i, thresh in enumerate(thresholds[::10]):  # Sample every 10th point
                if i < len(precision[::10]):
                    curve_data.append({
                        'threshold': float(thresh),
                        'precision': float(precision[::10][i]),
                        'recall': float(recall[::10][i]),
                        'f1': float(f1_scores[::10][i]) if i < len(f1_scores[::10]) else 0
                    })
                    
            return {
                'threshold': float(optimal_threshold),
                'precision': float(precision_opt),
                'recall': float(recall_opt),
                'f1': float(f1_opt),
                'fpr': float(fpr),
                'fnr': float(fnr),
                'curve': curve_data
            }
            
        except Exception as e:
            logger.error(f"Optimal threshold finding failed: {e}")
            # Return default values
            return {
                'threshold': 0.85,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'fpr': 0.0,
                'fnr': 0.0,
                'curve': []
            }
            
    def _match_predictions_with_gt(self, predictions: List[Dict], ground_truths: List[Dict]) -> List[Dict]:
        """Match predictions with ground truth by spot ID"""
        matched = []
        
        # Create ground truth lookup
        gt_lookup = {gt['spot_id']: gt for gt in ground_truths}
        
        for pred in predictions:
            spot_id = pred['spot_id']
            if spot_id in gt_lookup:
                matched_item = {
                    'spot_id': spot_id,
                    'confidence': pred['confidence'],
                    'predicted_class': pred['predicted_class'],
                    'true_class': gt_lookup[spot_id]['true_class']
                }
                matched.append(matched_item)
                
        # Add empty spots that weren't detected
        detected_spots = {pred['spot_id'] for pred in predictions}
        for gt in ground_truths:
            if gt['spot_id'] not in detected_spots:
                matched_item = {
                    'spot_id': gt['spot_id'],
                    'confidence': 0.0,  # No detection
                    'predicted_class': 'EMPTY',
                    'true_class': gt['true_class']
                }
                matched.append(matched_item)
                
        return matched
        
    def _save_calibration_result(self, result: CalibrationResult):
        """Save calibration result to file"""
        try:
            filename = f"calibration_{result.camera_id}_{result.zone_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.calibration_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(asdict(result), f, indent=2)
                
            logger.info(f"Calibration result saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save calibration result: {e}")
            
    def _update_camera_calibration(self, camera_id: str, zone_id: str, result: CalibrationResult):
        """Update camera calibration settings"""
        try:
            calibration = CameraCalibration(
                camera_id=camera_id,
                zone_id=zone_id,
                confidence_threshold=result.optimal_threshold,
                iou_threshold=0.45,  # Default IOU threshold
                brightness_adjustment=0.0,
                contrast_adjustment=0.0,
                last_calibrated=result.timestamp,
                performance_metrics={
                    'precision': result.precision_at_threshold,
                    'recall': result.recall_at_threshold,
                    'f1': result.f1_at_threshold,
                    'fpr': result.false_positive_rate,
                    'fnr': result.false_negative_rate
                }
            )
            
            self.calibrations[camera_id] = calibration
            self._save_calibrations()
            
        except Exception as e:
            logger.error(f"Failed to update camera calibration: {e}")
            
    def get_optimal_threshold(self, camera_id: str) -> float:
        """Get optimal confidence threshold for camera"""
        if camera_id in self.calibrations:
            return self.calibrations[camera_id].confidence_threshold
        else:
            # Return default threshold
            return self.config.get('detection_config', {}).get('confidence_threshold', 0.85)
            
    def calibrate_all_cameras(self, test_data_dir: str, model_path: str = "yolov8n.pt") -> Dict[str, CalibrationResult]:
        """Calibrate all cameras in the system"""
        logger.info("Starting calibration for all cameras...")
        
        results = {}
        
        for zone_id, zone_config in self.config.get('zones', {}).items():
            for camera_id in zone_config.get('camera_ids', []):
                try:
                    # Load test data for this camera
                    test_images, ground_truth = self._load_camera_test_data(test_data_dir, camera_id, zone_id)
                    
                    if test_images:
                        result = self.calibrate_camera(camera_id, zone_id, test_images, ground_truth, model_path)
                        results[f"{camera_id}_{zone_id}"] = result
                    else:
                        logger.warning(f"No test data found for {camera_id}_{zone_id}")
                        
                except Exception as e:
                    logger.error(f"Failed to calibrate {camera_id}_{zone_id}: {e}")
                    
        logger.info(f"Calibration completed. Results for {len(results)} cameras.")
        return results
        
    def _load_camera_test_data(self, test_data_dir: str, camera_id: str, zone_id: str) -> Tuple[List[str], List[Dict]]:
        """Load test images and ground truth for a specific camera"""
        test_dir = Path(test_data_dir)
        
        # Find test images for this camera
        camera_pattern = f"*{camera_id}*.jpg"
        image_files = list(test_dir.glob(camera_pattern))
        
        if not image_files:
            logger.warning(f"No test images found for camera {camera_id}")
            return [], []
            
        test_images = [str(img) for img in image_files]
        ground_truth = []
        
        # Load corresponding ground truth files
        for img_path in image_files:
            gt_file = img_path.with_suffix('.json')
            if gt_file.exists():
                try:
                    with open(gt_file, 'r') as f:
                        gt_data = json.load(f)
                        ground_truth.append(gt_data)
                except Exception as e:
                    logger.error(f"Failed to load ground truth {gt_file}: {e}")
                    ground_truth.append({})  # Empty ground truth
            else:
                logger.warning(f"No ground truth file found for {img_path}")
                ground_truth.append({})  # Empty ground truth
                
        return test_images, ground_truth
        
    def generate_calibration_report(self) -> str:
        """Generate comprehensive calibration report"""
        try:
            report = [
                "# YOLO26 Model Calibration Report",
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Camera Calibrations Summary",
                ""
            ]
            
            if not self.calibrations:
                report.append("No camera calibrations available.")
                return "\n".join(report)
                
            # Summary statistics
            thresholds = [cal.confidence_threshold for cal in self.calibrations.values()]
            precisions = [cal.performance_metrics.get('precision', 0) for cal in self.calibrations.values()]
            recalls = [cal.performance_metrics.get('recall', 0) for cal in self.calibrations.values()]
            f1_scores = [cal.performance_metrics.get('f1', 0) for cal in self.calibrations.values()]
            
            report.extend([
                f"**Total Cameras Calibrated:** {len(self.calibrations)}",
                f"**Average Confidence Threshold:** {np.mean(thresholds):.3f}",
                f"**Average Precision:** {np.mean(precisions):.3f}",
                f"**Average Recall:** {np.mean(recalls):.3f}",
                f"**Average F1-Score:** {np.mean(f1_scores):.3f}",
                "",
                "## Individual Camera Results",
                ""
            ])
            
            # Individual camera details
            for camera_id, calibration in self.calibrations.items():
                metrics = calibration.performance_metrics
                report.extend([
                    f"### Camera: {camera_id} (Zone: {calibration.zone_id})",
                    f"- **Optimal Threshold:** {calibration.confidence_threshold:.3f}",
                    f"- **Precision:** {metrics.get('precision', 0):.3f}",
                    f"- **Recall:** {metrics.get('recall', 0):.3f}",
                    f"- **F1-Score:** {metrics.get('f1', 0):.3f}",
                    f"- **False Positive Rate:** {metrics.get('fpr', 0):.3f}",
                    f"- **False Negative Rate:** {metrics.get('fnr', 0):.3f}",
                    f"- **Last Calibrated:** {calibration.last_calibrated}",
                    ""
                ])
                
            # Recommendations
            report.extend([
                "## Recommendations",
                ""
            ])
            
            low_precision_cameras = [cam_id for cam_id, cal in self.calibrations.items() 
                                   if cal.performance_metrics.get('precision', 0) < 0.8]
            low_recall_cameras = [cam_id for cam_id, cal in self.calibrations.items() 
                                if cal.performance_metrics.get('recall', 0) < 0.8]
            
            if low_precision_cameras:
                report.append(f"**Low Precision Cameras:** {', '.join(low_precision_cameras)}")
                report.append("- Consider increasing confidence threshold")
                report.append("- Review training data quality")
                report.append("")
                
            if low_recall_cameras:
                report.append(f"**Low Recall Cameras:** {', '.join(low_recall_cameras)}")
                report.append("- Consider decreasing confidence threshold")
                report.append("- Add more diverse training examples")
                report.append("")
                
            if not low_precision_cameras and not low_recall_cameras:
                report.append("✅ All cameras are performing well within acceptable ranges.")
                report.append("- System ready for production deployment")
                
            # Save report
            report_text = "\n".join(report)
            report_file = self.calibration_path / f"calibration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(report_file, 'w') as f:
                f.write(report_text)
                
            logger.info(f"Calibration report saved: {report_file}")
            return report_text
            
        except Exception as e:
            logger.error(f"Failed to generate calibration report: {e}")
            return "Error generating calibration report"

def main():
    """Main function for model calibration"""
    print("=== YOLO26 Model Calibration System ===")
    
    calibrator = ModelCalibrator()
    
    # Example: Calibrate a single camera
    # test_images = ["test1.jpg", "test2.jpg"]  # Your test images
    # ground_truth = [{"A-01": {"status": "OCCUPIED"}}, {"A-01": {"status": "EMPTY"}}]  # Ground truth
    # result = calibrator.calibrate_camera("cam_01", "zone_a", test_images, ground_truth)
    
    # Generate calibration report
    report = calibrator.generate_calibration_report()
    print("\n" + report)
    
    print(f"\nCalibration files saved in: {calibrator.calibration_path}")

if __name__ == "__main__":
    main()
