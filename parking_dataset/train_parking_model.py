"""
YOLOv8 Training Script for Parking Space Occupancy Detection
Trains a custom model to detect occupied and empty parking spaces
"""

import os
import yaml
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class ParkingModelTrainer:
    """Handles training of YOLOv8 model for parking space detection"""
    
    def __init__(self, dataset_path: str = "parking_dataset"):
        self.dataset_path = Path(dataset_path)
        self.models_path = self.dataset_path / "models"
        self.results_path = self.dataset_path / "results"
        self.logs_path = self.dataset_path / "logs"
        
        # Ensure directories exist
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.config = self._load_training_config()
        
    def _load_training_config(self) -> Dict:
        """Load training configuration"""
        config_path = self.dataset_path / "config" / "parking_zones.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                return full_config.get('training_config', {})
        else:
            # Default configuration
            return {
                'model_name': 'yolov8n.pt',
                'epochs': 100,
                'batch_size': 16,
                'imgsz': 640,
                'lr0': 0.001,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'patience': 20,
                'save_period': 10,
                'val_confidence': 0.5,
                'test_confidence': 0.85
            }
            
    def prepare_dataset(self):
        """Prepare and validate dataset for training"""
        print("[INFO] Preparing dataset for training...")
        
        # Check dataset structure
        required_dirs = [
            self.dataset_path / "images" / "train",
            self.dataset_path / "images" / "val",
            self.dataset_path / "labels" / "train",
            self.dataset_path / "labels" / "val"
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                print(f"[ERROR] Missing directory: {directory}")
                return False
                
        # Count dataset files
        train_images = list((self.dataset_path / "images" / "train").glob("*.jpg"))
        train_labels = list((self.dataset_path / "labels" / "train").glob("*.txt"))
        val_images = list((self.dataset_path / "images" / "val").glob("*.jpg"))
        val_labels = list((self.dataset_path / "labels" / "val").glob("*.txt"))
        
        print(f"[INFO] Dataset summary:")
        print(f"  Training: {len(train_images)} images, {len(train_labels)} labels")
        print(f"  Validation: {len(val_images)} images, {len(val_labels)} labels")
        
        # Validate dataset balance
        self._analyze_dataset_balance(train_labels, val_labels)
        
        # Create dataset.yaml if not exists
        self._create_dataset_yaml()
        
        return len(train_images) > 0 and len(val_images) > 0
        
    def _analyze_dataset_balance(self, train_labels: List[Path], val_labels: List[Path]):
        """Analyze class distribution in dataset"""
        train_counts = {0: 0, 1: 0}  # EMPTY, OCCUPIED
        val_counts = {0: 0, 1: 0}
        
        # Analyze training labels
        for label_path in train_labels:
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip():
                            class_id = int(line.split()[0])
                            train_counts[class_id] += 1
            except Exception as e:
                print(f"[WARNING] Error reading {label_path}: {e}")
                
        # Analyze validation labels
        for label_path in val_labels:
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip():
                            class_id = int(line.split()[0])
                            val_counts[class_id] += 1
            except Exception as e:
                print(f"[WARNING] Error reading {label_path}: {e}")
                
        print(f"[INFO] Class distribution:")
        print(f"  Training - EMPTY: {train_counts[0]}, OCCUPIED: {train_counts[1]}")
        print(f"  Validation - EMPTY: {val_counts[0]}, OCCUPIED: {val_counts[1]}")
        
        # Save analysis
        analysis = {
            'training': {'empty': train_counts[0], 'occupied': train_counts[1]},
            'validation': {'empty': val_counts[0], 'occupied': val_counts[1]},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.results_path / "dataset_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
            
    def _create_dataset_yaml(self):
        """Create dataset.yaml for YOLO training"""
        dataset_config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 2,  # Number of classes
            'names': {
                0: 'EMPTY',
                1: 'OCCUPIED'
            }
        }
        
        yaml_path = self.dataset_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
            
        print(f"[INFO] Created dataset configuration: {yaml_path}")
        
    def train_model(self, model_name: str = None, resume: bool = False) -> Optional[str]:
        """Train YOLOv8 model for parking detection"""
        if not self.prepare_dataset():
            print("[ERROR] Dataset preparation failed")
            return None
            
        model_name = model_name or self.config.get('model_name', 'yolov8n.pt')
        
        try:
            print(f"[INFO] Starting model training with {model_name}")
            
            # Load model
            model = YOLO(model_name)
            
            # Training parameters
            training_params = {
                'data': str(self.dataset_path / "dataset.yaml"),
                'epochs': self.config.get('epochs', 100),
                'batch': self.config.get('batch_size', 16),
                'imgsz': self.config.get('imgsz', 640),
                'lr0': self.config.get('lr0', 0.001),
                'device': self.config.get('device', 'auto'),
                'patience': self.config.get('patience', 20),
                'save_period': self.config.get('save_period', 10),
                'project': str(self.models_path),
                'name': f'parking_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'exist_ok': True,
                'verbose': True,
                'plots': True,
                'save': True
            }
            
            # Start training
            print(f"[INFO] Training parameters: {training_params}")
            
            if resume:
                # Resume from last checkpoint
                results = model.train(resume=True)
            else:
                results = model.train(**training_params)
                
            # Get the best model path
            best_model_path = results.save_dir / "weights" / "best.pt"
            
            if best_model_path.exists():
                print(f"[SUCCESS] Training completed! Best model saved at: {best_model_path}")
                
                # Save training summary
                self._save_training_summary(results, best_model_path)
                
                return str(best_model_path)
            else:
                print("[ERROR] Best model not found after training")
                return None
                
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            return None
            
    def _save_training_summary(self, results, model_path: Path):
        """Save training summary and metrics"""
        try:
            summary = {
                'model_path': str(model_path),
                'training_time': str(results.trainer.time),
                'best_fitness': results.results_dict.get('metrics/mAP50-95(B)', 0),
                'final_epoch': results.epoch,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save summary
            summary_path = self.results_path / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            print(f"[INFO] Training summary saved: {summary_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to save training summary: {e}")
            
    def evaluate_model(self, model_path: str, confidence_threshold: float = 0.5) -> Dict:
        """Evaluate trained model performance"""
        try:
            print(f"[INFO] Evaluating model: {model_path}")
            
            # Load trained model
            model = YOLO(model_path)
            
            # Run validation
            results = model.val(
                data=str(self.dataset_path / "dataset.yaml"),
                conf=confidence_threshold,
                split='val',
                device=self.config.get('device', 'auto')
            )
            
            # Extract metrics
            metrics = {
                'map50': results.box.map50,
                'map50_95': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr,
                'f1': 2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0,
                'confidence_threshold': confidence_threshold,
                'model_path': model_path,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"[INFO] Evaluation results:")
            print(f"  mAP@0.5: {metrics['map50']:.4f}")
            print(f"  mAP@0.5:0.95: {metrics['map50_95']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            
            # Save evaluation results
            eval_path = self.results_path / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(eval_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            return metrics
            
        except Exception as e:
            print(f"[ERROR] Model evaluation failed: {e}")
            return {}
            
    def test_model_on_images(self, model_path: str, test_images: List[str], 
                           confidence_threshold: float = 0.85) -> Dict:
        """Test model on specific images and generate detailed analysis"""
        try:
            model = YOLO(model_path)
            
            results_summary = {
                'total_images': len(test_images),
                'total_detections': 0,
                'occupied_detections': 0,
                'empty_detections': 0,
                'avg_confidence': 0,
                'detection_results': []
            }
            
            all_confidences = []
            
            for image_path in test_images:
                if not Path(image_path).exists():
                    print(f"[WARNING] Image not found: {image_path}")
                    continue
                    
                # Run inference
                results = model(image_path, conf=confidence_threshold, verbose=False)
                
                image_result = {
                    'image_path': image_path,
                    'detections': []
                }
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy())
                            class_name = model.names[cls]
                            
                            detection = {
                                'class': class_name,
                                'confidence': round(conf, 4),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            }
                            
                            image_result['detections'].append(detection)
                            all_confidences.append(conf)
                            
                            if class_name == 'OCCUPIED':
                                results_summary['occupied_detections'] += 1
                            else:
                                results_summary['empty_detections'] += 1
                                
                results_summary['detection_results'].append(image_result)
                
            # Calculate summary statistics
            results_summary['total_detections'] = results_summary['occupied_detections'] + results_summary['empty_detections']
            results_summary['avg_confidence'] = round(np.mean(all_confidences), 4) if all_confidences else 0
            
            # Save test results
            test_path = self.results_path / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(test_path, 'w') as f:
                json.dump(results_summary, f, indent=2)
                
            print(f"[INFO] Test results saved: {test_path}")
            return results_summary
            
        except Exception as e:
            print(f"[ERROR] Model testing failed: {e}")
            return {}
            
    def optimize_model(self, model_path: str, target_size: str = 'n') -> str:
        """Optimize model for better performance"""
        try:
            print(f"[INFO] Optimizing model: {model_path}")
            
            # Load model
            model = YOLO(model_path)
            
            # Export to different formats for optimization
            export_formats = ['onnx', 'engine']  # ONNX and TensorRT for better performance
            
            optimized_paths = []
            for format_name in export_formats:
                try:
                    exported_path = model.export(format=format_name, device=self.config.get('device', 'auto'))
                    if exported_path and Path(exported_path).exists():
                        optimized_paths.append(exported_path)
                        print(f"[INFO] Exported {format_name.upper()}: {exported_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to export {format_name}: {e}")
                    
            # Return the best optimized model path
            return optimized_paths[0] if optimized_paths else model_path
            
        except Exception as e:
            print(f"[ERROR] Model optimization failed: {e}")
            return model_path
            
    def generate_training_report(self, model_path: str):
        """Generate comprehensive training report"""
        try:
            print("[INFO] Generating training report...")
            
            # Load evaluation metrics
            eval_files = list(self.results_path.glob("evaluation_*.json"))
            if not eval_files:
                print("[WARNING] No evaluation files found")
                return
                
            latest_eval = max(eval_files, key=lambda x: x.stat().st_mtime)
            with open(latest_eval, 'r') as f:
                metrics = json.load(f)
                
            # Create visualizations
            self._create_performance_plots(metrics)
            
            # Generate text report
            report = f"""
# YOLO26 Parking Detection Model Training Report

## Model Information
- **Model Path**: {model_path}
- **Training Date**: {metrics.get('timestamp', 'Unknown')}
- **Confidence Threshold**: {metrics.get('confidence_threshold', 0.5)}

## Performance Metrics
- **mAP@0.5**: {metrics.get('map50', 0):.4f}
- **mAP@0.5:0.95**: {metrics.get('map50_95', 0):.4f}
- **Precision**: {metrics.get('precision', 0):.4f}
- **Recall**: {metrics.get('recall', 0):.4f}
- **F1-Score**: {metrics.get('f1', 0):.4f}

## Requirements Analysis
- **Required mAP@0.5**: 0.95
- **Achieved mAP@0.5**: {metrics.get('map50', 0):.4f}
- **Status**: {'✅ MET' if metrics.get('map50', 0) >= 0.95 else '❌ NOT MET'}

## Recommendations
"""
            
            if metrics.get('map50', 0) < 0.95:
                report += """
- Consider increasing training epochs
- Add more diverse training data
- Fine-tune hyperparameters
- Use data augmentation techniques
"""
            else:
                report += """
- Model meets performance requirements
- Ready for deployment
- Consider model optimization for faster inference
"""
                
            # Save report
            report_path = self.results_path / "training_report.md"
            with open(report_path, 'w') as f:
                f.write(report)
                
            print(f"[INFO] Training report saved: {report_path}")
            
        except Exception as e:
            print(f"[ERROR] Report generation failed: {e}")
            
    def _create_performance_plots(self, metrics: Dict):
        """Create performance visualization plots"""
        try:
            # Create metrics bar chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Performance metrics
            metric_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
            metric_values = [
                metrics.get('map50', 0),
                metrics.get('map50_95', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1', 0)
            ]
            
            bars = ax1.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            ax1.set_title('Model Performance Metrics')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
                        
            # Requirements comparison
            requirements = ['Required', 'Achieved']
            map_values = [0.95, metrics.get('map50', 0)]
            colors = ['red', 'green' if metrics.get('map50', 0) >= 0.95 else 'red']
            
            bars2 = ax2.bar(requirements, map_values, color=colors)
            ax2.set_title('mAP@0.5 Requirement Check')
            ax2.set_ylabel('mAP@0.5')
            ax2.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars2, map_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
                        
            plt.tight_layout()
            plot_path = self.results_path / "performance_metrics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[INFO] Performance plots saved: {plot_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to create performance plots: {e}")

def main():
    """Main training function"""
    print("=== YOLO26 Parking Model Training ===")
    
    # Initialize trainer
    trainer = ParkingModelTrainer()
    
    # Check if we should resume training
    resume_training = False  # Set to True to resume from checkpoint
    
    # Train model
    model_path = trainer.train_model(resume=resume_training)
    
    if model_path:
        print(f"[SUCCESS] Model training completed: {model_path}")
        
        # Evaluate model
        metrics = trainer.evaluate_model(model_path, confidence_threshold=0.5)
        
        # Optimize model
        optimized_path = trainer.optimize_model(model_path)
        
        # Generate report
        trainer.generate_training_report(model_path)
        
        print("[INFO] Training pipeline completed successfully!")
        print(f"Final model: {optimized_path}")
        print(f"Results saved in: {trainer.results_path}")
        
    else:
        print("[ERROR] Model training failed")

if __name__ == "__main__":
    main()
