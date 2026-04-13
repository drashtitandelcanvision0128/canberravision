"""
PPE (Helmet + Seatbelt) Detection Model Training Script
Trains YOLOv8 model for PPE detection with high accuracy

Usage:
    python train_ppe_model.py --epochs 100 --model yolov8m
"""

import os
import yaml
import json
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class PPEModelTrainer:
    """Trains YOLOv8 for PPE detection with optimized settings for high accuracy"""
    
    def __init__(self, dataset_path: str = "ppe_dataset/merged"):
        self.dataset_path = Path(dataset_path)
        self.models_path = Path("ppe_dataset/models")
        self.results_path = Path("ppe_dataset/results")
        self.logs_path = Path("ppe_dataset/logs")
        
        # Create directories
        for path in [self.models_path, self.results_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # High-accuracy training config
        self.config = {
            'model_name': 'yolov8m.pt',  # Medium size for better accuracy
            'epochs': 100,
            'batch_size': 8,  # Smaller batch for stability
            'imgsz': 640,
            'lr0': 0.001,
            'lrf': 0.01,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'patience': 25,  # Early stopping patience
            'save_period': 10,
            'val_confidence': 0.25,
            'test_confidence': 0.5,
            'optimizer': 'AdamW',  # Better optimizer
            'augment': True,
            'mosaic': 1.0,
            'mixup': 0.1,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'bgr': 0.0,
            'copy_paste': 0.1,
        }
        
        print(f"[INFO] Using device: {self.config['device']}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    
    def prepare_dataset(self) -> bool:
        """Validate and prepare dataset for training"""
        print("[INFO] Preparing PPE dataset...")
        
        # Check required directories
        required_dirs = [
            self.dataset_path / "images/train",
            self.dataset_path / "images/val",
            self.dataset_path / "labels/train",
            self.dataset_path / "labels/val"
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                print(f"[ERROR] Missing directory: {directory}")
                return False
        
        # Count dataset
        train_images = list((self.dataset_path / "images/train").glob("*.jpg"))
        val_images = list((self.dataset_path / "images/val").glob("*.jpg"))
        
        print(f"[INFO] Dataset count: {len(train_images)} train, {len(val_images)} val")
        
        if len(train_images) < 100:
            print("[WARNING] Very small dataset! Need at least 100+ images per class.")
        
        # Validate dataset.yaml
        yaml_path = self.dataset_path / "dataset.yaml"
        if not yaml_path.exists():
            print("[ERROR] dataset.yaml not found!")
            return False
        
        # Analyze class distribution
        self._analyze_dataset()
        
        return len(train_images) > 0 and len(val_images) > 0
    
    def _analyze_dataset(self):
        """Analyze class distribution"""
        train_labels = list((self.dataset_path / "labels/train").glob("*.txt"))
        
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # helmet, no_helmet, seatbelt, no_seatbelt
        
        for label_path in train_labels:
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            if class_id in class_counts:
                                class_counts[class_id] += 1
            except Exception as e:
                pass
        
        print("[INFO] Class distribution:")
        print(f"  helmet:      {class_counts[0]}")
        print(f"  no_helmet:   {class_counts[1]}")
        print(f"  seatbelt:    {class_counts[2]}")
        print(f"  no_seatbelt: {class_counts[3]}")
        
        # Check class balance
        total = sum(class_counts.values())
        if total > 0:
            for class_id, count in class_counts.items():
                pct = (count / total) * 100
                print(f"  Class {class_id}: {pct:.1f}%")
    
    def train_model(self, model_name: str = None, resume: bool = False) -> Optional[str]:
        """Train PPE detection model with high accuracy settings"""
        if not self.prepare_dataset():
            print("[ERROR] Dataset preparation failed!")
            return None
        
        model_name = model_name or self.config['model_name']
        
        try:
            print(f"\n{'='*60}")
            print(f"STARTING PPE MODEL TRAINING")
            print(f"{'='*60}")
            print(f"Model: {model_name}")
            print(f"Epochs: {self.config['epochs']}")
            print(f"Device: {self.config['device']}")
            
            # Load pretrained model
            model = YOLO(model_name)
            
            # Training with high-accuracy configuration
            training_params = {
                'data': str(self.dataset_path / "dataset.yaml"),
                'epochs': self.config['epochs'],
                'batch': self.config['batch_size'],
                'imgsz': self.config['imgsz'],
                'lr0': self.config['lr0'],
                'lrf': self.config['lrf'],
                'device': self.config['device'],
                'patience': self.config['patience'],
                'save_period': self.config['save_period'],
                'project': str(self.models_path),
                'name': f'ppe_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'exist_ok': True,
                'verbose': True,
                'plots': True,
                'save': True,
                'optimizer': self.config['optimizer'],
                'augment': self.config['augment'],
                'mosaic': self.config['mosaic'],
                'mixup': self.config['mixup'],
                'hsv_h': self.config['hsv_h'],
                'hsv_s': self.config['hsv_s'],
                'hsv_v': self.config['hsv_v'],
                'degrees': self.config['degrees'],
                'translate': self.config['translate'],
                'scale': self.config['scale'],
                'shear': self.config['shear'],
                'perspective': self.config['perspective'],
                'flipud': self.config['flipud'],
                'fliplr': self.config['fliplr'],
                'bgr': self.config['bgr'],
                'copy_paste': self.config['copy_paste'],
                'amp': True,  # Automatic Mixed Precision
                'cos_lr': True,  # Cosine learning rate
            }
            
            print(f"\n[CONFIG] Training parameters:")
            for key, value in training_params.items():
                if key not in ['project', 'name']:
                    print(f"  {key}: {value}")
            
            # Start training
            results = model.train(**training_params)
            
            # Get best model path
            best_model_path = results.save_dir / "weights" / "best.pt"
            
            if best_model_path.exists():
                print(f"\n[SUCCESS] Training completed!")
                print(f"Best model: {best_model_path}")
                
                # Save training info
                self._save_training_info(results, best_model_path)
                
                return str(best_model_path)
            else:
                print("[ERROR] Best model not found!")
                return None
                
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_training_info(self, results, model_path: Path):
        """Save training information"""
        info = {
            'model_path': str(model_path),
            'final_epoch': results.epoch,
            'best_fitness': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
            'map50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        info_path = self.results_path / f"training_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"[INFO] Training info saved: {info_path}")
    
    def evaluate_model(self, model_path: str) -> Dict:
        """Evaluate trained model"""
        print(f"\n[INFO] Evaluating model: {model_path}")
        
        try:
            model = YOLO(model_path)
            
            results = model.val(
                data=str(self.dataset_path / "dataset.yaml"),
                conf=0.25,
                device=self.config['device']
            )
            
            metrics = {
                'map50': float(results.box.map50),
                'map50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'model_path': model_path,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"\n[RESULTS] Evaluation metrics:")
            print(f"  mAP@0.5:     {metrics['map50']:.4f}")
            print(f"  mAP@0.5:0.95: {metrics['map50_95']:.4f}")
            print(f"  Precision:   {metrics['precision']:.4f}")
            print(f"  Recall:      {metrics['recall']:.4f}")
            
            # Save metrics
            metrics_path = self.results_path / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            return metrics
            
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            return {}
    
    def export_model(self, model_path: str):
        """Export model to multiple formats"""
        print(f"\n[INFO] Exporting model...")
        
        try:
            model = YOLO(model_path)
            
            formats_to_export = ['onnx', 'torchscript']
            
            exported_paths = []
            for fmt in formats_to_export:
                try:
                    exported = model.export(format=fmt)
                    exported_paths.append(exported)
                    print(f"[SUCCESS] Exported to {fmt}: {exported}")
                except Exception as e:
                    print(f"[WARNING] Export to {fmt} failed: {e}")
            
            return exported_paths
            
        except Exception as e:
            print(f"[ERROR] Export failed: {e}")
            return []

def main():
    parser = argparse.ArgumentParser(description="Train PPE Detection Model")
    parser.add_argument("--model", default="yolov8m.pt", help="Model to use (yolov8n/s/m/l)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    args = parser.parse_args()
    
    print("="*60)
    print("PPE DETECTION MODEL TRAINING")
    print("="*60)
    print("Classes: helmet, no_helmet, seatbelt, no_seatbelt")
    print("="*60)
    
    # Initialize trainer
    trainer = PPEModelTrainer()
    
    # Update config from args
    if args.model:
        trainer.config['model_name'] = args.model
    if args.epochs:
        trainer.config['epochs'] = args.epochs
    if args.batch:
        trainer.config['batch_size'] = args.batch
    if args.imgsz:
        trainer.config['imgsz'] = args.imgsz
    
    # Train
    best_model = trainer.train_model(resume=args.resume)
    
    if best_model:
        # Evaluate
        metrics = trainer.evaluate_model(best_model)
        
        # Export
        trainer.export_model(best_model)
        
        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETED")
        print("="*60)
        print(f"Best model: {best_model}")
        print(f"Results: ppe_dataset/results/")
        print("="*60)
    else:
        print("\n[FAILED] Training did not complete successfully")

if __name__ == "__main__":
    main()
