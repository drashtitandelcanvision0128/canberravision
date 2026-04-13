#!/usr/bin/env python3
"""
Seatbelt Detection Model Training Script
Trains YOLOv8 model specifically for seatbelt detection

Usage:
    python train_seatbelt_model.py --dataset ppe_dataset/seatbelt_manual
"""

import os
import yaml
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

class SeatbeltTrainer:
    """Trains YOLOv8 for seatbelt detection"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.models_path = Path("ppe_dataset/models")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.config = {
            'model_name': 'yolov8n.pt',  # Use nano for faster training
            'epochs': 50,
            'batch_size': 16,
            'imgsz': 640,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'patience': 15,
        }
        
        print(f"[INFO] Using device: {self.config['device']}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    
    def find_data_yaml(self):
        """Find data.yaml in dataset directory"""
        # Check common locations
        yaml_paths = [
            self.dataset_path / "data.yaml",
            self.dataset_path / "dataset.yaml",
        ]
        
        for yaml_path in yaml_paths:
            if yaml_path.exists():
                return yaml_path
        
        # Search recursively
        yaml_files = list(self.dataset_path.rglob("data.yaml"))
        if yaml_files:
            return yaml_files[0]
        
        return None
    
    def validate_dataset(self):
        """Validate dataset structure"""
        print("\n[INFO] Validating dataset...")
        
        # Find data.yaml
        data_yaml = self.find_data_yaml()
        if not data_yaml:
            print("[ERROR] data.yaml not found!")
            print(f"[INFO] Searched in: {self.dataset_path}")
            return False, None
        
        print(f"[INFO] Found data.yaml: {data_yaml}")
        
        # Load and check yaml
        try:
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
            
            print(f"[INFO] Dataset config:")
            print(f"  Classes: {data.get('nc', 'unknown')}")
            print(f"  Names: {data.get('names', [])}")
            print(f"  Train: {data.get('train', 'unknown')}")
            print(f"  Val: {data.get('val', 'unknown')}")
            
            # Check if paths exist
            train_path = self.dataset_path / data.get('train', '')
            val_path = self.dataset_path / data.get('val', '')
            
            if not train_path.exists():
                print(f"[WARNING] Train path not found: {train_path}")
                # Try to find train/images
                train_path = self.dataset_path / "train" / "images"
                if train_path.exists():
                    print(f"[INFO] Found train at: {train_path}")
                    data['train'] = 'train/images'
            
            if not val_path.exists():
                print(f"[WARNING] Val path not found: {val_path}")
                # Try to find valid/images
                val_path = self.dataset_path / "valid" / "images"
                if not val_path.exists():
                    val_path = self.dataset_path / "val" / "images"
                if val_path.exists():
                    print(f"[INFO] Found val at: {val_path}")
                    data['val'] = str(val_path.relative_to(self.dataset_path))
            
            # Update yaml with corrected paths
            data['path'] = str(self.dataset_path.absolute())
            with open(data_yaml, 'w') as f:
                yaml.dump(data, f)
            
            # Count images
            train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
            val_images = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))
            
            print(f"\n[INFO] Dataset statistics:")
            print(f"  Train images: {len(train_images)}")
            print(f"  Val images: {len(val_images)}")
            
            if len(train_images) == 0:
                print("[ERROR] No training images found!")
                return False, None
            
            return True, data_yaml
            
        except Exception as e:
            print(f"[ERROR] Failed to load data.yaml: {e}")
            return False, None
    
    def train(self):
        """Train seatbelt detection model"""
        print("\n" + "="*70)
        print("  STARTING SEATBELT MODEL TRAINING")
        print("="*70)
        
        # Validate dataset
        success, data_yaml = self.validate_dataset()
        if not success:
            print("\n[ERROR] Dataset validation failed!")
            print("[INFO] Please check:")
            print("  1. Dataset is extracted correctly")
            print("  2. data.yaml exists in the dataset folder")
            print("  3. Train and val folders have images")
            return None
        
        try:
            # Load pretrained model
            print(f"\n[INFO] Loading pretrained model: {self.config['model_name']}")
            model = YOLO(self.config['model_name'])
            
            # Training parameters
            training_params = {
                'data': str(data_yaml),
                'epochs': self.config['epochs'],
                'batch': self.config['batch_size'],
                'imgsz': self.config['imgsz'],
                'device': self.config['device'],
                'patience': self.config['patience'],
                'project': str(self.models_path),
                'name': f'seatbelt_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'exist_ok': True,
                'verbose': True,
                'plots': True,
                'save': True,
            }
            
            print(f"\n[CONFIG] Training parameters:")
            for key, value in training_params.items():
                if key not in ['project', 'name']:
                    print(f"  {key}: {value}")
            
            # Start training
            print(f"\n[INFO] Starting training...")
            results = model.train(**training_params)
            
            # Get best model path
            best_model_path = results.save_dir / "weights" / "best.pt"
            
            if best_model_path.exists():
                print(f"\n✅ Training completed!")
                print(f"Best model: {best_model_path}")
                
                # Copy to a standard location
                standard_path = self.models_path / "seatbelt_best.pt"
                import shutil
                shutil.copy(best_model_path, standard_path)
                print(f"Also copied to: {standard_path}")
                
                # Save training info
                info = {
                    'model_path': str(standard_path),
                    'original_path': str(best_model_path),
                    'epochs': results.epoch,
                    'timestamp': datetime.now().isoformat(),
                }
                info_path = self.models_path / "seatbelt_training_info.json"
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=2)
                
                return str(standard_path)
            else:
                print("[ERROR] Best model not found!")
                return None
                
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    parser = argparse.ArgumentParser(description="Train Seatbelt Detection Model")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--model", default="yolov8n.pt", help="Model to use (yolov8n/s/m/l)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    print("="*70)
    print("SEATBELT DETECTION MODEL TRAINING")
    print("="*70)
    
    # Initialize trainer
    trainer = SeatbeltTrainer(args.dataset)
    
    # Update config from args
    if args.model:
        trainer.config['model_name'] = args.model
    if args.epochs:
        trainer.config['epochs'] = args.epochs
    if args.batch:
        trainer.config['batch_size'] = args.batch
    
    # Train
    best_model = trainer.train()
    
    if best_model:
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Best model: {best_model}")
        print(f"\n📚 NEXT STEPS:")
        print(f"1. Update modules/ppe_detection.py to use this model")
        print(f"2. Change model path from 'yolov8n.pt' to '{best_model}'")
        print(f"3. Test the updated detection")
        print("="*70)
    else:
        print("\n[FAILED] Training did not complete successfully")
        print("\n[HELP] Common issues:")
        print("- Dataset path is incorrect")
        print("- data.yaml is missing or malformed")
        print("- Train/val folders don't have images")
        print("- Insufficient disk space or memory")

if __name__ == "__main__":
    import sys
    sys.exit(main() if main() is not None else 1)
