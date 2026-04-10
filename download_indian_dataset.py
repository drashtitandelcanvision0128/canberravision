r"""
Script to download Indian Number Plates Dataset from Kaggle
Dataset: https://www.kaggle.com/datasets/dataclusterlabs/indian-number-plates-dataset

Requirements:
1. Kaggle API credentials (kaggle.json)
2. kaggle Python package: pip install kaggle

Setup:
1. Get your Kaggle API key from: https://www.kaggle.com/settings/account
   - Click "Create New API Token" to download kaggle.json
2. Place kaggle.json in:
   - Windows: C:/Users/<YourUsername>/.kaggle/kaggle.json
   - Linux/Mac: ~/.kaggle/kaggle.json
3. Run this script: python download_indian_dataset.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Configuration
KAGGLE_DATASET = "dataclusterlabs/indian-number-plates-dataset"
DOWNLOAD_DIR = Path("training_data/indian_plates")

def check_kaggle_setup():
    """Check if Kaggle API is properly configured"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("[ERROR] Kaggle API credentials not found!")
        print(f"[INFO] Expected location: {kaggle_json}")
        print("\n[SETUP INSTRUCTIONS]:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token' to download kaggle.json")
        print(f"3. Create folder: {kaggle_dir}")
        print(f"4. Copy kaggle.json to: {kaggle_json}")
        print("5. Run this script again")
        return False
    
    print(f"[INFO] Found Kaggle credentials: {kaggle_json}")
    return True

def install_kaggle_package():
    """Install kaggle package if not available"""
    try:
        import kaggle
        print("[INFO] Kaggle package already installed")
        return True
    except ImportError:
        print("[INFO] Installing kaggle package...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
            print("[INFO] Kaggle package installed successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to install kaggle: {e}")
            return False

def download_dataset():
    """Download the Indian number plates dataset"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Create download directory
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Download directory: {DOWNLOAD_DIR.absolute()}")
        
        # Initialize API
        api = KaggleApi()
        api.authenticate()
        print("[INFO] Kaggle API authenticated successfully")
        
        # Download dataset
        print(f"[INFO] Downloading dataset: {KAGGLE_DATASET}")
        print("[INFO] This may take a few minutes...")
        
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(DOWNLOAD_DIR),
            unzip=True
        )
        
        print(f"[SUCCESS] Dataset downloaded to: {DOWNLOAD_DIR.absolute()}")
        
        # List downloaded files
        files = list(DOWNLOAD_DIR.rglob("*"))
        print(f"[INFO] Downloaded {len(files)} files")
        
        # Show directory structure
        print("\n[DATASET STRUCTURE]:")
        for item in DOWNLOAD_DIR.iterdir():
            if item.is_dir():
                file_count = len(list(item.rglob("*")))
                print(f"  📁 {item.name}/ ({file_count} items)")
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  📄 {item.name} ({size_mb:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def organize_for_training():
    """Organize downloaded dataset for training"""
    print("\n[INFO] Organizing dataset for training...")
    
    # Create YOLO format directories
    images_dir = Path("training_data/images/train")
    labels_dir = Path("training_data/labels/train")
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Created directories:")
    print(f"  - {images_dir}")
    print(f"  - {labels_dir}")
    
    # TODO: Add logic to convert dataset to YOLO format
    # This depends on the structure of the downloaded dataset
    
    print("[INFO] Dataset organization complete")
    print("[NOTE] You may need to manually organize images and labels based on the downloaded structure")

def main():
    print("=" * 60)
    print("Indian Number Plates Dataset Downloader")
    print("=" * 60)
    print()
    
    # Check Kaggle setup
    if not check_kaggle_setup():
        sys.exit(1)
    
    # Install kaggle package
    if not install_kaggle_package():
        sys.exit(1)
    
    # Download dataset
    if download_dataset():
        organize_for_training()
        print("\n" + "=" * 60)
        print("[SUCCESS] Dataset download complete!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"1. Check downloaded files in: {DOWNLOAD_DIR.absolute()}")
        print(f"2. Organize images into: training_data/images/train")
        print(f"3. Create corresponding labels in: training_data/labels/train")
        print(f"4. Run training with: train_parking_model.py")
    else:
        print("\n[FAILED] Dataset download failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
