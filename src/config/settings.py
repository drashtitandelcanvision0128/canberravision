"""
YOLO26 Configuration Settings
Central configuration management for the entire project.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INPUTS_DIR = PROJECT_ROOT / "inputs"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
PROCESSED_DIR = PROJECT_ROOT / "processed"

# GPU Configuration
GPU_CONFIG = {
    "use_gpu": True,
    "device_id": 0,
    "mixed_precision": True,
    "memory_fraction": 0.8,
    "enable_cuda_graphs": True
}

# YOLO Model Configuration
YOLO_CONFIG = {
    "available_models": ["yolo26n", "yolo26s", "yolo26m"],
    "default_model": "yolo26n",
    "model_extensions": [".pt"],
    "confidence_threshold": 0.4,
    "iou_threshold": 0.5,
    "image_sizes": [320, 640, 1024],
    "default_image_size": 640
}

# OCR Configuration
OCR_CONFIG = {
    "enabled": True,
    "primary_method": "paddleocr_gpu",
    "fallback_methods": ["paddleocr", "lighton", "tesseract"],
    "confidence_threshold": 0.3,
    "license_plate_confidence": 0.4,
    "languages": ["en", "hi"],
    "cache_enabled": True,
    "cache_size": 100
}

# Video Processing Configuration
VIDEO_CONFIG = {
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv"],
    "output_format": "mp4",
    "fps": 30,
    "quality": "high",
    "processing_modes": {
        "ultra_fast": {"skip_frames": 2, "batch_size": 16},
        "fast": {"skip_frames": 1, "batch_size": 12},
        "balanced": {"skip_frames": 1, "batch_size": 8}
    }
}

# Color Detection Configuration
COLOR_CONFIG = {
    "enabled": True,
    "method": "kmeans",
    "clusters": 8,
    "color_families": ["red", "blue", "green", "yellow", "purple", "neutral"],
    "confidence_threshold": 0.7
}

# UI Configuration
UI_CONFIG = {
    "title": "YOLO26 - Advanced Object Detection",
    "theme": "default",
    "show_confidence": True,
    "show_labels": True,
    "max_boxes": 100,
    "font_size": 0.5,
    "line_thickness": 1
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "[%(levelname)s] %(message)s",
    "file": PROJECT_ROOT / "logs" / "yolo26.log",
    "console": True
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "enable_caching": True,
    "cache_size": 50,
    "parallel_processing": True,
    "max_workers": 4,
    "batch_processing": True
}

# License Plate Configuration
LICENSE_PLATE_CONFIG = {
    "enabled": True,
    "indian_patterns": [
        r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$',  # MH12AB1234
        r'^[A-Z]{2}\d{2}\s?[A-Z]{1,2}\s?\d{4}$'  # MH 12 AB 1234
    ],
    "vehicle_classes": [
        'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'van',
        'taxi', 'ambulance', 'police', 'fire truck', 'tractor',
        'scooter', 'bike', 'auto', 'rickshaw', 'lorry'
    ],
    "min_confidence": 0.5,
    "validate_with_vehicles": True
}

# File Upload Configuration
UPLOAD_CONFIG = {
    "max_file_size": "100MB",
    "allowed_image_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "allowed_video_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv"],
    "auto_cleanup": True,
    "cleanup_interval": 24  # hours
}

def get_config(section: str) -> Dict:
    """Get configuration section by name."""
    configs = {
        "gpu": GPU_CONFIG,
        "yolo": YOLO_CONFIG,
        "ocr": OCR_CONFIG,
        "video": VIDEO_CONFIG,
        "color": COLOR_CONFIG,
        "ui": UI_CONFIG,
        "logging": LOGGING_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "license_plate": LICENSE_PLATE_CONFIG,
        "upload": UPLOAD_CONFIG
    }
    return configs.get(section, {})

def update_config(section: str, updates: Dict):
    """Update configuration section."""
    if section == "gpu":
        GPU_CONFIG.update(updates)
    elif section == "yolo":
        YOLO_CONFIG.update(updates)
    elif section == "ocr":
        OCR_CONFIG.update(updates)
    elif section == "video":
        VIDEO_CONFIG.update(updates)
    elif section == "color":
        COLOR_CONFIG.update(updates)
    elif section == "ui":
        UI_CONFIG.update(updates)
    elif section == "logging":
        LOGGING_CONFIG.update(updates)
    elif section == "performance":
        PERFORMANCE_CONFIG.update(updates)
    elif section == "license_plate":
        LICENSE_PLATE_CONFIG.update(updates)
    elif section == "upload":
        UPLOAD_CONFIG.update(updates)

def ensure_directories():
    """Create all necessary directories."""
    directories = [
        OUTPUTS_DIR, INPUTS_DIR, UPLOADS_DIR, PROCESSED_DIR,
        PROJECT_ROOT / "logs", DATA_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
# Initialize directories
ensure_directories()
