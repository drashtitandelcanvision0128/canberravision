# YOLO Car Plate Detection System - Structured Project

## 📁 Project Structure

```
YOLO26/
├── 📂 apps/                          # Main application files
│   ├── app.py                        # Main Gradio web application
│   ├── app_gpu.py                    # GPU-optimized version
│   ├── app_video_integration.py      # Video integration app
│   ├── gradio_car_plate_app.py       # Gradio interface
│   ├── modular_app_integration.py    # Modular integration
│   ├── force_gpu_app.py             # GPU forcing app
│   ├── FINAL_WORKING_DETECTOR.py    # Final working detector
│   ├── main.py                      # Main entry point
│   ├── demo_car_plate_detection.py  # Demo application
│   └── example_usage.py             # Usage examples
│
├── 📂 src/                           # Source code modules
│   ├── 📂 core/                      # Core functionality
│   │   ├── __init__.py
│   │   ├── detector.py               # Main detection logic
│   │   ├── exceptions.py             # Custom exceptions
│   │   └── processor.py              # Base processor
│   │
│   ├── 📂 ocr/                       # OCR functionality
│   │   ├── __init__.py
│   │   ├── base_ocr.py               # Base OCR class
│   │   ├── text_extractor.py         # Text extraction
│   │   └── optimized_paddleocr_gpu.py # GPU-optimized PaddleOCR
│   │
│   ├── 📂 processors/                # Processing modules
│   │   ├── __init__.py
│   │   ├── image_processor.py        # Image processing
│   │   ├── video_processor.py        # Video processing
│   │   ├── webcam_processing.py      # Webcam processing
│   │   ├── optimized_video_processing.py # Optimized video
│   │   ├── car_plate_video_processor.py # Car plate video
│   │   ├── video_output_handler.py   # Video output handling
│   │   └── [additional processors]
│   │
│   ├── 📂 utils/                     # Utility functions
│   │   └── __init__.py
│   │
│   └── 📂 config/                    # Configuration files
│       └── settings.py               # Application settings
│
├── 📂 modules/                       # Legacy modules (being migrated)
│   ├── __init__.py
│   ├── image_processing.py
│   ├── video_processing.py
│   ├── text_extraction.py
│   ├── utils.py
│   ├── webcam_processing.py
│   └── optimized_video_processing.py
│
├── 📂 models/                        # Model files
│   ├── yolo26n.pt                    # YOLOv8 nano model
│   ├── yolo26s.pt                    # YOLOv8 small model
│   ├── yolo26m.pt                    # YOLOv8 medium model
│   ├── yolov8n.pt                    # Original YOLOv8 nano
│   ├── yolov8s.pt                    # Original YOLOv8 small
│   ├── color_detector_mobilenetv2.pth
│   ├── color_shades_detector_mobilenetv2.pth
│   ├── gender_model.pth
│   └── lighton_ocr_info.json
│
├── 📂 tools/                         # Utility and testing tools
│   ├── color_training.py             # Color model training
│   ├── advanced_color_detection.py   # Advanced color detection
│   ├── system_test.py                # System testing
│   ├── diagnose_ocr.py               # OCR diagnostics
│   ├── force_gpu.py                  # GPU forcing utility
│   ├── quick_plate_fix.py            # Quick plate fixes
│   ├── japanese_plate_test.py        # Japanese plate testing
│   ├── international_license_plates.py # International plates
│   ├── international_integration.py  # International integration
│   └── process_specific_video.py     # Video processing tool
│
├── 📂 archive/                       # Archived/legacy code
│   ├── simple_working_detector.py
│   ├── working_plate_detector.py
│   ├── proper_plate_detector.py
│   ├── enhanced_detection.py
│   ├── enhanced_plate_detector.py
│   ├── direct_plate_detection.py
│   ├── simple_car_plate_detector.py
│   ├── simple_plate_display.py
│   ├── tesseract_plate_detector.py
│   ├── fallback_color_detector.py
│   ├── kmeans_color_detector.py
│   ├── lighton_ocr_integration.py
│   └── paddleocr_integration.py
│
├── 📂 outputs/                       # Output files
│   ├── 📂 videos/                    # Processed videos
│   └── 📂 json/                      # Detection results (JSON)
│
├── 📂 inputs/                        # Input files
├── 📂 uploads/                       # User uploads
├── 📂 processed_images/              # Processed images
├── 📂 processed_videos/              # Processed videos
├── 📂 temp/                          # Temporary files
├── 📂 tests/                         # Test files
├── 📂 docs/                          # Documentation
├── 📂 scripts/                       # Utility scripts
├── 📂 color_training_data/           # Training data
├── 📂 extracted_colors/              # Color extraction results
│
├── 📄 Configuration Files
│   ├── requirements.txt              # Python dependencies
│   ├── config.json                   # Application config
│   ├── .gitignore                    # Git ignore file
│   └── .gitattributes                # Git attributes
│
├── 📄 Batch Files
│   ├── start.bat                     # Main startup script
│   ├── start_clean.bat               # Clean startup
│   └── start_fast.bat                # Fast startup
│
├── 📄 Documentation
│   ├── README.md                     # Original README
│   ├── README_STRUCTURED.md          # This structured README
│   ├── CAR_PLATE_DETECTION_GUIDE.md  # Detection guide
│   ├── FAST_PROCESSING_GUIDE.md      # Fast processing guide
│   ├── IMPLEMENTATION_SUMMARY.md     # Implementation summary
│   └── SOLUTION_SUMMARY.md           # Solution summary
│
└── 📂 _trash/                        # Deleted files (git trash)
```

## 🚀 Quick Start

### 1. Main Applications
- **Primary App**: `apps/app.py` - Full-featured Gradio web application
- **GPU Version**: `apps/app_gpu.py` - GPU-optimized version
- **Video Integration**: `apps/app_video_integration.py` - Video-focused app

### 2. Entry Points
```bash
# Main application
python apps/app.py

# GPU-optimized version
python apps/app_gpu.py

# Quick start scripts
start.bat          # Standard startup
start_fast.bat     # Fast startup
start_clean.bat    # Clean startup
```

### 3. Core Components
- **Detection**: `src/core/detector.py` - Main detection logic
- **OCR**: `src/ocr/text_extractor.py` - Text extraction
- **Processing**: `src/processors/` - Image/video processing
- **Configuration**: `src/config/settings.py` - App settings

## 📋 Key Features

### 🔧 Detection Features
- YOLO-based car plate detection
- Multiple OCR engines (PaddleOCR, LightON OCR)
- GPU acceleration support
- Real-time processing
- Batch video processing

### 🎨 Color Detection
- Advanced color classification
- MobileNetV2-based models
- Custom color training

### 🌍 International Support
- Multiple license plate formats
- Japanese plate detection
- International plate integration

## 📊 Output Structure

### Videos
- **Input Videos**: `inputs/`
- **Processed Videos**: `outputs/videos/`
- **Legacy Videos**: Root directory (being migrated)

### JSON Results
- **Detection Results**: `outputs/json/`
- **Structured Data**: Timestamped detection results

## 🔧 Development

### Adding New Features
1. Core logic → `src/core/`
2. OCR functionality → `src/ocr/`
3. Processing modules → `src/processors/`
4. Utilities → `src/utils/`
5. Applications → `apps/`
6. Tools → `tools/`

### Testing
- System tests: `tools/system_test.py`
- OCR diagnostics: `tools/diagnose_ocr.py`
- Feature tests: `tests/`

## 📝 Migration Notes

### Completed
- ✅ Applications organized in `apps/`
- ✅ Core modules in `src/`
- ✅ Models consolidated in `models/`
- ✅ Outputs organized in `outputs/`
- ✅ Tools separated in `tools/`
- ✅ Legacy code archived in `archive/`

### In Progress
- 🔄 Module integration from `modules/` to `src/`
- 🔄 Import path updates
- 🔄 Configuration centralization

## 🤝 Contributing

1. Follow the directory structure
2. Add new features to appropriate `src/` subdirectories
3. Update documentation
4. Test with `tools/system_test.py`

## 📞 Support

For issues:
1. Check `tools/diagnose_ocr.py` for OCR issues
2. Run `tools/system_test.py` for system diagnostics
3. Review documentation in `docs/`

---

**Last Updated**: March 2026
**Version**: Structured v1.0
