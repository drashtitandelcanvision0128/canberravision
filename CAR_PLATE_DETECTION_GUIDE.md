# 🚗 Car & License Plate Video Detection System

A comprehensive system for detecting cars in videos and extracting license plate numbers in real-time.

## 🌟 Features

- **🚗 Real-time Car Detection**: Detects all types of vehicles (cars, trucks, buses, motorcycles, etc.)
- **📋 License Plate Recognition**: Extracts and reads license plate numbers from detected vehicles
- **🌍 International Support**: Recognizes license plates from multiple countries
- **🔥 GPU Acceleration**: Uses GPU for faster processing when available
- **📹 Video Processing**: Process entire videos with detailed results
- **🎬 Real-time Display**: See detections as they happen
- **📊 Comprehensive Reports**: Get detailed JSON reports with all detections
- **🖼️ Frame Saving**: Save frames with detections for review

## 📋 Requirements

### System Requirements
- Python 3.8+
- Windows/Linux/macOS
- GPU support (optional but recommended)

### Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- `ultralytics` - YOLO object detection
- `opencv-python` - Video processing
- `torch` - Deep learning framework
- `paddleocr` - License plate text recognition
- `gradio` - Web interface (optional)

## 🚀 Quick Start

### 1. Simple Video Processing
```python
from simple_car_plate_detector import process_video_for_cars_and_plates

# Process a video
results = process_video_for_cars_and_plates(
    video_path="your_video.mp4",
    show_realtime=True
)

print(f"Cars detected: {results['cars_detected']}")
print(f"Plates found: {results['plates_found']}")
print(f"Unique plates: {results['unique_plates']}")
```

### 2. Advanced Processing
```python
from car_plate_video_processor import CarPlateVideoProcessor

# Initialize processor
processor = CarPlateVideoProcessor(
    model_path="yolo26n.pt",  # or yolo26s.pt, yolo26m.pt
    use_gpu=True
)

# Process video with custom settings
results = processor.process_video(
    video_path="input.mp4",
    output_path="output.mp4",
    show_realtime=True,
    save_frames=True
)
```

### 3. Interactive Demo
```bash
python demo_car_plate_detection.py
```

### 4. Web Interface
```bash
python gradio_car_plate_app.py
```
Then open http://localhost:7860 in your browser

## 📖 Usage Examples

### Basic Video Processing
```python
# Process video and get results
results = process_video_for_cars_and_plates("traffic_video.mp4")

# Access results
cars = results['cars_detected']
plates = results['plates_found']
unique_plates = results['unique_plates']
output_video = results['video_info']['output_path']

print(f"Found {len(unique_plates)} unique license plates:")
for plate in unique_plates:
    print(f"  - {plate}")
```

### Advanced Configuration
```python
from car_plate_video_processor import CarPlateVideoProcessor

# Create processor with custom settings
processor = CarPlateVideoProcessor(
    model_path="yolo26s.pt",  # More accurate model
    use_gpu=True
)

# Process with specific settings
results = processor.process_video(
    video_path="highway_footage.mp4",
    output_path="processed_highway.mp4",
    show_realtime=False,  # Don't show live preview
    save_frames=True      # Save detection frames
)

# Get detailed plate information
for plate in results['all_plates']:
    print(f"Frame {plate['frame_number']}: {plate['text']} (confidence: {plate['confidence']:.2f})")
```

### Real-time Webcam Detection
```python
import cv2
from car_plate_video_processor import CarPlateVideoProcessor

processor = CarPlateVideoProcessor()
cap = cv2.VideoCapture(0)  # Use webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    result = processor._process_frame(frame, frame_number)
    annotated = processor._create_annotated_frame(frame, result)
    
    cv2.imshow('Live Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🎯 Model Options

### Available YOLO Models
- **yolo26n.pt**: Fastest, good for real-time
- **yolo26s.pt**: Balanced speed and accuracy
- **yolo26m.pt**: Most accurate, slower
- **yolov8s.pt**: Alternative model

### Choosing the Right Model
- **Real-time/Webcam**: Use `yolo26n.pt`
- **High Accuracy**: Use `yolo26m.pt`
- **Balanced**: Use `yolo26s.pt`

## 📊 Results Format

### Basic Results
```python
{
    'cars_detected': 15,
    'plates_found': 8,
    'unique_plates': ['ABC123', 'XYZ789', 'DEF456'],
    'video_info': {
        'input_path': 'input.mp4',
        'output_path': 'output.mp4'
    }
}
```

### Detailed Results
```python
{
    'video_info': {
        'output_path': 'processed_video.mp4',
        'processing_time': 45.2,
        'total_frames': 900,
        'fps_processed': 19.9
    },
    'detection_summary': {
        'total_cars_detected': 25,
        'total_plates_found': 12,
        'unique_plates_count': 8,
        'unique_plates': ['ABC123', 'XYZ789', ...]
    },
    'all_plates': [
        {
            'text': 'ABC123',
            'confidence': 0.92,
            'frame_number': 45,
            'bbox': [100, 200, 300, 250],
            'method': 'paddleocr',
            'device': 'GPU'
        },
        ...
    ],
    'most_common_plates': [('ABC123', 3), ('XYZ789', 2), ...]
}
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "Model not found" Error
```bash
# Download YOLO models
wget https://github.com/ultralytics/ultralytics/releases/download/v0.0.0/yolo26n.pt
wget https://github.com/ultralytics/ultralytics/releases/download/v0.0.0/yolo26s.pt
```

#### 2. "CUDA not available" Warning
- Install NVIDIA drivers
- Install CUDA Toolkit
- Install GPU version of PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 3. "PaddleOCR not available"
```bash
pip install paddlepaddle-gpu paddleocr
```

#### 4. Low Detection Accuracy
- Use larger model (`yolo26m.pt`)
- Lower confidence threshold
- Ensure good video quality and lighting

#### 5. Slow Processing
- Enable GPU acceleration
- Use smaller model (`yolo26n.pt`)
- Reduce video resolution

### Performance Tips

1. **For Speed**:
   - Use `yolo26n.pt` model
   - Enable GPU processing
   - Reduce video resolution

2. **For Accuracy**:
   - Use `yolo26m.pt` model
   - Ensure good video quality
   - Proper lighting conditions

3. **For Real-time**:
   - Use webcam mode
   - Process every 2-3 frames
   - Use GPU acceleration

## 🌍 International License Plates

The system supports license plates from multiple countries:

### Supported Countries
- 🇺🇸 USA (State-specific formats)
- 🇨🇦 Canada (Province formats)
- 🇬🇧 United Kingdom
- 🇩🇪 Germany
- 🇫🇷 France
- 🇮🇹 Italy
- 🇪🇸 Spain
- 🇮🇳 India
- 🇯🇵 Japan
- And many more...

### Plate Format Detection
The system automatically detects the country based on:
- Character patterns
- Format structure
- Special characters

## 📁 File Structure

```
YOLO26/
├── car_plate_video_processor.py    # Main processor class
├── simple_car_plate_detector.py    # Simple interface
├── demo_car_plate_detection.py     # Interactive demo
├── gradio_car_plate_app.py         # Web interface
├── system_test.py                  # System testing
├── yolo26n.pt                      # YOLO model files
├── yolo26s.pt
├── yolo26m.pt
├── outputs/                        # Processed videos
├── detected_frames/                # Saved detection frames
└── results/                        # JSON results
```

## 🤝 Contributing

### Adding New Features
1. Fork the repository
2. Create feature branch
3. Add your improvements
4. Test thoroughly
5. Submit pull request

### Reporting Issues
1. Check existing issues
2. Provide detailed description
3. Include error messages
4. Attach sample files if possible

## 📄 License

This project is licensed under the AGPL-3.0 License.

## 🆘 Support

For help and questions:
1. Check this guide
2. Run `python system_test.py` for diagnostics
3. Review error messages carefully
4. Check model files are present

## 🎯 Use Cases

### Traffic Monitoring
- Monitor busy intersections
- Count vehicles by type
- Track license plates

### Parking Management
- Monitor parking lots
- Track vehicle entries/exits
- License plate recognition

### Security
- Monitor restricted areas
- Alert on suspicious vehicles
- Track vehicle movements

### Research
- Traffic flow analysis
- Vehicle counting studies
- Pattern recognition

---

**Happy Detecting! 🚗📋**
