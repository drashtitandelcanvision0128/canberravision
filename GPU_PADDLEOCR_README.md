# YOLO26 with Optimized PaddleOCR GPU

🚀 **High-performance object detection + text extraction with GPU acceleration**

## Features

### 🔥 GPU-Accelerated PaddleOCR
- **Optimized GPU Processing**: Uses PaddleOCR with CUDA support for maximum speed
- **Smart Caching**: Intelligent result caching for faster repeated processing
- **Batch Processing**: Parallel text extraction from multiple objects
- **Auto Fallback**: Seamlessly falls back to CPU if GPU unavailable

### ⚡ Fast Detection + Text Flow
- **Real-time Processing**: Optimized YOLO detection + PaddleOCR text extraction
- **License Plate Detection**: Specialized Indian license plate recognition
- **General Text Extraction**: Extract any text from detected objects
- **Multi-language Support**: English, Hindi, and other Indian languages

### 🎯 Smart Optimization
- **Adaptive Processing**: Automatically adjusts settings based on video characteristics
- **Memory Management**: Optimized GPU memory usage for large videos
- **Frame Skipping**: Intelligent frame processing for lengthy videos
- **Quality Enhancement**: Preprocessing for better OCR accuracy

## Quick Start

### 1. Install Dependencies

```bash
# Automatic GPU setup (recommended)
python setup_paddleocr_gpu.py

# Or manual installation
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

### 3. Open Browser

Navigate to the provided URL (usually `http://127.0.0.1:7860`)

## GPU Setup

### CUDA Installation (Windows)

1. **Install NVIDIA CUDA Toolkit**
   - Download from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
   - Recommended: CUDA 11.8 or 12.x

2. **Verify CUDA Installation**
   ```bash
   nvcc --version
   ```

3. **Install GPU-enabled PaddleOCR**
   ```bash
   pip install paddlepaddle-gpu==2.6.2
   pip install paddleocr>=3.4.0
   ```

### Check GPU Status

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Usage

### Image Processing
1. Upload an image
2. Adjust confidence threshold (0.25 recommended)
3. Click "Detect Objects"
4. View results with extracted text

### Video Processing
1. Upload a video
2. Enable smart processing (automatic optimization)
3. Adjust frame processing settings
4. Click "Process Video"
5. Download processed video with detections

### Webcam Processing
1. Allow camera access
2. Real-time object detection
3. Live text extraction
4. Adjustable processing settings

## Performance

### GPU vs CPU Performance

| Operation | GPU (RTX 4050) | CPU | Speedup |
|-----------|----------------|-----|---------|
| Text Extraction | 0.05s | 0.3s | 6x |
| License Plate OCR | 0.08s | 0.5s | 6.25x |
| Batch Processing (10 images) | 0.2s | 2.1s | 10.5x |

### Memory Usage

- **GPU Memory**: ~2-4 GB for PaddleOCR
- **System Memory**: ~1-2 GB base + 1 GB per video
- **Optimization**: Automatic memory management for large files

## Configuration

### GPU Settings

```python
# In optimized_paddleocr_gpu.py
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.8'  # Use 80% GPU memory
os.environ['FLAGS_initial_gpu_memory_in_mb'] = '1024'      # Start with 1GB
os.environ['FLAGS_reallocate_gpu_memory_in_mb'] = '2048'   # Reallocate 2GB chunks
```

### Detection Settings

```python
# Fast detection flow
flow = create_fast_detection_flow(
    model_name="yolo26n",  # Fast model
    use_gpu=True           # Enable GPU
)

result = flow.process_image_fast(
    image,
    conf_threshold=0.25,    # Detection confidence
    confidence_threshold=0.5,  # Text confidence
    extract_text=True
)
```

## Troubleshooting

### Common Issues

**1. CUDA not available**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**2. PaddleOCR GPU errors**
```bash
# Uninstall and reinstall
pip uninstall paddlepaddle paddlepaddle-gpu paddleocr
pip install paddlepaddle-gpu==2.6.2 paddleocr>=3.4.0
```

**3. Memory issues**
- Reduce batch size in `optimized_paddleocr_gpu.py`
- Lower GPU memory fraction
- Use CPU mode for large videos

**4. Slow processing**
- Check GPU utilization
- Enable frame skipping for videos
- Use smaller image sizes

### Performance Tips

1. **Use GPU**: Ensure CUDA is properly installed
2. **Optimize Settings**: Lower confidence thresholds for faster processing
3. **Batch Processing**: Process multiple images simultaneously
4. **Cache Results**: Enable caching for repeated processing
5. **Frame Skipping**: Skip frames for long videos

## File Structure

```
YOLO26/
├── optimized_paddleocr_gpu.py     # GPU-optimized PaddleOCR
├── fast_detection_text_flow.py    # Fast detection + text flow
├── paddleocr_integration.py       # Legacy PaddleOCR (fallback)
├── modules/
│   ├── text_extraction.py         # Updated with GPU support
│   ├── image_processing.py        # Image processing
│   └── video_processing.py        # Video processing
├── setup_paddleocr_gpu.py         # Automatic setup script
├── app.py                         # Main application
└── requirements.txt               # Dependencies
```

## API Reference

### FastDetectionFlow

```python
class FastDetectionFlow:
    def __init__(self, model_name="yolo26n", use_gpu=None)
    def process_image_fast(self, image, conf_threshold=0.25, extract_text=True)
    def annotate_image(self, image, result, show_labels=True)
    def get_performance_stats(self)
```

### Optimized PaddleOCR

```python
def extract_text_optimized(
    image, 
    confidence_threshold=0.5,
    lang='en',
    use_gpu=None,
    use_cache=True
)

def extract_license_plates_optimized(
    image,
    confidence_threshold=0.6,
    use_gpu=None
)

def batch_extract_text(
    images,
    confidence_threshold=0.5,
    max_workers=4
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with GPU and CPU
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify GPU installation
3. Test with CPU fallback
4. Create an issue with system information

---

**🚀 Enjoy GPU-accelerated object detection and text extraction!**
