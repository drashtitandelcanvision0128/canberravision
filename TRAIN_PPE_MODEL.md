# PPE Detection Model Training Guide

## 📚 Dataset Requirements

You need these datasets for accurate PPE detection:

### 1. Helmet Detection Dataset
**Source**: Kaggle - Helmet Detection  
**URL**: https://www.kaggle.com/datasets/andrewmvd/helmet-detection  
**Classes**: helmet, no_helmet

### 2. Seatbelt Detection Dataset  
**Source**: Kaggle - Seat Belt Detection  
**URL**: https://www.kaggle.com/datasets/mohamedhanyyy/seat-belt-detection  
**Classes**: seatbelt, no_seatbelt

### 3. Additional Sources
- **Roboflow**: https://universe.roboflow.com/ (search "helmet detection", "seatbelt detection")
- **Open Images**: https://storage.googleapis.com/openimages/web/visualizer/index.html
- **Custom Images**: Capture your own images for better accuracy

---

## 📁 Directory Structure

Create this structure:

```
ppe_dataset/
├── raw/
│   ├── helmet/
│   │   ├── kaggle/          <- Download here
│   │   └── roboflow/        <- Download here
│   └── seatbelt/
│       ├── kaggle/          <- Download here
│       └── roboflow/        <- Download here
├── merged/
│   ├── images/
│   │   ├── train/           <- Training images
│   │   ├── val/             <- Validation images
│   │   └── test/            <- Test images
│   └── labels/
│       ├── train/           <- Training labels (YOLO format)
│       ├── val/             <- Validation labels
│       └── test/            <- Test labels
└── dataset.yaml
```

---

## 🔧 YOLO Label Format

Each image needs a `.txt` file with same name:

```
class_id center_x center_y width height
```

Example `image001.txt`:
```
0 0.5 0.3 0.2 0.25   <- helmet
2 0.4 0.6 0.1 0.3    <- seatbelt
```

**Class IDs**:
- 0 = helmet
- 1 = no_helmet  
- 2 = seatbelt
- 3 = no_seatbelt

---

## 🚀 Training Steps

### Step 1: Download Datasets

```bash
# Download helmet dataset from Kaggle
# Download seatbelt dataset from Kaggle
# Extract to ppe_dataset/raw/ directories
```

### Step 2: Prepare Dataset

```bash
python prepare_ppe_dataset.py
```

### Step 3: Train Model

```bash
python train_ppe_model.py
```

Or manually with YOLO:

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train
model.train(
    data='ppe_dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='ppe_detection',
    device=0  # GPU
)
```

---

## 🎯 Training Tips for 100% Accuracy

### 1. Data Quantity
- **Minimum**: 1000 images per class
- **Recommended**: 5000+ images per class
- **More data = Better accuracy**

### 2. Data Diversity
Include images with:
- Different lighting conditions (day, night, indoor)
- Various angles (front, side, back, top)
- Different helmet colors (yellow, white, red, blue, black)
- Various seatbelt types (3-point, 2-point, lap belts)
- Occlusions (partially hidden PPE)
- Different vehicle types

### 3. Annotation Quality
- Tight bounding boxes
- Include all instances
- Consistent labeling
- Check for errors

### 4. Class Balance
- Equal number of images for each class
- Avoid class imbalance
- Use data augmentation if needed

### 5. Hard Negative Mining
- Include images WITHOUT PPE
- Include images with similar objects (hats instead of helmets)
- Add confusing cases

---

## 🔄 Data Augmentation

Use these augmentations to increase dataset size:

```python
import albumentations as A

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.Blur(blur_limit=3, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
])
```

---

## 📊 Expected Results

After proper training:

| Metric | Value |
|--------|-------|
| mAP@0.5 | > 0.90 |
| Precision | > 0.92 |
| Recall | > 0.90 |
| Helmet Accuracy | > 95% |
| Seatbelt Accuracy | > 90% |

---

## 🧪 Testing Your Model

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/ppe_detection/weights/best.pt')

# Test on image
results = model('test_image.jpg')

# Show results
results[0].show()
```

---

## 🚀 Deployment

Copy trained model to app:

```bash
cp runs/detect/ppe_detection/weights/best.pt models/ppe_best.pt
```

Update `apps/app.py` to use new model:
```python
PPE_MODEL_PATH = 'models/ppe_best.pt'
```

---

## 📈 Improving Accuracy

### If helmet detection is poor:
1. Add more helmet variety (colors, types)
2. Include hard cases (hair visible, partial helmets)
3. Add more lighting conditions
4. Check annotation quality

### If seatbelt detection is poor:
1. Add more seatbelt variety (colors, positions)
2. Include vest-only images as negatives
3. Add more car interior angles
4. Include unbuckled cases

### General improvements:
1. Increase dataset size
2. Improve annotation quality
3. Train for more epochs (200-300)
4. Use larger model (yolov8s or yolov8m)
5. Add custom data from your environment

---

## 🆘 Troubleshooting

### "Low accuracy"
- Need more training data
- Check annotation quality
- Train for more epochs

### "False positives"
- Add hard negative images
- Increase class 1 (no_helmet) and class 3 (no_seatbelt) samples
- Check for annotation errors

### "Model not detecting"
- Lower confidence threshold
- Check if classes are correct
- Verify model loaded properly

---

## 📞 Need Help?

If you need assistance with:
- Dataset preparation
- Model training
- Deployment issues

Check Ultralytics docs: https://docs.ultralytics.com/

---

## 🎯 Quick Start

```bash
# 1. Download datasets from Kaggle links above
# 2. Extract to ppe_dataset/raw/
# 3. Run training:
python train_ppe_model.py

# 4. Test the model
python test_ppe_model.py
```

**Note**: Training requires good GPU. On CPU it will be very slow.
