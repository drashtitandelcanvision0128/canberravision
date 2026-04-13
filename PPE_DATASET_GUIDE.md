# 🦺 PPE Detection Dataset Guide (Helmet + Seatbelt)

Complete guide for downloading, preparing, and training PPE detection models.

---

## 📊 Best PPE Datasets

### 1️⃣ Helmet Detection Dataset
- **Source**: Kaggle - `andrewmvd/helmet-detection`
- **URL**: https://www.kaggle.com/datasets/andrewmvd/helmet-detection
- **Classes**: With Helmet, Without Helmet
- **Format**: Ready for YOLO
- **Images**: ~7,000+ images

### 2️⃣ Seatbelt Detection Dataset  
- **Source**: Kaggle - `mohamedhanyyy/seat-belt-detection`
- **URL**: https://www.kaggle.com/datasets/mohamedhanyyy/seat-belt-detection
- **Classes**: Seatbelt, No Seatbelt
- **Format**: Various formats available
- **Images**: ~1,000+ images

### 3️⃣ Hard Hat Workers (Alternative)
- **Source**: Kaggle - `dataclusterlabs/hard-hat-workers-detection`
- **URL**: https://www.kaggle.com/datasets/dataclusterlabs/hard-hat-workers-detection
- **Classes**: Person, Helmet, Vest
- **Industrial focused**

---

## 🚀 Quick Start

### Step 1: Download Datasets

```bash
# Using the download script
python download_ppe_dataset.py
```

Or **manual download**:
1. Visit Kaggle URLs above
2. Download datasets
3. Extract to `ppe_dataset/raw/helmet/` and `ppe_dataset/raw/seatbelt/`

### Step 2: Organize & Merge

```bash
# Automatically organizes and merges datasets
python download_ppe_dataset.py
```

This creates:
```
ppe_dataset/merged/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

### Step 3: Train Model

```bash
# Basic training
python train_ppe_model.py

# Custom parameters
python train_ppe_model.py --model yolov8l --epochs 150 --batch 16
```

---

## 🎯 Expected Accuracy

| Model | mAP@0.5 | Precision | Recall | Speed |
|-------|---------|-----------|--------|-------|
| YOLOv8n | ~75-80% | ~70% | ~75% | Fast |
| YOLOv8m | **85-92%** | **85%** | **88%** | Medium |
| YOLOv8l | **90-95%** | **90%** | **92%** | Slow |

**Note**: 
- ❌ 100% accuracy is **NOT possible**
- ✅ **85-95%** is considered very good
- ✅ **95%+** requires excellent dataset quality

---

## 🔧 Training Tips for High Accuracy

### 1. Dataset Quality
- **Minimum 500+ images per class**
- Diverse lighting conditions
- Different camera angles
- Various backgrounds
- Both close-up and far shots

### 2. Training Configuration
```python
# High accuracy settings
epochs = 100-150        # More epochs for convergence
model = 'yolov8m'       # Medium size (accuracy vs speed balance)
imgsz = 640             # Standard resolution
optimizer = 'AdamW'     # Better than SGD
augment = True          # Data augmentation ON
mosaic = 1.0            # Mosaic augmentation
mixup = 0.1             # Mixup augmentation
```

### 3. Hyperparameters
```python
lr0 = 0.001             # Initial learning rate
lrf = 0.01              # Final learning rate factor
patience = 25           # Early stopping patience
```

---

## 📁 Class Mapping

```yaml
names:
  0: helmet         # Person wearing helmet
  1: no_helmet      # Person without helmet  
  2: seatbelt       # Person wearing seatbelt
  3: no_seatbelt    # Person without seatbelt
```

---

## 🛠️ Troubleshooting

### Issue: Low Accuracy (< 70%)
**Solutions**:
- Increase training epochs
- Add more diverse training data
- Use larger model (yolov8m or yolov8l)
- Check annotation quality

### Issue: Model not detecting
**Solutions**:
- Lower confidence threshold (try 0.25)
- Check class IDs in annotations
- Verify dataset.yaml paths

### Issue: Out of memory
**Solutions**:
- Reduce batch size (try 4 or 8)
- Use smaller model (yolov8s)
- Close other applications

---

## 📦 Model Export

After training, export to multiple formats:

```python
from ultralytics import YOLO

model = YOLO("ppe_dataset/models/.../weights/best.pt")

# Export to ONNX (for deployment)
model.export(format='onnx')

# Export to TensorRT (for GPU)
model.export(format='engine')

# Export to TorchScript
model.export(format='torchscript')
```

---

## 🔗 Dataset Links Summary

| Dataset | Direct Link | Size |
|---------|-------------|------|
| Helmet Detection | https://www.kaggle.com/datasets/andrewmvd/helmet-detection | ~500 MB |
| Seatbelt Detection | https://www.kaggle.com/datasets/mohamedhanyyy/seat-belt-detection | ~200 MB |
| Hard Hat Workers | https://www.kaggle.com/datasets/dataclusterlabs/hard-hat-workers-detection | ~1 GB |

---

## ✅ Pre-Training Checklist

- [ ] Downloaded helmet dataset
- [ ] Downloaded seatbelt dataset  
- [ ] Organized images in `images/train` and `images/val`
- [ ] Created YOLO format labels in `labels/train` and `labels/val`
- [ ] Verified `dataset.yaml` exists and paths are correct
- [ ] Checked class distribution is balanced
- [ ] GPU available (check with `torch.cuda.is_available()`)
- [ ] At least 100 images per class minimum

---

## 🎓 Pro Tips

1. **Data Augmentation**: Essential for robust models
2. **Validation Set**: Keep 20% separate for testing
3. **Early Stopping**: Use patience=25 to prevent overfitting
4. **Mixed Precision**: Enable `amp=True` for faster training
5. **Cosine LR**: Use `cos_lr=True` for better convergence

---

## 📞 Need Help?

If you encounter issues:
1. Check `ppe_dataset/results/` for logs
2. Verify dataset structure with provided scripts
3. Review annotation files format
4. Ensure class IDs are correct (0-3)
