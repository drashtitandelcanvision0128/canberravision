# Seatbelt Dataset Download & Training Guide

## Problem
Automatic downloads from Roboflow/Kaggle are failing due to API restrictions and permissions.

## Solution: Manual Download + Training

### Step 1: Download Dataset Manually

1. **Visit Roboflow Universe:**
   - Go to: https://universe.roboflow.com/search?q=seatbelt
   - Look for a public seatbelt dataset with good ratings

2. **Select a Dataset:**
   Recommended datasets:
   - "Seat Belt Detection" by Paul Moran (popular)
   - "Seatbelt Detection" by Kannan S
   - Any dataset with YOLO format support

3. **Download:**
   - Click on the dataset
   - Click "Download Dataset"
   - Select format: **YOLOv8** (or YOLOv5/v7)
   - Click "Continue" and download

4. **Extract:**
   - Extract the downloaded zip to: `c:\canberravision\YOLO26\ppe_dataset\seatbelt_manual`

### Step 2: Verify Dataset Structure

After extraction, you should have:
```
ppe_dataset/seatbelt_manual/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

Or similar structure with train/val folders.

### Step 3: Run Training

Once dataset is downloaded and extracted, run:

```bash
python train_seatbelt_model.py --dataset ppe_dataset/seatbelt_manual
```

### Step 4: Update PPE Detection

After training completes, the model will be saved. Update `modules/ppe_detection.py` to use the new model:

1. Find the trained model in: `ppe_dataset/models/seatbelt_detection_*/weights/best.pt`
2. Update the model path in `ppe_detection.py`
3. Test the detection

## Alternative: Use Current Heuristic Detection

If training is not possible, the current heuristic seatbelt detection in `modules/ppe_detection.py` has been improved with:
- Expanded color range for seatbelts
- Multiple fallback detection methods
- More lenient criteria for strap detection

The current system should work reasonably well for most cases.

## Notes

- API Key provided: NxT14uIgw1lnMJfvk071
- The API key works but most public datasets have download restrictions
- Manual download is the most reliable method
- Training typically takes 1-2 hours depending on dataset size and GPU
