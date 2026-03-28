"""
Simple working car and plate detector for app.py integration
Shows plate text at TOP with car label
"""
import cv2
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
import pytesseract

# Load YOLO model
model = None

def load_model():
    global model
    if model is None:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov8n.pt', force_reload=False)
        model.conf = 0.4
    return model

def extract_plate_text(plate_crop):
    """Extract text from plate crop using Tesseract"""
    if plate_crop is None or plate_crop.size == 0:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # Resize for better OCR
    h, w = gray.shape
    if h < 30:
        scale = 30 / h
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR configs
    configs = [
        '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    ]
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(binary, config=config).strip()
            text = ''.join(c for c in text if c.isalnum()).upper()
            if len(text) >= 4:
                return text
        except:
            pass
    
    return None

def detect_license_plate_region(car_crop):
    """Detect license plate region within car crop"""
    if car_crop is None or car_crop.size == 0:
        return None
    
    h, w = car_crop.shape[:2]
    
    # Plate is typically in lower 35% of car, horizontally centered
    plate_y1 = int(h * 0.65)
    plate_y2 = int(h * 0.90)
    plate_x1 = int(w * 0.15)
    plate_x2 = int(w * 0.85)
    
    plate_crop = car_crop[plate_y1:plate_y2, plate_x1:plate_x2]
    
    if plate_crop.size == 0:
        return None
    
    return plate_crop

def process_frame_with_plate(frame):
    """Process single frame and detect cars with plates"""
    model = load_model()
    
    # YOLO detection
    results = model(frame)
    detections = results.pandas().xyxy[0]
    
    # Filter vehicles
    vehicles = ['car', 'truck', 'bus', 'motorcycle']
    cars = detections[detections['name'].isin(vehicles)]
    
    display = frame.copy()
    ih, iw = display.shape[:2]
    
    detections_list = []
    
    for idx, row in cars.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        label = row['name']
        
        # Get car crop for plate detection
        car_crop = frame[y1:y2, x1:x2]
        
        # Detect plate
        plate_crop = detect_license_plate_region(car_crop)
        plate_text = None
        
        if plate_crop is not None and plate_crop.size > 0:
            plate_text = extract_plate_text(plate_crop)
        
        # Draw car box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label with plate
        label_parts = [f"{label} {conf:.2f}"]
        if plate_text:
            label_parts.append(f"| PLATE:{plate_text}")
        
        text = " ".join(label_parts)
        
        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        ty = y1 - 8
        if ty - th - baseline < 0:
            ty = y1 + th + baseline + 8
        
        bg_x1, bg_y1 = x1, ty - th - baseline
        bg_x2, bg_y2 = x1 + tw + 6, ty + 4
        
        # Green background
        cv2.rectangle(display, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), -1)
        cv2.putText(display, text, (x1 + 3, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        
        detections_list.append({
            'label': label,
            'confidence': float(conf),
            'plate_text': plate_text,
            'bbox': [x1, y1, x2, y2]
        })
    
    return display, detections_list

def process_video(video_path, output_path=None):
    """Process video with car and plate detection"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"PLATE_VIDEO_{timestamp}.mp4"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    all_results = []
    frame_count = 0
    
    print(f"Processing video: {video_path}")
    print(f"Output: {output_path}")
    print(f"Resolution: {w}x{h}, FPS: {fps}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display, detections = process_frame_with_plate(frame)
        out.write(display)
        
        if detections:
            all_results.append({
                'frame': frame_count,
                'detections': detections
            })
            for d in detections:
                if d['plate_text']:
                    print(f"Frame {frame_count}: {d['plate_text']}")
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    
    # Save JSON results
    json_path = output_path.replace('.mp4', '.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Done! Processed {frame_count} frames")
    print(f"✓ Video saved: {output_path}")
    print(f"✓ Results saved: {json_path}")
    
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video = sys.argv[1]
    else:
        # Find latest compatible video
        import glob
        videos = glob.glob("compatible_video_*.mp4")
        video = videos[-1] if videos else "video.mp4"
    
    process_video(video)
