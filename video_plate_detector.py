"""
Standalone Video Processing with License Plate Detection
Command line tool for processing videos with vehicle and license plate detection
"""

import cv2
import numpy as np
import torch
import os
import sys
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

# Vehicle classes from COCO dataset
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


def detect_and_crop_plates(frame, model):
    """Detect vehicles and crop license plate regions"""
    try:
        results = model.predict(
            source=frame,
            conf=0.3,
            iou=0.5,
            imgsz=640,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False
        )
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        if not hasattr(result, 'boxes') or result.boxes is None:
            return []
        
        plates = []
        boxes = result.boxes
        
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            if cls in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                
                # Calculate license plate region (lower portion of vehicle)
                height = y2 - y1
                width = x2 - x1
                
                plate_y1 = y1 + int(height * 0.6)
                plate_y2 = y1 + int(height * 0.9)
                plate_x1 = x1 + int(width * 0.2)
                plate_x2 = x2 - int(width * 0.2)
                
                # Bounds check
                plate_y1 = max(0, plate_y1)
                plate_y2 = min(frame.shape[0], plate_y2)
                plate_x1 = max(0, plate_x1)
                plate_x2 = min(frame.shape[1], plate_x2)
                
                if plate_x2 > plate_x1 and plate_y2 > plate_y1:
                    cropped = frame[plate_y1:plate_y2, plate_x1:plate_x2]
                    plates.append({
                        'vehicle_bbox': (x1, y1, x2, y2),
                        'plate_bbox': (plate_x1, plate_y1, plate_x2, plate_y2),
                        'crop': cropped,
                        'vehicle_id': i,
                        'conf': float(boxes.conf[i])
                    })
        
        return plates
    except Exception as e:
        print(f"⚠️ Detection error: {e}")
        return []


def ocr_plate(plate_crop):
    """Extract text from license plate using OCR"""
    try:
        if plate_crop is None or plate_crop.size == 0:
            return None
        
        # Preprocess
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Try Tesseract
        try:
            import pytesseract
            text = pytesseract.image_to_string(
                denoised, 
                config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            text = text.strip().replace('\n', ' ')
            if text and len(text) >= 4:
                return text
        except:
            pass
        
        # Try EasyOCR
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            result = reader.readtext(denoised)
            if result:
                text = ' '.join([r[1] for r in result])
                if len(text) >= 4:
                    return text
        except:
            pass
        
        return None
    except Exception as e:
        return None


def process_video(video_path, model_name="yolo26n", mode="fast", save_crops=True):
    """
    Process video with license plate detection
    
    Args:
        video_path: Path to input video
        model_name: YOLO model name (yolo26n, yolov8n, etc.)
        mode: ultra_fast, fast, or balanced
        save_crops: Whether to save cropped plate images
    """
    print(f"🚀 Video Processing with License Plate Detection")
    print(f"📹 Input: {video_path}")
    print(f"🤖 Model: {model_name}")
    print(f"⚙️ Mode: {mode}")
    
    # Check video exists
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    # Load model
    model_path = f"models/{model_name}.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"⏳ Loading model...")
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Model loaded on {device}")
    
    # Mode settings
    settings = {
        'ultra_fast': {'conf': 0.4, 'imgsz': 256, 'skip': 3},
        'fast': {'conf': 0.35, 'imgsz': 320, 'skip': 2},
        'balanced': {'conf': 0.3, 'imgsz': 416, 'skip': 1}
    }
    cfg = settings.get(mode, settings['fast'])
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Setup output
    timestamp = int(time.time())
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"processed_{timestamp}.mp4")
    
    # Setup crops directory
    if save_crops:
        crops_dir = os.path.join(output_dir, f"plates_{timestamp}")
        os.makedirs(crops_dir, exist_ok=True)
        print(f"📁 Crops will be saved to: {crops_dir}")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process
    start_time = time.time()
    processed = 0
    actual = 0
    plates_found = 0
    plates_with_text = 0
    unique_plates = set()
    
    print(f"\n🎬 Starting processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed += 1
        
        if processed % cfg['skip'] != 0:
            continue
        
        actual += 1
        
        # Progress
        if actual % 50 == 0:
            elapsed = time.time() - start_time
            fps_proc = actual / elapsed
            pct = (processed / total_frames) * 100
            eta = (total_frames - processed) / (fps_proc * cfg['skip']) / 60
            print(f"📈 {pct:.1f}% | {fps_proc:.1f} FPS | ETA: {eta:.1f} min | Plates: {plates_found}")
        
        try:
            # Detect plates
            plates = detect_and_crop_plates(frame, model)
            annotated = frame.copy()
            
            if plates:
                plates_found += len(plates)
                
                for plate in plates:
                    # Draw vehicle box
                    vx1, vy1, vx2, vy2 = plate['vehicle_bbox']
                    cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
                    
                    # Draw plate box
                    px1, py1, px2, py2 = plate['plate_bbox']
                    cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 0, 0), 2)
                    
                    # OCR
                    plate_text = ocr_plate(plate['crop'])
                    
                    if plate_text:
                        plates_with_text += 1
                        unique_plates.add(plate_text)
                        
                        # Draw text
                        cv2.putText(annotated, f"LP: {plate_text}", (px1, py1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Save crop
                        if save_crops:
                            safe_text = "".join(c for c in plate_text if c.isalnum())
                            fname = f"plate_f{processed}_{safe_text}.jpg"
                            cv2.imwrite(os.path.join(crops_dir, fname), plate['crop'])
                    else:
                        # Save unknown plate
                        if save_crops:
                            fname = f"plate_f{processed}_unknown.jpg"
                            cv2.imwrite(os.path.join(crops_dir, fname), plate['crop'])
            
            out.write(annotated)
            
        except Exception as e:
            print(f"⚠️ Error on frame {processed}: {e}")
            out.write(frame)
    
    # Cleanup
    cap.release()
    out.release()
    
    # Results
    total_time = time.time() - start_time
    avg_fps = actual / total_time
    
    print(f"\n" + "="*50)
    print(f"✅ Processing Complete!")
    print(f"="*50)
    print(f"⏱️  Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"🚀 Speed: {avg_fps:.1f} FPS")
    print(f"📊 Frames: {actual}/{total_frames}")
    print(f"🔍 Plates Detected: {plates_found}")
    print(f"📝 Plates with Text: {plates_with_text}")
    print(f"🎯 Unique Plates: {len(unique_plates)}")
    
    if unique_plates:
        print(f"\n🚗 Detected License Plates:")
        for plate in sorted(unique_plates):
            print(f"   • {plate}")
    
    print(f"\n💾 Output Files:")
    print(f"   Video: {output_path}")
    if save_crops:
        print(f"   Crops: {crops_dir}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Process video with license plate detection'
    )
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--model', default='yolo26n',
                       help='YOLO model name (default: yolo26n)')
    parser.add_argument('--mode', default='fast',
                       choices=['ultra_fast', 'fast', 'balanced'],
                       help='Processing mode (default: fast)')
    parser.add_argument('--no-crops', action='store_true',
                       help='Disable saving cropped plate images')
    
    args = parser.parse_args()
    
    output = process_video(
        args.video,
        args.model,
        args.mode,
        save_crops=not args.no_crops
    )
    
    if output:
        print(f"\n🎉 Success! Video saved to: {output}")
    else:
        print(f"\n❌ Processing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
