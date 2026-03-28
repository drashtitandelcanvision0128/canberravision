"""
Simple Car and License Plate Detector
Uses existing app.py infrastructure for video processing
"""

import cv2
import numpy as np
import os
import time
import json
from datetime import datetime

def process_video_for_cars_and_plates(video_path, output_path=None, show_realtime=True):
    """
    Process video to detect cars and extract license plates using existing infrastructure
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video (auto-generated if None)
        show_realtime: Whether to show real-time processing
        
    Returns:
        Dictionary with detection results
    """
    
    print("🚗 Starting Car & License Plate Detection")
    print(f"📁 Input video: {video_path}")
    
    # Check if video exists
    if not os.path.exists(video_path):
        return {'error': f'Video file not found: {video_path}'}
    
    # Generate output path if not provided
    if output_path is None:
        timestamp = int(time.time())
        output_path = f"car_plate_detection_{timestamp}.mp4"
    
    try:
        # Try to use the existing app.py video processing function
        from app import process_video_optimized_fast
        
        print("🔄 Using existing video processing infrastructure...")
        
        # Process video with OCR enabled for license plate detection
        result_video, summary, json_results = process_video_optimized_fast(
            video_path=video_path,
            model_name="yolo26n",
            mode="fast",
            enable_ocr=True,  # Enable OCR for license plate detection
            ocr_every_n=1,    # Process every frame for better coverage
            force_gpu=True    # Use GPU if available
        )
        
        if result_video is None:
            return {'error': 'Video processing failed'}
        
        # Parse the JSON results to extract car and plate information
        parsed_results = parse_json_results(json_results)
        
        # Add video info
        parsed_results['video_info'] = {
            'input_path': video_path,
            'output_path': result_video,
            'summary': summary
        }
        
        # Save detailed results
        save_results_to_json(parsed_results, output_path.replace('.mp4', '_details.json'))
        
        print(f"✅ Processing completed!")
        print(f"📁 Output video: {result_video}")
        print(f"📋 Cars detected: {parsed_results.get('cars_detected', 0)}")
        print(f"🔢 Plates found: {parsed_results.get('plates_found', 0)}")
        print(f"🎯 Unique plates: {len(parsed_results.get('unique_plates', []))}")
        
        # Show unique plates found
        if parsed_results.get('unique_plates'):
            print("\n📋 Detected License Plates:")
            for i, plate in enumerate(parsed_results['unique_plates'][:10], 1):
                print(f"   {i}. {plate}")
        
        return parsed_results
        
    except ImportError:
        print("⚠️ Existing infrastructure not available, using fallback method...")
        return fallback_video_processing(video_path, output_path, show_realtime)
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return {'error': str(e)}

def parse_json_results(json_results):
    """Parse JSON results from app.py to extract car and plate information"""
    try:
        if not json_results:
            return {'cars_detected': 0, 'plates_found': 0, 'unique_plates': []}
        
        # Parse JSON string if needed
        if isinstance(json_results, str):
            data = json.loads(json_results)
        else:
            data = json_results
        
        # Extract text results (license plates)
        all_plates = []
        unique_plates = set()
        cars_detected = 0
        
        if 'all_detected_text' in data:
            for text_item in data['all_detected_text']:
                text = text_item.get('text', '').strip()
                confidence = text_item.get('confidence', 0)
                frame_number = text_item.get('frame_number', 0)
                
                if text and confidence > 0.3:  # Filter by confidence
                    # Check if it looks like a license plate
                    if is_likely_license_plate(text):
                        plate_info = {
                            'text': text,
                            'confidence': confidence,
                            'frame_number': frame_number,
                            'method': text_item.get('method', 'unknown')
                        }
                        all_plates.append(plate_info)
                        unique_plates.add(text)
        
        # Count cars from detections if available
        if 'total_detections' in data:
            cars_detected = data['total_detections']
        
        return {
            'cars_detected': cars_detected,
            'plates_found': len(all_plates),
            'unique_plates': list(unique_plates),
            'all_plates': all_plates,
            'raw_data': data
        }
        
    except Exception as e:
        print(f"❌ Failed to parse JSON results: {e}")
        return {'cars_detected': 0, 'plates_found': 0, 'unique_plates': []}

def is_likely_license_plate(text):
    """Check if text looks like a license plate"""
    import re
    
    # Remove spaces and special characters
    clean_text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
    
    # License plates typically have 5-10 characters
    if len(clean_text) < 4 or len(clean_text) > 12:
        return False
    
    # Check if it has both letters and numbers
    has_letters = bool(re.search(r'[A-Z]', clean_text))
    has_numbers = bool(re.search(r'[0-9]', clean_text))
    
    return has_letters and has_numbers

def fallback_video_processing(video_path, output_path, show_realtime):
    """Fallback video processing with ACTUAL detection using basic methods"""
    print("🔄 Using fallback video processing with detection...")
    
    # Try to import Tesseract
    try:
        import pytesseract
        if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        tesseract_available = True
        print("✅ Tesseract available for OCR")
    except:
        tesseract_available = False
        print("⚠️ Tesseract not available")
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Cannot open video'}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return {'error': 'Cannot create output video'}
        
        # Processing variables
        frame_count = 0
        cars_detected = 0
        plates_found = []
        unique_plates = set()
        
        print("🔄 Processing with detection...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            annotated = frame.copy()
            
            # Simple car detection using motion/pattern analysis
            # Look for rectangular shapes that could be cars
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use edge detection to find vehicle-like shapes
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            frame_cars = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect = w / h if h > 0 else 0
                
                # Car-like shape: width > height, reasonable size
                if 1.2 < aspect < 4.0 and w > 100 and h > 50 and w < width * 0.8:
                    frame_cars += 1
                    cars_detected += 1
                    
                    # Draw car box
                    cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(annotated, "CAR", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Look for license plate in lower part of car
                    if tesseract_available:
                        plate_y1 = y + int(h * 0.7)
                        plate_y2 = y + h - 5
                        plate_x1 = x + 10
                        plate_x2 = x + w - 10
                        
                        if plate_y2 > plate_y1 and plate_x2 > plate_x1:
                            plate_region = frame[plate_y1:plate_y2, plate_x1:plate_x2]
                            
                            if plate_region.size > 0:
                                # OCR
                                plate_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                                _, plate_binary = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                
                                try:
                                    text = pytesseract.image_to_string(plate_binary, config='--psm 7')
                                    text = text.strip().upper()
                                    text = re.sub(r'[^A-Z0-9\s-]', '', text)
                                    
                                    if text and 4 <= len(text) <= 12:
                                        # Check if it has letters or numbers
                                        if re.search(r'[A-Z0-9]', text):
                                            plates_found.append({
                                                'text': text,
                                                'frame': frame_count,
                                                'car_bbox': [x, y, x+w, y+h]
                                            })
                                            unique_plates.add(text)
                                            
                                            # Draw plate box
                                            cv2.rectangle(annotated, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 0, 255), 2)
                                            
                                            # Draw plate text
                                            label = f"PLATE: {text}"
                                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                            cv2.rectangle(annotated, (x, y+h+5), (x+tw+10, y+h+th+15), (0, 0, 255), -1)
                                            cv2.putText(annotated, label, (x+5, y+h+th+10), 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                except:
                                    pass
            
            # Frame info
            info = f"Frame: {frame_count} | Cars: {cars_detected} | Plates: {len(unique_plates)}"
            cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(annotated)
            
            # Show real-time
            if show_realtime:
                display = cv2.resize(annotated, (960, 540))
                cv2.imshow('Car & Plate Detection (Fallback)', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processed {frame_count}/{total_frames} ({progress:.1f}%) - Cars: {cars_detected}, Plates: {len(unique_plates)}")
        
        # Cleanup
        cap.release()
        out.release()
        if show_realtime:
            cv2.destroyAllWindows()
        
        results = {
            'cars_detected': cars_detected,
            'plates_found': len(plates_found),
            'unique_plates': list(unique_plates),
            'all_plates': plates_found,
            'video_info': {
                'input_path': video_path,
                'output_path': output_path,
                'frames_processed': frame_count
            }
        }
        
        print(f"✅ Fallback processing completed: {output_path}")
        print(f"🚗 Cars detected: {cars_detected}")
        print(f"📋 Plates found: {len(plates_found)}")
        if unique_plates:
            print(f"📋 Unique plates: {', '.join(sorted(unique_plates))}")
        
        return results
        
    except Exception as e:
        print(f"❌ Fallback processing failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def save_results_to_json(results, json_path):
    """Save results to JSON file"""
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📁 Results saved to: {json_path}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

def demo_simple_detection():
    """Simple demo function"""
    print("🚗 Simple Car & License Plate Detection Demo")
    print("=" * 50)
    
    # Look for video files
    video_files = [f for f in os.listdir('.') if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print("❌ No video files found. Please add a video to the current directory.")
        return
    
    print(f"Found videos: {video_files}")
    video_path = video_files[0]
    
    print(f"\n🎬 Processing: {video_path}")
    
    # Process video
    results = process_video_for_cars_and_plates(
        video_path=video_path,
        show_realtime=True
    )
    
    if 'error' in results:
        print(f"❌ Demo failed: {results['error']}")
        return
    
    # Display results
    print("\n📊 Results Summary:")
    print(f"🚗 Cars detected: {results.get('cars_detected', 0)}")
    print(f"📋 Plates found: {results.get('plates_found', 0)}")
    print(f"🔢 Unique plates: {len(results.get('unique_plates', []))}")
    
    if results.get('video_info', {}).get('output_path'):
        print(f"📁 Output video: {results['video_info']['output_path']}")

if __name__ == "__main__":
    demo_simple_detection()
