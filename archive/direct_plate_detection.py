"""
DIRECT License Plate Detection - Uses app.py directly without import issues
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_plate_detection(video_path):
    """Run detection using app.py directly"""
    
    print("=" * 70)
    print("🚗 DIRECT License Plate Detection")
    print("=" * 70)
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"📁 Video: {video_path}")
    print()
    
    # Execute app.py functions directly
    try:
        # Import required modules from app.py
        print("🔧 Loading modules...")
        
        # This will trigger app.py to load
        import app
        
        print("✅ Modules loaded")
        print()
        
        # Check if process_video_optimized_fast exists
        if hasattr(app, 'process_video_optimized_fast'):
            print("🚀 Starting detection with OCR enabled...")
            print("-" * 70)
            
            result = app.process_video_optimized_fast(
                video_path=video_path,
                model_name="yolo26n",
                mode="fast",
                enable_ocr=True,
                ocr_every_n=1,
                force_gpu=True
            )
            
            if result and len(result) >= 3:
                output_video, summary, json_data = result
                
                print()
                print("=" * 70)
                print("✅ PROCESSING COMPLETE!")
                print("=" * 70)
                print(f"📁 Output video: {output_video}")
                print(f"📊 Summary: {summary}")
                
                # Parse and display plates
                if json_data:
                    try:
                        import json
                        data = json.loads(json_data) if isinstance(json_data, str) else json_data
                        
                        if 'all_detected_text' in data:
                            texts = data['all_detected_text']
                            print(f"\n📋 Found {len(texts)} text detections:")
                            
                            plates = []
                            for item in texts:
                                text = item.get('text', '').strip()
                                if text and len(text) >= 4:
                                    plates.append(text)
                                    print(f"   • {text}")
                            
                            if plates:
                                print(f"\n🔢 Unique plates: {len(set(plates))}")
                            else:
                                print("\n⚠️ No license plates detected in text")
                    except Exception as e:
                        print(f"⚠️ Could not parse JSON: {e}")
            else:
                print("❌ Processing failed - no result returned")
        else:
            print("❌ process_video_optimized_fast not found in app.py")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Find a video to process
    videos = [f for f in os.listdir('.') if f.endswith('.mp4') and 'compatible' in f.lower()]
    
    if videos:
        print(f"Found {len(videos)} videos")
        run_plate_detection(videos[0])
    else:
        print("Usage: python direct_plate_detection.py")
        print("Or: python direct_plate_detection.py <video_path>")
