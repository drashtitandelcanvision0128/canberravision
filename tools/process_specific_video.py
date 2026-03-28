"""
Process specific video with proper license plate detection
Uses virtual environment Python with torch support
"""

import sys
import os

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def process_video_file(video_path):
    """Process a specific video file"""
    
    print("=" * 70)
    print("🚗 PROCESSING VIDEO:", video_path)
    print("=" * 70)
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    try:
        # Import app.py with all its dependencies (torch, etc.)
        print("🔧 Loading detection modules...")
        import app
        print("✅ Modules loaded successfully")
        print()
        
        # Check if the function exists
        if not hasattr(app, 'process_video_optimized_fast'):
            print("❌ Detection function not found")
            return None
        
        print("🚀 Starting detection with settings:")
        print("   • Model: yolo26n")
        print("   • Mode: fast")
        print("   • OCR: ENABLED")
        print("   • GPU: Forced")
        print()
        print("-" * 70)
        
        # Run detection
        result = app.process_video_optimized_fast(
            video_path=video_path,
            model_name="yolo26n",
            mode="fast",
            enable_ocr=True,
            ocr_every_n=1,
            force_gpu=True
        )
        
        if not result or len(result) < 3:
            print("❌ Processing failed - no result")
            return None
        
        output_video, summary, json_data = result
        
        print()
        print("=" * 70)
        print("✅ PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"📁 Output video: {output_video}")
        print(f"📊 Summary: {summary}")
        
        # Extract and display plates
        if json_data:
            import json
            try:
                data = json.loads(json_data) if isinstance(json_data, str) else json_data
                
                # Get all detected text
                all_text = data.get('all_detected_text', [])
                print(f"\n📋 Total text detections: {len(all_text)}")
                
                # Filter for license plates
                plates = []
                for item in all_text:
                    text = item.get('text', '').strip()
                    # License plates are typically 4-12 chars with letters and numbers
                    if text and 4 <= len(text) <= 15:
                        import re
                        if re.search(r'[A-Z0-9]', text.upper()):
                            plates.append(text)
                
                # Remove duplicates
                unique_plates = list(set(plates))
                
                if unique_plates:
                    print(f"\n🔢 FOUND {len(unique_plates)} UNIQUE LICENSE PLATES:")
                    print("-" * 70)
                    for i, plate in enumerate(sorted(unique_plates), 1):
                        print(f"   {i}. {plate}")
                    print("-" * 70)
                else:
                    print("\n⚠️ No license plates detected")
                    print("   (Try with better quality video or different angle)")
                
                # Show sample frames with text
                if all_text:
                    print(f"\n📝 Sample detections by frame:")
                    frames_with_text = {}
                    for item in all_text[:20]:  # First 20
                        frame = item.get('frame_number', 0)
                        text = item.get('text', '')
                        if text:
                            if frame not in frames_with_text:
                                frames_with_text[frame] = []
                            frames_with_text[frame].append(text)
                    
                    for frame, texts in sorted(frames_with_text.items())[:5]:
                        print(f"   Frame {frame}: {', '.join(texts[:3])}")
                
            except Exception as e:
                print(f"⚠️ Error parsing results: {e}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Use the latest processed video
    video = "processed_video_1773296566.mp4"
    
    if os.path.exists(video):
        process_video_file(video)
    else:
        # Try to find any compatible video
        import glob
        videos = glob.glob('compatible_video_*.mp4')
        if videos:
            print(f"Using: {videos[0]}")
            process_video_file(videos[0])
        else:
            print("No videos found")
