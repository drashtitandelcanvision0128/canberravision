"""
Example Usage of Car & License Plate Detection System
Demonstrates different ways to use the system
"""

import os
from simple_car_plate_detector import process_video_for_cars_and_plates

def example_1_basic_usage():
    """Example 1: Basic video processing"""
    print("📖 Example 1: Basic Video Processing")
    print("=" * 40)
    
    # Find a video file
    video_files = [f for f in os.listdir('.') if f.lower().endswith('.mp4') and 'compatible' in f]
    
    if not video_files:
        print("❌ No compatible video files found")
        return
    
    video_path = video_files[0]
    print(f"🎬 Processing video: {video_path}")
    
    # Process the video
    results = process_video_for_cars_and_plates(
        video_path=video_path,
        show_realtime=False  # Don't show live display for this example
    )
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Display results
    print(f"✅ Processing completed!")
    print(f"📁 Output video: {results['video_info']['output_path']}")
    print(f"🚗 Cars detected: {results.get('cars_detected', 0)}")
    print(f"📋 Plates found: {results.get('plates_found', 0)}")
    print(f"🔢 Unique plates: {len(results.get('unique_plates', []))}")
    
    # Show detected plates
    if results.get('unique_plates'):
        print("\n📋 Detected License Plates:")
        for i, plate in enumerate(results['unique_plates'], 1):
            print(f"   {i}. {plate}")

def example_2_batch_processing():
    """Example 2: Process multiple videos"""
    print("\n📖 Example 2: Batch Processing")
    print("=" * 40)
    
    # Find multiple video files
    video_files = [f for f in os.listdir('.') if f.lower().endswith('.mp4') and 'compatible' in f][:3]
    
    if not video_files:
        print("❌ No video files found")
        return
    
    print(f"🎬 Processing {len(video_files)} videos...")
    
    all_results = []
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path}")
        
        results = process_video_for_cars_and_plates(
            video_path=video_path,
            show_realtime=False
        )
        
        if 'error' not in results:
            all_results.append(results)
            print(f"   ✅ Cars: {results.get('cars_detected', 0)}, Plates: {results.get('plates_found', 0)}")
        else:
            print(f"   ❌ Error: {results['error']}")
    
    # Summary
    if all_results:
        total_cars = sum(r.get('cars_detected', 0) for r in all_results)
        total_plates = sum(r.get('plates_found', 0) for r in all_results)
        all_unique_plates = set()
        
        for result in all_results:
            all_unique_plates.update(result.get('unique_plates', []))
        
        print(f"\n📊 Batch Processing Summary:")
        print(f"   Videos processed: {len(all_results)}")
        print(f"   Total cars detected: {total_cars}")
        print(f"   Total plates found: {total_plates}")
        print(f"   Total unique plates: {len(all_unique_plates)}")

def example_3_custom_settings():
    """Example 3: Custom processing settings"""
    print("\n📖 Example 3: Custom Settings")
    print("=" * 40)
    
    # This example shows how you would use custom settings
    # when the full infrastructure is available
    
    print("🔧 Custom settings example (when full system is available):")
    print("""
    from car_plate_video_processor import CarPlateVideoProcessor
    
    # Create processor with custom settings
    processor = CarPlateVideoProcessor(
        model_path="yolo26s.pt",  # More accurate model
        use_gpu=True              # Use GPU acceleration
    )
    
    # Process with custom parameters
    results = processor.process_video(
        video_path="input.mp4",
        output_path="custom_output.mp4",
        show_realtime=True,
        save_frames=True
    )
    
    # Access detailed results
    for plate in results['all_plates']:
        print(f"Frame {plate['frame_number']}: {plate['text']}")
    """)

def example_4_results_analysis():
    """Example 4: Analyze results"""
    print("\n📖 Example 4: Results Analysis")
    print("=" * 40)
    
    # Process one video to get results
    video_files = [f for f in os.listdir('.') if f.lower().endswith('.mp4') and 'compatible' in f]
    
    if not video_files:
        print("❌ No video files found")
        return
    
    video_path = video_files[0]
    results = process_video_for_cars_and_plates(video_path, show_realtime=False)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print("📊 Results Analysis:")
    print(f"   Video file: {video_path}")
    print(f"   Cars detected: {results.get('cars_detected', 0)}")
    print(f"   Plates found: {results.get('plates_found', 0)}")
    print(f"   Unique plates: {len(results.get('unique_plates', []))}")
    
    # Analyze plates
    unique_plates = results.get('unique_plates', [])
    if unique_plates:
        print(f"\n🔢 License Plate Analysis:")
        print(f"   Total unique plates: {len(unique_plates)}")
        
        # Categorize plates by length
        short_plates = [p for p in unique_plates if len(p.replace(' ', '')) <= 6]
        long_plates = [p for p in unique_plates if len(p.replace(' ', '')) > 6]
        
        print(f"   Short plates (≤6 chars): {len(short_plates)}")
        print(f"   Long plates (>6 chars): {len(long_plates)}")
        
        # Show sample plates
        print(f"\n📋 Sample Plates:")
        for i, plate in enumerate(unique_plates[:5], 1):
            print(f"   {i}. {plate}")

def main():
    """Run all examples"""
    print("🚗 Car & License Plate Detection - Usage Examples")
    print("=" * 60)
    
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Batch Processing", example_2_batch_processing),
        ("Custom Settings", example_3_custom_settings),
        ("Results Analysis", example_4_results_analysis),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    print("\n✅ Examples completed!")
    print("\n📖 For more information, see CAR_PLATE_DETECTION_GUIDE.md")
    print("🌐 For web interface, run: python gradio_car_plate_app.py")
    print("🎮 For interactive demo, run: python demo_car_plate_detection.py")

if __name__ == "__main__":
    main()
