"""
Demo Script for Car and License Plate Video Detection
Simple demonstration of the enhanced video processor
"""

import cv2
import os
from car_plate_video_processor import CarPlateVideoProcessor, process_video_for_cars_and_plates

def demo_single_video():
    """Demonstrate processing a single video"""
    
    print("🚗 Car & License Plate Detection Demo")
    print("=" * 40)
    
    # Check for video file
    video_path = input("Enter video file path (or press Enter for default): ").strip()
    if not video_path:
        # Look for video files in current directory
        video_files = [f for f in os.listdir('.') if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if video_files:
            video_path = video_files[0]
            print(f"Using found video: {video_path}")
        else:
            print("❌ No video files found. Please provide a video path.")
            return
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"\n🎬 Processing video: {video_path}")
    
    # Process the video
    try:
        results = process_video_for_cars_and_plates(
            video_path=video_path,
            model_path="yolo26n.pt",
            output_path=None,  # Auto-generate
            show_realtime=True
        )
        
        if 'error' in results:
            print(f"❌ Processing failed: {results['error']}")
            return
        
        # Display results
        print("\n✅ Processing completed!")
        print(f"📁 Output video: {results['video_info']['output_path']}")
        print(f"⏱️  Processing time: {results['video_info']['processing_time']:.1f} seconds")
        print(f"🚗 Cars detected: {results['detection_summary']['total_cars_detected']}")
        print(f"📋 Plates found: {results['detection_summary']['total_plates_found']}")
        print(f"🔢 Unique plates: {results['detection_summary']['unique_plates_count']}")
        
        # Show unique plates
        if results['detection_summary']['unique_plates']:
            print("\n📋 Detected License Plates:")
            for i, plate in enumerate(results['detection_summary']['unique_plates'], 1):
                print(f"   {i}. {plate}")
        
        # Show most common plates
        if results['most_common_plates']:
            print("\n🏆 Most Common Plates:")
            for i, (plate, count) in enumerate(results['most_common_plates'][:5], 1):
                print(f"   {i}. {plate} (seen {count} times)")
        
        # Ask if user wants to play the output video
        play = input("\n▶️  Play output video? (y/n): ").strip().lower()
        if play == 'y':
            play_video(results['video_info']['output_path'])
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def play_video(video_path):
    """Play video using OpenCV"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        print(f"▶️  Playing: {video_path}")
        print("   Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('Output Video', frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("⏹️  Video playback ended")
        
    except Exception as e:
        print(f"❌ Playback failed: {e}")

def demo_webcam_detection():
    """Demonstrate real-time webcam detection"""
    print("\n📷 Real-time Webcam Detection Demo")
    print("=" * 40)
    
    try:
        # Initialize processor
        processor = CarPlateVideoProcessor(model_path="yolo26n.pt", use_gpu=True)
        
        # Open webcam
        cap = cv2.VideoCapture(0)  # Use default webcam
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        print("📷 Webcam started. Press 'q' to quit.")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame for performance
            if frame_count % 5 == 0:
                frame_result = processor._process_frame(frame, frame_count)
                annotated_frame = processor._create_annotated_frame(frame, frame_result)
            else:
                annotated_frame = frame
            
            cv2.imshow('Real-time Car & Plate Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("⏹️  Webcam detection ended")
        
    except Exception as e:
        print(f"❌ Webcam demo failed: {e}")

def show_usage_examples():
    """Show usage examples"""
    print("\n📖 Usage Examples:")
    print("=" * 30)
    
    print("\n1️⃣ Process a video file:")
    print("   from car_plate_video_processor import process_video_for_cars_and_plates")
    print("   results = process_video_for_cars_and_plates('video.mp4')")
    print("   print(f'Found {len(results[\"unique_plates\"])} unique plates')")
    
    print("\n2️⃣ Advanced usage with custom settings:")
    print("   processor = CarPlateVideoProcessor(model_path='yolo26s.pt', use_gpu=True)")
    print("   results = processor.process_video(")
    print("       video_path='input.mp4',")
    print("       output_path='output.mp4',")
    print("       show_realtime=False,")
    print("       save_frames=True")
    print("   )")
    
    print("\n3️⃣ Access specific results:")
    print("   # All detected plates")
    print("   plates = results['all_plates']")
    print("   ")
    print("   # Plates by frame")
    print("   plates_by_frame = results['plates_by_frame']")
    print("   ")
    print("   # Most common plates")
    print("   common_plates = results['most_common_plates']")
    
    print("\n4️⃣ Real-time webcam detection:")
    print("   processor = CarPlateVideoProcessor()")
    print("   # Then use webcam processing loop...")

def main():
    """Main demo menu"""
    print("🚗 Car & License Plate Detection System")
    print("=" * 50)
    
    while True:
        print("\n📋 Demo Menu:")
        print("1. Process Video File")
        print("2. Real-time Webcam Detection")
        print("3. Show Usage Examples")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            demo_single_video()
        elif choice == '2':
            demo_webcam_detection()
        elif choice == '3':
            show_usage_examples()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
