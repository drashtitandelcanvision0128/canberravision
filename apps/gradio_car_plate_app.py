"""
Gradio Web Interface for Car and License Plate Video Detection
User-friendly web interface for processing videos
"""

import gradio as gr
import cv2
import os
import json
import time
from datetime import datetime
from car_plate_video_processor import CarPlateVideoProcessor, process_video_for_cars_and_plates

# Global processor instance
processor = None

def initialize_processor():
    """Initialize the global processor"""
    global processor
    if processor is None:
        processor = CarPlateVideoProcessor(model_path="yolo26n.pt", use_gpu=True)
    return processor

def process_video_interface(video_file, model_name, confidence_threshold, show_realtime):
    """Process video for Gradio interface"""
    if video_file is None:
        return None, "❌ Please upload a video file", None
    
    try:
        # Initialize processor
        proc = initialize_processor()
        
        # Update model if needed
        if model_name != "yolo26n.pt":
            proc.model_path = model_name
            proc._initialize_model()
        
        print(f"[INFO] Processing video: {video_file}")
        
        # Process video
        start_time = time.time()
        results = proc.process_video(
            video_path=video_file,
            output_path=None,  # Auto-generate
            show_realtime=False,  # Disable real-time for web interface
            save_frames=True
        )
        
        if 'error' in results:
            return None, f"❌ Processing failed: {results['error']}", None
        
        # Create summary text
        summary = create_summary_text(results)
        
        # Check if output video exists
        output_video = results['video_info']['output_path']
        if os.path.exists(output_video):
            return output_video, summary, results
        else:
            return None, summary, results
            
    except Exception as e:
        error_msg = f"❌ Processing failed: {str(e)}"
        print(error_msg)
        return None, error_msg, None

def create_summary_text(results):
    """Create formatted summary text"""
    try:
        summary = f"""
## 🚗 Car & License Plate Detection Results

### 📊 Processing Summary
- **Processing Time**: {results['video_info']['processing_time']:.1f} seconds
- **Total Frames**: {results['video_info']['total_frames']}
- **Processing Speed**: {results['video_info']['fps_processed']:.1f} FPS
- **Frames with Detections**: {results['detection_summary']['frames_with_detections']}

### 🚗 Vehicle Detection
- **Total Cars Detected**: {results['detection_summary']['total_cars_detected']}

### 📋 License Plate Results
- **Total Plates Found**: {results['detection_summary']['total_plates_found']}
- **Unique Plates**: {results['detection_summary']['unique_plates_count']}

### 🔢 Detected License Plates
"""
        
        if results['detection_summary']['unique_plates']:
            for i, plate in enumerate(results['detection_summary']['unique_plates'], 1):
                summary += f"{i}. `{plate}`\n"
        else:
            summary += "No license plates detected.\n"
        
        if results['most_common_plates']:
            summary += "\n### 🏆 Most Common Plates\n"
            for i, (plate, count) in enumerate(results['most_common_plates'][:5], 1):
                summary += f"{i}. `{plate}` (seen {count} times)\n"
        
        # Add sample frame info
        if results['saved_frames']:
            summary += f"\n### 📁 Sample Frames\n"
            summary += f"Saved {len(results['saved_frames'])} frames with detections to `detected_frames/` folder.\n"
        
        return summary
        
    except Exception as e:
        return f"Error creating summary: {e}"

def get_frame_samples(results):
    """Get sample frames with detections"""
    if not results or 'saved_frames' not in results:
        return []
    
    frame_paths = results['saved_frames'][:10]  # Show up to 10 frames
    frames = []
    
    for path in frame_paths:
        if os.path.exists(path):
            frames.append(path)
    
    return frames

def download_results_json(results):
    """Create downloadable JSON results"""
    if not results:
        return None
    
    try:
        # Create JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"car_plate_results_{timestamp}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return json_path
        
    except Exception as e:
        print(f"Error creating JSON: {e}")
        return None

def create_interface():
    """Create Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .output-video {
        max-width: 100%;
        height: auto;
    }
    """
    
    with gr.Blocks(title="Car & License Plate Detection", theme=gr.themes.Soft(), css=css) as interface:
        gr.Markdown("""
        # 🚗 Car & License Plate Video Detection System
        
        Upload a video to detect cars and extract license plate numbers in real-time.
        
        **Features:**
        - 🚗 Real-time car detection
        - 📋 License plate recognition  
        - 🌍 International plate support
        - 🔥 GPU acceleration
        - 📊 Detailed results export
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Upload Video")
                video_input = gr.Video(
                    label="Upload Video File",
                    sources=["upload"],
                    type="filepath"
                )
                
                gr.Markdown("## ⚙️ Settings")
                model_dropdown = gr.Dropdown(
                    choices=["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolov8s.pt"],
                    value="yolo26n.pt",
                    label="Detection Model"
                )
                
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="Confidence Threshold"
                )
                
                realtime_checkbox = gr.Checkbox(
                    label="Show Real-time Processing (Desktop only)",
                    value=False
                )
                
                process_btn = gr.Button(
                    "🚀 Process Video",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("## 📊 Results")
                
                with gr.Tab("📹 Output Video"):
                    output_video = gr.Video(
                        label="Processed Video",
                        autoplay=False,
                        show_label=True
                    )
                
                with gr.Tab("📋 Detection Summary"):
                    summary_output = gr.Markdown(
                        label="Results Summary",
                        value="Upload a video and click 'Process Video' to see results."
                    )
                
                with gr.Tab("🖼️ Sample Frames"):
                    gallery_output = gr.Gallery(
                        label="Frames with Detections",
                        columns=3,
                        height=400,
                        show_label=True
                    )
                
                with gr.Tab("📁 Download Results"):
                    json_download = gr.File(
                        label="Download JSON Results",
                        visible=False
                    )
                    download_btn = gr.Button(
                        "📥 Generate Results JSON",
                        variant="secondary"
                    )
        
        # Hidden state for results
        results_state = gr.State()
        
        # Event handlers
        process_btn.click(
            fn=process_video_interface,
            inputs=[video_input, model_dropdown, confidence_slider, realtime_checkbox],
            outputs=[output_video, summary_output, results_state],
            show_progress=True
        ).then(
            fn=get_frame_samples,
            inputs=[results_state],
            outputs=[gallery_output]
        )
        
        download_btn.click(
            fn=download_results_json,
            inputs=[results_state],
            outputs=[json_download]
        )
        
        # Examples
        gr.Markdown("## 💡 Usage Examples")
        gr.Examples(
            examples=[
                # Add example videos if available
            ],
            inputs=[video_input],
            label="Example Videos"
        )
        
        # Instructions
        with gr.Accordion("📖 Instructions", open=False):
            gr.Markdown("""
            ### How to Use:
            
            1. **Upload Video**: Click 'Upload Video File' and select your video
            2. **Choose Settings**: 
               - Select detection model (yolo26n = fastest, yolo26m = most accurate)
               - Adjust confidence threshold (lower = more detections, higher = more accurate)
            3. **Process**: Click 'Process Video' to start detection
            4. **View Results**: 
               - Watch the processed video with detections
               - Check the summary for statistics
               - Browse sample frames with detections
               - Download detailed JSON results
            
            ### Tips:
            - For best results, use videos with clear lighting
            - Higher confidence thresholds reduce false positives
            - GPU processing is automatically enabled when available
            - Large videos may take several minutes to process
            
            ### Supported Formats:
            - Video: MP4, AVI, MOV, MKV
            - Plates: International format recognition
            """)
    
    return interface

def main():
    """Launch the Gradio interface"""
    print("🌐 Starting Car & License Plate Detection Web Interface...")
    
    # Initialize processor
    try:
        initialize_processor()
        print("✅ Processor initialized successfully")
    except Exception as e:
        print(f"⚠️ Processor initialization warning: {e}")
    
    # Create and launch interface
    interface = create_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
