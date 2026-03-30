#!/usr/bin/env python3
"""
Minimal working version of the app to test basic functionality
"""
import os
import sys
import gradio as gr

# Set environment variables
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "7860"
os.environ["APP_ENV"] = "production"

print("=" * 60)
print("YOLO Vision System - Minimal Version")
print("=" * 60)

# Create a working demo similar to the main app but simplified
demo = gr.Blocks(
    title="YOLO26 AI Vision",
    theme=gr.themes.Soft()
)

with demo:
    # Header
    gr.HTML("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, #2563eb 0%, #1e40af 100%); border-radius: 12px; margin-bottom: 20px; color: white;">
        <h1 style="margin: 0; font-size: 28px;">🚀 YOLO26 AI Vision</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">Advanced AI Vision Platform</p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.TabItem("Image Detection"):
            gr.Markdown("### Upload an image for instant AI-powered object detection")
            
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="Upload Image")
                    img_btn = gr.Button("🚀 Detect Objects", variant="primary", size="lg")
                    
                with gr.Column(scale=2):
                    img_output = gr.Image(type="pil", label="Detection Result")
                    img_summary = gr.Code(label="JSON Results", language="json", lines=10, value="{}")
            
            def simple_detect(image):
                if image is None:
                    return image, "{}"
                return image, '{"status": "Detection working!", "objects": []}'
            
            img_btn.click(simple_detect, inputs=[img_input], outputs=[img_output, img_summary])
        
        with gr.TabItem("Video Processing"):
            gr.Markdown("### Upload a video for AI-powered object detection")
            gr.Markdown("**Note**: This is a minimal test version. Full video processing will be available in the complete app.")
            
            vid_input = gr.Video(label="Upload Video")
            vid_btn = gr.Button("🚀 Process Video", variant="primary")
            vid_output = gr.Video(label="Processed Video")
            
            def simple_process(video):
                return video
            
            vid_btn.click(simple_process, inputs=[vid_input], outputs=[vid_output])

print(f"[INFO] Demo created successfully: {demo}")
print(f"[INFO] Demo title: {demo.title}")

# Test API info generation
try:
    api_info = demo.get_api_info()
    print(f"[INFO] API info generated: {len(api_info)} endpoints")
except Exception as e:
    print(f"[WARNING] API info generation failed: {e}")

# Launch the application
print(f"[INFO] Starting server on {os.environ.get('GRADIO_SERVER_NAME')}:{os.environ.get('GRADIO_SERVER_PORT')}")

try:
    demo.launch(
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=False,
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        prevent_thread_lock=False,
    )
    print("[SUCCESS] Application is ready!")
    print(f"[SUCCESS] Access at: https://p0g0wkk4wk8o4wcgcggs0kcc.senseword.com")
    
except Exception as e:
    print(f"[ERROR] Failed to launch: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
