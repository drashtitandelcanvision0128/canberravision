#!/usr/bin/env python3
"""
Minimal Gradio app to test if the basic interface works in container
"""
import os
import sys
import gradio as gr

# Set environment variables
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "7860"
os.environ["APP_ENV"] = "production"

print("[MINIMAL_APP] Starting minimal Gradio application...")

# Create a minimal demo
demo = gr.Blocks(
    title="YOLO26 AI Vision - Minimal Test",
    theme=gr.themes.Soft()
)

with demo:
    gr.Markdown("# 🚀 YOLO26 AI Vision - Minimal Test")
    gr.Markdown("## If you see this, the basic Gradio interface is working!")
    
    with gr.Tab("Test"):
        gr.Markdown("### Test Tab")
        gr.Textbox(label="Test Input", placeholder="Type something here...")
        gr.Button("Test Button")
        
    with gr.Tab("Info"):
        gr.Markdown(f"""
        ### Environment Information:
        - **Server Name**: {os.environ.get('GRADIO_SERVER_NAME', 'not set')}
        - **Server Port**: {os.environ.get('GRADIO_SERVER_PORT', 'not set')}
        - **App Environment**: {os.environ.get('APP_ENV', 'not set')}
        - **Gradio Version**: {gr.__version__}
        - **Python Version**: {sys.version.split()[0]}
        """)

print(f"[MINIMAL_APP] Demo created: {demo}")
print(f"[MINIMAL_APP] Demo title: {demo.title}")

# Launch the application
print(f"[MINIMAL_APP] Starting server on {os.environ.get('GRADIO_SERVER_NAME')}:{os.environ.get('GRADIO_SERVER_PORT')}")

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
    print("[MINIMAL_APP] ✅ Server started successfully!")
except Exception as e:
    print(f"[MINIMAL_APP] ❌ Failed to start server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
