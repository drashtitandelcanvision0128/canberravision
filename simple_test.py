#!/usr/bin/env python3
"""
Simple test app that matches your exact Coolify configuration
"""
import os
import sys
import gradio as gr

# Set exact environment variables from your Coolify setup
os.environ["GRADIO_SERVER_PORT"] = "7860"
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["APP_ENV"] = "production"

print("=" * 50)
print("YOLO Vision - Simple Test App")
print("=" * 50)
print(f"Server: {os.environ.get('GRADIO_SERVER_NAME')}:{os.environ.get('GRADIO_SERVER_PORT')}")
print(f"Environment: {os.environ.get('APP_ENV')}")

# Create a very simple demo to test basic functionality
demo = gr.Blocks(
    title="YOLO26 Test",
    theme=gr.themes.Soft()
)

with demo:
    gr.Markdown("# 🚀 YOLO26 AI Vision - TEST PAGE")
    gr.Markdown("## If you can see this page, the configuration is working!")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Test Information:")
            gr.Markdown(f"- **Server**: {os.environ.get('GRADIO_SERVER_NAME')}:{os.environ.get('GRADIO_SERVER_PORT')}")
            gr.Markdown(f"- **Environment**: {os.environ.get('APP_ENV')}")
            gr.Markdown(f"- **Gradio Version**: {gr.__version__}")
            gr.Markdown(f"- **Domain**: https://p0g0wkk4wk8o4wcgcggs0kcc.senseword.com")
            
        with gr.Column():
            gr.Markdown("### Test Controls:")
            test_input = gr.Textbox(label="Type something", placeholder="Test input...")
            test_output = gr.Textbox(label="Output", interactive=False)
            test_btn = gr.Button("Test Button", variant="primary")
            
            def test_function(text):
                return f"You typed: {text}" if text else "Please type something above"
            
            test_btn.click(test_function, inputs=[test_input], outputs=[test_output])

print(f"Demo created successfully: {demo}")
print(f"Demo title: {demo.title}")

# Launch with exact configuration for Coolify
print("Starting server...")
try:
    demo.launch(
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=False,
        server_name="0.0.0.0",
        server_port=7860,
        prevent_thread_lock=False,
    )
    print("✅ Server started successfully!")
    print("🌐 Access at: https://p0g0wkk4wk8o4wcgcggs0kcc.senseword.com")
except Exception as e:
    print(f"❌ Failed to start: {e}")
    import traceback
    traceback.print_exc()
