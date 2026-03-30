"""
Simple test to isolate tab rendering issue
"""
import gradio as gr

with gr.Blocks(title="Test") as demo:
    gr.Markdown("# Test Interface")
    
    with gr.Tabs():
        with gr.TabItem("Tab 1"):
            gr.Markdown("Content 1")
            gr.Textbox(label="Input 1")
        
        with gr.TabItem("Tab 2"):
            gr.Markdown("Content 2")
            gr.Button("Button 2")
        
        with gr.TabItem("Tab 3"):
            gr.Markdown("Content 3")
            gr.Image()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865)
