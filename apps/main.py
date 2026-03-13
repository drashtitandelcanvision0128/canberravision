"""
YOLO26 Main Application
Clean, organized main application with modular structure.
"""

import gradio as gr
import numpy as np
from PIL import Image
import traceback
from pathlib import Path

# Import our modular components
from src.core.detector import YOLODetector
from src.processors.image_processor import ImageProcessor
from src.config.settings import get_config, PROJECT_ROOT
from src.utils.logger import setup_logger, get_logger

# Setup logging
logger = setup_logger(__name__)


class YOLO26App:
    """
    Main YOLO26 application class.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.config = get_config('ui')
        self.yolo_config = get_config('yolo')
        
        # Initialize processors
        self.image_processor = ImageProcessor()
        
        # Model choices
        self.model_choices = self.yolo_config['available_models']
        self.image_size_choices = self.yolo_config['image_sizes']
        
        logger.info("YOLO26 Application initialized")
    
    def process_image(self, 
                     image: Image.Image,
                     model_name: str,
                     confidence: float,
                     iou_threshold: float,
                     image_size: int,
                     enable_ocr: bool,
                     enable_colors: bool,
                     show_labels: bool,
                     show_confidence: bool) -> tuple:
        """
        Process uploaded image.
        
        Args:
            image: Uploaded PIL Image
            model_name: YOLO model to use
            confidence: Confidence threshold
            iou_threshold: IOU threshold
            image_size: Image size for inference
            enable_ocr: Enable text extraction
            enable_colors: Enable color detection
            show_labels: Show object labels
            show_confidence: Show confidence scores
            
        Returns:
            Tuple of (annotated_image, results_text)
        """
        try:
            if image is None:
                return None, "Please upload an image first."
            
            logger.info(f"Processing image with model: {model_name}")
            
            # Convert PIL to BGR numpy array
            image_bgr = np.array(image)
            image_bgr = image_bgr[:, :, ::-1]  # RGB to BGR
            
            # Update processor settings
            self.image_processor.enable_ocr = enable_ocr
            self.image_processor.enable_colors = enable_colors
            self.image_processor.show_labels = show_labels
            self.image_processor.show_confidence = show_confidence
            
            # Process image
            processed_image, results = self.image_processor.process(
                image_bgr,
                conf_threshold=confidence,
                iou_threshold=iou_threshold,
                imgsz=image_size
            )
            
            # Generate results text
            results_text = self._format_results(results)
            
            logger.info("Image processing completed successfully")
            return processed_image, results_text
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return None, error_msg
    
    def _format_results(self, results: dict) -> str:
        """Format processing results for display."""
        lines = []
        
        # Header
        lines.append("🎯 **YOLO26 Detection Results**")
        lines.append("")
        
        # Detection Summary
        detection_summary = results.get('detection_summary', {})
        lines.append(f"📊 **Objects Detected:** {detection_summary.get('total_objects', 0)}")
        
        objects_by_class = detection_summary.get('objects_by_class', {})
        if objects_by_class:
            lines.append("**Objects by Class:**")
            for class_name, count in sorted(objects_by_class.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  • {class_name}: {count}")
        
        lines.append("")
        
        # Text Summary
        text_summary = results.get('text_summary', {})
        if text_summary:
            lines.append("📝 **Text Extraction:**")
            lines.append(f"  • Total text instances: {text_summary.get('total_text_instances', 0)}")
            lines.append(f"  • License plates found: {text_summary.get('license_plates_found', 0)}")
            lines.append(f"  • General text found: {text_summary.get('general_text_found', 0)}")
            
            # Show license plates
            license_plates = results.get('license_plates', [])
            if license_plates:
                lines.append("**License Plates:**")
                for plate in license_plates[:5]:  # Show top 5
                    lines.append(f"  • {plate['text']} (conf: {plate['confidence']:.2f})")
            
            lines.append("")
        
        # Color Summary
        color_summary = results.get('color_summary', {})
        if color_summary:
            lines.append("🎨 **Color Analysis:**")
            lines.append(f"  • Total colors found: {color_summary.get('total_colors_found', 0)}")
            
            dominant_color = color_summary.get('dominant_color')
            if dominant_color:
                lines.append(f"  • Dominant color: {dominant_color['color']} ({dominant_color['percentage']:.1f}%)")
            
            color_distribution = color_summary.get('color_distribution', {})
            if color_distribution:
                lines.append("**Color Distribution:**")
                for color_name, percentage in sorted(color_distribution.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"  • {color_name}: {percentage:.1f}%")
            
            lines.append("")
        
        # Processing Info
        lines.append("⚡ **Processing Information:**")
        lines.append(f"  • Processing time: {results.get('processing_time', 0):.2f} seconds")
        
        # Paths
        paths = results.get('paths', {})
        if paths:
            lines.append(f"  • Input saved: {paths.get('input', 'N/A')}")
            lines.append(f"  • Output saved: {paths.get('output', 'N/A')}")
        
        return "\\n".join(lines)
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        
        with gr.Blocks(
            title=self.config['title'],
            theme=gr.themes.Soft()
        ) as interface:
            
            gr.Markdown(f"# 🚀 {self.config['title']}")
            gr.Markdown("Advanced Object Detection with OCR and Color Analysis")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Input section
                    image_input = gr.Image(
                        label="Upload Image",
                        type="pil",
                        height=400
                    )
                    
                    # Settings section
                    with gr.Accordion("⚙️ Detection Settings", open=True):
                        model_dropdown = gr.Dropdown(
                            choices=self.model_choices,
                            value=self.yolo_config['default_model'],
                            label="YOLO Model"
                        )
                        
                        confidence_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=self.yolo_config['confidence_threshold'],
                            step=0.05,
                            label="Confidence Threshold"
                        )
                        
                        iou_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=self.yolo_config['iou_threshold'],
                            step=0.05,
                            label="IOU Threshold"
                        )
                        
                        image_size_dropdown = gr.Dropdown(
                            choices=self.image_size_choices,
                            value=self.yolo_config['default_image_size'],
                            label="Image Size"
                        )
                    
                    with gr.Accordion("🔧 Advanced Settings", open=False):
                        enable_ocr_checkbox = gr.Checkbox(
                            label="Enable Text Extraction (OCR)",
                            value=True
                        )
                        
                        enable_colors_checkbox = gr.Checkbox(
                            label="Enable Color Detection",
                            value=True
                        )
                        
                        show_labels_checkbox = gr.Checkbox(
                            label="Show Object Labels",
                            value=self.config['show_labels']
                        )
                        
                        show_confidence_checkbox = gr.Checkbox(
                            label="Show Confidence Scores",
                            value=self.config['show_confidence']
                        )
                    
                    # Process button
                    process_button = gr.Button(
                        "🚀 Process Image",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    # Output section
                    image_output = gr.Image(
                        label="Processed Image",
                        type="pil",
                        height=400
                    )
                    
                    results_output = gr.Markdown(
                        label="Results",
                        value="Upload an image and click 'Process Image' to see results."
                    )
            
            # Examples section
            gr.Markdown("## 📸 Examples")
            gr.Examples(
                examples=[
                    # Add example images here if available
                ],
                inputs=image_input,
                outputs=[image_output, results_output],
                fn=self.process_image,
                cache_examples=False
            )
            
            # Footer
            gr.Markdown("---")
            gr.Markdown(
                "💡 **Tips:**"
                "\\n• Use higher confidence for cleaner results"
                "\\n• Enable OCR for text detection and license plates"
                "\\n• Enable color analysis for object color information"
                "\\n• Different models offer speed vs accuracy trade-offs"
            )
            
            # Wire up the processing
            process_button.click(
                fn=self.process_image,
                inputs=[
                    image_input,
                    model_dropdown,
                    confidence_slider,
                    iou_slider,
                    image_size_dropdown,
                    enable_ocr_checkbox,
                    enable_colors_checkbox,
                    show_labels_checkbox,
                    show_confidence_checkbox
                ],
                outputs=[image_output, results_output]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the application."""
        interface = self.create_interface()
        
        # Default launch parameters
        launch_kwargs = {
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'share': False,
            'show_error': True,
            'show_tips': True,
            'inbrowser': True
        }
        
        # Update with user-provided kwargs
        launch_kwargs.update(kwargs)
        
        logger.info(f"Launching YOLO26 on {launch_kwargs['server_name']}:{launch_kwargs['server_port']}")
        
        interface.launch(**launch_kwargs)


def main():
    """Main entry point."""
    try:
        app = YOLO26App()
        app.launch()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ Error: {e}")
        print("Please check the logs for more details.")


if __name__ == "__main__":
    main()
