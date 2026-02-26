"""
Image processing module for YOLO26 object detection.
Handles all image processing, annotation, and text extraction.
"""

import time
import json
import numpy as np
import cv2
import PIL.Image as Image

# Import from our modules
from .text_extraction import extract_text_from_image_json, format_text_extraction_results
from .utils import get_model, _get_device, _annotate_with_color, _generate_detection_summary


def predict_image(
    img,
    conf_threshold,
    iou_threshold,
    model_name,
    show_labels,
    show_conf,
    imgsz,
    enable_resnet,
    max_boxes,
    enable_ocr,
):
    """Predicts objects in an image using a Ultralytics YOLO model with CUDA support and JSON-based text extraction."""
    model = get_model(model_name)
    device = _get_device()

    models = model if isinstance(model, list) else [model]

    all_results = []
    for m in models:
        r = m.predict(
            source=img,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
            half=True if device != "cpu" else False,  # Use FP16 on CUDA for speed
        )
        if r:
            all_results.append(r[0])

    if not all_results:
        return img, "No objects detected"

    # Convert PIL to BGR for OpenCV operations
    frame_rgb = np.array(img)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Save original input image to inputs folder
    import os
    import time
    timestamp = int(time.time())
    inputs_folder = os.path.join(os.getcwd(), "inputs")
    os.makedirs(inputs_folder, exist_ok=True)
    input_filename = f"input_image_{timestamp}.jpg"
    input_path = os.path.join(inputs_folder, input_filename)
    
    try:
        img.save(input_path)
        print(f"[INFO] Input image saved to: {input_path}")
    except Exception as e:
        print(f"[WARNING] Could not save input image: {e}")
    
    # Generate unique image ID for JSON text extraction
    image_id = f"img_{timestamp}"
    
    # Perform comprehensive text extraction if OCR is enabled
    json_text_results = None
    if enable_ocr:
        print(f"[DEBUG] Starting JSON-based text extraction for image {image_id}")
        json_text_results = extract_text_from_image_json(frame_bgr, image_id)
        print(f"[DEBUG] Text extraction completed for {image_id}")
    
    annotated_bgr = frame_bgr
    for idx, res in enumerate(all_results):
        annotated_bgr = _annotate_with_color(
            annotated_bgr,
            res,
            show_labels,
            show_conf,
            enable_resnet=bool(enable_resnet),
            max_boxes=int(max_boxes),
            resnet_every_n=1,
            stream_key_prefix=None,
            enable_ocr=False,  # Disable individual OCR since we're using JSON system
            ocr_every_n=1,
        )
    
    # If we have JSON text results, add text annotations from JSON
    if json_text_results and enable_ocr:
        from .utils import _annotate_from_json_results
        annotated_bgr = _annotate_from_json_results(annotated_bgr, json_text_results, show_labels)
    
    # Generate detection summary
    summaries = [
        _generate_detection_summary(r, enable_resnet=bool(enable_resnet), enable_ocr=False)
        for r in all_results
    ]
    summary = "\n\n".join([s for s in summaries if s])
    
    # Add JSON text extraction results to summary
    if json_text_results and enable_ocr:
        text_summary = format_text_extraction_results(json_text_results)
        summary = f"{summary}\n\n{text_summary}"
        
        # Also add raw JSON for debugging
        json_output = json.dumps(json_text_results, indent=2, ensure_ascii=False)
        summary = f"{summary}\n\n📋 **Raw JSON Data:**\n```json\n{json_output}\n```"
    
    # Convert back to PIL and save to outputs folder
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    result_image = Image.fromarray(annotated_rgb)
    
    # Save processed image to outputs folder
    outputs_folder = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outputs_folder, exist_ok=True)
    output_filename = f"processed_image_{timestamp}.jpg"
    output_path = os.path.join(outputs_folder, output_filename)
    
    try:
        result_image.save(output_path)
        print(f"[INFO] Processed image saved to: {output_path}")
    except Exception as e:
        print(f"[WARNING] Could not save processed image: {e}")
    
    return result_image, summary
