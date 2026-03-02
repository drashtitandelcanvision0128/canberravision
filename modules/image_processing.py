"""
Image processing module for YOLO26 object detection.
Handles all image processing, annotation, and text extraction.
"""

import time
import json
import numpy as np
import cv2
import PIL.Image as Image
from typing import Dict

# Import from our modules
from .text_extraction import extract_text_from_image_json, format_text_extraction_results
from .utils import get_model, _get_device, _annotate_with_color, _generate_detection_summary

# Import K-means color detector
try:
    from kmeans_color_detector import kmeans_detector, detect_image_colors, categorize_detected_object, analyze_scene
    KMEANS_COLOR_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] K-means color detector not available: {e}")
    KMEANS_COLOR_AVAILABLE = False


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
    enable_kmeans_colors=True,
):
    """Predicts objects in an image using a Ultralytics YOLO model with CUDA support, JSON-based text extraction, and K-means color detection."""
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
    
    # K-means color analysis
    kmeans_results = {}
    if enable_kmeans_colors and KMEANS_COLOR_AVAILABLE:
        try:
            print(f"[DEBUG] Starting K-means color analysis for image {image_id}")
            
            # Extract objects from YOLO results for color analysis
            objects = []
            for result in all_results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = result.names.get(class_id, f"class_{class_id}")
                        
                        objects.append({
                            'object_id': f"{class_name}_{i}",
                            'class_name': class_name,
                            'confidence': confidence,
                            'bounding_box': (int(x1), int(y1), int(x2), int(y2))
                        })
            
            if objects:
                # Perform scene color analysis
                scene_analysis = analyze_scene(frame_bgr, objects)
                kmeans_results['scene_analysis'] = scene_analysis
                
                # Analyze colors for each object
                object_colors = []
                for obj in objects:
                    x1, y1, x2, y2 = obj['bounding_box']
                    color_result = detect_image_colors(frame_bgr, (x1, y1, x2, y2))
                    
                    # Categorize object with color
                    categorized_obj = categorize_detected_object(
                        obj['class_name'], color_result, obj['confidence']
                    )
                    
                    object_colors.append({
                        'object_id': obj['object_id'],
                        'class_name': obj['class_name'],
                        'color_result': color_result,
                        'categorized_object': categorized_obj
                    })
                
                kmeans_results['object_colors'] = object_colors
                kmeans_results['success'] = True
                print(f"[DEBUG] K-means color analysis completed for {len(object_colors)} objects")
            else:
                kmeans_results['success'] = False
                kmeans_results['error'] = 'No objects detected for color analysis'
                
        except Exception as e:
            print(f"[ERROR] K-means color analysis failed: {e}")
            kmeans_results['success'] = False
            kmeans_results['error'] = str(e)
    else:
        kmeans_results['success'] = False
        kmeans_results['error'] = 'K-means color detection disabled or unavailable'
    
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
    
    # Add K-means color annotations if available
    if kmeans_results.get('success') and kmeans_results.get('object_colors'):
        try:
            for obj_color in kmeans_results['object_colors']:
                if 'bounding_box' in obj_color:
                    x1, y1, x2, y2 = obj_color['bounding_box']
                    color_info = obj_color['color_result']
                    categorized = obj_color['categorized_object']
                    
                    if color_info.get('success') and color_info.get('primary_color'):
                        primary_color = color_info['primary_color']
                        color_family = primary_color.get('family', 'Unknown')
                        color_shade = primary_color.get('shade', 'Unknown')
                        
                        # Draw color information
                        color_label = f"🎨 {color_shade}"
                        cv2.putText(annotated_bgr, color_label,
                                   (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        except Exception as e:
            print(f"[ERROR] Failed to add K-means color annotations: {e}")
    
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
    
    # Add K-means color results to summary
    if kmeans_results.get('success'):
        color_summary = _generate_kmeans_color_summary(kmeans_results)
        summary = f"{summary}\n\n{color_summary}"
    
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


def _generate_kmeans_color_summary(kmeans_results: Dict) -> str:
    """Generate summary for K-means color analysis results"""
    try:
        if not kmeans_results.get('success'):
            return "🎨 **K-means Color Analysis:** Failed - " + kmeans_results.get('error', 'Unknown error')
        
        summary_lines = []
        summary_lines.append("🎨 **Advanced K-means Color Analysis Complete!**")
        
        # Scene analysis
        scene_analysis = kmeans_results.get('scene_analysis', {})
        if scene_analysis:
            global_colors = scene_analysis.get('global_colors', {})
            if global_colors.get('dominant_colors'):
                primary_color = global_colors['dominant_colors'][0]
                summary_lines.append(f"🌈 **Primary Scene Color:** {primary_color.get('family', 'Unknown')} - {primary_color.get('shade', 'Unknown')}")
                summary_lines.append(f"🎯 **Color Confidence:** {primary_color.get('confidence', 0):.2f}")
            
            if scene_analysis.get('dominant_theme'):
                summary_lines.append(f"🌟 **Scene Theme:** {scene_analysis['dominant_theme']}")
            
            harmony = scene_analysis.get('scene_harmony', {})
            if harmony.get('harmony_type'):
                summary_lines.append(f"🎭 **Color Harmony:** {harmony['harmony_type']} (Score: {harmony.get('harmony_score', 0):.2f})")
        
        # Object colors
        object_colors = kmeans_results.get('object_colors', [])
        if object_colors:
            summary_lines.append(f"📊 **Objects Analyzed:** {len(object_colors)}")
            
            # Count color families
            color_families = {}
            for obj_color in object_colors:
                color_result = obj_color.get('color_result', {})
                if color_result.get('success') and color_result.get('primary_color'):
                    primary = color_result['primary_color']
                    family = primary.get('family', 'Unknown')
                    if family not in color_families:
                        color_families[family] = 0
                    color_families[family] += 1
            
            if color_families:
                summary_lines.append("**🎨 Color Distribution:**")
                for family, count in sorted(color_families.items(), key=lambda x: x[1], reverse=True):
                    summary_lines.append(f"  • {family}: {count} objects")
        
        summary_lines.append("")
        summary_lines.append("**🔬 Technical Details:**")
        summary_lines.append("  • Algorithm: K-means Clustering (8 clusters)")
        summary_lines.append("  • Color Families: 6 (Red, Blue, Green, Yellow, Purple, Neutral)")
        summary_lines.append("  • Color Shades: 56 total shades")
        summary_lines.append("  • Processing: Real-time analysis per object")
        
        return "\n".join(summary_lines)
        
    except Exception as e:
        print(f"[ERROR] Failed to generate K-means color summary: {e}")
        return f"🎨 **K-means Color Analysis:** Summary generation failed - {str(e)}"
