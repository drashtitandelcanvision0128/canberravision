"""
Webcam processing module for YOLO26 object detection.
Handles real-time webcam processing and streaming.
"""

import numpy as np
import cv2

# Import from our modules
from .utils import get_model, _get_device, _annotate_with_color


def predict_webcam(
    frame,
    conf_threshold,
    iou_threshold,
    model_name,
    show_labels,
    show_conf,
    imgsz,
    enable_resnet,
    max_boxes,
    resnet_every_n,
    enable_ocr,
    ocr_every_n,
):
    """Predicts objects in a webcam frame using a Ultralytics YOLO model with CUDA support."""
    if frame is None:
        return None

    try:
        # Validate frame dimensions
        if not isinstance(frame, np.ndarray):
            return frame
        
        if frame.size == 0:
            return frame

        # Check frame dimensions
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return frame

        # Use cached model for better streaming performance
        model = get_model(model_name)
        device = _get_device()

        models = model if isinstance(model, list) else [model]

        # Gradio webcam sends RGB, but Ultralytics YOLO expects BGR for OpenCV operations
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run inference with CUDA support
        all_results = []
        for m in models:
            r = m.predict(
                source=frame_bgr,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                device=device,
                verbose=False,
                half=True if device != "cpu" else False,
            )
            if r:
                all_results.append(r[0])

        if not all_results:
            return frame

        annotated_bgr = frame_bgr
        for res in all_results:
            annotated_bgr = _annotate_with_color(
                annotated_bgr,
                res,
                show_labels,
                show_conf,
                enable_resnet=bool(enable_resnet),
                max_boxes=int(max_boxes),
                resnet_every_n=int(resnet_every_n),
                stream_key_prefix="webcam",
                enable_ocr=bool(enable_ocr),
                ocr_every_n=int(ocr_every_n),
            )
        return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"[ERROR] Webcam prediction failed: {e}")
        return frame
