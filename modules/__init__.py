"""
YOLO26 Modules Package
Contains modular components for object detection, text extraction, and processing.
"""

from .image_processing import predict_image
from .video_processing import predict_video
from .optimized_video_processing import predict_video_optimized, benchmark_video_processing
from .webcam_processing import predict_webcam
from .text_extraction import (
    extract_text_from_image_json,
    format_text_extraction_results,
    _extract_text_ocr,
    _clean_general_text,
    _clean_license_plate_text,
    _is_valid_indian_license_plate
)
from .utils import (
    get_model,
    _get_device,
    _annotate_with_color,
    _generate_detection_summary,
    _classify_color_bgr,
    detect_license_plates_as_objects,
    detect_vehicles_in_image,
    _detect_car_color_around_plate
)

__all__ = [
    # Main processing functions
    'predict_image',
    'predict_video',
    'predict_video_optimized',
    'benchmark_video_processing',
    'predict_webcam',
    
    # Text extraction functions
    'extract_text_from_image_json',
    'format_text_extraction_results',
    '_extract_text_ocr',
    '_clean_general_text',
    '_clean_license_plate_text',
    '_is_valid_indian_license_plate',
    
    # Utility functions
    'get_model',
    '_get_device',
    '_classify_color_bgr',
    '_generate_detection_summary',
    '_annotate_with_color',
    '_annotate_from_json_results',
    'detect_license_plates_as_objects',
    'detect_vehicles_in_image',
    '_detect_car_color_around_plate'
]
