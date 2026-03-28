"""
Canberra Vision - Unified Detection System

Multi-Detection Engine integrating:
- PPE Detection (Helmet, Seatbelt, Safety Vest)
- Vehicle Detection (Type and Color)
- Number Plate Recognition (ANPR)
- Parking Slot Detection

Pipeline:
Input → Frame Extractor → Unified Detection Engine → Result Formatter → Database
"""

from .unified_detector import (
    UnifiedDetector,
    UnifiedDetectionResult,
    VehicleInfo,
    PPEInfo,
    PlateInfo,
    ParkingSlotInfo,
    get_unified_detector
)

from .frame_extractor import (
    FrameExtractor,
    InputSourceType,
    create_frame_extractor,
    webcam,
    video_file,
    image_file
)

from .result_formatter import (
    ResultFormatter,
    format_single_result,
    format_batch_results,
    print_formatted_result
)

from .database_service import (
    DatabaseService,
    get_database_service
)

__version__ = "1.0.0"
__author__ = "Canberra Vision"

__all__ = [
    # Detector
    'UnifiedDetector',
    'UnifiedDetectionResult',
    'VehicleInfo',
    'PPEInfo',
    'PlateInfo',
    'ParkingSlotInfo',
    'get_unified_detector',
    
    # Frame Extractor
    'FrameExtractor',
    'InputSourceType',
    'create_frame_extractor',
    'webcam',
    'video_file',
    'image_file',
    
    # Result Formatter
    'ResultFormatter',
    'format_single_result',
    'format_batch_results',
    'print_formatted_result',
    
    # Database Service
    'DatabaseService',
    'get_database_service'
]
