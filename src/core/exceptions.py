"""
Custom exceptions for YOLO26 project.
"""


class YOLOException(Exception):
    """Base exception for YOLO26 project."""
    pass


class ModelNotFoundError(YOLOException):
    """Raised when a model file is not found."""
    pass


class ProcessingError(YOLOException):
    """Raised when processing fails."""
    pass


class GPUError(YOLOException):
    """Raised when GPU operations fail."""
    pass


class OCRError(YOLOException):
    """Raised when OCR operations fail."""
    pass


class ConfigurationError(YOLOException):
    """Raised when there's a configuration error."""
    pass
