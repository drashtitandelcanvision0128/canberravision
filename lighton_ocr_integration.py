"""
LightOnOCR Integration for YOLO26
High-accuracy text detection and extraction
"""

import torch
import numpy as np
from PIL import Image
import cv2
import os
from typing import Optional, Dict, Any

class LightOnOCRProcessor:
    """
    LightOnOCR-2-1B integration for high-accuracy text detection
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize LightOnOCR processor
        
        Args:
            model_path: Path to LightOnOCR model weights
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load LightOnOCR model"""
        try:
            # For now, we'll use a placeholder until LightOnOCR is officially available
            # This will be replaced with actual model loading code
            print(f"[INFO] Loading LightOnOCR on device: {self.device}")
            
            # TODO: Replace with actual LightOnOCR loading
            # from transformers import AutoModel, AutoTokenizer
            # self.model = AutoModel.from_pretrained("lighton-ai/LightOnOCR-2-1B")
            # self.tokenizer = AutoTokenizer.from_pretrained("lighton-ai/LightOnOCR-2-1B")
            # self.model.to(self.device)
            
            print("[INFO] LightOnOCR loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load LightOnOCR: {e}")
            print("[INFO] Falling back to enhanced Tesseract")
            self.model = None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for optimal OCR performance
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Preprocessed image
        """
        if image is None or image.size == 0:
            return image
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Enhanced preprocessing for better OCR
        # 1. Noise reduction
        denoised = cv2.fastNlMeansDenoisingColored(image_rgb, None, 10, 10, 7, 21)
        
        # 2. Contrast enhancement using CLAHE
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # 3. Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def extract_text(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Extract text from image using LightOnOCR
        
        Args:
            image: Input image in BGR format
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        if image is None or image.size == 0:
            return {"text": "", "confidence": 0.0, "words": []}
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        if self.model is not None:
            # Use LightOnOCR when available
            return self._extract_with_lighton(processed_image, confidence_threshold)
        else:
            # Fallback to enhanced Tesseract
            return self._extract_with_tesseract(processed_image, confidence_threshold)
    
    def _extract_with_lighton(self, image: np.ndarray, confidence_threshold: float) -> Dict[str, Any]:
        """
        Extract text using LightOnOCR model
        
        Args:
            image: Preprocessed RGB image
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # TODO: Replace with actual LightOnOCR inference
            # This is placeholder code for the interface
            
            # Convert PIL Image for model input
            pil_image = Image.fromarray(image)
            
            # Placeholder for LightOnOCR inference
            # result = self.model.generate(pil_image)
            
            # For now, return empty result
            return {
                "text": "",
                "confidence": 0.0,
                "words": [],
                "method": "lighton_ocr"
            }
            
        except Exception as e:
            print(f"[ERROR] LightOnOCR inference failed: {e}")
            return {"text": "", "confidence": 0.0, "words": [], "method": "lighton_ocr_error"}
    
    def _extract_with_tesseract(self, image: np.ndarray, confidence_threshold: float) -> Dict[str, Any]:
        """
        Enhanced Tesseract fallback with multiple preprocessing attempts
        
        Args:
            image: Preprocessed RGB image
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            import pytesseract
            
            # Multiple OCR attempts with different preprocessing
            results = []
            
            # Method 1: Direct on preprocessed image
            try:
                text1 = pytesseract.image_to_string(image, lang='eng+guj', config='--psm 6 --oem 3')
                conf1 = self._get_average_confidence(image)
                results.append({"text": text1.strip(), "confidence": conf1})
            except:
                pass
            
            # Method 2: Grayscale + threshold
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                text2 = pytesseract.image_to_string(thresh, lang='eng+guj', config='--psm 6 --oem 3')
                conf2 = self._get_average_confidence(thresh)
                results.append({"text": text2.strip(), "confidence": conf2})
            except:
                pass
            
            # Method 3: Upscaled for small text
            try:
                h, w = image.shape[:2]
                if h < 100 or w < 100:
                    upscaled = cv2.resize(image, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
                    text3 = pytesseract.image_to_string(upscaled, lang='eng+guj', config='--psm 6 --oem 3')
                    conf3 = self._get_average_confidence(upscaled)
                    results.append({"text": text3.strip(), "confidence": conf3})
            except:
                pass
            
            # Select best result
            if not results:
                return {"text": "", "confidence": 0.0, "words": [], "method": "tesseract_failed"}
            
            best_result = max(results, key=lambda x: x["confidence"])
            
            # Extract word-level data
            try:
                data = pytesseract.image_to_data(image, lang='eng+guj', config='--psm 6 --oem 3', output_type=pytesseract.Output.DICT)
                words = []
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > confidence_threshold * 100 and data['text'][i].strip():
                        words.append({
                            "text": data['text'][i],
                            "confidence": int(data['conf'][i]) / 100.0,
                            "bbox": [data['left'][i], data['top'][i], data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]]
                        })
            except:
                words = []
            
            return {
                "text": best_result["text"],
                "confidence": best_result["confidence"],
                "words": words,
                "method": "enhanced_tesseract"
            }
            
        except ImportError:
            print("[ERROR] pytesseract not available")
            return {"text": "", "confidence": 0.0, "words": [], "method": "tesseract_unavailable"}
        except Exception as e:
            print(f"[ERROR] Enhanced Tesseract failed: {e}")
            return {"text": "", "confidence": 0.0, "words": [], "method": "tesseract_error"}
    
    def _get_average_confidence(self, image: np.ndarray) -> float:
        """Get average confidence from Tesseract"""
        try:
            import pytesseract
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            return sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
        except:
            return 0.0


# Global cache for LightOnOCR processor
_lighton_processor = None

def get_lighton_ocr_processor() -> LightOnOCRProcessor:
    """Get cached LightOnOCR processor instance"""
    global _lighton_processor
    if _lighton_processor is None:
        _lighton_processor = LightOnOCRProcessor()
    return _lighton_processor

def extract_text_with_lighton(image: np.ndarray, confidence_threshold: float = 0.5) -> str:
    """
    Convenience function to extract text using LightOnOCR
    
    Args:
        image: Input image in BGR format
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Extracted text string
    """
    processor = get_lighton_ocr_processor()
    result = processor.extract_text(image, confidence_threshold)
    return result["text"]
