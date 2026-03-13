"""
International License Plate Integration for YOLO26
This file provides integration functions to add worldwide license plate support
to your existing YOLO26 system without modifying the main app.py
"""

import cv2
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import json

# Import the international license plate system
try:
    from international_license_plates import (
        InternationalLicensePlateRecognizer, 
        extract_international_license_plates,
        get_country_plate_info
    )
    INTERNATIONAL_AVAILABLE = True
    print("[INFO] International license plate system available")
except ImportError:
    INTERNATIONAL_AVAILABLE = False
    print("[ERROR] International license plate system not available")

# Import enhanced detection for challenging images
try:
    from enhanced_detection import enhanced_license_plate_detection
    ENHANCED_DETECTION_AVAILABLE = True
    print("[INFO] Enhanced detection for challenging images available")
except ImportError:
    ENHANCED_DETECTION_AVAILABLE = False
    print("[ERROR] Enhanced detection not available")


class InternationalYOLO26Integration:
    """
    Integration class to add international license plate support to YOLO26
    """
    
    def __init__(self):
        self.recognizer = None
        if INTERNATIONAL_AVAILABLE:
            self.recognizer = InternationalLicensePlateRecognizer()
            print(f"[INFO] Loaded support for {len(self.recognizer.country_formats)} countries")
    
    def enhanced_text_extraction(self, image_bgr: np.ndarray, ocr_results: List[str]) -> Dict:
        """
        Enhanced text extraction with international license plate recognition
        
        Args:
            image_bgr: Input image in BGR format
            ocr_results: List of OCR text results from your existing system
            
        Returns:
            Enhanced results with international plate information
        """
        if not INTERNATIONAL_AVAILABLE:
            return {
                'error': 'International license plate system not available',
                'original_ocr_results': ocr_results
            }
        
        # Process OCR results through international recognizer
        international_results = extract_international_license_plates(image_bgr, ocr_results)
        
        # Enhance each detected plate with additional information
        enhanced_plates = []
        for plate in international_results['detected_plates']:
            enhanced_plate = {
                'text': plate['text'],
                'confidence': plate['confidence'],
                'method': plate['method'],
                'countries': plate['countries'],
                'most_likely_country': plate['countries'][0] if plate['countries'] else None,
                'country_examples': plate['countries'][0]['examples'] if plate['countries'] else [],
                'description': plate['countries'][0]['description'] if plate['countries'] else 'Unknown'
            }
            enhanced_plates.append(enhanced_plate)
        
        return {
            'international_plates': enhanced_plates,
            'statistics': international_results['statistics'],
            'total_unique_plates': international_results['total_unique_plates'],
            'supported_countries': international_results['supported_countries'],
            'original_ocr_results': ocr_results
        }
    
    def detect_plates_in_challenging_images(self, image_bgr: np.ndarray) -> Dict:
        """
        Enhanced detection for challenging (blurry/angled) images with international support
        
        Args:
            image_bgr: Input image in BGR format
            
        Returns:
            Detection results with international plate information
        """
        results = {
            'enhanced_detection_used': False,
            'plates_found': [],
            'international_plates': []
        }
        
        # Try enhanced detection first
        if ENHANCED_DETECTION_AVAILABLE:
            try:
                enhanced_result = enhanced_license_plate_detection(image_bgr)
                results['enhanced_detection_used'] = True
                results['enhanced_detection'] = enhanced_result
                
                if enhanced_result['plate_detected']:
                    # Extract text from detected plates using existing OCR
                    x1, y1, x2, y2 = enhanced_result['plate_bbox']
                    plate_crop = image_bgr[y1:y2, x1:x2]
                    
                    # Use your existing OCR function (you'll need to import this)
                    try:
                        from app import _extract_text_from_license_plate_crop
                        plate_text = _extract_text_from_license_plate_crop(plate_crop)
                        
                        if plate_text and INTERNATIONAL_AVAILABLE:
                            # Process through international recognizer
                            international_results = extract_international_license_plates(image_bgr, [plate_text])
                            results['international_plates'] = international_results['detected_plates']
                        
                        results['plates_found'].append({
                            'bbox': enhanced_result['plate_bbox'],
                            'color': enhanced_result['color'],
                            'text': plate_text,
                            'method': 'enhanced_detection'
                        })
                        
                    except ImportError:
                        print("[WARNING] Cannot import _extract_text_from_license_plate_crop from app")
                        results['plates_found'].append({
                            'bbox': enhanced_result['plate_bbox'],
                            'color': enhanced_result['color'],
                            'text': '',
                            'method': 'enhanced_detection_no_ocr'
                        })
                        
            except Exception as e:
                print(f"[ERROR] Enhanced detection failed: {e}")
        
        return results
    
    def validate_international_plate(self, plate_text: str) -> Dict:
        """
        Validate a license plate text against international formats
        
        Args:
            plate_text: License plate text to validate
            
        Returns:
            Validation results with country information
        """
        if not INTERNATIONAL_AVAILABLE:
            return {
                'valid': False,
                'error': 'International license plate system not available'
            }
        
        if not plate_text or len(plate_text.strip()) < 5:
            return {
                'valid': False,
                'reason': 'Plate text too short or empty'
            }
        
        # Detect country matches
        country_matches = self.recognizer.detect_country_from_plate(plate_text)
        
        return {
            'valid': len(country_matches) > 0,
            'country_matches': country_matches,
            'most_likely_country': country_matches[0] if country_matches else None,
            'confidence': country_matches[0]['confidence'] if country_matches else 0.0
        }
    
    def get_country_information(self, country_code: str) -> Dict:
        """
        Get detailed information about a country's license plate format
        
        Args:
            country_code: Country code (e.g., 'usa', 'uk', 'germany')
            
        Returns:
            Country license plate information
        """
        if not INTERNATIONAL_AVAILABLE:
            return {'error': 'International system not available'}
        
        return get_country_plate_info(country_code)
    
    def process_image_with_international_support(self, image_bgr: np.ndarray, existing_results: Dict = None) -> Dict:
        """
        Process an image with full international license plate support
        
        Args:
            image_bgr: Input image in BGR format
            existing_results: Results from your existing YOLO26 system
            
        Returns:
            Enhanced results with international plate information
        """
        final_results = {
            'original_results': existing_results,
            'international_enhancement': {
                'enabled': INTERNATIONAL_AVAILABLE,
                'plates_processed': 0,
                'countries_detected': [],
                'enhanced_plates': []
            }
        }
        
        if not INTERNATIONAL_AVAILABLE:
            return final_results
        
        # Extract OCR results from existing data
        ocr_texts = []
        
        if existing_results:
            # Extract text from existing results
            if 'text_extraction' in existing_results:
                text_data = existing_results['text_extraction']
                
                # Get license plates
                if 'license_plates' in text_data:
                    for plate in text_data['license_plates']:
                        if 'plate_text' in plate:
                            ocr_texts.append(plate['plate_text'])
                
                # Get general text
                if 'general_text' in text_data:
                    for text_item in text_data['general_text']:
                        if 'text' in text_item:
                            ocr_texts.append(text_item['text'])
        
        # Process through international system
        if ocr_texts:
            international_results = self.enhanced_text_extraction(image_bgr, ocr_texts)
            final_results['international_enhancement'].update({
                'plates_processed': len(international_results['international_plates']),
                'countries_detected': list(international_results['statistics'].get('countries', {}).keys()),
                'enhanced_plates': international_results['international_plates'],
                'statistics': international_results['statistics']
            })
        
        # Also try enhanced detection for challenging images
        enhanced_detection = self.detect_plates_in_challenging_images(image_bgr)
        if enhanced_detection['enhanced_detection_used']:
            final_results['enhanced_detection'] = enhanced_detection
        
        return final_results


# Convenience functions for easy integration
def add_international_support_to_results(image_bgr: np.ndarray, existing_results: Dict) -> Dict:
    """
    Add international license plate support to existing YOLO26 results
    
    Args:
        image_bgr: Input image in BGR format
        existing_results: Your existing YOLO26 detection results
        
    Returns:
        Enhanced results with international plate information
    """
    integration = InternationalYOLO26Integration()
    return integration.process_image_with_international_support(image_bgr, existing_results)


def validate_license_plate_international(plate_text: str) -> Dict:
    """
    Validate a license plate against international formats
    
    Args:
        plate_text: License plate text to validate
        
    Returns:
        Validation results with country information
    """
    integration = InternationalYOLO26Integration()
    return integration.validate_international_plate(plate_text)


def get_supported_countries() -> List[str]:
    """Get list of all supported countries"""
    if not INTERNATIONAL_AVAILABLE:
        return []
    
    recognizer = InternationalLicensePlateRecognizer()
    return list(recognizer.country_formats.keys())


def format_international_plate_for_display(plate_info: Dict) -> str:
    """
    Format international license plate information for display
    
    Args:
        plate_info: Plate information from international system
        
    Returns:
        Formatted string for display
    """
    if not plate_info or 'countries' not in plate_info or not plate_info['countries']:
        return f"Plate: {plate_info.get('text', 'Unknown')} (No country match)"
    
    text = plate_info['text']
    country = plate_info['countries'][0]
    
    formatted = f"🚗 Plate: {text}\n"
    formatted += f"🌍 Country: {country['country']} ({country['confidence']:.1%} confidence)\n"
    formatted += f"📝 Format: {country['description']}\n"
    formatted += f"💡 Examples: {', '.join(country['examples'])}"
    
    if len(plate_info['countries']) > 1:
        other_countries = [c['country'] for c in plate_info['countries'][1:3]]
        formatted += f"\n🔄 Also possible: {', '.join(other_countries)}"
    
    return formatted


# Example usage and testing
if __name__ == "__main__":
    print("International License Plate Integration for YOLO26")
    print("=" * 60)
    
    # Test the integration
    integration = InternationalYOLO26Integration()
    
    if INTERNATIONAL_AVAILABLE:
        print(f"✅ International system loaded with {len(get_supported_countries())} countries")
        print(f"📋 Supported countries: {', '.join(get_supported_countries()[:10])}...")
        
        # Test validation
        test_plates = [
            "ABC1234",    # US/Generic
            "AB12CDE",    # UK
            "B-AB123",    # Germany
            "1234ABC",    # Australia
            "京A12345",   # China
        ]
        
        print("\n🧪 Testing plate validation:")
        for plate in test_plates:
            result = validate_license_plate_international(plate)
            if result['valid']:
                country = result['most_likely_country']
                print(f"  ✅ {plate} → {country['country']} ({country['confidence']:.1%})")
            else:
                print(f"  ❌ {plate} → No match")
    else:
        print("❌ International system not available")
    
    if ENHANCED_DETECTION_AVAILABLE:
        print("✅ Enhanced detection for challenging images available")
    else:
        print("❌ Enhanced detection not available")
    
    print("\n📖 Usage:")
    print("1. Add to existing results: add_international_support_to_results(image, results)")
    print("2. Validate plate: validate_license_plate_international('ABC123')")
    print("3. Get country info: get_country_information('usa')")
    print("4. Format for display: format_international_plate_for_display(plate_info)")
