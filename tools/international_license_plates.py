"""
International License Plate Detection and Recognition System
Supports license plates from all countries worldwide
"""

import re
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CountryPlateFormat:
    """License plate format for different countries"""
    country: str
    patterns: List[str]
    examples: List[str]
    description: str

class InternationalLicensePlateRecognizer:
    """Handles license plate detection and recognition for all countries"""
    
    def __init__(self):
        self.country_formats = self._load_country_formats()
        self.ocr_configs = self._get_ocr_configs()
    
    def _load_country_formats(self) -> Dict[str, CountryPlateFormat]:
        """Load license plate formats for all countries"""
        formats = {
            # North America
            'usa': CountryPlateFormat(
                country='USA',
                patterns=[
                    r'^[A-Z]{1,3}[ -]?[A-Z0-9]{1,5}[ -]?[A-Z0-9]{1,3}$',  # Standard US
                    r'^[A-Z]{2}[0-9]{4,6}$',  # California style
                    r'^[0-9]{3}[ -]?[A-Z]{3}$',  # Some states
                ],
                examples=['ABC 1234', '1A-2345', 'XYZ 999', '123 ABC'],
                description='US State Plates'
            ),
            'canada': CountryPlateFormat(
                country='Canada',
                patterns=[
                    r'^[A-Z]{3}[ -]?[0-9]{3}$',  # Ontario
                    r'^[0-9]{3}[ -]?[A-Z]{3}$',  # Quebec
                    r'^[A-Z]{2}[ -]?[0-9]{4}$',  # Other provinces
                ],
                examples=['ABC 123', '123 ABC', 'AB 1234'],
                description='Canadian Province Plates'
            ),
            'mexico': CountryPlateFormat(
                country='Mexico',
                patterns=[
                    r'^[A-Z]{3}[ -]?[0-9]{3}[ -]?[A-Z]{2}$',  # Modern format
                    r'^[A-Z]{2}[ -]?[0-9]{4}[ -]?[A-Z]{2}$',  # Older format
                ],
                examples=['ABC-123-XY', 'AB-1234-CD'],
                description='Mexican State Plates'
            ),
            
            # Europe
            'uk': CountryPlateFormat(
                country='United Kingdom',
                patterns=[
                    r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$',  # Current format
                    r'^[A-Z][0-9]{3}[A-Z]{3}$',   # Older format
                    r'^[A-Z]{3}[0-9]{3}$',        # Pre-2001
                ],
                examples=['AB12 CDE', 'A123 BCD', 'ABC 123'],
                description='UK DVLA Plates'
            ),
            'germany': CountryPlateFormat(
                country='Germany',
                patterns=[
                    r'^[A-Z]{1,3}[ -]?[A-Z]{1,2}[ -]?[0-9]{1,4}[ -]?[A-Z]{0,2}$',
                    r'^[A-Z]{1,3}[ -]?[0-9]{1,4}[ -]?[A-Z]{0,2}$',
                ],
                examples=['B-AB 123', 'M-AB 1234', 'HH-AB 123'],
                description='German Registration Plates'
            ),
            'france': CountryPlateFormat(
                country='France',
                patterns=[
                    r'^[A-Z]{2}[ -]?[0-9]{3}[ -]?[A-Z]{2}$',  # Current
                    r'^[0-9]{4}[A-Z]{2}[0-9]{2}$',          # Older
                ],
                examples=['AB-123-CD', '1234 AB 56'],
                description='French Department Plates'
            ),
            'italy': CountryPlateFormat(
                country='Italy',
                patterns=[
                    r'^[A-Z]{2}[0-9]{3}[A-Z]{2}$',
                    r'^[A-Z]{2}[0-9]{5}$',
                ],
                examples=['AB123CD', 'AB12345'],
                description='Italian License Plates'
            ),
            'spain': CountryPlateFormat(
                country='Spain',
                patterns=[
                    r'^[0-9]{4}[ -]?[A-Z]{3}$',
                    r'^[A-Z]{1,2}[ -]?[0-9]{4}[ -]?[A-Z]{1,2}$',
                ],
                examples=['1234 ABC', 'A-1234-BC'],
                description='Spanish Provincial Plates'
            ),
            'netherlands': CountryPlateFormat(
                country='Netherlands',
                patterns=[
                    r'^[0-9]{2}[ -]?[A-Z]{3}[ -]?[0-9]{2}$',
                    r'^[A-Z]{2}[ -]?[0-9]{3}[ -]?[A-Z]{2}$',
                ],
                examples=['12-ABC-34', 'AB-123-CD'],
                description='Dutch License Plates'
            ),
            'belgium': CountryPlateFormat(
                country='Belgium',
                patterns=[
                    r'^[0-9]{3}[ -]?[A-Z]{3}$',
                    r'^[A-Z]{1}[ -]?[0-9]{3}[ -]?[A-Z]{3}$',
                ],
                examples=['123-ABC', '1-123-ABC'],
                description='Belgian Plates'
            ),
            'switzerland': CountryPlateFormat(
                country='Switzerland',
                patterns=[
                    r'^[A-Z]{2}[ -]?[0-9]{6}$',
                ],
                examples=['ZH-123456', 'BE 123456'],
                description='Swiss Cantonal Plates'
            ),
            'austria': CountryPlateFormat(
                country='Austria',
                patterns=[
                    r'^[A-Z]{1,3}[ -]?[0-9]{4,6}[A-Z]?$',
                ],
                examples=['W-12345', 'G-123456A'],
                description='Austrian District Plates'
            ),
            'poland': CountryPlateFormat(
                country='Poland',
                patterns=[
                    r'^[A-Z]{2,3}[ -]?[0-9]{4,5}[A-Z]{0,2}$',
                ],
                examples=['KR 12345', 'WA 1234AB'],
                description='Polish Voivodeship Plates'
            ),
            'sweden': CountryPlateFormat(
                country='Sweden',
                patterns=[
                    r'^[A-Z]{3}[ -]?[0-9]{3}[A-Z]?$',
                ],
                examples=['ABC 123', 'ABC123A'],
                description='Swedish Plates'
            ),
            'norway': CountryPlateFormat(
                country='Norway',
                patterns=[
                    r'^[A-Z]{2}[ -]?[0-9]{5}$',
                ],
                examples=['AB 12345', 'AB12345'],
                description='Norwegian County Plates'
            ),
            'denmark': CountryPlateFormat(
                country='Denmark',
                patterns=[
                    r'^[0-9]{2}[ -]?[A-Z]{3}[ -]?[0-9]{2}$',
                ],
                examples=['12-ABC-34'],
                description='Danish Plates'
            ),
            'finland': CountryPlateFormat(
                country='Finland',
                patterns=[
                    r'^[A-Z]{1,3}[ -]?[0-9]{1,3}[ -]?[A-Z]{3}$',
                ],
                examples=['ABC-123', 'A-1-ABC'],
                description='Finnish Plates'
            ),
            
            # Asia
            'japan': CountryPlateFormat(
                country='Japan',
                patterns=[
                    r'^[A-Z]{3,4}[ -]?[0-9]{4}$',
                    r'^[ひらがな]{1,2}[ -]?[0-9]{4}[ -]?[A-Z]{1,2}$',
                    r'^[A-Z]{2}[0-9]{4}$',
                ],
                examples=['品川 123 あ', 'ABC-1234', 'AB1234'],
                description='Japanese Prefectural Plates'
            ),
            'south_korea': CountryPlateFormat(
                country='South Korea',
                patterns=[
                    r'^[0-9]{2}[ -]?[가-힣]{1,2}[ -]?[0-9]{4}$',
                    r'^[0-9]{3}[ -]?[가-힣]{1,2}[ -]?[0-9]{4}$',
                ],
                examples=['12가1234', '123서4567'],
                description='Korean Regional Plates'
            ),
            'china': CountryPlateFormat(
                country='China',
                patterns=[
                    r'^[A-Z]{1}[0-9A-Z]{1}[ -]?[0-9A-Z]{5}$',
                    r'^[京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][0-9A-Z]{5}$',
                ],
                examples=['京A12345', '沪B123CD', '粤S12345'],
                description='Chinese Provincial Plates'
            ),
            'singapore': CountryPlateFormat(
                country='Singapore',
                patterns=[
                    r'^[A-Z]{3}[ -]?[0-9]{4}[A-Z]?$',
                ],
                examples=['SGH1234A', 'SK 1234 B'],
                description='Singapore Plates'
            ),
            'malaysia': CountryPlateFormat(
                country='Malaysia',
                patterns=[
                    r'^[A-Z]{1,3}[ -]?[0-9]{1,4}[ -]?[A-Z]{1,2}$',
                ],
                examples=['W 1234 A', 'ABC 5678 D'],
                description='Malaysian State Plates'
            ),
            'thailand': CountryPlateFormat(
                country='Thailand',
                patterns=[
                    r'^[ก-ฮ]{1,3}[ -]?[0-9]{1,4}[ -]?[0-9]{2}$',
                    r'^[0-9]{4}[ -]?[ก-ฮ]{2,3}$',
                ],
                examples=['กท 1234', 'กรุงเทพ 12345'],
                description='Thai Provincial Plates'
            ),
            
            # Middle East
            'uae': CountryPlateFormat(
                country='UAE',
                patterns=[
                    r'^[0-9]{5,6}$',
                    r'^[A-Z]{1,2}[ -]?[0-9]{5}$',
                ],
                examples=['12345', 'AB 12345'],
                description='UAE Emirate Plates'
            ),
            'saudi_arabia': CountryPlateFormat(
                country='Saudi Arabia',
                patterns=[
                    r'^[0-9]{4}[ -]?[A-Z]{3}$',
                    r'^[A-Z]{4}[ -]?[0-9]{4}$',
                ],
                examples=['1234 ABC', 'ABCD 1234'],
                description='Saudi Arabian Plates'
            ),
            
            # South America
            'brazil': CountryPlateFormat(
                country='Brazil',
                patterns=[
                    r'^[A-Z]{3}[ -]?[0-9]{4}[A-Z]?$',
                    r'^[A-Z]{3}[ -]?[0-9]{4}$',
                ],
                examples=['ABC-1234', 'ABC1234A'],
                description='Brazilian State Plates'
            ),
            'argentina': CountryPlateFormat(
                country='Argentina',
                patterns=[
                    r'^[A-Z]{3}[ -]?[0-9]{3}$',
                    r'^[A-Z]{2}[ -]?[0-9]{3}[ -]?[A-Z]{2}$',
                ],
                examples=['ABC 123', 'AB 123 CD'],
                description='Argentinean Provincial Plates'
            ),
            'chile': CountryPlateFormat(
                country='Chile',
                patterns=[
                    r'^[A-Z]{4}[ -]?[0-9]{2}$',
                    r'^[A-Z]{2}[ -]?[0-9]{4}$',
                ],
                examples=['ABCD 12', 'AB 1234'],
                description='Chilean Regional Plates'
            ),
            
            # Africa
            'south_africa': CountryPlateFormat(
                country='South Africa',
                patterns=[
                    r'^[A-Z]{2}[ -]?[0-9]{3}[ -]?[A-Z]{2}$',
                    r'^[A-Z]{2}[ -]?[0-9]{5}$',
                ],
                examples=['CA 123-45', 'GP 12345'],
                description='South African Provincial Plates'
            ),
            'egypt': CountryPlateFormat(
                country='Egypt',
                patterns=[
                    r'^[0-9]{4}[ -]?[A-Z]{3}$',
                    r'^[A-Z]{2}[ -]?[0-9]{5}$',
                ],
                examples=['1234 ABC', 'AB 12345'],
                description='Egyptian Governorate Plates'
            ),
            
            # Oceania
            'australia': CountryPlateFormat(
                country='Australia',
                patterns=[
                    r'^[0-9]{3}[ -]?[A-Z]{3}$',  # Most states
                    r'^[A-Z]{3}[ -]?[0-9]{3}$',  # Some states
                    r'^[A-Z]{2}[ -]?[0-9]{2}[ -]?[A-Z]{2}$',  # Victoria
                ],
                examples=['123 ABC', 'ABC 123', 'AB 12 CD'],
                description='Australian State Plates'
            ),
            'new_zealand': CountryPlateFormat(
                country='New Zealand',
                patterns=[
                    r'^[A-Z]{1,3}[ -]?[0-9]{4}[A-Z]?$',
                ],
                examples=['ABC1234', 'AB 1234 C'],
                description='New Zealand Plates'
            ),
        }
        
        # Add generic international patterns
        formats['generic'] = CountryPlateFormat(
            country='Generic International',
            patterns=[
                r'^[A-Z0-9]{5,12}$',  # Most international plates
                r'^[A-Z]{2,4}[ -]?[0-9]{2,6}[ -]?[A-Z]{0,3}$',  # Standard format
                r'^[0-9]{3,6}[ -]?[A-Z]{2,4}$',  # Number-first format
            ],
                examples=['ABC123', 'AB-123-CD', '123-ABC'],
                description='Generic International Format'
        )
        
        return formats
    
    def _get_ocr_configs(self) -> List[str]:
        """Get OCR configurations optimized for different plate styles"""
        return [
            # Standard alphanumeric
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            
            # Include common separators
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -',
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -',
            
            # More flexible (for unusual formats)
            r'--oem 3 --psm 6',
            r'--oem 3 --psm 11',
            r'--oem 3 --psm 13',
            
            # Single line assumption
            r'--oem 3 --psm 7',
            r'--oem 3 --psm 8',
        ]
    
    def detect_country_from_plate(self, plate_text: str) -> List[Dict]:
        """
        Detect which country(s) a license plate might belong to
        
        Args:
            plate_text: Extracted license plate text
            
        Returns:
            List of possible countries with confidence scores
        """
        plate_text = plate_text.upper().strip()
        possible_countries = []
        
        for country_code, format_info in self.country_formats.items():
            matches = 0
            total_patterns = len(format_info.patterns)
            
            for pattern in format_info.patterns:
                if re.match(pattern, plate_text):
                    matches += 1
            
            if matches > 0:
                confidence = matches / total_patterns
                possible_countries.append({
                    'country': format_info.country,
                    'code': country_code,
                    'confidence': confidence,
                    'description': format_info.description,
                    'examples': format_info.examples
                })
        
        # Sort by confidence
        possible_countries.sort(key=lambda x: x['confidence'], reverse=True)
        
        return possible_countries
    
    def extract_license_plates_from_text(self, text: str) -> List[Dict]:
        """
        Extract potential license plates from text and identify countries
        
        Args:
            text: Raw OCR text
            
        Returns:
            List of detected license plates with country information
        """
        detected_plates = []
        
        # Clean and split text
        words = self._clean_text(text)
        
        # Method 1: Check individual words
        for word in words:
            if len(word) >= 5:  # Minimum plate length
                country_matches = self.detect_country_from_plate(word)
                if country_matches:
                    detected_plates.append({
                        'text': word,
                        'countries': country_matches,
                        'method': 'individual_word',
                        'confidence': country_matches[0]['confidence']
                    })
        
        # Method 2: Check combined text chunks
        combined_text = ''.join(words)
        for i in range(len(combined_text) - 5):
            for length in range(5, min(15, len(combined_text) - i + 1)):
                chunk = combined_text[i:i+length]
                country_matches = self.detect_country_from_plate(chunk)
                
                if country_matches and country_matches[0]['confidence'] > 0.5:
                    # Avoid duplicates
                    is_duplicate = False
                    for existing in detected_plates:
                        if existing['text'] == chunk:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        detected_plates.append({
                            'text': chunk,
                            'countries': country_matches,
                            'method': 'combined_text',
                            'confidence': country_matches[0]['confidence']
                        })
        
        # Remove duplicates and sort by confidence
        unique_plates = []
        seen_texts = set()
        
        for plate in detected_plates:
            if plate['text'] not in seen_texts:
                seen_texts.add(plate['text'])
                unique_plates.append(plate)
        
        unique_plates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return unique_plates[:10]  # Return top 10 results
    
    def _clean_text(self, text: str) -> List[str]:
        """Clean OCR text and split into meaningful chunks"""
        # Remove common OCR errors
        text = re.sub(r'[O0]', '0', text)  # Replace O with 0
        text = re.sub(r'[I1l]', '1', text)  # Replace I/l with 1
        text = re.sub(r'[S5]', '5', text)   # Replace S with 5
        text = re.sub(r'[Z2]', '2', text)   # Replace Z with 2
        text = re.sub(r'[G6]', '6', text)   # Replace G with 6
        text = re.sub(r'[B8]', '8', text)   # Replace B with 8
        
        # Split into words and clean each
        words = []
        for word in text.split():
            # Keep only alphanumeric characters and common separators
            cleaned = re.sub(r'[^A-Z0-9 -]', '', word.upper())
            if cleaned and len(cleaned) >= 3:
                words.append(cleaned)
        
        return words
    
    def get_plate_statistics(self, detected_plates: List[Dict]) -> Dict:
        """Get statistics about detected plates"""
        if not detected_plates:
            return {'total_plates': 0, 'countries': {}, 'methods': {}}
        
        country_counts = {}
        method_counts = {}
        
        for plate in detected_plates:
            # Count countries
            if plate['countries']:
                top_country = plate['countries'][0]['country']
                country_counts[top_country] = country_counts.get(top_country, 0) + 1
            
            # Count methods
            method = plate['method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            'total_plates': len(detected_plates),
            'countries': country_counts,
            'methods': method_counts,
            'most_likely_country': max(country_counts.items(), key=lambda x: x[1]) if country_counts else None
        }
    
    def format_plate_output(self, plate_info: Dict) -> str:
        """Format license plate information for display"""
        text = plate_info['text']
        countries = plate_info['countries']
        
        if not countries:
            return f"Plate: {text} (Unknown format)"
        
        top_country = countries[0]
        output = f"Plate: {text}\n"
        output += f"Country: {top_country['country']} ({top_country['confidence']:.1%} confidence)\n"
        output += f"Description: {top_country['description']}\n"
        
        if len(countries) > 1:
            output += f"Other possibilities: {', '.join([c['country'] for c in countries[1:3]])}\n"
        
        output += f"Examples: {', '.join(top_country['examples'])}"
        
        return output


# Integration function for existing app.py
def extract_international_license_plates(image_bgr: np.ndarray, ocr_results: List[str]) -> Dict:
    """
    Extract and identify international license plates from image
    
    Args:
        image_bgr: Input image in BGR format
        ocr_results: List of OCR text results
        
    Returns:
        Dictionary containing detected plates and analysis
    """
    recognizer = InternationalLicensePlateRecognizer()
    
    all_detected_plates = []
    
    # Process each OCR result
    for ocr_text in ocr_results:
        if ocr_text and ocr_text.strip():
            plates = recognizer.extract_license_plates_from_text(ocr_text)
            all_detected_plates.extend(plates)
    
    # Remove duplicates
    unique_plates = []
    seen_texts = set()
    
    for plate in all_detected_plates:
        if plate['text'] not in seen_texts:
            seen_texts.add(plate['text'])
            unique_plates.append(plate)
    
    # Sort by confidence
    unique_plates.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Get statistics
    stats = recognizer.get_plate_statistics(unique_plates)
    
    return {
        'detected_plates': unique_plates,
        'statistics': stats,
        'total_unique_plates': len(unique_plates),
        'supported_countries': list(recognizer.country_formats.keys())
    }


def get_country_plate_info(country_code: str) -> Optional[Dict]:
    """Get detailed information about a country's license plate format"""
    recognizer = InternationalLicensePlateRecognizer()
    
    if country_code.lower() in recognizer.country_formats:
        format_info = recognizer.country_formats[country_code.lower()]
        return {
            'country': format_info.country,
            'patterns': format_info.patterns,
            'examples': format_info.examples,
            'description': format_info.description
        }
    
    return None


if __name__ == "__main__":
    # Test the international license plate recognizer
    recognizer = InternationalLicensePlateRecognizer()
    
    print("International License Plate Recognition System")
    print("=" * 50)
    print(f"Supported countries: {len(recognizer.country_formats)}")
    print("Countries:", list(recognizer.country_formats.keys()))
    
    # Test with sample plates
    test_plates = [
        "ABC1234",    # Generic/US
        "AB12CDE",    # UK
        "B-AB123",    # Germany
        "1234ABC",    # Australia
        "京A12345",   # China
        "12가1234",   # South Korea
    ]
    
    print("\nTest Results:")
    for plate in test_plates:
        countries = recognizer.detect_country_from_plate(plate)
        print(f"\nPlate: {plate}")
        if countries:
            for country in countries[:3]:
                print(f"  {country['country']}: {country['confidence']:.1%}")
        else:
            print("  No matches found")
