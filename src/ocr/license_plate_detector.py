"""
License Plate Detector Module
Intelligent license plate validation and detection logic.
"""

import re
from typing import Dict, List, Optional, Tuple, Any


class LicensePlateDetector:
    """
    Intelligent license plate detector with validation logic to distinguish
    between actual license plates and brand logos or other text.
    """
    
    def __init__(self):
        """Initialize license plate detector."""
        # Common license plate patterns (international)
        self.plate_patterns = [
            # Standard format: 2-3 letters + 2-4 numbers (e.g., "AB 123", "ABC 1234")
            r'^[A-Z]{2,3}\s?\d{2,4}$',
            # Reverse format: numbers + letters (e.g., "123 AB", "1234 ABC")
            r'^\d{2,4}\s?[A-Z]{2,3}$',
            # Mixed format with letters and numbers (e.g., "AB 12 CD", "123 AB 456")
            r'^[A-Z]{1,4}\s?\d{1,4}\s?[A-Z]{0,3}\s?\d{0,3}$',
            # Indian format: 2 letters + 1-2 numbers + 1-2 letters + 4 numbers
            r'^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{4}$',
            # Simple alphanumeric format (6-8 characters)
            r'^[A-Z0-9]{6,8}$',
            # Format with spaces (e.g., "BAD 231")
            r'^[A-Z]{3,4}\s?\d{1,4}$',
            # Bulgarian format: 1 letter + 4 numbers + 2 letters (e.g., "B 2228 HM")
            r'^[A-Z]{1,2}\s?\d{4}\s?[A-Z]{2}$',
            # European format: 1-2 letters + 3-4 numbers + 1-2 letters (e.g., "B 2228 HM", "CA 1234 AB")
            r'^[A-Z]{1,2}\s?\d{3,4}\s?[A-Z]{2,3}$',
            # German format: 1-3 letters + 1-2 letters + 1-4 numbers (e.g., "M AB 123")
            r'^[A-Z]{1,3}\s?[A-Z]{1,2}\s?\d{1,4}$',
            # UK format: 2 letters + 2 numbers + 3 letters (e.g., "AB12 ABC")
            r'^[A-Z]{2}\s?\d{2}\s?[A-Z]{3}$',
            # US format: 3 letters + 3-4 numbers (e.g., "ABC 123", "ABC 1234")
            r'^[A-Z]{3}\s?\d{3,4}$',
            # Generic format: letters-numbers-letters (e.g., "B 2228 HM")
            r'^[A-Z]{1,2}\s?\d{2,4}\s?[A-Z]{1,3}$',
            # NEW: Mixed alphanumeric like IM4U 555 (Malaysian/Singapore format)
            r'^[A-Z]{1,3}\d{1,2}[A-Z]{1,3}\s?\d{1,4}$',
            # NEW: Flexible format - 4-8 alphanumeric chars with at least 2 letters and 2 numbers
            r'^(?=(?:.*[A-Z]){2,})(?=(?:.*\d){2,})[A-Z0-9]{4,8}$',
        ]
        
        # Common brand names and logos to exclude
        self.brand_names = {
            'FORD', 'TOYOTA', 'HONDA', 'BMW', 'MERCEDES', 'AUDI', 'VOLKSWAGEN',
            'NISSAN', 'HYUNDAI', 'KIA', 'MAZDA', 'SUBARU', 'MITSUBISHI',
            'JEEP', 'DODGE', 'CHEVROLET', 'CADILLAC', 'LINCOLN', 'TESLA',
            'VOLVO', 'SAAB', 'MINI', 'SMART', 'FIAT', 'ALFA', 'JAGUAR',
            'LAND ROVER', 'PORSCHE', 'FERRARI', 'LAMBORGHINI', 'MASERATI',
            'BUGATTI', 'BENTLEY', 'ROLLS ROYCE', 'ASTON MARTIN', 'LOTUS',
            'MCLAREN', 'KOENIGSEGG', 'PAGANI', 'RIMAC', 'PININFARINA',
            'SUZUKI', 'ISUZU', 'DAIHATSU', 'DATSUN', 'INFINITI', 'ACURA',
            'LEXUS', 'GENESIS', 'CUPRA', 'DS', 'OPEL', 'VAUXHALL', 'PEUGEOT',
            'RENAULT', 'CITROEN', 'SKODA', 'SEAT', 'DACIA', 'LADA', 'GAZ',
            'UAZ', 'ZAZ', 'LUAZ', 'Moskvich', 'Zhiguli', 'Lada', 'Trabant',
            'Wartburg', 'FSO', 'FSM', 'ARO', 'Oltcit', 'Dacia', 'Proton',
            'Perodua', 'Chery', 'Geely', 'BYD', 'Great Wall', 'Haval',
            'Changan', 'FAW', 'Dongfeng', 'GAC', 'SAIC', 'BAIC', 'Chery',
            'Jac', 'Brilliance', 'Qoros', 'Luxgen', 'Nio', 'XPeng', 'Li Auto',
            'Aiways', 'Byton', 'Singulato', 'Leapmotor', 'Hozon', 'Skywell',
            'Weltmeister', 'Aiways', 'Qiantu', 'CH-Auto', 'Venucia', 'Ranz',
            'Concept', 'Everus', 'Gonow', 'Haima', 'Huapu', 'JAC', 'JMC',
            'Jinbei', 'Landwind', 'Qoros', 'Rely', 'Roewe', 'Wuling', 'Yuan',
            'Zotye', 'Tata', 'Mahindra', 'Ashok Leyland', 'Force', 'Bajaj',
            'TVS', 'Hero', 'Honda', 'Yamaha', 'Suzuki', 'Kawasaki', 'Royal Enfield',
            'KTM', 'Ducati', 'Aprilia', 'Moto Guzzi', 'Benelli', 'MV Agusta',
            'Bimota', 'Cagiva', 'Husqvarna', 'KTM', 'GasGas', 'Sherco',
            'Beta', 'TM', 'VOR', 'Husaberg', 'CCM', 'Norton', 'Triumph',
            'BSA', 'Ariel', 'Royal Enfield', 'Matchless', 'AJS', 'Velocette',
            'Vincent', 'Brough', 'Scott', 'Rudge', 'New Imperial', 'Zenith',
            'Coventry Eagle', 'Francis Barnett', 'James', 'Excelsior', 'EMC',
            'DOT', 'Cotton', 'Connaught', 'Clemson', 'Chater-Lea', 'ABC',
            'Radco', 'OK-Supreme', 'P&M', 'NUT', 'Montgomery', 'Martinsyde',
            'Douglas', 'Zenith', 'Coventry', 'Royal', 'Enfield', 'Indian',
            'Harley Davidson', 'Excelsior', 'Henderson', 'Merkel', 'Thor',
            'Yale', 'Columbia', 'Pope', 'Reading Standard', 'Flying Merkel',
            'Curtiss', 'Orient', 'Torch', 'Yale', 'Pierce', 'Populaire',
            'Rex', 'Royal', 'Sears', 'Thor', 'Twombly', 'US', 'Wagner',
            'Wayne', 'Whippet', 'Winton', 'Wood', 'Yale', 'Ace', 'Brennan',
            'Crosley', 'Davis', 'De Dion', 'De La Vergne', 'Duryea', 'E-M-F',
            'Elmore', 'EMF', 'Everitt', 'Flanders', 'Ford', 'Franklin',
            'Grant', 'Hupmobile', 'Jackson', 'Jeffery', 'Kissel', 'Lanchester',
            'Lozier', 'Lucas', 'Marmon', 'Maxwell', 'Mercer', 'Metzger',
            'Mitchell', 'Oakland', 'Oldsmobile', 'Overland', 'Packard',
            'Paige', 'Peerless', 'Pierce-Arrow', 'Pope', 'Premier', 'Rambler',
            'Reo', 'Rickenbacker', 'Robinson', 'Saxon', 'Scripps-Booth',
            'Simplex', 'Staver', 'Stearns', 'Stewart', 'Stoddard-Dayton',
            'Stutz', 'Thomas', 'Touchstone', 'Tudor', 'Winton', 'Woods',
            'White', 'Willys', 'Willys-Overland', 'Willys-Knight'
        }
        
        # Common non-plate words to exclude
        self.exclude_words = {
            'MODEL', 'TYPE', 'SERIES', 'EDITION', 'LIMITED', 'SPORT', 'EX',
            'LX', 'SE', 'LE', 'XLE', 'XLS', 'XL', 'ST', 'RS', 'GT', 'GTO',
            'GTX', 'GTR', 'GTI', 'GLI', 'GLX', 'GLE', 'GLS', 'ML', 'SLK',
            'SL', 'CL', 'CLK', 'CLS', 'AMG', 'M', 'CSL', 'M3', 'M5', 'M6',
            'Z4', 'Z3', 'X1', 'X3', 'X5', 'X6', 'X7', '1', '2', '3', '4',
            '5', '6', '7', '8', '9', '10', '12', '14', '16', '18', '20',
            '24', '28', '30', '32', '36', '40', '45', '50', '60', '70',
            '80', '90', '100', '110', '120', '130', '140', '150', '160',
            '170', '180', '190', '200', '210', '220', '230', '240', '250',
            '260', '270', '280', '290', '300', '310', '320', '330', '340',
            '350', '360', '370', '380', '390', '400', '410', '420', '430',
            '440', '450', '460', '470', '480', '490', '500', '510', '520',
            '530', '540', '550', '560', '570', '580', '590', '600', '610',
            '620', '630', '640', '650', '660', '670', '680', '690', '700',
            '710', '720', '730', '740', '750', '760', '770', '780', '790',
            '800', '810', '820', '830', '840', '850', '860', '870', '880',
            '890', '900', '910', '920', '930', '940', '950', '960', '970',
            '980', '990', '1000', 'V6', 'V8', 'V10', 'V12', 'I4', 'I6',
            'I8', 'DOHC', 'SOHC', 'OHV', 'CVVT', 'VVT', 'VTEC', 'VVTI',
            'MIVEC', 'CVT', 'DCT', 'AT', 'MT', 'AWD', 'RWD', 'FWD', '4WD',
            '2WD', 'ABS', 'EBD', 'ESP', 'TCS', 'HSA', 'HDC', 'LDWS',
            'ACC', 'LKA', 'BSD', 'RCTA', 'AVM', 'HUD', 'TPMS', 'SRS',
            'ECO', 'PRO', 'PLUS', 'PREMIUM', 'DELUXE', 'STANDARD', 'BASIC',
            'HYBRID', 'ELECTRIC', 'PLUGIN', 'HYBRID', 'DIESEL', 'GASOLINE',
            'PETROL', 'LPG', 'CNG', 'LNG', 'HYDROGEN', 'SOLAR', 'ETHANOL',
            'BIOFUEL', 'BIO', 'FLEX', 'FLEXFUEL', 'E85', 'E10', 'E15',
            'E20', 'E25', 'E30', 'E40', 'E50', 'E60', 'E70', 'E75', 'E80',
            'E85', 'E90', 'E95', 'E100', 'B2', 'B5', 'B10', 'B20', 'B30',
            'B40', 'B50', 'B60', 'B70', 'B80', 'B90', 'B100', 'B100',
            'B100', 'B100', 'B100', 'B100', 'B100', 'B100', 'B100',
            'B100', 'B100', 'B100', 'B100', 'B100', 'B100', 'B100'
        }
        
        print("[INFO] License Plate Detector initialized")
        print(f"[INFO] Loaded {len(self.plate_patterns)} license plate patterns")
        print(f"[INFO] Loaded {len(self.brand_names)} brand exclusions")
    
    def is_license_plate(self, text: str, confidence: float = 0.0) -> bool:
        """
        Check if text is a valid license plate.
        
        Args:
            text: Text to validate
            confidence: OCR confidence score (0-1)
            
        Returns:
            True if text is likely a license plate, False otherwise
        """
        if not text or not isinstance(text, str):
            return False
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        if not cleaned_text or len(cleaned_text) < 4:
            return False
        
        # Check if it's a brand name (immediate rejection)
        if self._is_brand_name(cleaned_text):
            print(f"[DEBUG] Rejected brand name: {cleaned_text}")
            return False
        
        # Check if it contains excluded words
        if self._contains_excluded_words(cleaned_text):
            print(f"[DEBUG] Rejected excluded word: {cleaned_text}")
            return False
        
        # Check if it matches license plate patterns
        if self._matches_plate_patterns(cleaned_text):
            print(f"[DEBUG] ✅ Valid license plate: {cleaned_text}")
            return True
        
        # Additional heuristics for edge cases
        if self._is_likely_plate_by_heuristics(cleaned_text, confidence):
            print(f"[DEBUG] ✅ Likely license plate by heuristics: {cleaned_text}")
            return True
        
        print(f"[DEBUG] ❌ Not a license plate: {cleaned_text}")
        return False
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for validation."""
        # Remove extra whitespace and convert to uppercase
        cleaned = re.sub(r'\s+', ' ', text.strip().upper())
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[^\w\s]', '', cleaned)  # Remove special characters
        cleaned = re.sub(r'[IO]', '0', cleaned)   # Replace I/O with 0 (common OCR confusion)
        
        return cleaned
    
    def _is_brand_name(self, text: str) -> bool:
        """Check if text matches known brand names."""
        # Direct match
        if text in self.brand_names:
            return True
        
        # Partial match for multi-word brands
        for brand in self.brand_names:
            if ' ' in brand and brand in text:
                return True
        
        # Check if text is just a brand name without numbers
        if text.isalpha() and len(text) >= 3:
            # Check if it's likely a brand name (all letters, no numbers)
            text_lower = text.lower()
            for brand in self.brand_names:
                if text_lower == brand.lower():
                    return True
        
        return False
    
    def _contains_excluded_words(self, text: str) -> bool:
        """Check if text contains excluded words."""
        words = text.split()
        for word in words:
            if word in self.exclude_words:
                return True
        return False
    
    def _matches_plate_patterns(self, text: str) -> bool:
        """Check if text matches any license plate pattern."""
        for pattern in self.plate_patterns:
            if re.match(pattern, text):
                return True
        return False
    
    def _is_likely_plate_by_heuristics(self, text: str, confidence: float) -> bool:
        """
        Additional heuristics to determine if text is likely a license plate.
        This catches edge cases that don't match standard patterns.
        """
        # Must have both letters and numbers (most plates do)
        has_letters = bool(re.search(r'[A-Z]', text))
        has_numbers = bool(re.search(r'\d', text))
        
        if not (has_letters and has_numbers):
            return False
        
        # Check length (most plates are 4-10 characters after cleaning)
        if len(text) < 4 or len(text) > 12:
            return False
        
        # Check character composition (should be mostly alphanumeric)
        alnum_ratio = sum(c.isalnum() or c.isspace() for c in text) / len(text)
        if alnum_ratio < 0.7:
            return False
        
        # Avoid single words followed by single numbers (likely model names)
        if re.match(r'^[A-Z]+\s?\d$', text):
            return False
        
        # Prefer reasonable letter-to-number ratios
        letter_count = sum(c.isalpha() for c in text)
        number_count = sum(c.isdigit() for c in text)
        
        # Most plates have a reasonable balance
        if letter_count > 0 and number_count > 0:
            ratio = max(letter_count, number_count) / min(letter_count, number_count)
            if ratio <= 6:  # Not too skewed
                # Higher confidence OCR results are more reliable
                if confidence > 0.7:
                    return True
                elif confidence > 0.5 and ratio <= 4:
                    return True
        
        # Special case: Bulgarian format (1 letter + 4 numbers + 2 letters) like "B 2228 HM"
        # Pattern: [A-Z]{1,2} [0-9]{4} [A-Z]{2}
        bulgarian_pattern = r'^[A-Z]{1,2}\s?\d{4}\s?[A-Z]{2}$'
        if re.match(bulgarian_pattern, text):
            print(f"[DEBUG] ✅ Bulgarian format license plate detected: {text}")
            return True
        
        # Special case: European format with spaces (e.g., "B 2228 HM", "CA 1234 AB")
        european_pattern = r'^[A-Z]{1,2}\s+\d{3,4}\s+[A-Z]{2,3}$'
        if re.match(european_pattern, text):
            print(f"[DEBUG] ✅ European format license plate detected: {text}")
            return True
        
        return False
    
    def extract_license_plate_candidates(self, text_results: List[Dict]) -> List[Dict]:
        """
        Filter OCR results to find likely license plates.
        
        Args:
            text_results: List of OCR results with text and confidence
            
        Returns:
            Filtered list containing only likely license plates
        """
        candidates = []
        
        for result in text_results:
            text = result.get('text', '')
            confidence = result.get('confidence', 0.0)
            
            if self.is_license_plate(text, confidence):
                candidates.append(result)
        
        # Sort by confidence (highest first)
        candidates.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        print(f"[INFO] Found {len(candidates)} license plate candidates from {len(text_results)} text results")
        return candidates
    
    def get_info(self) -> Dict:
        """Get detector information."""
        return {
            'patterns_count': len(self.plate_patterns),
            'brand_exclusions_count': len(self.brand_names),
            'exclude_words_count': len(self.exclude_words),
            'version': '1.0.0'
        }
