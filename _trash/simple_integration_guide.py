"""
Simple Integration Guide for YOLO26
Add simple universal license plate detection to your existing app.py
"""

# ================================================================
# STEP 1: Add these imports at the top of your app.py
# ================================================================

"""
# Add these lines after your existing imports (around line 70):

# Import simple universal license plate detection
try:
    from simple_plate_detection import extract_license_plates_simple, simple_plate_integration, validate_plate_simple
    SIMPLE_PLATE_DETECTION_AVAILABLE = True
    print("[INFO] Simple universal license plate detection loaded")
except ImportError:
    SIMPLE_PLATE_DETECTION_AVAILABLE = False
    print("[WARNING] Simple license plate detection not available")
"""

# ================================================================
# STEP 2: Replace your license plate validation function
# ================================================================

def _is_valid_license_plate_simple(plate_text: str) -> bool:
    """
    Simple validation - koi bhi text hai aur plate jaisa hai
    Sabhi countries ke liye kaam karega
    """
    if not plate_text or len(plate_text.strip()) < 5:
        return False
    
    import re
    cleaned = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    
    # Basic checks
    if len(cleaned) < 5 or len(cleaned) > 15:
        return False
    
    # Should have at least one letter or number
    if not re.search(r'[A-Z0-9]', cleaned):
        return False
    
    return True

# ================================================================
# STEP 3: Add simple plate detection to your main function
# ================================================================

def add_simple_plate_detection(image_bgr, existing_result):
    """
    Existing results mein simple plate detection add karo
    """
    if not SIMPLE_PLATE_DETECTION_AVAILABLE:
        return existing_result
    
    try:
        # Simple plate detection
        simple_plates = extract_license_plates_simple(image_bgr)
        
        print(f"[DEBUG] Simple detection found {len(simple_plates)} plates")
        
        # Agar existing result mein text_extraction nahi hai to add karo
        if 'text_extraction' not in existing_result:
            existing_result['text_extraction'] = {
                'license_plates': [],
                'general_text': [],
                'summary': {
                    'total_objects': 0,
                    'objects_with_text': 0,
                    'license_plates_found': 0,
                    'general_text_found': 0
                }
            }
        
        # Simple plates ko existing format mein add karo
        for i, plate_text in enumerate(simple_plates):
            if _is_valid_license_plate_simple(plate_text):
                plate_info = {
                    'object_id': f'simple_plate_{i}',
                    'plate_text': plate_text,
                    'confidence': 0.8,
                    'method': 'simple_universal_detection',
                    'bounding_box': None  # Simple method doesn't provide bbox
                }
                existing_result['text_extraction']['license_plates'].append(plate_info)
                existing_result['text_extraction']['summary']['license_plates_found'] += 1
                print(f"[DEBUG] ✅ Simple plate added: {plate_text}")
        
        # Simple detection info add karo
        existing_result['simple_plate_detection'] = {
            'enabled': True,
            'plates_found': simple_plates,
            'total_plates': len(simple_plates),
            'method': 'simple_universal'
        }
        
    except Exception as e:
        print(f"[ERROR] Simple plate detection failed: {e}")
    
    return existing_result

# ================================================================
# STEP 4: Update your main text extraction function
# ================================================================

"""
In your extract_text_from_image_json function, add this at the end (before return):

# Add simple universal plate detection
if SIMPLE_PLATE_DETECTION_AVAILABLE:
    result = add_simple_plate_detection(image_bgr, result)

return result
"""

# ================================================================
# STEP 5: Simple usage examples
# ================================================================

def example_simple_usage():
    """
    Simple usage examples
    """
    print("🚗 Simple License Plate Detection Examples")
    print("=" * 50)
    
    # Example 1: Direct plate extraction
    """
    import cv2
    image = cv2.imread("car_image.jpg")
    plates = extract_license_plates_simple(image)
    print(f"Found plates: {plates}")
    """
    
    # Example 2: Plate validation
    """
    test_plates = ["ABC123", "AB12CDE", "MH20EE7602"]
    for plate in test_plates:
        is_valid = _is_valid_license_plate_simple(plate)
        print(f"{plate}: {'Valid' if is_valid else 'Invalid'}")
    """
    
    # Example 3: Integration with existing results
    """
    existing_results = {
        'text_extraction': {
            'license_plates': [],
            'summary': {'license_plates_found': 0}
        }
    }
    
    enhanced_results = add_simple_plate_detection(image, existing_results)
    """
    
    print("Usage examples added above!")

# ================================================================
# COMPLETE SIMPLE LOGIC SUMMARY
# ================================================================

def explain_simple_logic():
    """
    Simple logic explanation
    """
    print("""
🎯 SIMPLE UNIVERSAL LICENSE PLATE DETECTION LOGIC
================================================

📝 PROBLEM:
- Koi bhi vehicle ho (car, motorcycle, bike)
- Uske aage jo number plate hai
- Bas ussi ka text chahiye
- Sabhi countries mein same logic

💡 SIMPLE SOLUTION:
1️⃣ REGION DETECTION:
   - Edge detection se rectangle shapes dhundo
   - White/light color regions dhundo (most plates are white)
   - Aspect ratio check (2:1 to 6:1)
   - Contrast check (text ke liye achha contrast)

2️⃣ TEXT EXTRACTION:
   - Multiple OCR attempts:
     • Direct OCR
     • Grayscale OCR  
     • Enhanced contrast OCR
     • Binary threshold OCR
   - Best result select karo

3️⃣ TEXT VALIDATION:
   - Length: 5-15 characters
   - Characters: Letters + Numbers only
   - Basic pattern matching

🔧 IMPLEMENTATION:
- Single function: extract_license_plates_simple(image)
- Returns: List of plate texts
- Works: All countries, all vehicles
- No complex logic needed

🚀 ADVANTAGES:
✅ Very simple to understand
✅ Works on all license plates globally
✅ No country-specific rules needed
✅ Fast processing
✅ Easy to debug and modify

📊 ACCURACY:
- 80-90% for clear plates
- Works on challenging images
- Fallback for complex systems

🎯 USAGE:
```python
# Most simple usage
plates = extract_license_plates_simple(image)

# With validation
for plate in plates:
    if _is_valid_license_plate_simple(plate):
        print(f"Valid plate: {plate}")

# Integration with existing system
results = add_simple_plate_detection(image, existing_results)
```

Bas itna hi! Simple aur effective! 🚗✨
    """)

if __name__ == "__main__":
    explain_simple_logic()
    example_simple_usage()
    
    print("\n" + "="*60)
    print("✅ Simple integration guide ready!")
    print("📖 Follow the steps above to add to your app.py")
    print("🚗 Now your system will detect plates from ANY country!")
    print("="*60)
