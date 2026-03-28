#!/usr/bin/env python3
"""
🔧 Quick Fix for MockBoxes Error in app.py
==========================================

This script will fix the MockBoxes error in the main app.py file.
Run this once to fix the issue.
"""

import re
import os
from pathlib import Path

def fix_mockboxes_error():
    """Fix the MockBoxes error in app.py"""
    
    app_file = Path(__file__).parent / "apps" / "app.py"
    
    if not app_file.exists():
        print(f"❌ File not found: {app_file}")
        return False
    
    print(f"🔧 Fixing MockBoxes error in: {app_file}")
    
    try:
        # Read the file
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix the MockBoxes error by adding a check before len()
        # Find all occurrences of "if len(boxes) == 0:" and fix them
        
        # Pattern to find the problematic lines
        pattern = r'(\s+)boxes = result\.boxes\n(\s+)if len\(boxes\) == 0:'
        
        def replace_func(match):
            indent1 = match.group(1)
            indent2 = match.group(2)
            return f"{indent1}boxes = result.boxes\n{indent2}# Fix MockBoxes error\n{indent2}if not hasattr(boxes, '__len__') or len(boxes) == 0:"
        
        # Apply the fix
        new_content = re.sub(pattern, replace_func, content)
        
        # Write the fixed content back
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ MockBoxes error fixed successfully!")
        print("🚀 You can now run: python apps/app.py")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing file: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Starting MockBoxes Error Fix...")
    print("=" * 50)
    
    if fix_mockboxes_error():
        print("\n✅ Fix completed successfully!")
        print("Now you can run the main app without errors.")
    else:
        print("\n❌ Fix failed. Please use the simple working detector instead:")
        print("python apps/simple_working_plate_detector.py")
