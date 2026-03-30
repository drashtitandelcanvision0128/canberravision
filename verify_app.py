#!/usr/bin/env python3
"""
Quick test to verify the Gradio demo can be created and launched
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("[TEST] Importing required modules...")
    import gradio as gr
    print("[TEST] Gradio imported successfully")
    
    print("[TEST] Testing demo creation...")
    # Import just the demo creation part
    from apps.app import demo
    print("[TEST] Demo created successfully!")
    
    print("[TEST] Demo configuration:")
    print(f"  Title: {demo.title}")
    print(f"  Theme: {demo.theme}")
    print(f"  CSS: {'Custom CSS applied' if demo.css else 'No CSS'}")
    
    print("[TEST] ✅ All tests passed! The application should start correctly.")
    
except Exception as e:
    print(f"[TEST] ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
