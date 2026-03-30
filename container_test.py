#!/usr/bin/env python3
"""
Test script to simulate the exact container environment and debug the 404 issue
"""
import os
import sys

# Set the exact environment variables from your Coolify setup
os.environ["APP_NAME"] = "YOLO Car Plate Detection"
os.environ["APP_VERSION"] = "2.0"
os.environ["APP_ENV"] = "production"
os.environ["GRADIO_SERVER_PORT"] = "7860"
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("=" * 60)
print("CONTAINER ENVIRONMENT SIMULATION")
print("=" * 60)

print("\n[ENV] Environment Variables:")
print(f"  APP_ENV: {os.environ.get('APP_ENV')}")
print(f"  GRADIO_SERVER_PORT: {os.environ.get('GRADIO_SERVER_PORT')}")
print(f"  GRADIO_SERVER_NAME: {os.environ.get('GRADIO_SERVER_NAME')}")

print(f"\n[SYS] Python Version: {sys.version}")
print(f"[SYS] Current Directory: {os.getcwd()}")

try:
    print("\n[IMPORT] Testing basic imports...")
    import gradio as gr
    print(f"  ✅ Gradio {gr.__version__} imported")
    
    import torch
    print(f"  ✅ PyTorch {torch.__version__} imported")
    
    print("\n[DEMO] Testing demo creation...")
    
    # Add the path to import apps
    sys.path.insert(0, os.getcwd())
    
    # Try to import the demo
    from apps.app import demo
    print(f"  ✅ Demo imported: {type(demo)}")
    print(f"  ✅ Demo title: {demo.title}")
    
    # Test if demo has the required methods
    if hasattr(demo, 'launch'):
        print(f"  ✅ Demo has launch method")
    else:
        print(f"  ❌ Demo missing launch method")
        
    # Test launch configuration (without actually launching)
    print("\n[LAUNCH] Testing launch configuration...")
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    
    print(f"  Server will bind to: {server_name}:{server_port}")
    
    # This is the exact configuration that will be used
    launch_config = {
        "share": False,
        "show_error": True,
        "quiet": False,
        "inbrowser": False,
        "server_name": server_name,
        "server_port": server_port,
        "prevent_thread_lock": False,
    }
    print(f"  ✅ Launch configuration ready")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Application should work!")
    print("=" * 60)
    
except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    print("This suggests a missing dependency in the container!")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
