#!/usr/bin/env python3
"""
Quick deployment test for YOLO26 to verify fixes work
"""

import os
import sys
import subprocess
import time

def test_deployment():
    """Test the deployment with our fixes"""
    
    print("=" * 60)
    print("YOLO26 Deployment Test")
    print("=" * 60)
    
    # Set production environment
    os.environ['APP_ENV'] = 'production'
    os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'
    os.environ['GRADIO_SERVER_PORT'] = '7860'
    
    print("✅ Environment variables set")
    print(f"   APP_ENV: {os.environ.get('APP_ENV')}")
    print(f"   Server: {os.environ.get('GRADIO_SERVER_NAME')}:{os.environ.get('GRADIO_SERVER_PORT')}")
    
    # Test imports
    try:
        import gradio as gr
        print(f"✅ Gradio {gr.__version__} imported")
    except Exception as e:
        print(f"❌ Gradio import failed: {e}")
        return False
    
    try:
        from apps.app import demo
        print("✅ Demo imported successfully")
    except Exception as e:
        print(f"❌ Demo import failed: {e}")
        return False
    
    # Test API info
    try:
        api_info = demo.get_api_info()
        print(f"✅ API info generated: {len(api_info)} endpoints")
    except Exception as e:
        print(f"❌ API info failed: {e}")
        return False
    
    print("\n🚀 All tests passed! Deployment should work correctly.")
    print("\nNext steps:")
    print("1. Push changes to Git")
    print("2. Redeploy on Coolify")
    print("3. Check browser console for 'Applying tab selection fix...' message")
    print("4. Verify tabs are visible and clickable")
    
    return True

if __name__ == "__main__":
    test_deployment()
