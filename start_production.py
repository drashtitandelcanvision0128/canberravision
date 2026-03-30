#!/usr/bin/env python3
"""
Production startup script for YOLO26 AI Vision Platform on Coolify
This script handles production-specific configuration and error handling
"""

import os
import sys
import logging
import traceback
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Set up production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_production_environment():
    """Configure production environment variables and settings"""
    
    # Set production environment
    os.environ['APP_ENV'] = 'production'
    os.environ['ENV'] = 'production'
    
    # Force CPU mode (Coolify servers typically don't have GPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Set Gradio production settings
    os.environ.setdefault('GRADIO_SERVER_NAME', '0.0.0.0')
    os.environ.setdefault('GRADIO_SERVER_PORT', '7860')
    
    # Optimize for production
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    
    logger.info("Production environment configured")
    logger.info(f"Server: {os.environ['GRADIO_SERVER_NAME']}:{os.environ['GRADIO_SERVER_PORT']}")
    logger.info(f"Working directory: {os.getcwd()}")

def test_imports():
    """Test critical imports before starting the application"""
    
    try:
        import gradio as gr
        logger.info(f"✅ Gradio {gr.__version__} imported successfully")
    except Exception as e:
        logger.error(f"❌ Gradio import failed: {e}")
        return False
    
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} imported successfully")
        logger.info(f"   CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        logger.error(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        logger.info(f"✅ OpenCV {cv2.__version__} imported successfully")
    except Exception as e:
        logger.error(f"❌ OpenCV import failed: {e}")
        return False
    
    return True

def test_demo_creation():
    """Test demo creation with full error reporting"""
    
    try:
        # Add app directory to path
        app_dir = Path('.')
        if str(app_dir) not in sys.path:
            sys.path.insert(0, str(app_dir))
        
        logger.info("Testing demo creation...")
        
        # Import demo
        from apps.app import demo
        logger.info(f"✅ Demo imported: {type(demo)}")
        
        # Test API info generation
        api_info = demo.get_api_info()
        logger.info(f"✅ API info generated: {len(api_info)} endpoints")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo creation failed: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return False

def main():
    """Main production startup function"""
    
    logger.info("=" * 60)
    logger.info("YOLO26 AI Vision Platform - Production Startup")
    logger.info("=" * 60)
    
    # Setup production environment
    setup_production_environment()
    
    # Test imports
    if not test_imports():
        logger.error("❌ Import tests failed. Exiting.")
        sys.exit(1)
    
    # Test demo creation
    if not test_demo_creation():
        logger.error("❌ Demo creation failed. Exiting.")
        sys.exit(1)
    
    # Start the application
    logger.info("🚀 Starting production application...")
    
    try:
        # Import and run the app
        from apps.app import demo
        
        logger.info("✅ Application starting successfully")
        logger.info(f"🌐 Server will be available at: http://{os.environ['GRADIO_SERVER_NAME']}:{os.environ['GRADIO_SERVER_PORT']}")
        
        # Launch with production settings
        demo.launch(
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=False,  # Never open browser in production
            server_name=os.environ['GRADIO_SERVER_NAME'],
            server_port=int(os.environ['GRADIO_SERVER_PORT']),
            allowed_paths=['.'],
            prevent_thread_lock=False
        )
        
    except Exception as e:
        logger.error(f"❌ Application failed to start: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
