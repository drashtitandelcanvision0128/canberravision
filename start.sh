#!/bin/bash

# Startup script for YOLO Vision System in Coolify
echo "=========================================="
echo "YOLO Vision System - Starting Up"
echo "=========================================="

# Print environment variables for debugging
echo "[STARTUP] Environment Variables:"
echo "  APP_ENV: ${APP_ENV:-not set}"
echo "  GRADIO_SERVER_PORT: ${GRADIO_SERVER_PORT:-not set}"
echo "  GRADIO_SERVER_NAME: ${GRADIO_SERVER_NAME:-not set}"
echo "  PYTHONPATH: ${PYTHONPATH:-not set}"

# Print system info
echo "[STARTUP] System Info:"
echo "  Python: $(python --version)"
echo "  Working Directory: $(pwd)"
echo "  User: $(whoami)"

# Wait a moment for everything to be ready
echo "[STARTUP] Waiting 3 seconds before starting application..."
sleep 3

# Change to app directory
cd /app

# Test Gradio import first
echo "[STARTUP] Testing Gradio import..."
python -c "import gradio as gr; print(f'Gradio version: {gr.__version__}')" || {
    echo "[ERROR] Gradio import failed!"
    exit 1
}

# Test demo creation
echo "[STARTUP] Testing demo creation..."
python -c "
import sys, os
sys.path.insert(0, '/app')
try:
    from apps.app import demo
    print(f'Demo created successfully: {type(demo)}')
    print(f'Demo title: {demo.title}')
except Exception as e:
    print(f'Demo creation failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" || {
    echo "[ERROR] Demo creation failed!"
    exit 1
}

# Start the application
echo "[STARTUP] Starting Gradio application..."
exec python apps/app.py
