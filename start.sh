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

# Test demo creation with full error output
echo "[STARTUP] Testing demo creation..."
python -c "
import sys, os, traceback
sys.path.insert(0, '/app')
try:
    print('[DEBUG] Starting demo import...')
    from apps.app import demo
    print(f'[DEBUG] Demo imported successfully: {type(demo)}')
    print(f'[DEBUG] Demo title: {demo.title}')
    print(f'[DEBUG] Demo has launch method: {hasattr(demo, \"launch\")}')
except Exception as e:
    print(f'[ERROR] Demo creation failed: {e}')
    print('[ERROR] Full traceback:')
    traceback.print_exc()
    exit(1)
" || {
    echo "[ERROR] Demo creation failed! Check the error output above."
    exit 1
}

# Start the application with error handling
echo "[STARTUP] Starting Gradio application..."
echo "[STARTUP] If this fails, the error will be shown below:"

# Run the app and capture any errors
if python apps/app.py; then
    echo "[STARTUP] ✅ Application started successfully!"
else
    echo "[ERROR] ❌ Application failed to start!"
    echo "[ERROR] Check the error output above for details."
    exit 1
fi
