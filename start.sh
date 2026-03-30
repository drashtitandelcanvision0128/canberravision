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

# Wait a moment for everything to be ready
echo "[STARTUP] Waiting 3 seconds before starting application..."
sleep 3

# Change to app directory
cd /app

# Start the application
echo "[STARTUP] Starting Gradio application..."
exec python apps/app.py
