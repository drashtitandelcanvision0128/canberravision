# ============================================================
# Canberra Vision Detection System - Production Dockerfile
# Fixed: gradio/huggingface_hub compatibility + Debian bookworm slim
# ============================================================
# bookworm = stable Debian; avoids trixie’s huge recommended stacks (helps Coolify RAM/disk).
FROM python:3.10-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV APP_ENV=production
ENV PORT=7860
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

# Install system dependencies (Debian bookworm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    pkg-config \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# -------------------------------------------------------
# Step 1: Pin huggingface_hub first to avoid HfFolder error
# (HfFolder was removed in huggingface_hub>=0.25)
# -------------------------------------------------------
RUN pip install --no-cache-dir "huggingface_hub==0.24.7"

# -------------------------------------------------------
# Step 2: Install gradio (compatible with pinned hf_hub)
# Must install AFTER huggingface_hub to avoid override
# -------------------------------------------------------
# Force reinstall of FastAPI and Pydantic v2 to ensure compatibility with gradio 4.25.0
RUN pip install --no-cache-dir --force-reinstall "starlette==0.36.3" "jinja2==3.1.2" "fastapi==0.110.0" "pydantic==2.10.6"
# gradio-client must match gradio (4.32.2 → 0.17.0). Do not downgrade gradio-client.
RUN pip install --no-cache-dir "gradio==4.32.2"

# -------------------------------------------------------
# Step 3: Core dependencies
# -------------------------------------------------------
RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    pillow \
    opencv-python-headless \
    psycopg2-binary \
    python-dotenv \
    scipy \
    scikit-learn \
    imageio-ffmpeg \
    pytesseract \
    sentencepiece \
    "protobuf==3.20.2" \
    "setuptools==68.0.0"

# -------------------------------------------------------
# Step 4: PyTorch CPU (no CUDA needed on Coolify server)
# -------------------------------------------------------
RUN pip install --no-cache-dir \
    "torch==2.1.2" \
    "torchvision==0.16.2" \
    --index-url https://download.pytorch.org/whl/cpu

# -------------------------------------------------------
# Step 5: ML models
# -------------------------------------------------------
RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    ultralytics \
    "transformers==4.37.2" \
    timm

# Base YOLO weights for PPE fallback when best_ppe.pt is not mounted (models/ is gitignored locally).
# Download before removing opencv-python (ultralytics needs cv2 during import).
RUN mkdir -p /app/models && python -c "import os; os.chdir('/app/models'); from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Ultralytics pulls opencv-python (GUI); keep headless only for runtime.
RUN pip uninstall -y opencv-python 2>/dev/null || true && \
    pip install --no-cache-dir opencv-python-headless

# -------------------------------------------------------
# Step 6: PaddleOCR - CPU version (server has no GPU)
# Use || true so build doesn't fail if paddle unavailable
# -------------------------------------------------------
RUN pip install --no-cache-dir "numpy<2.0.0" "paddlepaddle==2.6.2" || \
    echo "WARNING: paddlepaddle failed - OCR will be limited"
RUN pip install --no-cache-dir "numpy<2.0.0" "paddleocr>=2.7.0" || \
    echo "WARNING: paddleocr failed - OCR will be limited"

# License plate ALPR (import: fast_alpr). Image/webcam/video use YOLO if this fails.
RUN pip install --no-cache-dir "onnxruntime>=1.16.0" "fast-alpr" || \
    echo "WARNING: fast-alpr install failed — plate OCR falls back to YOLO path in app"

# -------------------------------------------------------
# Copy application
# -------------------------------------------------------
COPY . .

# Cache buster - forces rebuild when Gradio versions change
RUN echo "Build timestamp: $(date)" > /app/build_info.txt

# Create necessary directories
RUN mkdir -p uploads processed processed_images processed_videos \
    temp_gradio inputs outputs logs

# Make startup scripts executable
RUN chmod +x start.sh start_production.py

RUN chmod -R 755 /app

# Expose Gradio port
EXPOSE 7860

# Health check for Coolify (simple port check)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:7860/ || exit 1

# Use production startup script for Coolify
CMD ["python", "start_production.py"]
