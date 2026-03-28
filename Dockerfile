# ============================================================
# Canberra Vision Detection System - Production Dockerfile
# Fixed for Debian Trixie + Coolify deployment
# ============================================================
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV APP_ENV=production
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0

# Set working directory
WORKDIR /app

# Install system dependencies (Debian Trixie compatible)
RUN apt-get update && apt-get install -y \
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

# ---- Install core dependencies first (lighter, faster) ----
RUN pip install --no-cache-dir \
    gradio==4.44.1 \
    numpy \
    pillow \
    opencv-python-headless \
    psycopg2-binary \
    python-dotenv

# ---- Install ML dependencies (heavier) ----
RUN pip install --no-cache-dir \
    torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    ultralytics \
    transformers \
    timm \
    scikit-learn \
    scipy \
    imageio-ffmpeg \
    pytesseract \
    sentencepiece \
    protobuf==3.20.2 \
    setuptools==68.0.0

# ---- Install PaddleOCR (CPU version for server without GPU) ----
RUN pip install --no-cache-dir paddlepaddle==2.6.2 || \
    echo "WARNING: paddlepaddle install failed, OCR may be limited"

RUN pip install --no-cache-dir "paddleocr>=2.7.0" || \
    echo "WARNING: paddleocr install failed, OCR may be limited"

# Copy the entire application
COPY . .

# Create necessary directories
RUN mkdir -p uploads processed processed_images processed_videos temp_gradio inputs outputs logs

# Set permissions
RUN chmod -R 755 /app

# Expose main Gradio port
EXPOSE 7860

# Healthcheck - wait up to 5 min for heavy ML model loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command
CMD ["python", "apps/app.py"]
