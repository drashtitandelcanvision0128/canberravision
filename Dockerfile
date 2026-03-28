# ============================================================
# Canberra Vision Detection System - Production Dockerfile
# Fixed: gradio/huggingface_hub compatibility + Debian Trixie
# ============================================================
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV APP_ENV=production
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0

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

# -------------------------------------------------------
# Step 1: Pin huggingface_hub first to avoid HfFolder error
# (HfFolder was removed in huggingface_hub>=0.25)
# -------------------------------------------------------
RUN pip install --no-cache-dir "huggingface_hub==0.24.7"

# -------------------------------------------------------
# Step 2: Install gradio (compatible with pinned hf_hub)
# Must install AFTER huggingface_hub to avoid override
# -------------------------------------------------------
RUN pip install --no-cache-dir "gradio==4.32.2"
RUN pip install --no-cache-dir "gradio-client==0.8.1"

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

# -------------------------------------------------------
# Step 6: PaddleOCR - CPU version (server has no GPU)
# Use || true so build doesn't fail if paddle unavailable
# -------------------------------------------------------
RUN pip install --no-cache-dir "numpy<2.0.0" "paddlepaddle==2.6.2" || \
    echo "WARNING: paddlepaddle failed - OCR will be limited"
RUN pip install --no-cache-dir "numpy<2.0.0" "paddleocr>=2.7.0" || \
    echo "WARNING: paddleocr failed - OCR will be limited"

# -------------------------------------------------------
# Copy application
# -------------------------------------------------------
COPY . .

# Create necessary directories
RUN mkdir -p uploads processed processed_images processed_videos \
    temp_gradio inputs outputs logs

RUN chmod -R 755 /app

# Expose Gradio port
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://127.0.0.1:7860/ || exit 1

CMD ["python", "apps/app.py"]
