# DeepForensics AI - Hugging Face Spaces & Cloud Production Dockerfile
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PORT=7860

# Install system dependencies: FFmpeg (audio extraction), OpenCV dependencies, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create Hugging Face Spaces non-root user (UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Pre-install CPU-only PyTorch (drastically reduces image size & build time)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining packages
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, templates, and application files
COPY --chown=user . $HOME/app

# Ensure output directories exist with full read/write permissions
RUN mkdir -p outputs/preprocessing_demo outputs/web_uploads outputs/web_reports /tmp/matplotlib && \
    chmod -R 777 outputs /tmp/matplotlib

USER user

# Expose Hugging Face default port
EXPOSE 7860

# Start FastAPI production server binding to dynamic $PORT
CMD ["sh", "-c", "python app.py"]
