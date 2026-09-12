# DeepForensics AI - Production Dockerfile
FROM python:3.11-slim

# Install system dependencies for OpenCV and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and application
COPY . .

# Expose server port
EXPOSE 8000

# Run FastAPI production server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
