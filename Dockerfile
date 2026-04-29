# Base image: Use Python 3.9 slim (lightweight version of Python)
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the source tree and model weights to the container
COPY . /app
RUN mkdir -p /app/.cxas/weights && \
    cp /app/cxas/weights/UNet_resnet50_default.pth /app/.cxas/weights/UNet_ResNet50_default.pth

# Set environment variables
ENV CXAS_PATH=/app/

# Install system-level dependencies (libGL for OpenCV and other dependencies)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install setuptools wheel && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir streamlit opencv-python-headless pillow numpy && \
    pip install --no-cache-dir -e .

# Expose the default port for Streamlit (8501)
EXPOSE 8501

# Command to run the Streamlit app on container startup
CMD ["streamlit", "run", "interactive_cxas_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
