# Dockerfile.runpods
# This image already has CUDA, PyTorch, and common ML libraries installed
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1

WORKDIR /app


# Install YOLO-specific system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (PyTorch already installed in base image!)
COPY requirements_runpods.txt .
RUN pip install --no-cache-dir -r requirements_runpods.txt

# Copy handler
COPY handler.py .

# Run handler
CMD ["python", "-u", "handler.py"]