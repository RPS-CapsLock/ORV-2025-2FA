# Use official Python 3.11 slim image
FROM python:3.11-slim

# Install system-level dependencies required for OpenCV, face recognition, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["python", "API.py"]
