# 5G Core Network Emulator - Docker Image
# Base image for all 5G Network Functions

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY core_network/ ./core_network/
COPY ran/ ./ran/
COPY ptp/ ./ptp/

# Default command (overridden by docker-compose)
CMD ["python", "core_network/main.py"]
