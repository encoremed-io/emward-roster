# syntax=docker/dockerfile:1.7

# Base image with Python (force amd64)
FROM --platform=linux/amd64 python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Use BuildKit cache for pip and prefer wheels
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install --prefer-binary -r requirements.txt --extra-index-url ${TORCH_INDEX_URL}

# Copy all project files
COPY . .

# Expose Streamlit default port
EXPOSE 8501
# EXPOSE 8000  # (uncomment if running FastAPI on 8000)

# Create non-root user
RUN useradd -u 10001 -m appuser
USER 10001:10001

# Default CMD (gunicorn)
CMD ["gunicorn", "main:app"]
