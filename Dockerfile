# syntax=docker/dockerfile:1.7

# Base image with Python
FROM python:3.11-slim

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

# Use BuildKit cache for pip and prefer wheels to avoid source builds
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install --prefer-binary -r requirements.txt

# Copy all project files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Expose FastAPI port
# EXPOSE 8000

# Create non-root user
RUN useradd -u 10001 -m appuser
USER 10001:10001

# Gunicorn with Uvicorn workers (multiple processes)
# Tune via env: GUNICORN_WORKERS, GUNICORN_TIMEOUT
CMD ["gunicorn", "main:app"]