# Base image with Python
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Create non-root user
RUN useradd -u 10001 -m appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Expose FastAPI port
# EXPOSE 8000

USER 10001:10001

# Gunicorn with Uvicorn workers (multiple processes)
# Tune via env: GUNICORN_WORKERS, GUNICORN_TIMEOUT
CMD ["gunicorn", "main:app"]