# Base image with Python
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

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
EXPOSE 8000

# Gunicorn with Uvicorn workers (multiple processes)
# Tune via env: GUNICORN_WORKERS, GUNICORN_TIMEOUT
CMD ["sh", "-c", "gunicorn main:app -k uvicorn.workers.UvicornWorker -w ${GUNICORN_WORKERS:-4} -b 0.0.0.0:8000 --timeout ${GUNICORN_TIMEOUT:-180} --keep-alive 5 --access-logfile - --error-logfile -"]