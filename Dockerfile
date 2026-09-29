# Cricket Biomechanics AI - Docker Image
#
# Build:
#   docker build -t cricket-biomechanics .
#
# Run:
#   docker run -p 5000:5000 -v "$(pwd)/data:/app/data" cricket-biomechanics
#
# The container runs the Flask web app on port 5000. Mount a volume at
# /app/data to persist input_videos/, output_data/, output_videos/ and
# reports/ across container restarts.

FROM python:3.11-slim

# System dependencies: FFmpeg for video processing, libGL for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p input_videos output_data output_videos reports uploads models

# Environment variables (override at runtime with -e or --env-file)
ENV CBAI_PREPROCESS_ENABLE=true \
    CBAI_MAX_WORKERS=4 \
    CBAI_PIPELINE_TIMEOUT=1800 \
    FLASK_SECRET_KEY=change-me-in-production

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/api/stats')" || exit 1

# Run the web server
CMD ["python", "run_server.py"]
