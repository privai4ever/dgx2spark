# Multi-stage Dockerfile for DGX Spark Chat (ARM64)
# Optimized for easypanel deployment

FROM python:3.12-slim-bookworm

LABEL maintainer="DGX Spark Chat"
LABEL description="Grok-style chat UI for TensorRT-LLM multi-node inference"

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY static ./static

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check - verify API is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/api/models || exit 1

# Expose port
EXPOSE 7860

# Run uvicorn server
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
