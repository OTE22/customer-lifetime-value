# =============================================================================
# CLV Prediction System - Dockerfile
# Multi-stage build for production deployment
# =============================================================================

# Stage 1: Builder
FROM python:3.11-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Stage 2: Production
FROM python:3.11-slim as production

# Labels
LABEL maintainer="Ali Abbass <ali.abbass@ote22.dev>"
LABEL version="2.0.0"
LABEL description="Customer Lifetime Value Prediction System"

# Install runtime dependencies (libgomp needed by LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CLV_ENVIRONMENT=production \
    CLV_API_HOST=0.0.0.0 \
    CLV_API_PORT=8000

# Create non-root user for security
RUN groupadd --gid 1000 clvuser && \
    useradd --uid 1000 --gid clvuser --shell /bin/bash --create-home clvuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=clvuser:clvuser . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data && \
    chown -R clvuser:clvuser /app

# Switch to non-root user
USER clvuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]


# Stage 3: Development
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    httpx \
    black \
    isort \
    flake8 \
    mypy

# Switch back to app user
USER clvuser

# Development command with reload
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
