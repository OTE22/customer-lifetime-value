#!/bin/bash
# =============================================================================
# Deployment Script for CLV Prediction System
# Run this script to deploy/update the application on EC2
# =============================================================================

set -e  # Exit on error

# Configuration
APP_NAME="clv-prediction"
IMAGE_NAME="clv-prediction:latest"
CONTAINER_NAME="clv-app"
DATA_DIR="/home/ubuntu/clv-data"
MODELS_DIR="/home/ubuntu/clv-models"
LOGS_DIR="/home/ubuntu/clv-logs"
ENV_FILE="/home/ubuntu/.env.production"

echo "=========================================="
echo "CLV Prediction System - Deployment Script"
echo "=========================================="

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: Environment file not found at $ENV_FILE"
    echo "Please create the environment file first."
    exit 1
fi

# Load environment variables
export $(cat $ENV_FILE | grep -v '^#' | xargs)

# Pull latest image from ECR (if using)
if [ -n "$ECR_REGISTRY" ]; then
    echo "[1/5] Logging in to ECR..."
    aws ecr get-login-password --region ${AWS_REGION:-us-east-1} | docker login --username AWS --password-stdin $ECR_REGISTRY
    
    echo "[2/5] Pulling latest image..."
    docker pull $ECR_REGISTRY/$APP_NAME:latest
    IMAGE_NAME="$ECR_REGISTRY/$APP_NAME:latest"
else
    echo "[1/5] Building Docker image locally..."
    docker build -t $IMAGE_NAME .
fi

# Stop and remove existing container
echo "[3/5] Stopping existing container..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Run new container
echo "[4/5] Starting new container..."
docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -p 8000:8000 \
    --env-file $ENV_FILE \
    -v $DATA_DIR:/app/data \
    -v $MODELS_DIR:/app/models \
    -v $LOGS_DIR:/app/logs \
    --health-cmd="curl -f http://localhost:8000/api/health || exit 1" \
    --health-interval=30s \
    --health-timeout=10s \
    --health-retries=3 \
    $IMAGE_NAME

# Wait for container to be healthy
echo "[5/5] Waiting for application to be healthy..."
for i in {1..30}; do
    if curl -sf http://localhost:8000/api/health > /dev/null 2>&1; then
        echo ""
        echo "=========================================="
        echo "✅ Deployment Successful!"
        echo "=========================================="
        echo ""
        echo "Application is running at:"
        echo "  - http://$(curl -s ifconfig.me):8000"
        echo "  - http://localhost:8000"
        echo ""
        echo "Container logs: docker logs -f $CONTAINER_NAME"
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "=========================================="
echo "❌ Deployment Failed!"
echo "=========================================="
echo "Container did not become healthy in time."
echo "Check logs: docker logs $CONTAINER_NAME"
exit 1
