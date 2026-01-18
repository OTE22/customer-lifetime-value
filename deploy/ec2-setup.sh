#!/bin/bash
# =============================================================================
# EC2 Initial Setup Script for CLV Prediction System
# Run this script ONCE on a fresh EC2 instance (Ubuntu 22.04 LTS recommended)
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "CLV Prediction System - EC2 Setup Script"
echo "=========================================="

# Update system packages
echo "[1/8] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Docker
echo "[2/8] Installing Docker..."
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add ubuntu user to docker group
sudo usermod -aG docker ubuntu
newgrp docker

# Install Docker Compose
echo "[3/8] Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install AWS CLI
echo "[4/8] Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Create application directories
echo "[5/8] Creating application directories..."
sudo mkdir -p /home/ubuntu/clv-data
sudo mkdir -p /home/ubuntu/clv-models
sudo mkdir -p /home/ubuntu/clv-logs
sudo chown -R ubuntu:ubuntu /home/ubuntu/clv-*

# Install Nginx (reverse proxy)
echo "[6/8] Installing Nginx..."
sudo apt install -y nginx
sudo systemctl enable nginx

# Configure Firewall
echo "[7/8] Configuring UFW firewall..."
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow 8000/tcp    # API (for testing)
sudo ufw --force enable

# Create swap file (useful for small instances)
echo "[8/8] Creating swap file..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "=========================================="
echo "EC2 Setup Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "1. Configure AWS credentials: aws configure"
echo "2. Copy your .env file to the server"
echo "3. Deploy using: ./deploy.sh"
echo ""
echo "Server Info:"
echo "  - Docker version: $(docker --version)"
echo "  - Docker Compose version: $(docker-compose --version)"
echo "  - AWS CLI version: $(aws --version)"
