# AWS EC2 Deployment Guide for CLV Prediction System

## Prerequisites

Before deploying, ensure you have:
- AWS Account with appropriate permissions
- AWS CLI installed locally
- SSH key pair for EC2 access
- Domain name (optional, for HTTPS)

---

## Step 1: Create AWS Resources

### 1.1 Create ECR Repository (for Docker images)

```bash
# Create ECR repository
aws ecr create-repository \
    --repository-name clv-prediction \
    --region us-east-1

# Note the repository URI from output:
# 123456789012.dkr.ecr.us-east-1.amazonaws.com/clv-prediction
```

### 1.2 Create S3 Bucket (for DVC and MLflow artifacts)

```bash
# Create S3 bucket
aws s3 mb s3://your-clv-bucket --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket your-clv-bucket \
    --versioning-configuration Status=Enabled
```

### 1.3 Create IAM User for Deployments

```bash
# Create user
aws iam create-user --user-name clv-deployer

# Attach policies
aws iam attach-user-policy --user-name clv-deployer \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess

aws iam attach-user-policy --user-name clv-deployer \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create access keys
aws iam create-access-key --user-name clv-deployer
# Save the output! You'll need AccessKeyId and SecretAccessKey
```

---

## Step 2: Launch EC2 Instance

### 2.1 Launch Instance via AWS Console

1. Go to **EC2 Dashboard** → **Launch Instance**
2. **Settings**:
   - **Name**: CLV-Prediction-Server
   - **AMI**: Ubuntu Server 22.04 LTS (64-bit x86)
   - **Instance type**: t3.medium (or larger for production)
   - **Key pair**: Select or create an SSH key pair
   - **Security group**: Allow ports 22, 80, 443, 8000
   - **Storage**: 30 GB gp3

3. Click **Launch instance**

### 2.2 Configure Security Group

Allow inbound traffic:
| Type | Port | Source | Description |
|------|------|--------|-------------|
| SSH | 22 | Your IP | SSH access |
| HTTP | 80 | Anywhere | Web traffic |
| HTTPS | 443 | Anywhere | Secure web |
| Custom TCP | 8000 | Anywhere | API (testing) |
| Custom TCP | 5000 | Your IP | MLflow UI |

---

## Step 3: Initial Server Setup

### 3.1 Connect to EC2

```bash
# Connect via SSH
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### 3.2 Run Setup Script

```bash
# Download setup script
curl -O https://raw.githubusercontent.com/your-repo/clv/main/deploy/ec2-setup.sh

# Or copy manually:
scp -i your-key.pem deploy/ec2-setup.sh ubuntu@your-ec2-ip:~/

# Make executable and run
chmod +x ec2-setup.sh
./ec2-setup.sh
```

### 3.3 Configure AWS CLI

```bash
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region: us-east-1
# - Default output format: json
```

---

## Step 4: Deploy Application

### 4.1 Create Environment File

```bash
# Copy from template
nano ~/.env.production

# Add your values:
CLV_ENV=production
CLV_API_HOST=0.0.0.0
CLV_API_PORT=8000
AWS_ACCESS_KEY_ID=your-key-id
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com
```

### 4.2 Deploy via Docker

**Option A: Pull from ECR (CI/CD deployed)**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin YOUR_ECR_REGISTRY

# Pull and run
docker pull $ECR_REGISTRY/clv-prediction:latest
docker run -d \
    --name clv-app \
    --restart unless-stopped \
    -p 8000:8000 \
    --env-file ~/.env.production \
    -v /home/ubuntu/clv-data:/app/data \
    $ECR_REGISTRY/clv-prediction:latest
```

**Option B: Build locally on EC2**
```bash
# Clone repository
git clone https://github.com/your-repo/clv.git
cd clv

# Build and run
docker build -t clv-prediction .
docker run -d \
    --name clv-app \
    --restart unless-stopped \
    -p 8000:8000 \
    --env-file ~/.env.production \
    -v /home/ubuntu/clv-data:/app/data \
    clv-prediction
```

### 4.3 Verify Deployment

```bash
# Check container is running
docker ps

# Check logs
docker logs clv-app

# Test API health
curl http://localhost:8000/api/health
```

---

## Step 5: Configure Domain & SSL (Optional)

### 5.1 Point Domain to EC2

1. Get your EC2 Elastic IP or public IP
2. In your DNS provider, add:
   - **A Record**: `your-domain.com` → EC2 IP
   - **A Record**: `api.your-domain.com` → EC2 IP

### 5.2 Setup SSL with Let's Encrypt

```bash
# Install Certbot
sudo apt install -y certbot

# Get certificate
sudo certbot certonly --standalone -d your-domain.com

# Certificates saved to:
# /etc/letsencrypt/live/your-domain.com/fullchain.pem
# /etc/letsencrypt/live/your-domain.com/privkey.pem

# Setup auto-renewal
sudo systemctl enable certbot.timer
```

### 5.3 Update Nginx for HTTPS

```bash
# Copy SSL certs to nginx directory
sudo mkdir -p /home/ubuntu/clv/deploy/ssl
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem /home/ubuntu/clv/deploy/ssl/
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem /home/ubuntu/clv/deploy/ssl/

# Restart with docker-compose
cd /home/ubuntu/clv/deploy
docker-compose -f docker-compose.prod.yml up -d
```

---

## Step 6: Setup GitHub Actions Secrets

In your GitHub repository, go to **Settings** → **Secrets and variables** → **Actions**.

Add these secrets:
| Secret Name | Value |
|-------------|-------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key |
| `AWS_REGION` | `us-east-1` |
| `EC2_HOST` | Your EC2 public IP or domain |
| `EC2_SSH_KEY` | Contents of your .pem file |
| `MLFLOW_TRACKING_URI` | MLflow server URL |

---

## Step 7: Monitoring & Maintenance

### Check Application Status
```bash
# Container status
docker ps
docker stats clv-app

# Application logs
docker logs -f clv-app --tail 100

# Nginx access logs
docker logs clv-nginx
```

### Update Application
```bash
# Pull latest and restart
cd /home/ubuntu/clv/deploy
./deploy.sh
```

### Backup Data
```bash
# Backup models
aws s3 sync /home/ubuntu/clv-models s3://your-bucket/backups/models/

# Backup data
aws s3 sync /home/ubuntu/clv-data s3://your-bucket/backups/data/
```

---

## Troubleshooting

### Container won't start
```bash
docker logs clv-app  # Check for error messages
```

### Port already in use
```bash
sudo lsof -i :8000
sudo kill -9 <PID>
```

### Out of memory
```bash
# Add swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### ECR login failed
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin REGISTRY
```

---

## Cost Estimation

| Resource | Type | Estimated Monthly Cost |
|----------|------|----------------------|
| EC2 | t3.medium | ~$30 |
| EBS Storage | 30 GB gp3 | ~$3 |
| S3 | 10 GB | ~$0.25 |
| ECR | 5 GB images | ~$0.50 |
| Data Transfer | 10 GB | ~$1 |
| **Total** | | **~$35/month** |

---

## Support

For issues or questions:
1. Check the logs: `docker logs clv-app`
2. Review GitHub Actions runs for deployment errors
3. Open an issue on the repository
