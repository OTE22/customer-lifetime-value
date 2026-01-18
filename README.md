# 🎯 Customer Lifetime Value (CLV) Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/XGBoost-1.7+-red.svg" alt="XGBoost">
  <img src="https://img.shields.io/badge/Docker-Ready-blue.svg" alt="Docker">
  <img src="https://img.shields.io/badge/MLflow-2.10+-orange.svg" alt="MLflow">
  <img src="https://img.shields.io/badge/DVC-3.30+-9cf.svg" alt="DVC">
  <img src="https://img.shields.io/badge/AWS-EC2_Ready-FF9900.svg" alt="AWS">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  <strong>Production-ready ML system for predicting customer lifetime value in e-commerce</strong>
</p>

---

## 📑 Table of Contents

| Section | Description |
|---------|-------------|
| [🚀 Quick Start](#-quick-start) | Get running in 2 minutes |
| [✨ Features](#-features) | What this system does |
| [📡 API Reference](#-api-reference) | All endpoints with examples |
| [🐳 Docker](#-docker) | Container deployment |
| [🔬 MLOps](#-mlops) | MLflow, DVC, CI/CD setup |
| [☁️ AWS Deployment](#-aws-deployment) | Production deployment |
| [🎯 Meta Ads](#-meta-ads-integration) | Budget optimization |
| [⚙️ Configuration](#-configuration) | All environment variables |
| [🧪 Testing](#-testing) | Run tests |
| [📁 Project Structure](#-project-structure) | File organization |

---

## 🚀 Quick Start

### Option 1: Setup Script (Recommended)

**Windows:**
```batch
git clone https://github.com/OTE22/customer-lifetime-value.git
cd customer-lifetime-value
.\setup.bat
```

**Linux/Mac:**
```bash
git clone https://github.com/OTE22/customer-lifetime-value.git
cd customer-lifetime-value
chmod +x setup.sh && ./setup.sh
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python data/generate_data.py

# Start API server
python -m uvicorn backend.api:app --reload --port 8000
```

### Option 3: Docker

```bash
docker-compose up -d
```

### 🌐 Access Points

| URL | Description |
|-----|-------------|
| http://localhost:8000 | Dashboard UI |
| http://localhost:8000/api/docs | Swagger API Docs |
| http://localhost:8000/api/redoc | ReDoc API Docs |
| http://localhost:8000/api/health | Health Check |

---

## ✨ Features

### 🤖 Machine Learning Models

| Model | Description |
|-------|-------------|
| **XGBoost** | State-of-the-art gradient boosting |
| **LightGBM** | Fast distributed ML |
| **Random Forest** | Robust ensemble predictions |
| **Gradient Boosting** | Captures complex patterns |
| **Weighted Ensemble** | Combines all models for best accuracy |

### 📊 Feature Engineering (80+ Features)

| Category | Examples |
|----------|----------|
| **RFM** | `recency_score`, `frequency_score`, `monetary_score`, `rfm_weighted` |
| **Behavioral** | `purchase_velocity`, `churn_risk_score`, `customer_maturity` |
| **Temporal** | `first_purchase_season`, `is_peak_season`, `days_to_major_holiday` |
| **Interactions** | `source_category_interaction`, `campaign_value_interaction` |
| **Acquisition** | `cac_efficiency`, `cac_roi`, `source_quality_score` |

### 🎯 Prediction Confidence

| Feature | Description |
|---------|-------------|
| **Confidence Intervals** | 90%, 95%, 99% prediction bounds |
| **Uncertainty Estimation** | Model ensemble variance analysis |
| **Temporal Validation** | Time-based train/val/test split (70/15/15) |

---

## 📡 API Reference

### All Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check with component status |
| `GET` | `/api/status` | Detailed status + cache stats |
| `GET` | `/api/customers` | List customers (paginated) |
| `GET` | `/api/customers/{id}` | Get single customer |
| `POST` | `/api/predict` | Predict CLV for customer |
| `POST` | `/api/predict/batch` | Batch predictions |
| `GET` | `/api/segments` | Segment distribution |
| `GET` | `/api/metrics` | Model performance metrics |
| `POST` | `/api/model/train` | Train/retrain model |
| `GET` | `/api/meta-ads/audiences` | Audience segments |
| `GET` | `/api/meta-ads/budget-allocation` | Budget recommendations |
| `GET` | `/api/meta-ads/strategy` | Full Meta Ads strategy |
| `GET` | `/api/dashboard/summary` | Dashboard data |
| `GET` | `/api/cache/stats` | Cache statistics |
| `POST` | `/api/cache/clear` | Clear cache |

### Example: Predict CLV

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "total_orders": 5,
    "total_spent": 450,
    "avg_order_value": 90,
    "days_since_first_purchase": 180,
    "days_since_last_purchase": 15,
    "num_categories": 3,
    "acquisition_source": "Meta Ads",
    "campaign_type": "Prospecting",
    "acquisition_cost": 45,
    "email_engagement_rate": 0.65,
    "return_rate": 0.05
  }'
```

**Response:**
```json
{
  "predicted_clv": 892.50,
  "segment": "High-CLV",
  "confidence": "High",
  "recommended_cac": 267.75
}
```

### Example: Get Budget Allocation

```bash
curl "http://localhost:8000/api/meta-ads/budget-allocation?total_budget=10000"
```

---

## 🐳 Docker

### Quick Start
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI backend |
| `frontend` | 3000 | Nginx dashboard |
| `redis` | 6379 | Cache (optional) |

### Build Commands
```bash
# Build production image
docker build -t clv-prediction .

# Build development image
docker build --target development -t clv-prediction:dev .

# Run standalone
docker run -p 8000:8000 clv-prediction
```

---

## 🔬 MLOps

### MLflow (Experiment Tracking)

```bash
# Set tracking URI (optional - uses local ./mlruns by default)
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000

# Train with MLflow logging
python scripts/train_models.py

# View MLflow UI
mlflow ui --port 5000
# Open http://localhost:5000
```

**What gets logged:**
- Training parameters (n_estimators, max_depth, etc.)
- Metrics (RMSE, MAE, R²)
- Model artifacts
- Feature importance

### DVC (Data Version Control)

```bash
# Initialize DVC (first time only)
dvc init

# Configure S3 remote (edit .dvc/config)
dvc remote modify s3remote url s3://your-bucket/clv-prediction

# Add data to DVC
dvc add data/customers.csv

# Push data to remote
dvc push

# Pull data from remote
dvc pull

# Run full pipeline
dvc repro
```

### GitHub Actions CI/CD

**Workflows:**
| Workflow | Trigger | Actions |
|----------|---------|---------|
| `ci.yml` | Push to main/develop | Lint → Test → Build Docker → Push to ECR |
| `deploy.yml` | Release/Manual | Deploy to EC2 |

**Required GitHub Secrets:**
| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key |
| `AWS_REGION` | e.g., `us-east-1` |
| `EC2_HOST` | EC2 public IP address |
| `EC2_SSH_KEY` | SSH private key (full content) |

---

## ☁️ AWS Deployment

### Prerequisites
1. AWS Account with ECR and EC2 access
2. EC2 instance (Ubuntu 22.04, t3.medium+)
3. GitHub repository with secrets configured

### Quick Deploy (3 Steps)

**Step 1: Setup EC2**
```bash
scp deploy/ec2-setup.sh ubuntu@YOUR-EC2-IP:~/
ssh ubuntu@YOUR-EC2-IP
chmod +x ec2-setup.sh && ./ec2-setup.sh
```

**Step 2: Configure Environment**
```bash
cp .env.production.example ~/.env.production
nano ~/.env.production  # Add your values
```

**Step 3: Deploy**
```bash
./deploy.sh
```

📖 **Full guide:** [deploy/EC2_DEPLOYMENT_GUIDE.md](deploy/EC2_DEPLOYMENT_GUIDE.md)

### Production Docker Compose
```bash
cd deploy
docker-compose -f docker-compose.prod.yml up -d
```

---

## 🎯 Meta Ads Integration

### Budget Allocation (3:2:1 Rule)

| Segment | Budget % | CAC Target | Strategy |
|---------|----------|------------|----------|
| **High-CLV** | 50% | 30% of CLV | Value optimization, 1% lookalikes |
| **Growth-Potential** | 35% | 30% of CLV | Conversion optimization |
| **Low-CLV** | 15% | 30% of CLV | Cost caps, testing only |

### API Usage

```bash
# Get audience segments
curl "http://localhost:8000/api/meta-ads/audiences"

# Get budget allocation
curl "http://localhost:8000/api/meta-ads/budget-allocation?total_budget=10000"

# Get full strategy
curl "http://localhost:8000/api/meta-ads/strategy?total_budget=10000"
```

---

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Application
CLV_ENVIRONMENT=development  # development, staging, production
CLV_DEBUG=true
CLV_SECRET_KEY=your-secret-key-here

# API
CLV_API_HOST=0.0.0.0
CLV_API_PORT=8000
CLV_API_WORKERS=4
CLV_RATE_LIMIT=100

# Cache
CLV_CACHE_ENABLED=true
CLV_CACHE_TYPE=memory  # memory, redis
CLV_CACHE_TTL=3600
CLV_REDIS_URL=redis://localhost:6379

# Logging
CLV_LOG_LEVEL=INFO
CLV_LOG_FILE=logs/clv_api.log

# MLflow (optional)
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Training Parameters

Edit `params.yaml`:

```yaml
train:
  use_temporal_split: true
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  n_estimators: 100
  max_depth: 10
  learning_rate: 0.1
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_clv.py -v
```

---

## 📁 Project Structure

```
clv/
├── backend/                    # Python API & ML
│   ├── api.py                 # FastAPI endpoints
│   ├── clv_predictor.py       # Prediction pipeline
│   ├── feature_engineering.py # 80+ features
│   ├── ml_models.py           # XGBoost, LightGBM, RF
│   ├── mlflow_config.py       # MLflow integration
│   ├── meta_ads_integration.py# Meta Ads optimization
│   ├── data_processor.py      # Data cleaning
│   ├── config.py              # Configuration
│   ├── cache.py               # LRU/Redis caching
│   ├── middleware.py          # Rate limiting
│   └── schemas.py             # Pydantic models
├── frontend/                   # Dashboard UI
│   ├── index.html
│   ├── css/styles.css
│   └── js/app.js
├── scripts/                    # Training scripts
│   ├── train_models.py        # MLflow-tracked training
│   └── evaluate_models.py     # Model evaluation
├── deploy/                     # Deployment
│   ├── EC2_DEPLOYMENT_GUIDE.md
│   ├── ec2-setup.sh
│   ├── deploy.sh
│   ├── docker-compose.prod.yml
│   └── nginx.prod.conf
├── .github/workflows/          # CI/CD
│   ├── ci.yml
│   └── deploy.yml
├── .dvc/config                # DVC S3 config
├── data/                       # Sample data
├── tests/                      # Unit tests
├── docker-compose.yml
├── Dockerfile
├── dvc.yaml                   # DVC pipeline
├── params.yaml                # Training params
├── requirements.txt
├── setup.bat / setup.sh       # Setup scripts
└── README.md
```

---

## 📈 Model Performance

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Random Forest | $142 | $198 | 0.72 |
| Gradient Boosting | $135 | $185 | 0.75 |
| **Ensemble** | **$128** | **$176** | **0.78** |

---

## 👤 Author

**Ali Abbass (OTE22)**
- GitHub: [@OTE22](https://github.com/OTE22)
- Repository: [customer-lifetime-value](https://github.com/OTE22/customer-lifetime-value)

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

<p align="center">
  Made with ❤️ by <strong>Ali Abbass (OTE22)</strong>
</p>
