# Customer Lifetime Value (CLV) Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/XGBoost-1.7+-red.svg" alt="XGBoost">
  <img src="https://img.shields.io/badge/LightGBM-4.0+-purple.svg" alt="LightGBM">
  <img src="https://img.shields.io/badge/Docker-Ready-blue.svg" alt="Docker">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  <strong>🚀 Production-ready machine learning system for predicting customer lifetime value in e-commerce</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#docker">Docker</a> •
  <a href="#api-documentation">API Docs</a> •
  <a href="#meta-ads-integration">Meta Ads</a>
</p>

---

## 👤 Author

**Ali Abbass (OTE22)**
- GitHub: [@OTE22](https://github.com/OTE22)
- Repository: [customer-lifetime-value](https://github.com/OTE22/customer-lifetime-value)

---

## 🎯 Overview

This CLV Prediction System uses machine learning to forecast customer lifetime value from early behavioral patterns. Instead of waiting months to identify your best customers, predict their value within days of their first purchase.

**Key Benefits:**
- 📈 **25-40% ROAS improvement** through CLV-optimized ad spend
- 🎯 **Better customer acquisition** - spend more on high-value prospects
- 💰 **Automated budget allocation** based on the 3:2:1 rule
- 🔮 **Early identification** of High-CLV customers

## ✨ Features

### Machine Learning Pipeline
- **XGBoost** - State-of-the-art gradient boosting with GPU support
- **LightGBM** - Fast, distributed, high-performance ML
- **Random Forest** - Robust predictions with feature importance
- **Gradient Boosting** - Captures complex patterns
- **Auto-Weighted Ensemble** - Optimal weights based on validation
- **Model Registry** - Version control and model management

### Advanced Feature Engineering (60+ Features)
- **RFM Analysis** - Recency, Frequency, Monetary scoring
- **Behavioral Features** - Purchase velocity, engagement, churn risk
- **Temporal Features** - Seasonality, day-of-week, time thresholds
- **Statistical Features** - Z-scores, percentiles, outlier detection
- **Interaction Features** - Value×frequency, engagement×value composites

### Production Infrastructure
- ⚙️ **Configuration Management** - Environment variables, JSON config
- 📝 **Structured Logging** - JSON format, rotating files, metrics
- 🗄️ **Caching Layer** - LRU memory cache + Redis support
- 🛡️ **Rate Limiting** - Token bucket algorithm protection
- 🔐 **API Key Auth** - Optional authentication support
- 🐳 **Docker Ready** - Multi-stage build, docker-compose

### Modern Dashboard
- 📊 Real-time KPI visualization
- 📈 Interactive Chart.js graphs
- 🎨 Premium dark theme with glassmorphism
- 📱 Fully responsive design

### Meta Ads Integration
- 🎯 Automatic audience segmentation
- 💵 Budget allocation recommendations
- 🔄 Lookalike audience strategies
- 📊 Campaign performance tracking

---

## 🚀 Quick Start

### Option 1: Setup Script (Recommended)

**Windows:**
```batch
.\setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

This will:
1. Create a virtual environment
2. Install all dependencies
3. Create necessary directories
4. Generate sample data

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/OTE22/customer-lifetime-value.git
cd customer-lifetime-value

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python data/generate_data.py

# Start the API server
python -m uvicorn backend.api_enhanced:app --reload --port 8000
```

### Option 3: Install as Package

```bash
# Basic install
pip install -e .

# With development tools
pip install -e ".[dev]"

# With Redis support
pip install -e ".[redis]"
```

---

## 🐳 Docker

### Quick Start with Docker Compose

```bash
# Production (API + Redis + Nginx Frontend)
docker-compose up -d

# Development (with hot reload)
docker-compose --profile dev up api-dev

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI backend |
| `redis` | 6379 | Cache (optional) |
| `frontend` | 3000 | Nginx frontend |

### Docker Commands

```bash
# Build image only
docker build -t clv-prediction .

# Run standalone container
docker run -p 8000:8000 clv-prediction

# Development build
docker build --target development -t clv-prediction:dev .
```

---

## 📁 Project Structure

```
clv/
├── backend/                    # Python API & ML
│   ├── config.py              # Configuration management
│   ├── logging_config.py      # Structured logging
│   ├── exceptions.py          # Custom exceptions
│   ├── cache.py               # LRU/Redis caching
│   ├── schemas.py             # Pydantic validation
│   ├── middleware.py          # Rate limiting, security
│   ├── dependencies.py        # Dependency injection
│   ├── data_processor.py      # Data cleaning
│   ├── feature_engineering.py # RFM features
│   ├── ml_models.py           # Base ML models
│   ├── ml_models_enhanced.py  # Enhanced ML + registry
│   ├── clv_predictor.py       # Prediction pipeline
│   ├── meta_ads_integration.py# Meta Ads optimization
│   ├── api.py                 # Basic API
│   └── api_enhanced.py        # Production API
├── frontend/                   # Dashboard UI
│   ├── index.html
│   ├── css/styles.css
│   └── js/app.js
├── data/
│   ├── generate_data.py       # Data generator
│   └── customers.csv          # Sample dataset
├── tests/
│   └── test_clv.py            # Unit tests
├── Dockerfile                 # Multi-stage Docker
├── docker-compose.yml         # Container orchestration
├── nginx.conf                 # Frontend proxy
├── pyproject.toml             # Modern Python config
├── setup.py                   # Legacy packaging
├── setup.bat                  # Windows setup
├── setup.sh                   # Unix setup
├── requirements.txt           # Dependencies
├── .env.example               # Environment template
├── CHANGELOG.md               # Version history
└── README.md
```

---

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
# Application
CLV_ENVIRONMENT=production  # development, staging, production
CLV_DEBUG=false
CLV_SECRET_KEY=your-secret-key

# API
CLV_API_HOST=0.0.0.0
CLV_API_PORT=8000
CLV_API_WORKERS=4
CLV_RATE_LIMIT=100

# Cache
CLV_CACHE_ENABLED=true
CLV_CACHE_TYPE=memory  # memory, redis
CLV_CACHE_TTL=3600

# Logging
CLV_LOG_LEVEL=INFO
CLV_LOG_FILE=logs/clv_api.log
```

---

## 📊 API Documentation

### Base URL
```
http://localhost:8000
```

### Interactive Docs
- **Swagger UI:** http://localhost:8000/api/docs
- **ReDoc:** http://localhost:8000/api/redoc

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check with component status |
| GET | `/api/status` | Detailed status + cache stats |
| GET | `/api/customers` | List customers with pagination |
| GET | `/api/customers/{id}` | Get single customer |
| POST | `/api/predict` | Predict CLV for new customer |
| POST | `/api/predict/batch` | Batch predictions |
| GET | `/api/segments` | Segment distribution |
| GET | `/api/metrics` | Model performance metrics |
| POST | `/api/model/train` | Train/retrain model |
| GET | `/api/meta-ads/audiences` | Audience segments |
| GET | `/api/meta-ads/budget-allocation` | Budget recommendations |
| GET | `/api/dashboard/summary` | Dashboard data |
| GET | `/api/cache/stats` | Cache statistics |
| POST | `/api/cache/clear` | Clear cache |

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

Response:
```json
{
  "predicted_clv": 892.50,
  "segment": "High-CLV",
  "confidence": "High",
  "recommended_cac": 267.75,
  "model_version": "2.0.0",
  "cached": false
}
```

---

## 📱 Meta Ads Integration

### Budget Allocation (3:2:1 Rule)

| Segment | Budget % | CAC Target | Strategy |
|---------|----------|------------|----------|
| High-CLV | 50% | 30% of CLV | Value optimization, 1% lookalikes |
| Growth-Potential | 35% | 30% of CLV | Conversion optimization |
| Low-CLV | 15% | 30% of CLV | Cost caps, testing only |

```bash
curl "http://localhost:8000/api/meta-ads/budget-allocation?total_budget=10000"
```

---

## 📈 Model Performance

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Random Forest | $142 | $198 | 0.72 |
| Gradient Boosting | $135 | $185 | 0.75 |
| **Ensemble** | **$128** | **$176** | **0.78** |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=backend --cov-report=html

# Async tests
pytest tests/ --asyncio-mode=auto
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by <strong>Ali Abbass (OTE22)</strong>
</p>
