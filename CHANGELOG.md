# Changelog

All notable changes to the CLV Prediction System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-17

### Added
- **Docker Support**: Multi-stage Dockerfile with production and development targets
- **Docker Compose**: Complete orchestration with API, Redis, and Nginx frontend
- **Configuration Management**: Centralized settings with environment variable support (`config.py`)
- **Structured Logging**: JSON formatter, rotating file handlers, metrics logging (`logging_config.py`)
- **Exception Handling**: Custom exception hierarchy with 40+ error codes (`exceptions.py`)
- **Caching Layer**: LRU memory cache and Redis support with TTL (`cache.py`)
- **Pydantic Schemas**: 25+ request/response models with validation (`schemas.py`)
- **Enhanced ML Models**: Hyperparameter tuning, cross-validation, model registry (`ml_models_enhanced.py`)
- **Production Middleware**: Rate limiting, request logging, security headers (`middleware.py`)
- **Dependency Injection**: Service container, API key auth, health checking (`dependencies.py`)
- **Production API**: Full-featured API with all middleware integrated (`api_enhanced.py`)
- **Setup Scripts**: `setup.bat` (Windows) and `setup.sh` (Unix/Mac)
- **Project Configuration**: `pyproject.toml` and `setup.py` with modern packaging

### Changed
- Upgraded from v1.0.0 to v2.0.0
- Updated `requirements.txt` with production dependencies
- Enhanced `__init__.py` to export all modules
- Author information: Ali Abbass (OTE22)

### Security
- Non-root Docker user
- Rate limiting (token bucket algorithm)
- Security headers (XSS, CSRF protection)
- API key authentication support

## [1.0.0] - 2026-01-17

### Added
- Initial release
- Core ML models (Random Forest, Gradient Boosting, Ensemble)
- Data processing and feature engineering
- FastAPI REST API
- Meta Ads integration
- Modern frontend dashboard
- 1000-row fake customer dataset
