"""
Customer Lifetime Value (CLV) Prediction System
A production-ready ML system for predicting customer lifetime value in e-commerce.

Author: Ali Abbass (OTE22)
"""

__version__ = "2.0.0"
__author__ = "Ali Abbass (OTE22)"

# Core modules
from .config import get_config, CLVConfig
from .logging_config import get_logger, LoggerFactory
from .exceptions import CLVException, ValidationError, ModelNotTrainedError
from .cache import get_cache, CacheManager, cached

# Data processing
from .data_processor import DataProcessor
from .feature_engineering import AdvancedFeatureEngineer, engineer_features

# ML models
from .ml_models import (
    ProductionEnsemble, 
    XGBoostCLVModel, 
    LightGBMCLVModel,
    get_available_models
)

# Prediction pipeline
from .clv_predictor import CLVPredictor

# Meta Ads
from .meta_ads_integration import MetaAdsIntegration

# API components
from .dependencies import ServiceContainer, get_service_container

__all__ = [
    # Config
    "get_config",
    "CLVConfig",
    # Logging
    "get_logger",
    "LoggerFactory",
    # Exceptions
    "CLVException",
    "ValidationError", 
    "ModelNotTrainedError",
    # Cache
    "get_cache",
    "CacheManager",
    "cached",
    # Data
    "DataProcessor",
    "AdvancedFeatureEngineer",
    "engineer_features",
    # Models
    "ProductionEnsemble",
    "XGBoostCLVModel",
    "LightGBMCLVModel",
    "get_available_models",
    # Predictor
    "CLVPredictor",
    # Meta Ads
    "MetaAdsIntegration",
    # Services
    "ServiceContainer",
    "get_service_container",
]
