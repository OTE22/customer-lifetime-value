"""
Customer Lifetime Value (CLV) Prediction System
A production-ready ML system for predicting customer lifetime value in e-commerce.

Author: Ali Abbass (OTE22)
"""

__version__ = "2.0.0"
__author__ = "Ali Abbass (OTE22)"

from .cache import CacheManager, cached, get_cache

# Prediction pipeline
from .clv_predictor import CLVPredictor

# Core modules
from .config import CLVConfig, get_config

# Data processing
from .data_processor import DataProcessor

# API components
from .dependencies import ServiceContainer, get_service_container
from .exceptions import CLVException, ModelNotTrainedError, ValidationError
from .feature_engineering import AdvancedFeatureEngineer, engineer_features
from .logging_config import LoggerFactory, get_logger

# Meta Ads
from .meta_ads_integration import MetaAdsIntegration

# ML models
from .ml_models import LightGBMCLVModel, ProductionEnsemble, XGBoostCLVModel, get_available_models

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
