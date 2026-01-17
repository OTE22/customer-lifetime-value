"""
Dependency Injection for CLV Prediction System
FastAPI dependencies for services, caching, and configuration.
"""

from typing import Optional, Generator, AsyncGenerator
from functools import lru_cache
from fastapi import Depends, Request, HTTPException, Header
import asyncio

from .config import get_config, CLVConfig
from .logging_config import get_logger, LoggerFactory
from .cache import CacheManager, get_cache
from .exceptions import UnauthorizedError, ForbiddenError


logger = get_logger(__name__)


# Configuration Dependency
@lru_cache()
def get_settings() -> CLVConfig:
    """Get cached application settings."""
    return get_config()


def get_config_dependency() -> CLVConfig:
    """Dependency to inject configuration."""
    return get_settings()


# Cache Dependency
def get_cache_manager() -> CacheManager:
    """Dependency to inject cache manager."""
    return get_cache()


# Request Context
async def get_request_id(request: Request) -> Optional[str]:
    """Get request ID from middleware."""
    return getattr(request.state, 'request_id', None)


# Pagination Dependencies
class PaginationParams:
    """Pagination parameters extracted from query."""
    
    def __init__(
        self,
        page: int = 1,
        page_size: int = 100,
        config: CLVConfig = Depends(get_config_dependency)
    ):
        self.page = max(1, page)
        self.page_size = min(max(1, page_size), config.api.max_page_size)
        
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        return self.page_size


# API Key Authentication (optional)
class APIKeyValidator:
    """Validates API keys for protected endpoints."""
    
    # In production, these would be stored in a database
    _valid_keys = {
        "demo-key-12345": {"name": "Demo", "scopes": ["read", "predict"]},
        "admin-key-67890": {"name": "Admin", "scopes": ["read", "write", "predict", "admin"]}
    }
    
    @classmethod
    def validate(cls, api_key: str) -> Optional[dict]:
        """Validate an API key and return its metadata."""
        return cls._valid_keys.get(api_key)
    
    @classmethod
    def has_scope(cls, api_key: str, scope: str) -> bool:
        """Check if an API key has a specific scope."""
        key_data = cls.validate(api_key)
        if not key_data:
            return False
        return scope in key_data.get("scopes", [])


async def get_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """Extract API key from header (optional)."""
    return x_api_key


async def require_api_key(
    api_key: Optional[str] = Depends(get_api_key)
) -> str:
    """Require a valid API key."""
    if not api_key:
        raise UnauthorizedError("API key required. Include X-API-Key header.")
    
    key_data = APIKeyValidator.validate(api_key)
    if not key_data:
        raise UnauthorizedError("Invalid API key")
    
    return api_key


def require_scope(scope: str):
    """Create a dependency that requires a specific scope."""
    async def check_scope(api_key: str = Depends(require_api_key)):
        if not APIKeyValidator.has_scope(api_key, scope):
            raise ForbiddenError(f"This endpoint requires '{scope}' scope")
        return api_key
    return check_scope


# Service Container
class ServiceContainer:
    """Container for application services with lazy initialization."""
    
    _instance: Optional["ServiceContainer"] = None
    
    def __init__(self, config: Optional[CLVConfig] = None):
        self._config = config or get_config()
        self._predictor = None
        self._data_processor = None
        self._feature_engineer = None
        self._meta_ads = None
        self._model_registry = None
        self._initialized = False
    
    @classmethod
    def get_instance(cls) -> "ServiceContainer":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ServiceContainer()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance."""
        if cls._instance:
            cls._instance._initialized = False
        cls._instance = None
    
    @property
    def config(self) -> CLVConfig:
        return self._config
    
    @property
    def predictor(self):
        """Get or create CLV predictor."""
        if self._predictor is None:
            from .clv_predictor import CLVPredictor
            self._predictor = CLVPredictor()
        return self._predictor
    
    @property
    def data_processor(self):
        """Get or create data processor."""
        if self._data_processor is None:
            from .data_processor import DataProcessor
            self._data_processor = DataProcessor()
        return self._data_processor
    
    @property
    def feature_engineer(self):
        """Get or create feature engineer."""
        if self._feature_engineer is None:
            from .feature_engineering import FeatureEngineer
            self._feature_engineer = FeatureEngineer()
        return self._feature_engineer
    
    @property
    def meta_ads(self):
        """Get or create Meta Ads integration."""
        if self._meta_ads is None:
            from .meta_ads_integration import MetaAdsIntegration
            self._meta_ads = MetaAdsIntegration()
        return self._meta_ads
    
    @property
    def model_registry(self):
        """Get or create model registry."""
        if self._model_registry is None:
            from .ml_models_enhanced import ModelRegistry
            self._model_registry = ModelRegistry()
        return self._model_registry
    
    def initialize(self, force: bool = False) -> bool:
        """Initialize all services and load models."""
        if self._initialized and not force:
            return True
        
        try:
            logger.info("Initializing service container...")
            
            # Initialize logging
            LoggerFactory.setup(self._config.logging)
            
            # Pre-load services (lazy loading will happen on first access)
            _ = self.data_processor
            _ = self.feature_engineer
            
            # Try to load existing model if available
            try:
                self.predictor.load_model()
                logger.info("Loaded existing model")
            except Exception as e:
                logger.warning(f"No existing model found: {e}")
            
            self._initialized = True
            logger.info("Service container initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            return False


def get_service_container() -> ServiceContainer:
    """Dependency to inject service container."""
    return ServiceContainer.get_instance()


def get_predictor(
    container: ServiceContainer = Depends(get_service_container)
):
    """Dependency to inject CLV predictor."""
    return container.predictor


def get_data_processor(
    container: ServiceContainer = Depends(get_service_container)
):
    """Dependency to inject data processor."""
    return container.data_processor


def get_feature_engineer(
    container: ServiceContainer = Depends(get_service_container)
):
    """Dependency to inject feature engineer."""
    return container.feature_engineer


def get_meta_ads(
    container: ServiceContainer = Depends(get_service_container)
):
    """Dependency to inject Meta Ads integration."""
    return container.meta_ads


# Health Check Dependencies
class HealthChecker:
    """Health check utility."""
    
    def __init__(self, container: ServiceContainer):
        self.container = container
    
    async def check_all(self) -> dict:
        """Check all components health."""
        import time
        
        components = []
        overall_status = "healthy"
        
        # Check data availability
        try:
            start = time.time()
            processor = self.container.data_processor
            # Try to access data
            if hasattr(processor, 'raw_data') and processor.raw_data is not None:
                data_status = "healthy"
            else:
                data_status = "degraded"
                overall_status = "degraded"
            latency = (time.time() - start) * 1000
            
            components.append({
                "name": "data",
                "status": data_status,
                "latency_ms": round(latency, 2)
            })
        except Exception as e:
            components.append({
                "name": "data",
                "status": "unhealthy",
                "message": str(e)
            })
            overall_status = "unhealthy"
        
        # Check model
        try:
            start = time.time()
            predictor = self.container.predictor
            model_status = "healthy" if predictor.model_trained else "degraded"
            if model_status == "degraded":
                overall_status = "degraded"
            latency = (time.time() - start) * 1000
            
            components.append({
                "name": "model",
                "status": model_status,
                "latency_ms": round(latency, 2)
            })
        except Exception as e:
            components.append({
                "name": "model",
                "status": "unhealthy",
                "message": str(e)
            })
            overall_status = "unhealthy"
        
        # Check cache
        try:
            cache = get_cache()
            cache_stats = cache.get_stats()
            cache_status = "healthy" if cache_stats.get("enabled") else "disabled"
            
            components.append({
                "name": "cache",
                "status": cache_status,
                "stats": cache_stats
            })
        except Exception as e:
            components.append({
                "name": "cache",
                "status": "unhealthy",
                "message": str(e)
            })
        
        return {
            "status": overall_status,
            "components": components
        }


def get_health_checker(
    container: ServiceContainer = Depends(get_service_container)
) -> HealthChecker:
    """Dependency to inject health checker."""
    return HealthChecker(container)


# Startup/Shutdown Events
async def startup_event():
    """Run on application startup."""
    logger.info("Starting CLV Prediction API...")
    container = ServiceContainer.get_instance()
    container.initialize()
    logger.info("Application started successfully")


async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down CLV Prediction API...")
    
    # Clear cache
    try:
        cache = get_cache()
        cache.clear()
    except:
        pass
    
    # Reset services
    ServiceContainer.reset()
    CacheManager.reset_instance()
    
    logger.info("Application shutdown complete")
