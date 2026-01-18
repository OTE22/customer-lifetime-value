"""
Configuration Management for CLV Prediction System
Centralized settings management with environment variable support.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class Environment(Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    type: str = "sqlite"  # sqlite, postgresql, mysql
    host: str = "localhost"
    port: int = 5432
    name: str = "clv_predictions"
    user: str = ""
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10

    @property
    def connection_string(self) -> str:
        if self.type == "sqlite":
            return f"sqlite:///{self.name}.db"
        return f"{self.type}://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class CacheConfig:
    """Cache configuration settings."""

    enabled: bool = True
    type: str = "memory"  # memory, redis
    host: str = "localhost"
    port: int = 6379
    ttl_seconds: int = 3600  # 1 hour default
    max_size: int = 1000  # Max items in memory cache


@dataclass
class ModelConfig:
    """ML model configuration settings."""

    model_dir: str = "models"
    ensemble_weights: Dict[str, float] = field(
        default_factory=lambda: {"gradient_boosting": 0.40, "random_forest": 0.35, "linear": 0.25}
    )
    random_forest_estimators: int = 100
    random_forest_max_depth: int = 15
    gradient_boosting_estimators: int = 100
    gradient_boosting_learning_rate: float = 0.1
    gradient_boosting_max_depth: int = 6
    retrain_threshold_days: int = 30
    min_samples_for_training: int = 100
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class APIConfig:
    """API configuration settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    request_timeout: int = 30  # seconds
    max_page_size: int = 1000
    default_page_size: int = 100


@dataclass
class LoggingConfig:
    """Logging configuration settings."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "logs/clv_api.log"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    json_format: bool = False


@dataclass
class CLVConfig:
    """Main configuration class for CLV Prediction System."""

    environment: Environment = Environment.DEVELOPMENT
    project_name: str = "CLV Prediction System"
    version: str = "1.0.0"
    debug: bool = False
    secret_key: str = "change-this-in-production"

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Data paths
    data_dir: str = "data"
    default_data_file: str = "customers.csv"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir) / self.default_data_file

    @property
    def model_path(self) -> Path:
        return Path(self.model.model_dir)

    @classmethod
    def from_env(cls) -> "CLVConfig":
        """Create configuration from environment variables."""
        env_str = os.getenv("CLV_ENVIRONMENT", "development").lower()
        environment = (
            Environment(env_str)
            if env_str in [e.value for e in Environment]
            else Environment.DEVELOPMENT
        )

        config = cls(
            environment=environment,
            debug=os.getenv("CLV_DEBUG", "false").lower() == "true",
            secret_key=os.getenv("CLV_SECRET_KEY", "change-this-in-production"),
        )

        # Database config from env
        config.database.type = os.getenv("CLV_DB_TYPE", config.database.type)
        config.database.host = os.getenv("CLV_DB_HOST", config.database.host)
        config.database.port = int(os.getenv("CLV_DB_PORT", config.database.port))
        config.database.name = os.getenv("CLV_DB_NAME", config.database.name)
        config.database.user = os.getenv("CLV_DB_USER", config.database.user)
        config.database.password = os.getenv("CLV_DB_PASSWORD", config.database.password)

        # Cache config from env
        config.cache.enabled = os.getenv("CLV_CACHE_ENABLED", "true").lower() == "true"
        config.cache.type = os.getenv("CLV_CACHE_TYPE", config.cache.type)
        config.cache.host = os.getenv("CLV_CACHE_HOST", config.cache.host)
        config.cache.port = int(os.getenv("CLV_CACHE_PORT", config.cache.port))
        config.cache.ttl_seconds = int(os.getenv("CLV_CACHE_TTL", config.cache.ttl_seconds))

        # API config from env
        config.api.host = os.getenv("CLV_API_HOST", config.api.host)
        config.api.port = int(os.getenv("CLV_API_PORT", config.api.port))
        config.api.workers = int(os.getenv("CLV_API_WORKERS", config.api.workers))
        config.api.rate_limit_requests = int(
            os.getenv("CLV_RATE_LIMIT", config.api.rate_limit_requests)
        )

        # Logging config from env
        config.logging.level = os.getenv("CLV_LOG_LEVEL", config.logging.level)
        config.logging.file_path = os.getenv("CLV_LOG_FILE", config.logging.file_path)

        # Data paths from env
        config.data_dir = os.getenv("CLV_DATA_DIR", config.data_dir)
        config.default_data_file = os.getenv("CLV_DATA_FILE", config.default_data_file)

        # Model config from env
        config.model.model_dir = os.getenv("CLV_MODEL_DIR", config.model.model_dir)

        return config

    @classmethod
    def from_json(cls, filepath: str) -> "CLVConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        config = cls()

        # Map JSON to config attributes
        if "environment" in data:
            config.environment = Environment(data["environment"])
        if "debug" in data:
            config.debug = data["debug"]
        if "secret_key" in data:
            config.secret_key = data["secret_key"]

        # Sub-configurations
        if "database" in data:
            for key, value in data["database"].items():
                if hasattr(config.database, key):
                    setattr(config.database, key, value)

        if "cache" in data:
            for key, value in data["cache"].items():
                if hasattr(config.cache, key):
                    setattr(config.cache, key, value)

        if "model" in data:
            for key, value in data["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)

        if "api" in data:
            for key, value in data["api"].items():
                if hasattr(config.api, key):
                    setattr(config.api, key, value)

        if "logging" in data:
            for key, value in data["logging"].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment.value,
            "project_name": self.project_name,
            "version": self.version,
            "debug": self.debug,
            "database": {
                "type": self.database.type,
                "host": self.database.host,
                "port": self.database.port,
                "name": self.database.name,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "type": self.cache.type,
                "ttl_seconds": self.cache.ttl_seconds,
            },
            "model": {
                "model_dir": self.model.model_dir,
                "ensemble_weights": self.model.ensemble_weights,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "workers": self.api.workers,
                "rate_limit_requests": self.api.rate_limit_requests,
            },
            "logging": {
                "level": self.logging.level,
                "file_path": self.logging.file_path,
            },
        }


# Global configuration instance
_config: Optional[CLVConfig] = None


def get_config() -> CLVConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = CLVConfig.from_env()
    return _config


def set_config(config: CLVConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to None."""
    global _config
    _config = None
