"""
Logging Infrastructure for CLV Prediction System
Provides structured logging with multiple handlers and formatters.
"""

import json
import logging
import logging.handlers
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import LoggingConfig, get_config


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)

        return json.dumps(log_obj)


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that adds context to all log messages."""

    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        # Add extra fields to the record
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"]["extra_fields"] = self.extra
        return msg, kwargs

    def with_context(self, **context) -> "ContextLogger":
        """Create a new logger with additional context."""
        new_extra = {**self.extra, **context}
        return ContextLogger(self.logger, new_extra)


class LoggerFactory:
    """Factory for creating configured loggers."""

    _initialized: bool = False
    _loggers: Dict[str, ContextLogger] = {}

    @classmethod
    def setup(cls, config: Optional[LoggingConfig] = None) -> None:
        """Set up the logging infrastructure."""
        if cls._initialized:
            return

        if config is None:
            config = get_config().logging

        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.level.upper()))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level.upper()))

        if config.json_format:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(config.format))

        root_logger.addHandler(console_handler)

        # File handler (if path specified)
        if config.file_path:
            log_path = Path(config.file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                config.file_path, maxBytes=config.max_bytes, backupCount=config.backup_count
            )
            file_handler.setLevel(getattr(logging, config.level.upper()))

            if config.json_format:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(config.format))

            root_logger.addHandler(file_handler)

        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str, **context) -> ContextLogger:
        """Get or create a logger with the given name."""
        if not cls._initialized:
            cls.setup()

        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = ContextLogger(logger, context)

        return cls._loggers[name]

    @classmethod
    def reset(cls) -> None:
        """Reset the logger factory."""
        cls._initialized = False
        cls._loggers.clear()


def get_logger(name: str, **context) -> ContextLogger:
    """Convenience function to get a logger."""
    return LoggerFactory.get_logger(name, **context)


class LogMetrics:
    """Utility class for logging metrics and performance data."""

    def __init__(self, logger: ContextLogger):
        self.logger = logger

    def log_prediction(
        self, customer_id: str, predicted_clv: float, segment: str, latency_ms: float
    ) -> None:
        """Log a prediction event."""
        self.logger.info(
            f"Prediction completed for customer {customer_id}",
            extra={
                "extra_fields": {
                    "event": "prediction",
                    "customer_id": customer_id,
                    "predicted_clv": predicted_clv,
                    "segment": segment,
                    "latency_ms": latency_ms,
                }
            },
        )

    def log_model_training(
        self, model_name: str, samples: int, metrics: Dict[str, float], duration_seconds: float
    ) -> None:
        """Log a model training event."""
        self.logger.info(
            f"Model {model_name} trained on {samples} samples",
            extra={
                "extra_fields": {
                    "event": "model_training",
                    "model_name": model_name,
                    "samples": samples,
                    "metrics": metrics,
                    "duration_seconds": duration_seconds,
                }
            },
        )

    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
        user_agent: Optional[str] = None,
    ) -> None:
        """Log an API request."""
        self.logger.info(
            f"{method} {path} - {status_code}",
            extra={
                "extra_fields": {
                    "event": "api_request",
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "user_agent": user_agent,
                }
            },
        )

    def log_error(
        self, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an error event."""
        self.logger.error(
            f"{error_type}: {error_message}",
            extra={
                "extra_fields": {
                    "event": "error",
                    "error_type": error_type,
                    "error_message": error_message,
                    "context": context or {},
                }
            },
        )
