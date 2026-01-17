"""
Exception Handling for CLV Prediction System
Custom exceptions and error handling utilities.
"""

from typing import Optional, Any, Dict
from enum import Enum
import traceback


class ErrorCode(Enum):
    """Error codes for the CLV system."""
    # Validation errors (1000-1999)
    VALIDATION_ERROR = 1000
    INVALID_CUSTOMER_ID = 1001
    INVALID_FEATURES = 1002
    MISSING_REQUIRED_FIELD = 1003
    INVALID_DATE_FORMAT = 1004
    VALUE_OUT_OF_RANGE = 1005
    
    # Data errors (2000-2999)
    DATA_NOT_FOUND = 2000
    CUSTOMER_NOT_FOUND = 2001
    DATA_LOAD_ERROR = 2002
    DATA_PROCESSING_ERROR = 2003
    INSUFFICIENT_DATA = 2004
    
    # Model errors (3000-3999)
    MODEL_NOT_FOUND = 3000
    MODEL_NOT_TRAINED = 3001
    MODEL_TRAINING_ERROR = 3002
    MODEL_PREDICTION_ERROR = 3003
    MODEL_SAVE_ERROR = 3004
    MODEL_LOAD_ERROR = 3005
    FEATURE_MISMATCH = 3006
    
    # API errors (4000-4999)
    RATE_LIMIT_EXCEEDED = 4000
    UNAUTHORIZED = 4001
    FORBIDDEN = 4002
    RESOURCE_NOT_FOUND = 4003
    METHOD_NOT_ALLOWED = 4004
    REQUEST_TIMEOUT = 4005
    
    # System errors (5000-5999)
    INTERNAL_ERROR = 5000
    DATABASE_ERROR = 5001
    CACHE_ERROR = 5002
    CONFIGURATION_ERROR = 5003
    SERVICE_UNAVAILABLE = 5004


class CLVException(Exception):
    """Base exception for CLV Prediction System."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result = {
            "error": True,
            "error_code": self.error_code.value,
            "error_name": self.error_code.name,
            "message": self.message,
        }
        
        if self.details:
            result["details"] = self.details
            
        if self.cause:
            result["cause"] = str(self.cause)
            
        return result
    
    def __repr__(self) -> str:
        return f"CLVException({self.error_code.name}: {self.message})"


# Validation Exceptions
class ValidationError(CLVException):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details={
                "field": field,
                "value": str(value) if value is not None else None,
                **(details or {})
            }
        )


class MissingRequiredFieldError(ValidationError):
    """Raised when a required field is missing."""
    
    def __init__(self, field: str):
        super().__init__(
            message=f"Required field '{field}' is missing",
            field=field
        )
        self.error_code = ErrorCode.MISSING_REQUIRED_FIELD


class ValueOutOfRangeError(ValidationError):
    """Raised when a value is out of acceptable range."""
    
    def __init__(
        self,
        field: str,
        value: Any,
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None
    ):
        range_str = ""
        if min_value is not None and max_value is not None:
            range_str = f" Must be between {min_value} and {max_value}"
        elif min_value is not None:
            range_str = f" Must be >= {min_value}"
        elif max_value is not None:
            range_str = f" Must be <= {max_value}"
            
        super().__init__(
            message=f"Value {value} for '{field}' is out of range.{range_str}",
            field=field,
            value=value,
            details={
                "min_value": min_value,
                "max_value": max_value
            }
        )
        self.error_code = ErrorCode.VALUE_OUT_OF_RANGE


# Data Exceptions
class DataError(CLVException):
    """Base exception for data-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DATA_NOT_FOUND,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)


class CustomerNotFoundError(DataError):
    """Raised when a customer is not found."""
    
    def __init__(self, customer_id: str):
        super().__init__(
            message=f"Customer '{customer_id}' not found",
            error_code=ErrorCode.CUSTOMER_NOT_FOUND,
            details={"customer_id": customer_id}
        )


class DataLoadError(DataError):
    """Raised when data cannot be loaded."""
    
    def __init__(self, filepath: str, cause: Optional[Exception] = None):
        super().__init__(
            message=f"Failed to load data from '{filepath}'",
            error_code=ErrorCode.DATA_LOAD_ERROR,
            details={"filepath": filepath}
        )
        self.cause = cause


class InsufficientDataError(DataError):
    """Raised when there is not enough data for an operation."""
    
    def __init__(self, required: int, available: int, operation: str = "operation"):
        super().__init__(
            message=f"Insufficient data for {operation}. Required: {required}, Available: {available}",
            error_code=ErrorCode.INSUFFICIENT_DATA,
            details={
                "required": required,
                "available": available,
                "operation": operation
            }
        )


# Model Exceptions
class ModelError(CLVException):
    """Base exception for model-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        model_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            error_code,
            details={
                "model_name": model_name,
                **(details or {})
            }
        )


class ModelNotTrainedError(ModelError):
    """Raised when trying to use an untrained model."""
    
    def __init__(self, model_name: str = "CLVPredictor"):
        super().__init__(
            message=f"Model '{model_name}' has not been trained. Call train() first.",
            error_code=ErrorCode.MODEL_NOT_TRAINED,
            model_name=model_name
        )


class ModelNotFoundError(ModelError):
    """Raised when a saved model cannot be found."""
    
    def __init__(self, model_path: str):
        super().__init__(
            message=f"Model not found at '{model_path}'",
            error_code=ErrorCode.MODEL_NOT_FOUND,
            details={"model_path": model_path}
        )


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    
    def __init__(self, model_name: str, cause: Optional[Exception] = None):
        super().__init__(
            message=f"Failed to train model '{model_name}'",
            error_code=ErrorCode.MODEL_TRAINING_ERROR,
            model_name=model_name
        )
        self.cause = cause


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""
    
    def __init__(self, model_name: str, cause: Optional[Exception] = None):
        super().__init__(
            message=f"Prediction failed for model '{model_name}'",
            error_code=ErrorCode.MODEL_PREDICTION_ERROR,
            model_name=model_name
        )
        self.cause = cause


class FeatureMismatchError(ModelError):
    """Raised when input features don't match expected features."""
    
    def __init__(
        self,
        expected_features: int,
        received_features: int,
        missing: Optional[list] = None
    ):
        super().__init__(
            message=f"Feature mismatch. Expected {expected_features} features, got {received_features}",
            error_code=ErrorCode.FEATURE_MISMATCH,
            details={
                "expected_features": expected_features,
                "received_features": received_features,
                "missing_features": missing
            }
        )


# API Exceptions
class APIError(CLVException):
    """Base exception for API-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)
        self.status_code = status_code


class RateLimitExceededError(APIError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, retry_after: int = 60):
        super().__init__(
            message=f"Rate limit exceeded. Please retry after {retry_after} seconds.",
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            status_code=429,
            details={"retry_after": retry_after}
        )


class UnauthorizedError(APIError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            error_code=ErrorCode.UNAUTHORIZED,
            status_code=401
        )


class ForbiddenError(APIError):
    """Raised when access is denied."""
    
    def __init__(self, message: str = "Access denied"):
        super().__init__(
            message=message,
            error_code=ErrorCode.FORBIDDEN,
            status_code=403
        )


class ResourceNotFoundError(APIError):
    """Raised when a requested resource is not found."""
    
    def __init__(self, resource: str, resource_id: Optional[str] = None):
        message = f"{resource} not found"
        if resource_id:
            message = f"{resource} '{resource_id}' not found"
            
        super().__init__(
            message=message,
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            status_code=404,
            details={"resource": resource, "resource_id": resource_id}
        )


# Utility Functions
def format_exception(exc: Exception) -> Dict[str, Any]:
    """Format any exception for API response."""
    if isinstance(exc, CLVException):
        return exc.to_dict()
    
    return {
        "error": True,
        "error_code": ErrorCode.INTERNAL_ERROR.value,
        "error_name": ErrorCode.INTERNAL_ERROR.name,
        "message": str(exc),
        "type": type(exc).__name__
    }


def wrap_exception(exc: Exception, context: Optional[str] = None) -> CLVException:
    """Wrap a standard exception in a CLVException."""
    if isinstance(exc, CLVException):
        return exc
    
    message = str(exc)
    if context:
        message = f"{context}: {message}"
    
    return CLVException(
        message=message,
        error_code=ErrorCode.INTERNAL_ERROR,
        cause=exc,
        details={"traceback": traceback.format_exc()}
    )
