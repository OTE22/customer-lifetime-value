"""
API Middleware for CLV Prediction System
Request logging, rate limiting, and error handling.
"""

import time
import uuid
from datetime import datetime
from typing import Callable, Dict, Any
from collections import defaultdict
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import HTTPException

from .config import get_config
from .logging_config import get_logger, LogMetrics
from .exceptions import CLVException

logger = get_logger(__name__)
metrics = LogMetrics(logger)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to add request context (ID, timing) to all requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        # Add to headers
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - request.state.start_time) * 1000
        
        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all requests."""
    
    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/api/health", "/api/docs", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        start_time = time.time()
        
        # Get request info
        method = request.method
        path = request.url.path
        user_agent = request.headers.get("user-agent")
        
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log the request
        metrics.log_api_request(
            method=method,
            path=path,
            status_code=response.status_code,
            latency_ms=duration_ms,
            user_agent=user_agent
        )
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiting middleware."""
    
    def __init__(
        self,
        app,
        requests_per_period: int = None,
        period_seconds: int = None,
        exclude_paths: list = None
    ):
        super().__init__(app)
        config = get_config().api
        self.max_requests = requests_per_period or config.rate_limit_requests
        self.period = period_seconds or config.rate_limit_period
        self.exclude_paths = exclude_paths or ["/api/health"]
        
        # Token bucket state: {client_id: (tokens, last_update)}
        self._buckets: Dict[str, tuple] = defaultdict(lambda: (self.max_requests, time.time()))
        self._lock = asyncio.Lock()
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Try API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        
        # Fall back to IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        client = request.client
        if client:
            return f"ip:{client.host}"
        
        return "unknown"
    
    async def _check_rate_limit(self, client_id: str) -> tuple:
        """Check and update rate limit. Returns (allowed, remaining, reset_time)."""
        async with self._lock:
            tokens, last_update = self._buckets[client_id]
            now = time.time()
            
            # Refill tokens based on time passed
            time_passed = now - last_update
            tokens_to_add = time_passed * (self.max_requests / self.period)
            tokens = min(self.max_requests, tokens + tokens_to_add)
            
            if tokens >= 1:
                # Allow request
                tokens -= 1
                self._buckets[client_id] = (tokens, now)
                remaining = int(tokens)
                reset_time = int(now + self.period)
                return True, remaining, reset_time
            else:
                # Deny request
                self._buckets[client_id] = (tokens, now)
                retry_after = int(self.period - time_passed)
                return False, 0, retry_after
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        allowed, remaining, reset_time = await self._check_rate_limit(client_id)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_id}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "error_code": 4000,
                    "error_name": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Please retry after {reset_time} seconds.",
                    "retry_after": reset_time
                },
                headers={
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": str(reset_time)
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global exception handling middleware."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
            
        except CLVException as e:
            # Handle custom exceptions
            status_code = getattr(e, 'status_code', 500)
            if hasattr(e, 'error_code'):
                if e.error_code.value >= 4000 and e.error_code.value < 5000:
                    status_code = 400 + (e.error_code.value % 100)
            
            logger.error(f"CLV Exception: {e.message}", exc_info=True)
            
            response_data = e.to_dict()
            response_data["request_id"] = getattr(request.state, 'request_id', None)
            
            return JSONResponse(
                status_code=status_code,
                content=response_data
            )
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": True,
                    "error_code": e.status_code * 10,
                    "error_name": "HTTP_ERROR",
                    "message": e.detail,
                    "request_id": getattr(request.state, 'request_id', None)
                }
            )
            
        except Exception as e:
            # Handle unexpected exceptions
            logger.error(f"Unhandled exception: {e}", exc_info=True)
            
            # In production, don't expose internal errors
            config = get_config()
            message = str(e) if config.debug else "An internal error occurred"
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "error_code": 5000,
                    "error_name": "INTERNAL_ERROR",
                    "message": message,
                    "request_id": getattr(request.state, 'request_id', None)
                }
            )


class CORSMiddleware:
    """Custom CORS middleware with configuration."""
    
    def __init__(
        self,
        app,
        allow_origins: list = None,
        allow_methods: list = None,
        allow_headers: list = None,
        allow_credentials: bool = True,
        max_age: int = 600
    ):
        self.app = app
        config = get_config().api
        self.allow_origins = allow_origins or config.cors_origins
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        self.allow_headers = allow_headers or ["*"]
        self.allow_credentials = allow_credentials
        self.max_age = max_age
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        origin = request.headers.get("origin")
        
        # Handle preflight
        if request.method == "OPTIONS":
            response = Response(status_code=204)
            self._add_cors_headers(response, origin)
            await response(scope, receive, send)
            return
        
        # Process request
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                
                if origin and self._is_origin_allowed(origin):
                    headers[b"access-control-allow-origin"] = origin.encode()
                    headers[b"access-control-allow-credentials"] = b"true"
                    headers[b"access-control-expose-headers"] = b"X-Request-ID, X-Response-Time-Ms"
                
                message["headers"] = list(headers.items())
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
    
    def _is_origin_allowed(self, origin: str) -> bool:
        if "*" in self.allow_origins:
            return True
        return origin in self.allow_origins
    
    def _add_cors_headers(self, response: Response, origin: str):
        if origin and self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
            response.headers["Access-Control-Allow-Credentials"] = str(self.allow_credentials).lower()
            response.headers["Access-Control-Max-Age"] = str(self.max_age)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        
        return response


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware to handle response compression hints."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add compression hint for large responses
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > 1024:
            response.headers["Content-Encoding-Hint"] = "gzip"
        
        return response
