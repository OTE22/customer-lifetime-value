"""
Production-Ready FastAPI Application for CLV Prediction System
Enhanced API with middleware, validation, caching, and comprehensive endpoints.
"""

import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Path as PathParam, Body, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import pandas as pd


from .config import get_config, CLVConfig
from .logging_config import get_logger, LogMetrics, LoggerFactory
from .schemas import (
    CustomerResponse, CustomerListResponse, PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse, SegmentAnalysis, SegmentStats,
    ModelPerformance, FeatureImportance, BudgetAllocationRequest, BudgetAllocationResponse,
    MetaAdsStrategy, AudienceSegment, BudgetAllocation, HealthCheck, ComponentHealth,
    HealthStatus, DashboardResponse, DashboardSummary, ErrorResponse, CacheStats,
    SegmentType, ConfidenceLevel
)
from .exceptions import (
    CLVException, CustomerNotFoundError, ModelNotTrainedError,
    ValidationError, format_exception
)
from .middleware import (
    RequestContextMiddleware, RequestLoggingMiddleware, RateLimitMiddleware,
    ErrorHandlerMiddleware, SecurityHeadersMiddleware
)
from .dependencies import (
    get_config_dependency, get_cache_manager, get_predictor, get_data_processor,
    get_feature_engineer, get_meta_ads, get_health_checker, get_service_container,
    PaginationParams, ServiceContainer, startup_event, shutdown_event,
    get_api_key, require_api_key
)
from .cache import CacheManager, get_cache

# Initialize logger
logger = get_logger(__name__)
metrics_logger = LogMetrics(logger)

# Application start time for uptime tracking
APP_START_TIME = datetime.utcnow()


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    await startup_event()
    yield
    await shutdown_event()


def create_app(config: Optional[CLVConfig] = None) -> FastAPI:
    """Factory function to create the FastAPI application."""
    
    if config is None:
        config = get_config()
    
    # Initialize logging
    LoggerFactory.setup(config.logging)
    
    # Create application
    app = FastAPI(
        title=config.project_name,
        version=config.version,
        description="Production-ready Customer Lifetime Value Prediction API",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware (order matters - first added is outermost)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestContextMiddleware)
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-Ms", "X-RateLimit-Remaining"]
    )
    
    return app


# Create application instance
app = create_app()

# Mount static files for frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    # Mount css and js directories at their expected paths
    css_dir = FRONTEND_DIR / "css"
    js_dir = FRONTEND_DIR / "js"
    
    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")
    
    # Also mount entire frontend for other static assets
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")



# Root route for frontend
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend index.html."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(
        status_code=404,
        content={"message": "Frontend not found. Place index.html in frontend/ directory."}
    )


@app.get("/index.html", include_in_schema=False)
async def serve_index():
    """Serve index.html explicitly."""
    return await serve_frontend()


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get(
    "/api/health",
    response_model=HealthCheck,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check(
    checker = Depends(get_health_checker),
    config: CLVConfig = Depends(get_config_dependency)
):
    """Check the health status of the API and its components."""
    health_data = await checker.check_all()
    
    uptime = (datetime.utcnow() - APP_START_TIME).total_seconds()
    
    components = [
        ComponentHealth(
            name=comp["name"],
            status=HealthStatus(comp["status"]) if comp["status"] in ["healthy", "degraded", "unhealthy"] else HealthStatus.UNHEALTHY,
            latency_ms=comp.get("latency_ms"),
            message=comp.get("message")
        )
        for comp in health_data["components"]
    ]
    
    return HealthCheck(
        status=HealthStatus(health_data["status"]),
        timestamp=datetime.utcnow(),
        version=config.version,
        uptime_seconds=uptime,
        components=components
    )


@app.get(
    "/api/status",
    tags=["Health"],
    summary="Detailed status information"
)
async def get_status(
    config: CLVConfig = Depends(get_config_dependency),
    cache: CacheManager = Depends(get_cache_manager)
):
    """Get detailed status information including cache stats."""
    cache_stats = cache.get_stats()
    uptime = (datetime.utcnow() - APP_START_TIME).total_seconds()
    
    return {
        "application": config.project_name,
        "version": config.version,
        "environment": config.environment.value,
        "uptime_seconds": uptime,
        "uptime_human": str(timedelta(seconds=int(uptime))),
        "cache": cache_stats,
        "config": {
            "rate_limit": config.api.rate_limit_requests,
            "page_size_max": config.api.max_page_size
        }
    }


# =============================================================================
# Customer Endpoints
# =============================================================================

@app.get(
    "/api/customers",
    response_model=CustomerListResponse,
    tags=["Customers"],
    summary="List all customers"
)
async def list_customers(
    segment: Optional[str] = Query(None, description="Filter by segment"),
    source: Optional[str] = Query(None, description="Filter by acquisition source"),
    min_clv: Optional[float] = Query(None, ge=0, description="Minimum CLV"),
    max_clv: Optional[float] = Query(None, ge=0, description="Maximum CLV"),
    sort_by: str = Query("predicted_clv", description="Sort field"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    pagination: PaginationParams = Depends(),
    predictor = Depends(get_predictor)
):
    """Get paginated list of customers with predictions."""
    try:
        if not predictor.is_trained:
            # Return raw data without predictions
            processor = predictor.data_processor
            if processor.raw_data is None:
                processor.load_data(predictor.data_path)
            df = processor.raw_data.copy()
        else:
            df = predictor.get_all_predictions()
        
        # Apply filters
        if segment and 'predicted_segment' in df.columns:
            df = df[df['predicted_segment'] == segment]
        elif segment and 'customer_segment' in df.columns:
            df = df[df['customer_segment'] == segment]
        
        if source and 'acquisition_source' in df.columns:
            df = df[df['acquisition_source'] == source]
        
        clv_col = 'predicted_clv' if 'predicted_clv' in df.columns else 'actual_clv'
        if min_clv is not None and clv_col in df.columns:
            df = df[df[clv_col] >= min_clv]
        if max_clv is not None and clv_col in df.columns:
            df = df[df[clv_col] <= max_clv]
        
        total = len(df)
        
        # Sort
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=(sort_order == "asc"))
        
        # Paginate
        df_page = df.iloc[pagination.offset:pagination.offset + pagination.limit].copy()
        
        # Convert all timestamp columns to strings for JSON serialization
        for col in df_page.columns:
            if pd.api.types.is_datetime64_any_dtype(df_page[col]):
                df_page[col] = df_page[col].astype(str)
        
        # Replace NaN values with None for proper JSON serialization
        df_page = df_page.where(pd.notnull(df_page), None)
        
        # Convert to records and clean up any remaining NaN/nan strings
        customers = df_page.to_dict('records')
        for customer in customers:
            for key, value in customer.items():
                if pd.isna(value) or value == 'nan' or value == 'NaT':
                    customer[key] = None
                # Convert None to "None" string for enum fields
                if key in ['campaign_type', 'acquisition_source'] and customer[key] is None:
                    customer[key] = "None"
        
        return CustomerListResponse(
            customers=customers,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
            has_more=(pagination.offset + pagination.limit) < total
        )
        
    except Exception as e:
        logger.error(f"Error listing customers: {e}")
        raise


@app.get(
    "/api/customers/{customer_id}",
    response_model=CustomerResponse,
    tags=["Customers"],
    summary="Get customer by ID"
)
async def get_customer(
    customer_id: str = PathParam(..., min_length=1, max_length=50),
    predictor = Depends(get_predictor),
    cache: CacheManager = Depends(get_cache_manager)
):
    """Get detailed customer information with predictions."""
    # Check cache
    cached = cache.get_prediction(customer_id)
    if cached:
        return CustomerResponse(**cached)
    
    try:
        if not predictor.is_trained:
            processor = predictor.data_processor
            if processor.raw_data is None:
                processor.load_data(predictor.data_path)
            df = processor.raw_data
        else:
            df = predictor.get_all_predictions()
        
        customer_data = df[df['customer_id'] == customer_id]
        
        if customer_data.empty:
            raise CustomerNotFoundError(customer_id)
        
        # Get customer as dict
        result = customer_data.iloc[0].to_dict()
        
        # Convert timestamps to strings and handle NaN values
        for key, value in result.items():
            # Convert Timestamp to string
            if pd.api.types.is_datetime64_any_dtype(type(value)) or hasattr(value, 'isoformat'):
                result[key] = str(value)
            # Convert NaN to None or "None" for enum fields
            if pd.isna(value) or value == 'nan' or value == 'NaT':
                if key in ['campaign_type', 'acquisition_source']:
                    result[key] = "None"
                else:
                    result[key] = None
        
        # Cache result
        cache.set_prediction(customer_id, result)
        
        return CustomerResponse(**result)
        
    except CustomerNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error getting customer {customer_id}: {e}")
        raise


# =============================================================================
# Prediction Endpoints
# =============================================================================

@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Predict CLV for a customer"
)
async def predict_clv(
    request: PredictionRequest,
    predictor = Depends(get_predictor),
    cache: CacheManager = Depends(get_cache_manager),
    config: CLVConfig = Depends(get_config_dependency)
):
    """Predict Customer Lifetime Value based on features."""
    start_time = time.time()
    
    try:
        # Create DataFrame from request
        features = pd.DataFrame([request.model_dump()])
        
        # Convert enums to strings
        features['acquisition_source'] = features['acquisition_source'].apply(
            lambda x: x.value if hasattr(x, 'value') else x
        )
        features['campaign_type'] = features['campaign_type'].apply(
            lambda x: x.value if hasattr(x, 'value') else x
        )
        
        if not predictor.is_trained:
            # Train model first
            logger.info("Training model for first prediction...")
            predictor.train()
        
        # Make prediction
        prediction = predictor.predict_single(features.iloc[0].to_dict())
        
        # Determine confidence
        clv = prediction['predicted_clv']
        if clv >= 500:
            segment = SegmentType.HIGH_CLV
            confidence = ConfidenceLevel.HIGH
        elif clv >= 150:
            segment = SegmentType.GROWTH_POTENTIAL
            confidence = ConfidenceLevel.MEDIUM
        else:
            segment = SegmentType.LOW_CLV
            confidence = ConfidenceLevel.LOW
        
        # Calculate recommended CAC (30% of CLV)
        recommended_cac = round(clv * 0.3, 2)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Log prediction
        metrics_logger.log_prediction(
            customer_id="new_customer",
            predicted_clv=clv,
            segment=segment.value,
            latency_ms=latency_ms
        )
        
        return PredictionResponse(
            predicted_clv=round(clv, 2),
            segment=segment,
            confidence=confidence,
            recommended_cac=recommended_cac,
            model_version=getattr(predictor, 'model_version', config.version),
            cached=False
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise


@app.post(
    "/api/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Batch predict CLV for multiple customers"
)
async def batch_predict(
    request: BatchPredictionRequest,
    predictor = Depends(get_predictor)
):
    """Predict CLV for multiple customers in batch."""
    start_time = time.time()
    
    try:
        predictions = []
        
        for customer in request.customers:
            features = pd.DataFrame([customer.model_dump()])
            
            # Convert enums
            features['acquisition_source'] = features['acquisition_source'].apply(
                lambda x: x.value if hasattr(x, 'value') else x
            )
            features['campaign_type'] = features['campaign_type'].apply(
                lambda x: x.value if hasattr(x, 'value') else x
            )
            
            if not predictor.is_trained:
                predictor.train()
            
            pred = predictor.predict_single(features.iloc[0].to_dict())
            clv = pred['predicted_clv']
            
            if clv >= 500:
                segment = SegmentType.HIGH_CLV
                confidence = ConfidenceLevel.HIGH
            elif clv >= 150:
                segment = SegmentType.GROWTH_POTENTIAL
                confidence = ConfidenceLevel.MEDIUM
            else:
                segment = SegmentType.LOW_CLV
                confidence = ConfidenceLevel.LOW
            
            predictions.append(PredictionResponse(
                predicted_clv=round(clv, 2),
                segment=segment,
                confidence=confidence,
                recommended_cac=round(clv * 0.3, 2),
                cached=False
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise


# =============================================================================
# Segment Endpoints
# =============================================================================

@app.get(
    "/api/segments",
    response_model=SegmentAnalysis,
    tags=["Segments"],
    summary="Get segment analysis"
)
async def get_segments(
    predictor = Depends(get_predictor)
):
    """Get customer segment distribution and statistics."""
    try:
        if not predictor.is_trained:
            predictor.train()
        
        segments_data = predictor.get_segment_analysis()
        
        segments = {}
        total_value = 0
        total_customers = 0
        
        for seg_name, seg_data in segments_data['segments'].items():
            count = seg_data['count']
            avg_clv = seg_data.get('avg_predicted_clv', seg_data.get('avg_clv', 0))
            total_val = seg_data.get('total_predicted_value', seg_data.get('total_value', 0))
            
            segments[seg_name] = SegmentStats(
                count=count,
                avg_predicted_clv=round(avg_clv, 2),
                total_predicted_value=round(total_val, 2)
            )
            
            total_value += total_val
            total_customers += count
        
        # Calculate percentages
        for seg in segments.values():
            seg.percentage = round((seg.count / total_customers) * 100, 1) if total_customers > 0 else 0
        
        return SegmentAnalysis(
            segments=segments,
            total_customers=total_customers,
            total_value=round(total_value, 2)
        )
        
    except Exception as e:
        logger.error(f"Error getting segments: {e}")
        raise


# =============================================================================
# Model Endpoints
# =============================================================================

@app.get(
    "/api/metrics",
    response_model=ModelPerformance,
    tags=["Model"],
    summary="Get model performance metrics"
)
async def get_model_metrics(
    predictor = Depends(get_predictor),
    config: CLVConfig = Depends(get_config_dependency)
):
    """Get model performance metrics and feature importance."""
    try:
        if not predictor.is_trained:
            raise ModelNotTrainedError()
        
        metrics = predictor.get_model_metrics()
        feature_importance = predictor.get_feature_importance()
        
        from .schemas import ModelMetrics
        
        return ModelPerformance(
            model_name="CLVEnsemble",
            version=config.version,
            trained_at=getattr(predictor, 'trained_at', None),
            samples_trained=metrics.get('samples_trained', 0),
            metrics=ModelMetrics(
                mae=metrics.get('mae', 0),
                rmse=metrics.get('rmse', 0),
                r2=metrics.get('r2', 0)
            ),
            feature_importance=[
                FeatureImportance(
                    feature=row['feature'],
                    importance=round(row['importance'], 4),
                    rank=i + 1
                )
                for i, row in feature_importance.head(10).iterrows()
            ]
        )
        
    except ModelNotTrainedError:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise


@app.post(
    "/api/model/train",
    tags=["Model"],
    summary="Train or retrain the model"
)
async def train_model(
    force: bool = Query(False, description="Force retraining even if model exists"),
    predictor = Depends(get_predictor)
):
    """Train or retrain the CLV prediction model."""
    try:
        start_time = time.time()
        
        if predictor.is_trained and not force:
            return {
                "status": "skipped",
                "message": "Model already trained. Use force=true to retrain."
            }
        
        predictor.train()
        
        duration = time.time() - start_time
        metrics = predictor.get_model_metrics()
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "training_time_seconds": round(duration, 2),
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


# =============================================================================
# Meta Ads Endpoints
# =============================================================================

@app.get(
    "/api/meta-ads/audiences",
    tags=["Meta Ads"],
    summary="Get audience segments for Meta Ads"
)
async def get_audiences(
    meta_ads = Depends(get_meta_ads),
    predictor = Depends(get_predictor)
):
    """Get audience segments optimized for Meta Ads targeting."""
    try:
        if not predictor.is_trained:
            predictor.train()
        
        predictions = predictor.get_all_predictions()
        meta_ads.predictions_df = predictions
        
        audiences = meta_ads.generate_audience_segments()
        
        return {
            "audiences": audiences,
            "total_customers": len(predictions),
            "recommendations": [
                "Use High-CLV segment for value-optimized campaigns",
                "Target Growth-Potential with upsell offers",
                "Apply cost caps to Low-CLV acquisition"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error generating audiences: {e}")
        raise


@app.get(
    "/api/meta-ads/budget-allocation",
    response_model=BudgetAllocationResponse,
    tags=["Meta Ads"],
    summary="Get budget allocation recommendations"
)
async def get_budget_allocation(
    total_budget: float = Query(..., gt=0, le=10000000, description="Total marketing budget"),
    meta_ads = Depends(get_meta_ads),
    predictor = Depends(get_predictor)
):
    """Get recommended budget allocation across segments."""
    try:
        if not predictor.is_trained:
            predictor.train()
        
        predictions = predictor.get_all_predictions()
        meta_ads.predictions_df = predictions
        
        allocation = meta_ads.generate_budget_allocation(total_budget)
        
        return BudgetAllocationResponse(
            total_budget=total_budget,
            allocation={
                seg: BudgetAllocation(
                    budget=round(data['budget'], 2),
                    budget_percentage=data['budget_percentage'],
                    expected_acquisitions=data['expected_acquisitions'],
                    expected_roas=data['expected_roas'],
                    optimal_cac=round(data.get('optimal_cac', data['budget'] / max(data['expected_acquisitions'], 1)), 2)
                )
                for seg, data in allocation['allocation'].items()
            },
            summary=allocation.get('summary', {})
        )
        
    except Exception as e:
        logger.error(f"Error calculating budget allocation: {e}")
        raise


# =============================================================================
# Dashboard Endpoints
# =============================================================================

@app.get(
    "/api/dashboard/summary",
    response_model=DashboardResponse,
    tags=["Dashboard"],
    summary="Get dashboard summary data"
)
async def get_dashboard_summary(
    predictor = Depends(get_predictor),
    cache: CacheManager = Depends(get_cache_manager)
):
    """Get aggregated dashboard data."""
    # Check cache
    cached = cache.get("dashboard:summary")
    if cached:
        return DashboardResponse(**cached)
    
    try:
        if not predictor.is_trained:
            predictor.train()
        
        df = predictor.get_all_predictions()
        
        clv_col = 'predicted_clv' if 'predicted_clv' in df.columns else 'actual_clv'
        seg_col = 'predicted_segment' if 'predicted_segment' in df.columns else 'customer_segment'
        
        # Summary stats
        total_customers = len(df)
        avg_clv = df[clv_col].mean() if clv_col in df.columns else 0
        total_value = df[clv_col].sum() if clv_col in df.columns else 0
        
        high_clv_count = len(df[df[seg_col] == 'High-CLV']) if seg_col in df.columns else 0
        high_clv_pct = (high_clv_count / total_customers * 100) if total_customers > 0 else 0
        
        # Segment distribution
        segments = df[seg_col].value_counts().to_dict() if seg_col in df.columns else {}
        
        # Acquisition sources
        sources = df['acquisition_source'].value_counts().to_dict() if 'acquisition_source' in df.columns else {}
        
        # CLV distribution
        if clv_col in df.columns:
            clv_dist = {
                "$0-100": len(df[df[clv_col] < 100]),
                "$100-250": len(df[(df[clv_col] >= 100) & (df[clv_col] < 250)]),
                "$250-500": len(df[(df[clv_col] >= 250) & (df[clv_col] < 500)]),
                "$500-1000": len(df[(df[clv_col] >= 500) & (df[clv_col] < 1000)]),
                "$1000+": len(df[df[clv_col] >= 1000])
            }
        else:
            clv_dist = {}
        
        # Top customers - convert timestamps to strings for JSON serialization
        if clv_col in df.columns:
            top_df = df.nlargest(5, clv_col).copy()
            # Convert all timestamp columns to strings
            for col in top_df.columns:
                if pd.api.types.is_datetime64_any_dtype(top_df[col]):
                    top_df[col] = top_df[col].astype(str)
            # Replace NaN values with None
            top_df = top_df.where(pd.notnull(top_df), None)
            top_customers = top_df.to_dict('records')
            # Clean up any remaining nan strings and enum fields
            for customer in top_customers:
                for key, value in customer.items():
                    if pd.isna(value) or value == 'nan' or value == 'NaT':
                        customer[key] = None
                    # Convert None to "None" string for enum fields
                    if key in ['campaign_type', 'acquisition_source'] and customer[key] is None:
                        customer[key] = "None"
        else:
            top_customers = []
        
        result = {
            "summary": {
                "total_customers": total_customers,
                "avg_clv": round(avg_clv, 2),
                "total_value": round(total_value, 2),
                "high_clv_percentage": round(high_clv_pct, 1)
            },
            "segments": segments,
            "acquisition_sources": sources,
            "clv_distribution": clv_dist,
            "top_customers": top_customers
        }
        
        # Cache for 5 minutes
        cache.set("dashboard:summary", result, ttl=300)
        
        return DashboardResponse(**result)
        
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise


# =============================================================================
# Cache Management Endpoints
# =============================================================================

@app.get(
    "/api/cache/stats",
    response_model=CacheStats,
    tags=["Admin"],
    summary="Get cache statistics"
)
async def get_cache_stats(
    cache: CacheManager = Depends(get_cache_manager)
):
    """Get current cache statistics."""
    stats = cache.get_stats()
    return CacheStats(**stats)


@app.post(
    "/api/cache/clear",
    tags=["Admin"],
    summary="Clear the cache"
)
async def clear_cache(
    cache: CacheManager = Depends(get_cache_manager)
):
    """Clear all cached data."""
    cache.clear()
    return {"status": "success", "message": "Cache cleared"}


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(CLVException)
async def clv_exception_handler(request: Request, exc: CLVException):
    """Handle CLV-specific exceptions."""
    return JSONResponse(
        status_code=getattr(exc, 'status_code', 500),
        content=exc.to_dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "error_code": exc.status_code * 10,
            "message": exc.detail
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    config = get_config()
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_code": 5000,
            "message": str(exc) if config.debug else "Internal server error"
        }
    )


# Export app for uvicorn
if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(
        "backend.api_enhanced:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.debug,
        workers=config.api.workers if not config.debug else 1
    )
