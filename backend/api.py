"""
FastAPI REST API for CLV Prediction System
Provides endpoints for predictions, segmentation, and Meta Ads integration.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data_processor import DataProcessor
from backend.feature_engineering import FeatureEngineer
from backend.clv_predictor import CLVPredictor
from backend.meta_ads_integration import MetaAdsIntegration

# Initialize FastAPI app
app = FastAPI(
    title="CLV Prediction API",
    description="Machine Learning API for Customer Lifetime Value Prediction",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
DATA_PATH = str(Path(__file__).parent.parent / "data" / "customers.csv")
predictor: Optional[CLVPredictor] = None
customers_df: Optional[pd.DataFrame] = None
meta_integration: Optional[MetaAdsIntegration] = None


# Pydantic models
class CustomerPredictionRequest(BaseModel):
    """Request model for single customer prediction."""
    total_orders: int = Field(ge=1, description="Number of orders")
    total_spent: float = Field(ge=0, description="Total amount spent")
    avg_order_value: float = Field(ge=0, description="Average order value")
    days_since_first_purchase: int = Field(ge=0, description="Days since first purchase")
    days_since_last_purchase: int = Field(ge=0, description="Days since last purchase")
    num_categories: int = Field(ge=1, description="Number of product categories")
    acquisition_source: str = Field(description="Acquisition source")
    campaign_type: str = Field(default="None", description="Campaign type")
    acquisition_cost: float = Field(ge=0, description="Customer acquisition cost")
    email_engagement_rate: float = Field(ge=0, le=1, description="Email engagement rate")
    return_rate: float = Field(ge=0, le=1, description="Product return rate")


class PredictionResponse(BaseModel):
    """Response model for CLV prediction."""
    predicted_clv: float
    segment: str
    confidence: str
    recommended_cac: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_trained: bool
    data_loaded: bool
    timestamp: str


class MetricsResponse(BaseModel):
    """Model metrics response."""
    ensemble: Dict[str, float]
    random_forest: Dict[str, float]
    gradient_boosting: Dict[str, float]
    linear: Dict[str, float]


# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize predictor and load data on startup."""
    global predictor, customers_df, meta_integration
    
    try:
        # Initialize predictor
        predictor = CLVPredictor(
            model_dir=str(Path(__file__).parent.parent / "models"),
            data_path=DATA_PATH
        )
        
        # Train if data exists
        if Path(DATA_PATH).exists():
            print(f"Loading data from {DATA_PATH}")
            predictor.train(DATA_PATH, save_models=True)
            
            # Load data for serving
            processor = DataProcessor(DATA_PATH)
            customers_df = processor.load_data()
            customers_df = processor.clean_data()
            
            # Initialize Meta integration
            meta_integration = MetaAdsIntegration()
            
            print("CLV Predictor initialized successfully!")
        else:
            print(f"Warning: Data file not found at {DATA_PATH}")
            
    except Exception as e:
        print(f"Error during startup: {e}")


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "CLV Prediction API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        model_trained=predictor is not None and predictor.is_trained,
        data_loaded=customers_df is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/customers", tags=["Customers"])
async def get_customers(
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    segment: Optional[str] = None
):
    """Get list of customers with predictions."""
    global customers_df, predictor
    
    if customers_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = customers_df.copy()
    
    # Add predictions
    if predictor is not None and predictor.is_trained:
        df = predictor.predict_batch(df)
    
    # Filter by segment if provided
    if segment:
        segment_col = 'predicted_segment' if 'predicted_segment' in df.columns else 'customer_segment'
        df = df[df[segment_col] == segment]
    
    # Paginate
    total = len(df)
    df = df.iloc[offset:offset + limit]
    
    # Convert to records
    records = df.to_dict('records')
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "customers": records
    }


@app.get("/api/customers/{customer_id}", tags=["Customers"])
async def get_customer(customer_id: str):
    """Get single customer details with prediction."""
    global customers_df, predictor
    
    if customers_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    customer = customers_df[customers_df['customer_id'] == customer_id]
    
    if customer.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    # Get prediction
    if predictor is not None and predictor.is_trained:
        customer = predictor.predict_batch(customer)
    
    return customer.iloc[0].to_dict()


@app.post("/api/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_clv(request: CustomerPredictionRequest):
    """Predict CLV for a new customer."""
    global predictor
    
    if predictor is None or not predictor.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    try:
        customer_data = request.dict()
        result = predictor.predict_single(customer_data)
        
        return PredictionResponse(
            predicted_clv=result['predicted_clv'],
            segment=result['segment'],
            confidence=result['confidence'],
            recommended_cac=result['recommended_cac']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/segments", tags=["Segments"])
async def get_segments():
    """Get customer segment distribution and statistics."""
    global customers_df, predictor
    
    if customers_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = customers_df.copy()
    
    # Add predictions
    if predictor is not None and predictor.is_trained:
        df = predictor.predict_batch(df)
        summary = predictor.get_segment_summary(df)
    else:
        # Use existing segments
        summary = {}
        for segment in ['High-CLV', 'Growth-Potential', 'Low-CLV']:
            segment_df = df[df['customer_segment'] == segment]
            summary[segment] = {
                'count': len(segment_df),
                'percentage': round(len(segment_df) / len(df) * 100, 2),
                'avg_clv': round(segment_df['actual_clv'].mean(), 2) if 'actual_clv' in segment_df.columns else 0,
                'total_value': round(segment_df['actual_clv'].sum(), 2) if 'actual_clv' in segment_df.columns else 0
            }
    
    return {
        "segments": summary,
        "total_customers": len(df)
    }


@app.get("/api/metrics", tags=["Metrics"])
async def get_metrics():
    """Get model performance metrics."""
    global predictor
    
    if predictor is None or not predictor.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    metrics = predictor.get_model_metrics()
    
    return {
        "metrics": metrics,
        "best_model": "ensemble",
        "feature_importance": predictor.get_feature_importance()[:10]
    }


@app.get("/api/meta-ads/audiences", tags=["Meta Ads"])
async def get_meta_audiences():
    """Get audience segments for Meta Ads targeting."""
    global customers_df, predictor, meta_integration
    
    if customers_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = customers_df.copy()
    
    # Add predictions
    if predictor is not None and predictor.is_trained:
        df = predictor.predict_batch(df)
    
    # Generate audiences
    if meta_integration is None:
        meta_integration = MetaAdsIntegration()
    
    meta_integration.set_predictions(df)
    audiences = meta_integration.generate_audience_segments(df)
    
    result = {}
    for segment, audience_df in audiences.items():
        result[segment] = {
            'count': len(audience_df),
            'avg_predicted_clv': round(audience_df['predicted_clv'].mean(), 2) if 'predicted_clv' in audience_df.columns else 0,
            'targeting_priority': audience_df['targeting_priority'].iloc[0] if len(audience_df) > 0 else 'Unknown',
            'lookalike_recommendation': audience_df['lookalike_recommendation'].iloc[0] if len(audience_df) > 0 else 'Unknown',
            'bidding_strategy': audience_df['bidding_strategy'].iloc[0] if len(audience_df) > 0 else 'Unknown'
        }
    
    return {
        "audiences": result,
        "lookalike_recommendations": meta_integration.create_lookalike_recommendations()
    }


@app.get("/api/meta-ads/budget-allocation", tags=["Meta Ads"])
async def get_budget_allocation(
    total_budget: float = Query(default=10000, description="Total advertising budget")
):
    """Get budget allocation recommendations."""
    global customers_df, predictor, meta_integration
    
    if customers_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = customers_df.copy()
    
    # Add predictions
    if predictor is not None and predictor.is_trained:
        df = predictor.predict_batch(df)
    
    # Generate allocation
    if meta_integration is None:
        meta_integration = MetaAdsIntegration()
    
    allocation = meta_integration.generate_budget_allocation(total_budget, df)
    
    return allocation


@app.get("/api/meta-ads/strategy", tags=["Meta Ads"])
async def get_full_strategy(
    total_budget: float = Query(default=10000, description="Total advertising budget")
):
    """Get complete Meta Ads strategy with all recommendations."""
    global customers_df, predictor, meta_integration
    
    if customers_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = customers_df.copy()
    
    # Add predictions
    if predictor is not None and predictor.is_trained:
        df = predictor.predict_batch(df)
    
    # Generate full strategy
    if meta_integration is None:
        meta_integration = MetaAdsIntegration()
    
    strategy = meta_integration.get_full_strategy(total_budget, df)
    
    return strategy


@app.get("/api/dashboard/summary", tags=["Dashboard"])
async def get_dashboard_summary():
    """Get summary data for dashboard."""
    global customers_df, predictor
    
    if customers_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = customers_df.copy()
    
    # Add predictions if model is trained
    if predictor is not None and predictor.is_trained:
        df = predictor.predict_batch(df)
        clv_col = 'predicted_clv'
    else:
        clv_col = 'actual_clv'
    
    # Calculate summary statistics
    total_customers = len(df)
    avg_clv = round(df[clv_col].mean(), 2)
    total_value = round(df[clv_col].sum(), 2)
    
    # Segment distribution
    segment_col = 'predicted_segment' if 'predicted_segment' in df.columns else 'customer_segment'
    segment_counts = df[segment_col].value_counts().to_dict()
    
    # Acquisition source distribution
    source_counts = df['acquisition_source'].value_counts().to_dict()
    
    # CLV distribution
    clv_bins = [0, 100, 250, 500, 1000, float('inf')]
    clv_labels = ['$0-100', '$100-250', '$250-500', '$500-1000', '$1000+']
    df['clv_range'] = pd.cut(df[clv_col], bins=clv_bins, labels=clv_labels)
    clv_distribution = df['clv_range'].value_counts().to_dict()
    
    # Top customers
    top_customers = df.nlargest(10, clv_col)[['customer_id', clv_col, segment_col, 'total_orders', 'acquisition_source']].to_dict('records')
    
    return {
        "summary": {
            "total_customers": total_customers,
            "avg_clv": avg_clv,
            "total_value": total_value,
            "high_clv_percentage": round(segment_counts.get('High-CLV', 0) / total_customers * 100, 1)
        },
        "segments": segment_counts,
        "acquisition_sources": source_counts,
        "clv_distribution": clv_distribution,
        "top_customers": top_customers
    }


# Run with: uvicorn backend.api:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
