"""
Pydantic Schemas for CLV Prediction System
Data validation and serialization models.
"""

import re
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SegmentType(str, Enum):
    """Customer segment types."""

    HIGH_CLV = "High-CLV"
    GROWTH_POTENTIAL = "Growth-Potential"
    LOW_CLV = "Low-CLV"


class AcquisitionSource(str, Enum):
    """Customer acquisition sources."""

    META_ADS = "Meta Ads"
    GOOGLE_ADS = "Google Ads"
    EMAIL = "Email"
    DIRECT = "Direct"
    ORGANIC = "Organic"
    REFERRAL = "Referral"


class CampaignType(str, Enum):
    """Marketing campaign types."""

    PROSPECTING = "Prospecting"
    RETARGETING = "Retargeting"
    BRAND = "Brand"
    NONE = "None"


class ConfidenceLevel(str, Enum):
    """Prediction confidence levels."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


# Base Models
class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    model_config = ConfigDict(
        from_attributes=True, populate_by_name=True, str_strip_whitespace=True
    )


# Customer Schemas
class CustomerBase(BaseSchema):
    """Base customer data."""

    customer_id: str = Field(..., min_length=1, max_length=50)

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, v: str) -> str:
        if not re.match(r"^[A-Za-z0-9_-]+$", v):
            raise ValueError("Customer ID must be alphanumeric with underscores/dashes only")
        return v


class CustomerFeatures(BaseSchema):
    """Customer features for prediction."""

    total_orders: int = Field(..., ge=0, le=10000, description="Total number of orders")
    total_spent: float = Field(..., ge=0, le=10000000, description="Total amount spent")
    avg_order_value: float = Field(..., ge=0, le=100000, description="Average order value")
    days_since_first_purchase: int = Field(
        ..., ge=0, le=3650, description="Days since first purchase"
    )
    days_since_last_purchase: int = Field(
        ..., ge=0, le=3650, description="Days since last purchase"
    )
    num_categories: int = Field(default=1, ge=1, le=20, description="Number of product categories")
    acquisition_source: AcquisitionSource = Field(default=AcquisitionSource.DIRECT)
    campaign_type: CampaignType = Field(default=CampaignType.NONE)
    acquisition_cost: float = Field(
        default=0, ge=0, le=10000, description="Customer acquisition cost"
    )
    email_engagement_rate: float = Field(
        default=0.0, ge=0, le=1, description="Email engagement rate (0-1)"
    )
    return_rate: float = Field(default=0.0, ge=0, le=1, description="Product return rate (0-1)")

    @field_validator("avg_order_value", mode="before")
    @classmethod
    def calculate_avg_if_zero(cls, v, info):
        if v == 0 and "total_spent" in info.data and "total_orders" in info.data:
            orders = info.data["total_orders"]
            if orders > 0:
                return info.data["total_spent"] / orders
        return v

    @model_validator(mode="after")
    def validate_date_consistency(self):
        if self.days_since_last_purchase > self.days_since_first_purchase:
            raise ValueError(
                "days_since_last_purchase cannot be greater than days_since_first_purchase"
            )
        return self


class CustomerCreate(CustomerBase, CustomerFeatures):
    """Schema for creating a new customer."""

    first_purchase_date: Optional[date] = None
    last_purchase_date: Optional[date] = None


class CustomerResponse(CustomerBase, CustomerFeatures):
    """Schema for customer response."""

    first_purchase_date: Optional[str] = None
    last_purchase_date: Optional[str] = None
    predicted_clv: Optional[float] = None
    predicted_segment: Optional[SegmentType] = None
    customer_segment: Optional[str] = None
    actual_clv: Optional[float] = None


class CustomerListResponse(BaseSchema):
    """Response for paginated customer list."""

    customers: List[CustomerResponse]
    total: int
    page: int = 1
    page_size: int = 100
    has_more: bool = False


# Prediction Schemas
class PredictionRequest(CustomerFeatures):
    """Request schema for CLV prediction."""

    pass


class PredictionResponse(BaseSchema):
    """Response schema for CLV prediction."""

    predicted_clv: float = Field(..., ge=0)
    segment: SegmentType
    confidence: ConfidenceLevel
    recommended_cac: float = Field(..., ge=0)
    feature_contributions: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    cached: bool = False

    @field_validator("recommended_cac")
    @classmethod
    def validate_cac(cls, v, info):
        if "predicted_clv" in info.data:
            max_cac = info.data["predicted_clv"] * 0.5
            if v > max_cac:
                return max_cac
        return v


class BatchPredictionRequest(BaseSchema):
    """Request for batch predictions."""

    customers: List[PredictionRequest] = Field(..., min_length=1, max_length=1000)
    include_features: bool = False


class BatchPredictionResponse(BaseSchema):
    """Response for batch predictions."""

    predictions: List[PredictionResponse]
    total: int
    processing_time_ms: float


# Segment Schemas
class SegmentStats(BaseSchema):
    """Statistics for a customer segment."""

    count: int
    avg_clv: float = Field(alias="avg_predicted_clv")
    total_value: float = Field(alias="total_predicted_value")
    percentage: float = 0.0
    avg_orders: Optional[float] = None
    avg_tenure_days: Optional[float] = None


class SegmentAnalysis(BaseSchema):
    """Complete segment analysis response."""

    segments: Dict[str, SegmentStats]
    total_customers: int
    total_value: float


# Model Metrics Schemas
class ModelMetrics(BaseSchema):
    """Model performance metrics."""

    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Square Error")
    r2: float = Field(..., ge=-1, le=1, description="R-squared score")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error")


class FeatureImportance(BaseSchema):
    """Feature importance entry."""

    feature: str
    importance: float = Field(..., ge=0, le=1)
    rank: Optional[int] = None


class ModelPerformance(BaseSchema):
    """Complete model performance response."""

    model_name: str
    version: str
    trained_at: Optional[datetime] = None
    samples_trained: int
    metrics: ModelMetrics
    feature_importance: List[FeatureImportance]


# Meta Ads Schemas
class AudienceSegment(BaseSchema):
    """Meta Ads audience segment."""

    segment_name: str
    customer_count: int
    avg_clv: float
    total_value: float
    lookalike_recommendation: str
    targeting_strategy: str


class BudgetAllocation(BaseSchema):
    """Budget allocation for a segment."""

    budget: float = Field(..., ge=0)
    budget_percentage: float = Field(..., ge=0, le=100)
    expected_acquisitions: int = Field(..., ge=0)
    expected_roas: float = Field(..., ge=0)
    optimal_cac: float = Field(..., ge=0)


class BudgetAllocationRequest(BaseSchema):
    """Request for budget allocation."""

    total_budget: float = Field(..., gt=0, le=10000000)
    custom_weights: Optional[Dict[str, float]] = None


class BudgetAllocationResponse(BaseSchema):
    """Response for budget allocation."""

    total_budget: float
    allocation: Dict[str, BudgetAllocation]
    summary: Dict[str, Any]


class MetaAdsStrategy(BaseSchema):
    """Complete Meta Ads strategy."""

    audiences: List[AudienceSegment]
    budget_allocation: BudgetAllocationResponse
    recommendations: List[str]


# Health & Status Schemas
class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseSchema):
    """Health status of a component."""

    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None


class HealthCheck(BaseSchema):
    """Complete health check response."""

    status: HealthStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    uptime_seconds: float
    components: List[ComponentHealth] = []


class CacheStats(BaseSchema):
    """Cache statistics."""

    enabled: bool
    type: Optional[str] = None
    size: Optional[int] = None
    hits: Optional[int] = None
    misses: Optional[int] = None
    hit_rate: Optional[float] = None


# Dashboard Schemas
class DashboardSummary(BaseSchema):
    """Dashboard summary statistics."""

    total_customers: int
    avg_clv: float
    total_value: float
    high_clv_percentage: float


class DashboardResponse(BaseSchema):
    """Complete dashboard data response."""

    summary: DashboardSummary
    segments: Dict[str, int]
    acquisition_sources: Dict[str, int]
    clv_distribution: Dict[str, int]
    top_customers: List[CustomerResponse]


# Error Schemas
class ErrorDetail(BaseSchema):
    """Error detail."""

    field: Optional[str] = None
    message: str
    code: Optional[str] = None


class ErrorResponse(BaseSchema):
    """Standard error response."""

    error: bool = True
    error_code: int
    error_name: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    request_id: Optional[str] = None


# Pagination
class PaginationParams(BaseSchema):
    """Pagination parameters."""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=100, ge=1, le=1000)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


# API Key (for future auth)
class APIKey(BaseSchema):
    """API key information."""

    key: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    scopes: List[str] = []
    is_active: bool = True
