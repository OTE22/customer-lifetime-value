"""
Advanced Feature Engineering Module for CLV Prediction System
Production-ready feature engineering with RFM, behavioral, temporal, 
statistical, and interaction features.

Author: Ali Abbass (OTE22)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats

from .logging_config import get_logger
from .config import get_config

logger = get_logger(__name__)


class FeatureCategory(Enum):
    """Categories of features for organization."""
    RFM = "rfm"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    STATISTICAL = "statistical"
    ACQUISITION = "acquisition"
    INTERACTION = "interaction"
    DERIVED = "derived"


@dataclass
class FeatureMetadata:
    """Metadata for an engineered feature."""
    name: str
    category: FeatureCategory
    description: str
    importance: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class AdvancedFeatureEngineer:
    """
    Production-ready feature engineering for CLV prediction.
    
    Implements:
    - RFM (Recency, Frequency, Monetary) scoring with multiple methods
    - Behavioral features (purchase patterns, engagement)
    - Temporal features (seasonality, trends, day-of-week)
    - Statistical features (z-scores, percentiles, rolling stats)
    - Interaction features (feature combinations)
    - Acquisition features (channel, campaign effectiveness)
    """
    
    def __init__(
        self,
        reference_date: Optional[datetime] = None,
        n_rfm_bins: int = 5,
        enable_advanced_stats: bool = True
    ):
        """
        Initialize the Advanced Feature Engineer.
        
        Args:
            reference_date: Reference date for recency calculations (default: today)
            n_rfm_bins: Number of bins for RFM scoring (default: 5)
            enable_advanced_stats: Whether to compute advanced statistical features
        """
        self.reference_date = reference_date or datetime.now()
        self.n_rfm_bins = n_rfm_bins
        self.enable_advanced_stats = enable_advanced_stats
        
        # Feature tracking
        self._feature_metadata: Dict[str, FeatureMetadata] = {}
        self._feature_columns: List[str] = []
        self._original_columns: List[str] = []
        
        # Feature statistics for normalization
        self._feature_stats: Dict[str, Dict[str, float]] = {}
        
    def _register_feature(
        self,
        name: str,
        category: FeatureCategory,
        description: str
    ) -> None:
        """Register a new feature with metadata."""
        self._feature_metadata[name] = FeatureMetadata(
            name=name,
            category=category,
            description=description
        )
        if name not in self._feature_columns:
            self._feature_columns.append(name)
    
    # =========================================================================
    # RFM Features
    # =========================================================================
    
    def calculate_rfm_scores(
        self,
        df: pd.DataFrame,
        method: str = "quantile"
    ) -> pd.DataFrame:
        """
        Calculate RFM scores with multiple methods.
        
        Args:
            df: Customer DataFrame
            method: Scoring method ('quantile', 'kmeans', 'fixed')
            
        Returns:
            DataFrame with RFM features
        """
        df = df.copy()
        logger.info(f"Calculating RFM scores using {method} method...")
        
        n = self.n_rfm_bins
        
        # For small datasets (single customer prediction), use fixed bins
        use_fixed_bins = len(df) < n or method == "fixed"
        
        # === RECENCY SCORE ===
        if 'days_since_last_purchase' in df.columns:
            if use_fixed_bins:
                # Fixed bins based on business logic
                df['recency_score'] = pd.cut(
                    df['days_since_last_purchase'],
                    bins=[-1, 7, 30, 90, 180, float('inf')],
                    labels=[5, 4, 3, 2, 1]
                ).astype(float).fillna(3)
            else:
                try:
                    df['recency_score'] = pd.qcut(
                        df['days_since_last_purchase'].rank(method='first'),
                        q=n, labels=range(n, 0, -1), duplicates='drop'
                    ).astype(float)
                except ValueError:
                    # Fallback to fixed bins if qcut fails
                    df['recency_score'] = pd.cut(
                        df['days_since_last_purchase'],
                        bins=[-1, 7, 30, 90, 180, float('inf')],
                        labels=[5, 4, 3, 2, 1]
                    ).astype(float).fillna(3)
            
            # Normalized recency (0-1, lower is better)
            max_recency = df['days_since_last_purchase'].max()
            if max_recency > 0:
                df['recency_normalized'] = 1 - (df['days_since_last_purchase'] / max_recency)
            else:
                df['recency_normalized'] = 1.0
            
            self._register_feature('recency_score', FeatureCategory.RFM, 'RFM recency score (1-5)')
            self._register_feature('recency_normalized', FeatureCategory.RFM, 'Normalized recency (0-1)')
        
        # === FREQUENCY SCORE ===
        if 'total_orders' in df.columns:
            if use_fixed_bins:
                df['frequency_score'] = pd.cut(
                    df['total_orders'],
                    bins=[-1, 1, 3, 6, 12, float('inf')],
                    labels=[1, 2, 3, 4, 5]
                ).astype(float).fillna(3)
            else:
                try:
                    df['frequency_score'] = pd.qcut(
                        df['total_orders'].rank(method='first'),
                        q=n, labels=range(1, n + 1), duplicates='drop'
                    ).astype(float)
                except ValueError:
                    df['frequency_score'] = pd.cut(
                        df['total_orders'],
                        bins=[-1, 1, 3, 6, 12, float('inf')],
                        labels=[1, 2, 3, 4, 5]
                    ).astype(float).fillna(3)
            
            # Log-transformed frequency (handles skewness)
            df['frequency_log'] = np.log1p(df['total_orders'])
            
            # Normalized frequency
            max_freq = df['total_orders'].max()
            if max_freq > 0:
                df['frequency_normalized'] = df['total_orders'] / max_freq
            else:
                df['frequency_normalized'] = 1.0
            
            self._register_feature('frequency_score', FeatureCategory.RFM, 'RFM frequency score (1-5)')
            self._register_feature('frequency_log', FeatureCategory.RFM, 'Log-transformed order count')
            self._register_feature('frequency_normalized', FeatureCategory.RFM, 'Normalized frequency (0-1)')
        
        # === MONETARY SCORE ===
        if 'total_spent' in df.columns:
            if use_fixed_bins:
                df['monetary_score'] = pd.cut(
                    df['total_spent'],
                    bins=[-1, 50, 150, 400, 1000, float('inf')],
                    labels=[1, 2, 3, 4, 5]
                ).astype(float).fillna(3)
            else:
                try:
                    df['monetary_score'] = pd.qcut(
                        df['total_spent'].rank(method='first'),
                        q=n, labels=range(1, n + 1), duplicates='drop'
                    ).astype(float)
                except ValueError:
                    df['monetary_score'] = pd.cut(
                        df['total_spent'],
                        bins=[-1, 50, 150, 400, 1000, float('inf')],
                        labels=[1, 2, 3, 4, 5]
                    ).astype(float).fillna(3)
            
            # Log-transformed monetary (handles skewness)
            df['monetary_log'] = np.log1p(df['total_spent'])
            
            # Normalized monetary
            max_monetary = df['total_spent'].max()
            if max_monetary > 0:
                df['monetary_normalized'] = df['total_spent'] / max_monetary
            else:
                df['monetary_normalized'] = 1.0
            
            self._register_feature('monetary_score', FeatureCategory.RFM, 'RFM monetary score (1-5)')
            self._register_feature('monetary_log', FeatureCategory.RFM, 'Log-transformed total spent')
            self._register_feature('monetary_normalized', FeatureCategory.RFM, 'Normalized monetary (0-1)')
        
        # === COMPOSITE RFM SCORES ===
        required_scores = ['recency_score', 'frequency_score', 'monetary_score']
        if all(col in df.columns for col in required_scores):
            # Simple sum
            df['rfm_score'] = df['recency_score'] + df['frequency_score'] + df['monetary_score']
            
            # Weighted RFM (emphasize monetary and frequency for CLV)
            df['rfm_weighted'] = (
                df['recency_score'] * 0.2 +
                df['frequency_score'] * 0.35 +
                df['monetary_score'] * 0.45
            )
            
            # RFM string for segmentation (e.g., "555", "311")
            df['rfm_string'] = (
                df['recency_score'].astype(int).astype(str) +
                df['frequency_score'].astype(int).astype(str) +
                df['monetary_score'].astype(int).astype(str)
            )
            
            # RFM segment
            df['rfm_segment'] = df.apply(self._assign_rfm_segment, axis=1)
            
            self._register_feature('rfm_score', FeatureCategory.RFM, 'Sum of RFM scores')
            self._register_feature('rfm_weighted', FeatureCategory.RFM, 'Weighted RFM (CLV-optimized)')
        
        logger.info("RFM features calculated")
        return df
    
    def _assign_rfm_segment(self, row: pd.Series) -> str:
        """Assign detailed RFM segment based on scores."""
        r = row.get('recency_score', 3)
        f = row.get('frequency_score', 3)
        m = row.get('monetary_score', 3)
        
        # Champions: Best customers
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        # Loyal: High frequency and monetary, good recency
        elif f >= 4 and m >= 4:
            return 'Loyal Customers'
        # Potential Loyalists: Recent with good frequency
        elif r >= 4 and f >= 3:
            return 'Potential Loyalists'
        # Recent: New customers with few orders
        elif r >= 4 and f <= 2:
            return 'Recent Customers'
        # Promising: Recent, low frequency but potential
        elif r >= 3 and f >= 2 and m >= 2:
            return 'Promising'
        # Need Attention: Above average but declining
        elif r >= 2 and f >= 3 and m >= 3:
            return 'Need Attention'
        # About to Sleep: Below average recency
        elif r <= 2 and f >= 2:
            return 'About to Sleep'
        # At Risk: Haven't purchased recently
        elif r <= 2 and f >= 4:
            return 'At Risk'
        # Cannot Lose: Was loyal, haven't seen recently
        elif r <= 2 and f >= 4 and m >= 4:
            return 'Cannot Lose Them'
        # Hibernating: Low recency and frequency
        elif r <= 2 and f <= 2:
            return 'Hibernating'
        # Lost: Very low scores
        elif r <= 1 and f <= 1:
            return 'Lost'
        else:
            return 'Others'
    
    # =========================================================================
    # Behavioral Features
    # =========================================================================
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive behavioral features.
        
        Features include:
        - Purchase velocity and patterns
        - Engagement metrics
        - Customer lifecycle indicators
        - Product diversity metrics
        """
        df = df.copy()
        logger.info("Creating behavioral features...")
        
        # === PURCHASE VELOCITY FEATURES ===
        if 'total_orders' in df.columns and 'days_since_first_purchase' in df.columns:
            # Orders per month (purchase velocity)
            months_active = (df['days_since_first_purchase'] / 30.44).clip(lower=0.5)
            df['purchase_velocity'] = df['total_orders'] / months_active
            
            # Orders per week
            weeks_active = (df['days_since_first_purchase'] / 7).clip(lower=1)
            df['orders_per_week'] = df['total_orders'] / weeks_active
            
            # Intensity: orders in recent period vs lifetime
            if 'days_since_last_purchase' in df.columns:
                recent_activity = df['days_since_last_purchase']
                tenure = df['days_since_first_purchase']
                df['recency_ratio'] = np.where(
                    tenure > 0,
                    1 - (recent_activity / tenure),
                    0
                )
            
            self._register_feature('purchase_velocity', FeatureCategory.BEHAVIORAL, 'Orders per month')
            self._register_feature('orders_per_week', FeatureCategory.BEHAVIORAL, 'Orders per week')
            self._register_feature('recency_ratio', FeatureCategory.BEHAVIORAL, 'Activity recency ratio')
        
        # === INTER-PURCHASE TIMING ===
        if 'total_orders' in df.columns and 'days_since_first_purchase' in df.columns:
            df['avg_days_between_purchases'] = np.where(
                df['total_orders'] > 1,
                df['days_since_first_purchase'] / (df['total_orders'] - 1),
                df['days_since_first_purchase']
            )
            
            # Purchase regularity (coefficient of variation - simulated)
            # Lower = more regular purchasing
            df['purchase_regularity'] = np.where(
                df['avg_days_between_purchases'] > 0,
                1 / (1 + df['avg_days_between_purchases'] / 30),
                0
            )
            
            self._register_feature('avg_days_between_purchases', FeatureCategory.BEHAVIORAL, 'Average days between orders')
            self._register_feature('purchase_regularity', FeatureCategory.BEHAVIORAL, 'Purchase regularity score')
        
        # === SPENDING PATTERNS ===
        if 'total_spent' in df.columns and 'total_orders' in df.columns:
            # Spending per month
            if 'days_since_first_purchase' in df.columns:
                months_active = (df['days_since_first_purchase'] / 30.44).clip(lower=0.5)
                df['spending_velocity'] = df['total_spent'] / months_active
                self._register_feature('spending_velocity', FeatureCategory.BEHAVIORAL, 'Spending per month')
            
            # AOV consistency (variance indicator)
            if 'avg_order_value' in df.columns:
                median_aov = df['avg_order_value'].median()
                df['aov_deviation'] = abs(df['avg_order_value'] - median_aov) / median_aov
                self._register_feature('aov_deviation', FeatureCategory.BEHAVIORAL, 'AOV deviation from median')
        
        # === ENGAGEMENT FEATURES ===
        if 'email_engagement_rate' in df.columns:
            # Email engagement buckets
            df['email_engagement_bucket'] = pd.cut(
                df['email_engagement_rate'],
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=[1, 2, 3, 4, 5]
            ).astype(float)
            
            self._register_feature('email_engagement_bucket', FeatureCategory.BEHAVIORAL, 'Email engagement tier')
        
        if 'return_rate' in df.columns:
            # Return behavior indicator
            df['is_returner'] = (df['return_rate'] > 0.1).astype(int)
            df['return_risk'] = np.where(
                df['return_rate'] > 0.3, 2,
                np.where(df['return_rate'] > 0.15, 1, 0)
            )
            
            self._register_feature('is_returner', FeatureCategory.BEHAVIORAL, 'Has high return rate')
            self._register_feature('return_risk', FeatureCategory.BEHAVIORAL, 'Return risk level (0-2)')
        
        # === COMPOSITE ENGAGEMENT SCORE ===
        if 'email_engagement_rate' in df.columns and 'return_rate' in df.columns:
            df['engagement_score'] = (
                df['email_engagement_rate'] * 0.6 +
                (1 - df['return_rate']) * 0.4
            )
            self._register_feature('engagement_score', FeatureCategory.BEHAVIORAL, 'Composite engagement (0-1)')
        
        # === CUSTOMER LIFECYCLE ===
        if 'days_since_first_purchase' in df.columns:
            df['tenure_days'] = df['days_since_first_purchase']
            df['tenure_months'] = df['days_since_first_purchase'] / 30.44
            df['tenure_years'] = df['days_since_first_purchase'] / 365.25
            
            # Tenure bucket
            df['tenure_bucket'] = pd.cut(
                df['days_since_first_purchase'],
                bins=[0, 30, 90, 180, 365, 730, float('inf')],
                labels=[1, 2, 3, 4, 5, 6]
            ).astype(float)
            
            # Customer maturity score
            df['customer_maturity'] = np.minimum(df['tenure_days'] / 365, 1.0)
            
            self._register_feature('tenure_months', FeatureCategory.BEHAVIORAL, 'Customer tenure in months')
            self._register_feature('tenure_bucket', FeatureCategory.BEHAVIORAL, 'Tenure bucket (1-6)')
            self._register_feature('customer_maturity', FeatureCategory.BEHAVIORAL, 'Maturity score (0-1)')
        
        # === PRODUCT DIVERSITY ===
        if 'num_categories' in df.columns:
            df['is_multi_category'] = (df['num_categories'] >= 3).astype(int)
            df['category_diversity'] = np.minimum(df['num_categories'] / 5, 1.0)
            
            self._register_feature('is_multi_category', FeatureCategory.BEHAVIORAL, 'Shops 3+ categories')
            self._register_feature('category_diversity', FeatureCategory.BEHAVIORAL, 'Category diversity score')
        
        # === HIGH VALUE INDICATORS ===
        if 'avg_order_value' in df.columns:
            aov_75 = df['avg_order_value'].quantile(0.75)
            aov_90 = df['avg_order_value'].quantile(0.90)
            df['is_high_aov'] = (df['avg_order_value'] >= aov_75).astype(int)
            df['is_premium'] = (df['avg_order_value'] >= aov_90).astype(int)
            
            self._register_feature('is_high_aov', FeatureCategory.BEHAVIORAL, 'Top 25% by AOV')
            self._register_feature('is_premium', FeatureCategory.BEHAVIORAL, 'Top 10% by AOV')
        
        # === CHURN RISK ===
        if 'days_since_last_purchase' in df.columns:
            mean_recency = df['days_since_last_purchase'].mean()
            std_recency = df['days_since_last_purchase'].std()
            
            df['churn_risk_score'] = np.where(
                df['days_since_last_purchase'] > mean_recency + 2 * std_recency, 3,
                np.where(df['days_since_last_purchase'] > mean_recency + std_recency, 2,
                np.where(df['days_since_last_purchase'] > mean_recency, 1, 0))
            )
            
            df['is_at_risk'] = (df['churn_risk_score'] >= 2).astype(int)
            
            self._register_feature('churn_risk_score', FeatureCategory.BEHAVIORAL, 'Churn risk (0-3)')
            self._register_feature('is_at_risk', FeatureCategory.BEHAVIORAL, 'At churn risk')
        
        logger.info("Behavioral features created")
        return df
    
    # =========================================================================
    # Temporal Features
    # =========================================================================
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Features include:
        - Day of week, month, quarter of first/last purchase
        - Seasonality indicators
        - Time since thresholds
        """
        df = df.copy()
        logger.info("Creating temporal features...")
        
        # === FIRST PURCHASE TEMPORAL ===
        if 'first_purchase_date' in df.columns:
            try:
                first_date = pd.to_datetime(df['first_purchase_date'])
                
                df['first_purchase_dow'] = first_date.dt.dayofweek
                df['first_purchase_month'] = first_date.dt.month
                df['first_purchase_quarter'] = first_date.dt.quarter
                df['first_purchase_is_weekend'] = (first_date.dt.dayofweek >= 5).astype(int)
                
                # Season
                df['first_purchase_season'] = first_date.dt.month.map({
                    12: 1, 1: 1, 2: 1,  # Winter
                    3: 2, 4: 2, 5: 2,   # Spring
                    6: 3, 7: 3, 8: 3,   # Summer
                    9: 4, 10: 4, 11: 4  # Fall
                })
                
                self._register_feature('first_purchase_dow', FeatureCategory.TEMPORAL, 'First purchase day of week')
                self._register_feature('first_purchase_month', FeatureCategory.TEMPORAL, 'First purchase month')
                self._register_feature('first_purchase_quarter', FeatureCategory.TEMPORAL, 'First purchase quarter')
                self._register_feature('first_purchase_is_weekend', FeatureCategory.TEMPORAL, 'First purchase on weekend')
                self._register_feature('first_purchase_season', FeatureCategory.TEMPORAL, 'First purchase season')
                
            except Exception as e:
                logger.warning(f"Could not parse first_purchase_date: {e}")
        
        # === LAST PURCHASE TEMPORAL ===
        if 'last_purchase_date' in df.columns:
            try:
                last_date = pd.to_datetime(df['last_purchase_date'])
                
                df['last_purchase_dow'] = last_date.dt.dayofweek
                df['last_purchase_month'] = last_date.dt.month
                df['last_purchase_quarter'] = last_date.dt.quarter
                df['last_purchase_is_weekend'] = (last_date.dt.dayofweek >= 5).astype(int)
                
                self._register_feature('last_purchase_dow', FeatureCategory.TEMPORAL, 'Last purchase day of week')
                self._register_feature('last_purchase_month', FeatureCategory.TEMPORAL, 'Last purchase month')
                
            except Exception as e:
                logger.warning(f"Could not parse last_purchase_date: {e}")
        
        # === TIME THRESHOLDS ===
        if 'days_since_last_purchase' in df.columns:
            df['purchased_last_7d'] = (df['days_since_last_purchase'] <= 7).astype(int)
            df['purchased_last_30d'] = (df['days_since_last_purchase'] <= 30).astype(int)
            df['purchased_last_90d'] = (df['days_since_last_purchase'] <= 90).astype(int)
            df['inactive_180d'] = (df['days_since_last_purchase'] > 180).astype(int)
            
            self._register_feature('purchased_last_7d', FeatureCategory.TEMPORAL, 'Purchased in last 7 days')
            self._register_feature('purchased_last_30d', FeatureCategory.TEMPORAL, 'Purchased in last 30 days')
            self._register_feature('purchased_last_90d', FeatureCategory.TEMPORAL, 'Purchased in last 90 days')
            self._register_feature('inactive_180d', FeatureCategory.TEMPORAL, 'Inactive for 180+ days')
        
        logger.info("Temporal features created")
        return df
    
    # =========================================================================
    # Statistical Features
    # =========================================================================
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features for anomaly detection and normalization.
        
        Features include:
        - Z-scores for key metrics
        - Percentile ranks
        - Outlier indicators
        """
        df = df.copy()
        
        if not self.enable_advanced_stats:
            return df
            
        logger.info("Creating statistical features...")
        
        # Columns to create z-scores for
        zscore_columns = ['total_spent', 'total_orders', 'avg_order_value']
        
        for col in zscore_columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                
                if std > 0:
                    df[f'{col}_zscore'] = (df[col] - mean) / std
                    df[f'{col}_percentile'] = df[col].rank(pct=True)
                    
                    # Outlier detection
                    df[f'{col}_is_outlier'] = (abs(df[f'{col}_zscore']) > 3).astype(int)
                    
                    self._register_feature(f'{col}_zscore', FeatureCategory.STATISTICAL, f'Z-score of {col}')
                    self._register_feature(f'{col}_percentile', FeatureCategory.STATISTICAL, f'Percentile rank of {col}')
                    self._register_feature(f'{col}_is_outlier', FeatureCategory.STATISTICAL, f'Outlier flag for {col}')
        
        # === COMPOSITE OUTLIER SCORE ===
        outlier_cols = [c for c in df.columns if c.endswith('_is_outlier')]
        if outlier_cols:
            df['outlier_score'] = df[outlier_cols].sum(axis=1)
            self._register_feature('outlier_score', FeatureCategory.STATISTICAL, 'Number of outlier flags')
        
        logger.info("Statistical features created")
        return df
    
    # =========================================================================
    # Acquisition Features
    # =========================================================================
    
    def create_acquisition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on customer acquisition channel and campaign.
        """
        df = df.copy()
        logger.info("Creating acquisition features...")
        
        # === SOURCE QUALITY SCORES ===
        if 'acquisition_source' in df.columns:
            # Quality score based on historical performance
            source_quality = {
                'Referral': 0.95,
                'Email': 0.85,
                'Organic': 0.80,
                'Google Ads': 0.75,
                'Meta Ads': 0.70,
                'Direct': 0.65
            }
            df['source_quality_score'] = df['acquisition_source'].map(source_quality).fillna(0.5)
            
            # One-hot encoding style scores
            df['is_paid_acquisition'] = df['acquisition_source'].isin(['Meta Ads', 'Google Ads']).astype(int)
            df['is_organic'] = df['acquisition_source'].isin(['Organic', 'Direct', 'Referral']).astype(int)
            df['is_referral'] = (df['acquisition_source'] == 'Referral').astype(int)
            
            self._register_feature('source_quality_score', FeatureCategory.ACQUISITION, 'Source quality (0-1)')
            self._register_feature('is_paid_acquisition', FeatureCategory.ACQUISITION, 'Acquired via paid channel')
            self._register_feature('is_organic', FeatureCategory.ACQUISITION, 'Acquired organically')
            self._register_feature('is_referral', FeatureCategory.ACQUISITION, 'Acquired via referral')
        
        # === CAMPAIGN EFFECTIVENESS ===
        if 'campaign_type' in df.columns:
            campaign_scores = {
                'Retargeting': 0.85,
                'Brand': 0.75,
                'Prospecting': 0.65,
                'None': 0.50
            }
            df['campaign_effectiveness'] = df['campaign_type'].map(campaign_scores).fillna(0.5)
            
            df['is_retargeted'] = (df['campaign_type'] == 'Retargeting').astype(int)
            df['is_prospecting'] = (df['campaign_type'] == 'Prospecting').astype(int)
            
            self._register_feature('campaign_effectiveness', FeatureCategory.ACQUISITION, 'Campaign type effectiveness')
            self._register_feature('is_retargeted', FeatureCategory.ACQUISITION, 'Acquired via retargeting')
        
        # === CAC EFFICIENCY ===
        if 'acquisition_cost' in df.columns:
            df['has_cac'] = (df['acquisition_cost'] > 0).astype(int)
            
            if 'total_spent' in df.columns:
                df['cac_efficiency'] = np.where(
                    df['acquisition_cost'] > 0,
                    df['total_spent'] / df['acquisition_cost'],
                    df['total_spent'].median()
                )
                
                df['cac_roi'] = np.where(
                    df['acquisition_cost'] > 0,
                    (df['total_spent'] - df['acquisition_cost']) / df['acquisition_cost'],
                    0
                )
                
                # CAC payback indicator
                df['cac_paid_back'] = (df['total_spent'] > df['acquisition_cost']).astype(int)
                
                self._register_feature('cac_efficiency', FeatureCategory.ACQUISITION, 'Revenue / CAC ratio')
                self._register_feature('cac_roi', FeatureCategory.ACQUISITION, 'CAC return on investment')
                self._register_feature('cac_paid_back', FeatureCategory.ACQUISITION, 'CAC recovered flag')
        
        logger.info("Acquisition features created")
        return df
    
    # =========================================================================
    # Interaction Features
    # =========================================================================
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features by combining existing features.
        """
        df = df.copy()
        logger.info("Creating interaction features...")
        
        # === VALUE x FREQUENCY INTERACTIONS ===
        if 'total_spent' in df.columns and 'total_orders' in df.columns:
            # Value per order trend
            if 'avg_order_value' in df.columns:
                median_aov = df['avg_order_value'].median()
                df['value_frequency_score'] = (
                    (df['avg_order_value'] / median_aov) * 
                    np.log1p(df['total_orders'])
                )
                self._register_feature('value_frequency_score', FeatureCategory.INTERACTION, 'Value x frequency composite')
        
        # === ENGAGEMENT x VALUE ===
        if 'engagement_score' in df.columns and 'monetary_normalized' in df.columns:
            df['engaged_value_score'] = df['engagement_score'] * df['monetary_normalized']
            self._register_feature('engaged_value_score', FeatureCategory.INTERACTION, 'Engagement x monetary value')
        
        # === RECENCY x FREQUENCY ===
        if 'recency_normalized' in df.columns and 'frequency_normalized' in df.columns:
            df['rf_score'] = df['recency_normalized'] * df['frequency_normalized']
            self._register_feature('rf_score', FeatureCategory.INTERACTION, 'Recency x frequency product')
        
        # === TENURE x VALUE ===
        if 'customer_maturity' in df.columns and 'monetary_normalized' in df.columns:
            df['mature_value_score'] = df['customer_maturity'] * df['monetary_normalized']
            self._register_feature('mature_value_score', FeatureCategory.INTERACTION, 'Maturity x value score')
        
        # === SOURCE x VALUE ===
        if 'source_quality_score' in df.columns and 'monetary_normalized' in df.columns:
            df['source_value_score'] = df['source_quality_score'] * df['monetary_normalized']
            self._register_feature('source_value_score', FeatureCategory.INTERACTION, 'Source quality x value')
        
        # === CLV PREDICTOR COMPOSITE ===
        composite_features = ['rfm_weighted', 'engagement_score', 'purchase_velocity', 'cac_efficiency']
        available = [f for f in composite_features if f in df.columns]
        
        if len(available) >= 2:
            # Normalize and combine
            normalized = df[available].apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10))
            df['clv_predictor_composite'] = normalized.mean(axis=1)
            self._register_feature('clv_predictor_composite', FeatureCategory.INTERACTION, 'CLV predictor composite score')
        
        logger.info("Interaction features created")
        return df
    
    # =========================================================================
    # Main Pipeline
    # =========================================================================
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        include_rfm: bool = True,
        include_behavioral: bool = True,
        include_temporal: bool = True,
        include_statistical: bool = True,
        include_acquisition: bool = True,
        include_interaction: bool = True
    ) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            df: Raw customer DataFrame
            include_*: Flags to include/exclude feature categories
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info(f"Starting feature engineering on {len(df)} records...")
        self._original_columns = df.columns.tolist()
        
        result = df.copy()
        
        if include_rfm:
            result = self.calculate_rfm_scores(result)
        
        if include_behavioral:
            result = self.create_behavioral_features(result)
        
        if include_temporal:
            result = self.create_temporal_features(result)
        
        if include_statistical:
            result = self.create_statistical_features(result)
        
        if include_acquisition:
            result = self.create_acquisition_features(result)
        
        if include_interaction:
            result = self.create_interaction_features(result)
        
        # Fill any NaN values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)
        
        logger.info(f"Feature engineering complete. Created {len(self._feature_columns)} new features.")
        return result
    
    def get_feature_names(self, category: Optional[FeatureCategory] = None) -> List[str]:
        """Get list of feature names, optionally filtered by category."""
        if category is None:
            return self._feature_columns.copy()
        
        return [
            name for name, meta in self._feature_metadata.items()
            if meta.category == category
        ]
    
    def get_numeric_features(self, df: pd.DataFrame) -> List[str]:
        """Get list of numeric feature columns suitable for ML."""
        exclude = ['customer_id', 'first_purchase_date', 'last_purchase_date',
                   'product_categories', 'customer_segment', 'actual_clv',
                   'rfm_string', 'rfm_segment', 'tenure_bucket']
        
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric if c not in exclude]
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of all engineered features."""
        category_counts = {}
        for meta in self._feature_metadata.values():
            cat = meta.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            'total_features': len(self._feature_columns),
            'features_by_category': category_counts,
            'feature_list': list(self._feature_metadata.keys())
        }


# Convenience function
def engineer_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """One-line feature engineering."""
    engineer = AdvancedFeatureEngineer()
    return engineer.fit_transform(df, **kwargs)
