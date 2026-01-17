"""
Feature Engineering Module
Creates RFM features and behavioral features for CLV prediction.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for CLV prediction from customer data."""
    
    def __init__(self, reference_date: Optional[datetime] = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            reference_date: Date to calculate recency from. Defaults to today.
        """
        self.reference_date = reference_date or datetime.now()
        self.feature_columns = []
        
    def calculate_rfm_scores(
        self, 
        df: pd.DataFrame,
        n_bins: int = 5
    ) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) scores.
        
        Args:
            df: DataFrame with customer data.
            n_bins: Number of bins for RFM scoring (default 5).
            
        Returns:
            DataFrame with RFM scores added.
        """
        df = df.copy()
        logger.info("Calculating RFM scores...")
        
        # Recency score (lower is better, so reverse scoring)
        if 'days_since_last_purchase' in df.columns:
            df['recency_score'] = pd.qcut(
                df['days_since_last_purchase'].rank(method='first'), 
                q=n_bins, 
                labels=range(n_bins, 0, -1),
                duplicates='drop'
            ).astype(int)
        else:
            df['recency_score'] = n_bins // 2
        
        # Frequency score (higher is better)
        if 'total_orders' in df.columns:
            df['frequency_score'] = pd.qcut(
                df['total_orders'].rank(method='first'), 
                q=n_bins, 
                labels=range(1, n_bins + 1),
                duplicates='drop'
            ).astype(int)
        else:
            df['frequency_score'] = n_bins // 2
        
        # Monetary score (higher is better)
        if 'total_spent' in df.columns:
            df['monetary_score'] = pd.qcut(
                df['total_spent'].rank(method='first'), 
                q=n_bins, 
                labels=range(1, n_bins + 1),
                duplicates='drop'
            ).astype(int)
        else:
            df['monetary_score'] = n_bins // 2
        
        # Combined RFM score
        df['rfm_score'] = df['recency_score'] + df['frequency_score'] + df['monetary_score']
        
        # RFM segment
        df['rfm_segment'] = df.apply(self._assign_rfm_segment, axis=1)
        
        logger.info("RFM scores calculated")
        return df
    
    def _assign_rfm_segment(self, row: pd.Series) -> str:
        """Assign RFM segment based on scores."""
        r, f, m = row['recency_score'], row['frequency_score'], row['monetary_score']
        
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'Recent Customers'
        elif r <= 2 and f >= 4:
            return 'At Risk'
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Lost'
        else:
            return 'Potential Loyalists'
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral features from customer data.
        
        Args:
            df: DataFrame with customer data.
            
        Returns:
            DataFrame with behavioral features added.
        """
        df = df.copy()
        logger.info("Creating behavioral features...")
        
        # Purchase velocity (orders per month of tenure)
        if 'total_orders' in df.columns and 'days_since_first_purchase' in df.columns:
            months_active = (df['days_since_first_purchase'] / 30).clip(lower=1)
            df['purchase_velocity'] = df['total_orders'] / months_active
        
        # Average days between purchases
        if 'total_orders' in df.columns and 'days_since_first_purchase' in df.columns:
            df['avg_days_between_purchases'] = np.where(
                df['total_orders'] > 1,
                df['days_since_first_purchase'] / (df['total_orders'] - 1),
                df['days_since_first_purchase']
            )
        
        # Customer tenure bucket
        if 'days_since_first_purchase' in df.columns:
            df['tenure_bucket'] = pd.cut(
                df['days_since_first_purchase'],
                bins=[0, 30, 90, 180, 365, float('inf')],
                labels=['New', '1-3 Months', '3-6 Months', '6-12 Months', '12+ Months']
            )
        
        # Engagement score (combination of email engagement and low return rate)
        if 'email_engagement_rate' in df.columns and 'return_rate' in df.columns:
            df['engagement_score'] = (
                df['email_engagement_rate'] * 0.7 + 
                (1 - df['return_rate']) * 0.3
            )
        
        # High-value indicator
        if 'avg_order_value' in df.columns:
            aov_threshold = df['avg_order_value'].quantile(0.75)
            df['is_high_value'] = (df['avg_order_value'] >= aov_threshold).astype(int)
        
        # Multi-category buyer
        if 'num_categories' in df.columns:
            df['is_multi_category'] = (df['num_categories'] >= 3).astype(int)
        
        # Churn risk indicator
        if 'days_since_last_purchase' in df.columns:
            mean_recency = df['days_since_last_purchase'].mean()
            std_recency = df['days_since_last_purchase'].std()
            df['churn_risk'] = np.where(
                df['days_since_last_purchase'] > mean_recency + std_recency,
                1, 0
            )
        
        logger.info("Behavioral features created")
        return df
    
    def create_acquisition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on customer acquisition.
        
        Args:
            df: DataFrame with customer data.
            
        Returns:
            DataFrame with acquisition features added.
        """
        df = df.copy()
        logger.info("Creating acquisition features...")
        
        # Acquisition source encoding (for numeric analysis)
        if 'acquisition_source' in df.columns:
            source_quality = {
                'Meta Ads': 0.7,
                'Google Ads': 0.75,
                'Email': 0.85,
                'Referral': 0.9,
                'Organic': 0.8,
                'Direct': 0.65
            }
            df['source_quality_score'] = df['acquisition_source'].map(source_quality).fillna(0.5)
        
        # Campaign effectiveness indicator
        if 'campaign_type' in df.columns:
            campaign_scores = {
                'Prospecting': 0.6,
                'Retargeting': 0.8,
                'Brand': 0.75,
                'None': 0.5
            }
            df['campaign_effectiveness'] = df['campaign_type'].map(campaign_scores).fillna(0.5)
        
        # Customer Acquisition Cost efficiency
        if 'acquisition_cost' in df.columns and 'total_spent' in df.columns:
            df['cac_efficiency'] = np.where(
                df['acquisition_cost'] > 0,
                df['total_spent'] / df['acquisition_cost'],
                df['total_spent']
            )
        
        # Paid vs organic flag
        if 'acquisition_source' in df.columns:
            paid_sources = ['Meta Ads', 'Google Ads']
            df['is_paid_acquisition'] = df['acquisition_source'].isin(paid_sources).astype(int)
        
        logger.info("Acquisition features created")
        return df
    
    def prepare_feature_matrix(
        self, 
        df: pd.DataFrame,
        include_rfm: bool = True,
        include_behavioral: bool = True,
        include_acquisition: bool = True
    ) -> pd.DataFrame:
        """
        Create complete feature matrix applying all feature engineering.
        
        Args:
            df: Raw customer DataFrame.
            include_rfm: Whether to include RFM features.
            include_behavioral: Whether to include behavioral features.
            include_acquisition: Whether to include acquisition features.
            
        Returns:
            DataFrame with all engineered features.
        """
        logger.info("Preparing complete feature matrix...")
        
        result = df.copy()
        
        if include_rfm:
            result = self.calculate_rfm_scores(result)
        
        if include_behavioral:
            result = self.create_behavioral_features(result)
        
        if include_acquisition:
            result = self.create_acquisition_features(result)
        
        # Store feature columns
        self.feature_columns = [
            col for col in result.columns 
            if col not in ['customer_id', 'first_purchase_date', 'last_purchase_date',
                          'product_categories', 'customer_segment', 'actual_clv']
        ]
        
        logger.info(f"Feature matrix prepared with {len(self.feature_columns)} features")
        return result
    
    def get_numeric_features(self, df: pd.DataFrame) -> List[str]:
        """Get list of numeric feature columns."""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics for all features."""
        numeric_cols = self.get_numeric_features(df)
        
        return {
            'total_features': len(numeric_cols),
            'feature_names': numeric_cols,
            'statistics': df[numeric_cols].describe().to_dict()
        }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to apply all feature engineering.
    
    Args:
        df: Raw customer DataFrame.
        
    Returns:
        DataFrame with all engineered features.
    """
    engineer = FeatureEngineer()
    return engineer.prepare_feature_matrix(df)
