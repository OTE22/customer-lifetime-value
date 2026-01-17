"""
Data Processor Module
Handles loading, cleaning, and preprocessing of customer data for CLV prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles all data loading and preprocessing operations."""
    
    def __init__(self, filepath: Optional[str] = None):
        """Initialize DataProcessor with optional data filepath."""
        self.filepath = filepath
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load customer data from CSV file.
        
        Args:
            filepath: Path to CSV file. Uses instance filepath if not provided.
            
        Returns:
            DataFrame with raw customer data.
        """
        filepath = filepath or self.filepath
        if not filepath:
            raise ValueError("No filepath provided")
        
        logger.info(f"Loading data from {filepath}")
        
        self.raw_data = pd.read_csv(filepath)
        
        # Parse dates
        date_columns = ['first_purchase_date', 'last_purchase_date']
        for col in date_columns:
            if col in self.raw_data.columns:
                self.raw_data[col] = pd.to_datetime(self.raw_data[col])
        
        logger.info(f"Loaded {len(self.raw_data)} customer records")
        return self.raw_data
    
    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean customer data by handling missing values and outliers.
        
        Args:
            df: DataFrame to clean. Uses raw_data if not provided.
            
        Returns:
            Cleaned DataFrame.
        """
        df = df.copy() if df is not None else self.raw_data.copy()
        
        if df is None:
            raise ValueError("No data to clean. Load data first.")
        
        logger.info("Cleaning data...")
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {col} missing values with median: {median_val}")
        
        # Handle categorical missing values
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
        
        # Remove outliers using IQR method for monetary columns
        monetary_columns = ['total_spent', 'avg_order_value', 'actual_clv']
        for col in monetary_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing
                df[col] = df[col].clip(lower=max(0, lower_bound), upper=upper_bound)
        
        # Ensure non-negative values for counts
        count_columns = ['total_orders', 'days_since_first_purchase', 'days_since_last_purchase']
        for col in count_columns:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        
        # Ensure rates are between 0 and 1
        rate_columns = ['email_engagement_rate', 'return_rate']
        for col in rate_columns:
            if col in df.columns:
                df[col] = df[col].clip(lower=0, upper=1)
        
        logger.info(f"Cleaned data: {len(df)} records")
        self.processed_data = df
        return df
    
    def preprocess_for_training(
        self, 
        df: Optional[pd.DataFrame] = None,
        target_column: str = 'actual_clv'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for ML training by encoding categoricals and splitting features/target.
        
        Args:
            df: DataFrame to preprocess.
            target_column: Name of target variable column.
            
        Returns:
            Tuple of (features DataFrame, target Series).
        """
        df = df.copy() if df is not None else self.processed_data.copy()
        
        if df is None:
            raise ValueError("No processed data available. Run clean_data first.")
        
        logger.info("Preprocessing data for training...")
        
        # Extract target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        y = df[target_column].copy()
        
        # Select feature columns (exclude IDs, dates, target, segment)
        exclude_columns = [
            'customer_id', 
            'first_purchase_date', 
            'last_purchase_date',
            'product_categories',
            'customer_segment',
            target_column
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        X = df[feature_columns].copy()
        
        # One-hot encode categorical columns
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_columns:
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        
        logger.info(f"Created feature matrix with shape: {X.shape}")
        
        return X, y
    
    def get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """Get list of feature names from preprocessed DataFrame."""
        return X.columns.tolist()
    
    def validate_data_quality(self, df: Optional[pd.DataFrame] = None) -> dict:
        """
        Validate data quality and return quality metrics.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Dictionary with quality metrics.
        """
        df = df if df is not None else self.raw_data
        
        if df is None:
            raise ValueError("No data to validate")
        
        quality_report = {
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'numeric_stats': {},
            'categorical_stats': {}
        }
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            quality_report['numeric_stats'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std())
            }
        
        # Categorical column statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['customer_id', 'product_categories']:
                quality_report['categorical_stats'][col] = df[col].value_counts().to_dict()
        
        return quality_report


def load_and_preprocess(filepath: str) -> Tuple[pd.DataFrame, pd.Series, DataProcessor]:
    """
    Convenience function to load and preprocess data in one step.
    
    Args:
        filepath: Path to customer data CSV.
        
    Returns:
        Tuple of (features, target, processor instance).
    """
    processor = DataProcessor(filepath)
    processor.load_data()
    processor.clean_data()
    X, y = processor.preprocess_for_training()
    return X, y, processor
