#!/usr/bin/env python3
"""
Model Evaluation Script for CLV Prediction System
Used by DVC pipeline for model evaluation.
"""
import os
import sys
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.feature_engineering import AdvancedFeatureEngineer
from backend.ml_models import CLVModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_models(data_path: str, models_dir: str):
    """
    Evaluate trained models on test data.
    
    Args:
        data_path: Path to test data
        models_dir: Directory containing trained models
    """
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Feature engineering
    logger.info("Running feature engineering...")
    feature_engineer = AdvancedFeatureEngineer()
    df_features = feature_engineer.fit_transform(df)
    
    # Prepare features
    target_col = 'actual_clv' if 'actual_clv' in df_features.columns else 'total_spent'
    feature_cols = [col for col in df_features.columns 
                   if col not in ['customer_id', 'actual_clv', 'predicted_clv', 
                                 'customer_segment', 'predicted_segment',
                                 'first_purchase_date', 'last_purchase_date',
                                 'product_categories', 'acquisition_source', 'campaign_type']]
    
    numeric_features = df_features[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = df_features[numeric_features].fillna(0)
    y = df_features[target_col]
    
    # Load trained models
    logger.info(f"Loading models from {models_dir}")
    trainer = CLVModelTrainer()
    trainer.load_all_models(models_dir)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = trainer.predict_ensemble(X)
    
    # Calculate metrics
    metrics = {
        "evaluation_samples": len(X),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "mae": float(mean_absolute_error(y, predictions)),
        "r2": float(r2_score(y, predictions)),
        "mape": float(mean_absolute_percentage_error(y, predictions)) * 100,
        
        # Segment-level metrics
        "high_clv_threshold": 500,
        "growth_clv_threshold": 150,
    }
    
    # Segment accuracy
    actual_segments = y.apply(lambda x: 'High-CLV' if x >= 500 else ('Growth-Potential' if x >= 150 else 'Low-CLV'))
    pred_segments = pd.Series(predictions).apply(lambda x: 'High-CLV' if x >= 500 else ('Growth-Potential' if x >= 150 else 'Low-CLV'))
    segment_accuracy = (actual_segments == pred_segments).mean()
    metrics["segment_accuracy"] = float(segment_accuracy)
    
    # Per-segment metrics
    for segment in ['High-CLV', 'Growth-Potential', 'Low-CLV']:
        mask = actual_segments == segment
        if mask.sum() > 0:
            seg_rmse = np.sqrt(mean_squared_error(y[mask], predictions[mask]))
            metrics[f"{segment.lower().replace('-', '_')}_rmse"] = float(seg_rmse)
            metrics[f"{segment.lower().replace('-', '_')}_count"] = int(mask.sum())
    
    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/evaluation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation metrics saved to metrics/evaluation_metrics.json")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Samples Evaluated: {metrics['evaluation_samples']}")
    print(f"RMSE:              ${metrics['rmse']:.2f}")
    print(f"MAE:               ${metrics['mae']:.2f}") 
    print(f"R²:                {metrics['r2']:.4f}")
    print(f"MAPE:              {metrics['mape']:.1f}%")
    print(f"Segment Accuracy:  {metrics['segment_accuracy']*100:.1f}%")
    print("="*50)
    
    return metrics


def main():
    """Main entry point."""
    data_path = os.environ.get("DATA_PATH", "data/customers.csv")
    models_dir = os.environ.get("MODELS_DIR", "models")
    
    metrics = evaluate_models(data_path, models_dir)
    
    # Exit with error if R² is too low
    if metrics['r2'] < 0.5:
        logger.warning(f"R² score ({metrics['r2']:.4f}) is below threshold (0.5)")
        sys.exit(1)
    
    print("\n✅ Evaluation Complete!")


if __name__ == "__main__":
    main()
