#!/usr/bin/env python3
"""
Training Script for CLV Models with MLflow Integration
Used by DVC pipeline for reproducible training runs.
"""
import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.feature_engineering import AdvancedFeatureEngineer
from backend.ml_models import CLVModelTrainer
from backend.mlflow_config import get_mlflow_config, MLflowConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_file: str = "params.yaml") -> dict:
    """Load parameters from params.yaml."""
    with open(params_file, 'r') as f:
        return yaml.safe_load(f)


def train_models(data_path: str, output_dir: str, params: dict):
    """
    Train CLV models with MLflow tracking.
    
    Args:
        data_path: Path to feature data CSV
        output_dir: Directory to save models
        params: Training parameters
    """
    # Setup MLflow
    mlflow_config = get_mlflow_config()
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Feature engineering
    logger.info("Running feature engineering...")
    feature_engineer = AdvancedFeatureEngineer()
    df_features = feature_engineer.fit_transform(df)
    
    # Prepare features and target
    target_col = 'actual_clv' if 'actual_clv' in df_features.columns else 'total_spent'
    feature_cols = [col for col in df_features.columns 
                   if col not in ['customer_id', 'actual_clv', 'predicted_clv', 
                                 'customer_segment', 'predicted_segment',
                                 'first_purchase_date', 'last_purchase_date',
                                 'product_categories', 'acquisition_source', 'campaign_type']]
    
    # Filter to numeric columns only
    numeric_features = df_features[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = df_features[numeric_features].fillna(0)
    y = df_features[target_col]
    
    logger.info(f"Features: {len(numeric_features)}, Samples: {len(X)}")
    
    # Split data
    test_size = params.get('train', {}).get('test_size', 0.2)
    random_state = params.get('train', {}).get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize trainer
    trainer = CLVModelTrainer()
    
    # Start MLflow run
    with mlflow_config.start_run(
        run_name=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        tags={"stage": "training"}
    ):
        # Log parameters
        mlflow_config.log_params({
            "n_samples": len(X),
            "n_features": len(numeric_features),
            "test_size": test_size,
            "random_state": random_state,
            **params.get('train', {})
        })
        
        # Train models
        logger.info("Training models...")
        results = trainer.train_ensemble(X_train, y_train)
        
        # Evaluate on test set
        logger.info("Evaluating models...")
        predictions = trainer.predict_ensemble(X_test)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
            "n_train": len(X_train),
            "n_test": len(X_test)
        }
        
        logger.info(f"Metrics: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}")
        
        # Log metrics to MLflow
        mlflow_config.log_metrics(metrics)
        
        # Save metrics to file (for DVC)
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Get feature importance
        feature_importance = trainer.get_feature_importance('ensemble', numeric_features)
        if feature_importance is not None:
            importance_data = feature_importance.to_dict('records')
            with open("metrics/feature_importance.json", 'w') as f:
                json.dump(importance_data, f, indent=2)
            mlflow_config.log_artifact("metrics/feature_importance.json")
        
        # Save models
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_all_models(output_dir)
        logger.info(f"Models saved to {output_dir}")
        
        # Log model to MLflow
        mlflow_config.log_artifact(output_dir, "models")
        
        # Set tags
        mlflow_config.set_tag("best_rmse", f"{metrics['rmse']:.2f}")
        mlflow_config.set_tag("model_type", "ensemble")
    
    logger.info("Training complete!")
    return metrics


def main():
    """Main entry point."""
    # Load parameters
    params = load_params()
    
    # Paths
    data_path = os.environ.get("DATA_PATH", "data/customers.csv")
    output_dir = os.environ.get("OUTPUT_DIR", "models")
    
    # Train
    metrics = train_models(data_path, output_dir, params)
    
    print(f"\n✅ Training Complete!")
    print(f"   RMSE: {metrics['rmse']:.2f}")
    print(f"   MAE:  {metrics['mae']:.2f}")
    print(f"   R²:   {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
