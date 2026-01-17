"""
Machine Learning Models Module
Implements Random Forest, XGBoost, and ensemble models for CLV prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import pickle
import json
from pathlib import Path
import logging

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLVModelTrainer:
    """Trains and evaluates ML models for CLV prediction."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize model trainer.
        
        Args:
            model_dir: Directory to save trained models.
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.scaler = StandardScaler()
        self.models = {}
        self.metrics = {}
        
    def train_random_forest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 15,
        random_state: int = 42
    ) -> RandomForestRegressor:
        """
        Train Random Forest regressor for CLV prediction.
        
        Args:
            X: Feature matrix.
            y: Target variable.
            n_estimators: Number of trees.
            max_depth: Maximum tree depth.
            random_state: Random seed.
            
        Returns:
            Trained Random Forest model.
        """
        logger.info("Training Random Forest model...")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        model.fit(X, y)
        self.models['random_forest'] = model
        
        logger.info("Random Forest training complete")
        return model
    
    def train_gradient_boosting(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        random_state: int = 42
    ) -> GradientBoostingRegressor:
        """
        Train Gradient Boosting (XGBoost-like) regressor.
        
        Args:
            X: Feature matrix.
            y: Target variable.
            n_estimators: Number of boosting stages.
            learning_rate: Learning rate.
            max_depth: Maximum tree depth.
            random_state: Random seed.
            
        Returns:
            Trained Gradient Boosting model.
        """
        logger.info("Training Gradient Boosting model...")
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )
        
        model.fit(X, y)
        self.models['gradient_boosting'] = model
        
        logger.info("Gradient Boosting training complete")
        return model
    
    def train_linear_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        alpha: float = 1.0
    ) -> Ridge:
        """
        Train Ridge regression as baseline model.
        
        Args:
            X: Feature matrix.
            y: Target variable.
            alpha: Regularization strength.
            
        Returns:
            Trained Ridge model.
        """
        logger.info("Training Ridge regression model...")
        
        # Scale features for linear model
        X_scaled = self.scaler.fit_transform(X)
        
        model = Ridge(alpha=alpha)
        model.fit(X_scaled, y)
        self.models['linear'] = model
        
        logger.info("Ridge regression training complete")
        return model
    
    def train_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Train ensemble of all models with weighted averaging.
        
        Args:
            X: Feature matrix.
            y: Target variable.
            weights: Model weights for ensemble (default: 40% GB, 35% RF, 25% Linear).
            
        Returns:
            Dictionary containing all trained models and weights.
        """
        logger.info("Training ensemble model...")
        
        # Default weights based on best practices
        if weights is None:
            weights = {
                'gradient_boosting': 0.40,
                'random_forest': 0.35,
                'linear': 0.25
            }
        
        # Train all component models
        self.train_random_forest(X, y)
        self.train_gradient_boosting(X, y)
        self.train_linear_model(X, y)
        
        ensemble = {
            'models': self.models,
            'weights': weights,
            'scaler': self.scaler
        }
        
        self.models['ensemble'] = ensemble
        
        logger.info("Ensemble training complete")
        return ensemble
    
    def predict_ensemble(
        self,
        X: pd.DataFrame,
        ensemble: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Make predictions using the ensemble model.
        
        Args:
            X: Feature matrix.
            ensemble: Ensemble dictionary. Uses stored ensemble if not provided.
            
        Returns:
            Weighted average predictions.
        """
        ensemble = ensemble or self.models.get('ensemble')
        
        if ensemble is None:
            raise ValueError("No ensemble model available. Train ensemble first.")
        
        models = ensemble['models']
        weights = ensemble['weights']
        scaler = ensemble['scaler']
        
        predictions = np.zeros(len(X))
        
        # Random Forest prediction
        rf_pred = models['random_forest'].predict(X)
        predictions += weights['random_forest'] * rf_pred
        
        # Gradient Boosting prediction
        gb_pred = models['gradient_boosting'].predict(X)
        predictions += weights['gradient_boosting'] * gb_pred
        
        # Linear model prediction (needs scaled input)
        X_scaled = scaler.transform(X)
        linear_pred = models['linear'].predict(X_scaled)
        predictions += weights['linear'] * linear_pred
        
        return predictions
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate model performance with multiple metrics.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test targets.
            model_name: Name for storing metrics.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Handle ensemble predictions
        if model_name == 'ensemble':
            y_pred = self.predict_ensemble(X_test)
        elif model_name == 'linear':
            X_scaled = self.scaler.transform(X_test)
            y_pred = model.predict(X_scaled)
        else:
            y_pred = model.predict(X_test)
        
        metrics = {
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'r2': float(r2_score(y_test, y_pred)),
            'mape': float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100)
        }
        
        self.metrics[model_name] = metrics
        
        logger.info(f"{model_name} - MAE: ${metrics['mae']:.2f}, RMSE: ${metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
        
        return metrics
    
    def get_feature_importance(
        self,
        model_name: str = 'random_forest',
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Args:
            model_name: Name of the model to get importance from.
            feature_names: List of feature names.
            
        Returns:
            DataFrame with feature importances sorted by importance.
        """
        model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model '{model_name}' doesn't have feature importance")
        
        importance = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(
        self,
        model_name: str,
        filepath: Optional[str] = None
    ) -> str:
        """
        Save trained model to disk.
        
        Args:
            model_name: Name of model to save.
            filepath: Custom filepath. Defaults to model_dir/model_name.pkl.
            
        Returns:
            Path where model was saved.
        """
        model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        if filepath is None:
            filepath = self.model_dir / f"{model_name}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to {filepath}")
        return str(filepath)
    
    def load_model(self, filepath: str, model_name: str) -> Any:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved model.
            model_name: Name to store model under.
            
        Returns:
            Loaded model.
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        self.models[model_name] = model
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def save_all_models(self) -> Dict[str, str]:
        """
        Save all trained models.
        
        Returns:
            Dictionary mapping model names to save paths.
        """
        paths = {}
        for name in self.models.keys():
            paths[name] = self.save_model(name)
        return paths
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all stored evaluation metrics."""
        return self.metrics


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[CLVModelTrainer, Dict[str, Dict[str, float]]]:
    """
    Convenience function to train and evaluate all models.
    
    Args:
        X: Feature matrix.
        y: Target variable.
        test_size: Fraction of data for testing.
        random_state: Random seed.
        
    Returns:
        Tuple of (trainer instance, metrics dictionary).
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train models
    trainer = CLVModelTrainer()
    trainer.train_ensemble(X_train, y_train)
    
    # Evaluate all models
    for model_name in ['random_forest', 'gradient_boosting', 'linear']:
        trainer.evaluate_model(
            trainer.models[model_name],
            X_test,
            y_test,
            model_name
        )
    
    # Evaluate ensemble
    trainer.evaluate_model(
        trainer.models['ensemble'],
        X_test,
        y_test,
        'ensemble'
    )
    
    return trainer, trainer.get_all_metrics()
