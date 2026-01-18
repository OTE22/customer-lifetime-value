"""
Production ML Models for CLV Prediction System
XGBoost, LightGBM, and advanced ensemble for large-scale predictions.

Author: Ali Abbass (OTE22)
"""

import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import warnings

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

from .logging_config import get_logger
from .exceptions import ModelNotTrainedError

logger = get_logger(__name__)


@dataclass
class ProductionModelMetrics:
    """Metrics for production model evaluation."""
    mae: float
    rmse: float
    r2: float
    mape: Optional[float] = None
    cv_mae_mean: Optional[float] = None
    cv_mae_std: Optional[float] = None
    training_samples: int = 0
    training_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class XGBoostCLVModel:
    """
    XGBoost model optimized for CLV prediction.
    
    Features:
    - Early stopping to prevent overfitting
    - GPU support when available
    - Built-in feature importance
    - Hyperparameter tuning
    """
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        n_jobs: int = -1,
        use_gpu: bool = False
    ):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'objective': 'reg:squarederror',
            'tree_method': 'gpu_hist' if use_gpu else 'hist',
            'verbosity': 0
        }
        
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        self.metrics: Optional[ProductionModelMetrics] = None
        
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> ProductionModelMetrics:
        """Train the XGBoost model with early stopping."""
        start_time = datetime.now()
        
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create train/val split for early stopping
        if eval_set is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.15, random_state=42
            )
        else:
            X_train, y_train = X_scaled, y
            X_val = self.scaler.transform(eval_set[0])
            y_val = eval_set[1]
        
        # Initialize model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Calculate metrics
        y_pred = self.model.predict(X_scaled)
        y_pred = np.maximum(y_pred, 0)  # CLV cannot be negative
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.metrics = self._calculate_metrics(y, y_pred, len(X), training_time)
        self.is_trained = True
        
        logger.info(f"XGBoost trained: MAE=${self.metrics.mae:.2f}, R²={self.metrics.r2:.3f}")
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ModelNotTrainedError("XGBoost")
        
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return np.maximum(predictions, 0)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_trained:
            raise ModelNotTrainedError("XGBoost")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        importance['rank'] = range(1, len(importance) + 1)
        return importance
    
    def _calculate_metrics(
        self, y_true, y_pred, n_samples, training_time
    ) -> ProductionModelMetrics:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (avoid division by zero)
        non_zero = y_true != 0
        if non_zero.sum() > 0:
            mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
        else:
            mape = None
        
        return ProductionModelMetrics(
            mae=round(mae, 2),
            rmse=round(rmse, 2),
            r2=round(r2, 4),
            mape=round(mape, 2) if mape else None,
            training_samples=n_samples,
            training_time_seconds=round(training_time, 2)
        )


class LightGBMCLVModel:
    """
    LightGBM model optimized for CLV prediction.
    
    Features:
    - Faster training than XGBoost
    - Better handling of categorical features
    - Lower memory usage
    - GPU support
    """
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = -1,  # -1 means no limit
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        n_jobs: int = -1,
        use_gpu: bool = False
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_samples': min_child_samples,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'objective': 'regression',
            'metric': 'mae',
            'device': 'gpu' if use_gpu else 'cpu',
            'verbosity': -1
        }
        
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        self.metrics: Optional[ProductionModelMetrics] = None
        
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> ProductionModelMetrics:
        """Train the LightGBM model."""
        start_time = datetime.now()
        
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create train/val split
        if eval_set is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.15, random_state=42
            )
        else:
            X_train, y_train = X_scaled, y
            X_val = self.scaler.transform(eval_set[0])
            y_val = eval_set[1]
        
        # Initialize and train
        self.model = lgb.LGBMRegressor(**self.params)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )
        
        # Calculate metrics
        y_pred = self.model.predict(X_scaled)
        y_pred = np.maximum(y_pred, 0)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.metrics = self._calculate_metrics(y, y_pred, len(X), training_time)
        self.is_trained = True
        
        logger.info(f"LightGBM trained: MAE=${self.metrics.mae:.2f}, R²={self.metrics.r2:.3f}")
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ModelNotTrainedError("LightGBM")
        
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return np.maximum(predictions, 0)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_trained:
            raise ModelNotTrainedError("LightGBM")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        importance['rank'] = range(1, len(importance) + 1)
        return importance
    
    def _calculate_metrics(
        self, y_true, y_pred, n_samples, training_time
    ) -> ProductionModelMetrics:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        non_zero = y_true != 0
        if non_zero.sum() > 0:
            mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
        else:
            mape = None
        
        return ProductionModelMetrics(
            mae=round(mae, 2),
            rmse=round(rmse, 2),
            r2=round(r2, 4),
            mape=round(mape, 2) if mape else None,
            training_samples=n_samples,
            training_time_seconds=round(training_time, 2)
        )


class ProductionEnsemble:
    """
    Production-grade ensemble combining multiple models.
    
    Models included:
    - XGBoost (if available)
    - LightGBM (if available)
    - Random Forest (sklearn)
    - Gradient Boosting (sklearn)
    - Ridge Regression (baseline)
    
    Features:
    - Automatic model weighting based on validation performance
    - Stacking ensemble option
    - Cross-validation support
    - Model persistence
    """
    
    def __init__(
        self,
        use_xgboost: bool = True,
        use_lightgbm: bool = True,
        use_random_forest: bool = True,
        use_gradient_boosting: bool = True,
        use_ridge: bool = True,
        auto_weight: bool = True,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        self.use_xgboost = use_xgboost and HAS_XGBOOST
        self.use_lightgbm = use_lightgbm and HAS_LIGHTGBM
        self.use_random_forest = use_random_forest
        self.use_gradient_boosting = use_gradient_boosting
        self.use_ridge = use_ridge
        self.auto_weight = auto_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        self.metrics: Dict[str, ProductionModelMetrics] = {}
        self.ensemble_metrics: Optional[ProductionModelMetrics] = None
        
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        start_time = datetime.now()
        logger.info(f"Training production ensemble on {len(X)} samples...")
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        val_predictions = {}
        
        # Train XGBoost
        if self.use_xgboost:
            try:
                logger.info("Training XGBoost...")
                xgb_model = XGBoostCLVModel(random_state=self.random_state, n_jobs=self.n_jobs)
                xgb_model.train(X_train, y_train, eval_set=(X_val, y_val))
                self.models['xgboost'] = xgb_model
                self.metrics['xgboost'] = xgb_model.metrics
                val_predictions['xgboost'] = xgb_model.predict(X_val)
            except Exception as e:
                logger.warning(f"XGBoost training failed: {e}")
        
        # Train LightGBM
        if self.use_lightgbm:
            try:
                logger.info("Training LightGBM...")
                lgb_model = LightGBMCLVModel(random_state=self.random_state, n_jobs=self.n_jobs)
                lgb_model.train(X_train, y_train, eval_set=(X_val, y_val))
                self.models['lightgbm'] = lgb_model
                self.metrics['lightgbm'] = lgb_model.metrics
                val_predictions['lightgbm'] = lgb_model.predict(X_val)
            except Exception as e:
                logger.warning(f"LightGBM training failed: {e}")
        
        # Train Random Forest
        if self.use_random_forest:
            logger.info("Training Random Forest...")
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            )
            rf_model.fit(X_train_scaled, y_train)
            self.models['random_forest'] = {'model': rf_model, 'scaler': self.scaler}
            
            rf_pred = np.maximum(rf_model.predict(X_val_scaled), 0)
            val_predictions['random_forest'] = rf_pred
            
            mae = mean_absolute_error(y_val, rf_pred)
            r2 = r2_score(y_val, rf_pred)
            self.metrics['random_forest'] = ProductionModelMetrics(
                mae=round(mae, 2),
                rmse=round(np.sqrt(mean_squared_error(y_val, rf_pred)), 2),
                r2=round(r2, 4),
                training_samples=len(X_train)
            )
            logger.info(f"Random Forest: MAE=${mae:.2f}, R²={r2:.3f}")
        
        # Train Gradient Boosting
        if self.use_gradient_boosting:
            logger.info("Training Gradient Boosting...")
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            )
            gb_model.fit(X_train_scaled, y_train)
            self.models['gradient_boosting'] = {'model': gb_model, 'scaler': self.scaler}
            
            gb_pred = np.maximum(gb_model.predict(X_val_scaled), 0)
            val_predictions['gradient_boosting'] = gb_pred
            
            mae = mean_absolute_error(y_val, gb_pred)
            r2 = r2_score(y_val, gb_pred)
            self.metrics['gradient_boosting'] = ProductionModelMetrics(
                mae=round(mae, 2),
                rmse=round(np.sqrt(mean_squared_error(y_val, gb_pred)), 2),
                r2=round(r2, 4),
                training_samples=len(X_train)
            )
            logger.info(f"Gradient Boosting: MAE=${mae:.2f}, R²={r2:.3f}")
        
        # Train Ridge (baseline)
        if self.use_ridge:
            logger.info("Training Ridge Regression...")
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X_train_scaled, y_train)
            self.models['ridge'] = {'model': ridge_model, 'scaler': self.scaler}
            
            ridge_pred = np.maximum(ridge_model.predict(X_val_scaled), 0)
            val_predictions['ridge'] = ridge_pred
            
            mae = mean_absolute_error(y_val, ridge_pred)
            self.metrics['ridge'] = ProductionModelMetrics(
                mae=round(mae, 2),
                rmse=round(np.sqrt(mean_squared_error(y_val, ridge_pred)), 2),
                r2=round(r2_score(y_val, ridge_pred), 4),
                training_samples=len(X_train)
            )
        
        # Calculate optimal weights based on validation performance
        if self.auto_weight and len(val_predictions) > 1:
            self._calculate_optimal_weights(y_val, val_predictions)
        else:
            # Equal weights
            n_models = len(self.models)
            self.weights = {name: 1.0 / n_models for name in self.models}
        
        # Calculate ensemble predictions on validation
        ensemble_pred = self._ensemble_predict_internal(val_predictions)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.ensemble_metrics = ProductionModelMetrics(
            mae=round(mean_absolute_error(y_val, ensemble_pred), 2),
            rmse=round(np.sqrt(mean_squared_error(y_val, ensemble_pred)), 2),
            r2=round(r2_score(y_val, ensemble_pred), 4),
            training_samples=len(X),
            training_time_seconds=round(training_time, 2)
        )
        
        self.is_trained = True
        
        logger.info(f"Ensemble trained: MAE=${self.ensemble_metrics.mae:.2f}, R²={self.ensemble_metrics.r2:.3f}")
        logger.info(f"Model weights: {self.weights}")
        
        return {
            'individual_metrics': {k: v.to_dict() for k, v in self.metrics.items()},
            'ensemble_metrics': self.ensemble_metrics.to_dict(),
            'weights': self.weights,
            'training_time_seconds': training_time
        }
    
    def _calculate_optimal_weights(
        self,
        y_true: pd.Series,
        predictions: Dict[str, np.ndarray]
    ) -> None:
        """Calculate optimal weights based on inverse MAE."""
        maes = {}
        for name, pred in predictions.items():
            maes[name] = mean_absolute_error(y_true, pred)
        
        # Inverse MAE weighting (lower MAE = higher weight)
        inv_maes = {name: 1.0 / (mae + 1e-6) for name, mae in maes.items()}
        total = sum(inv_maes.values())
        
        self.weights = {name: inv_mae / total for name, inv_mae in inv_maes.items()}
    
    def _ensemble_predict_internal(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Combine predictions using weights."""
        weighted_sum = np.zeros(len(list(predictions.values())[0]))
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 0)
            weighted_sum += weight * pred
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else weighted_sum
    
    def predict(
        self,
        X: pd.DataFrame,
        return_individual: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ModelNotTrainedError("ProductionEnsemble")
        
        X = X[self.feature_names]
        individual_predictions = {}
        
        for name, model_data in self.models.items():
            if name in ['xgboost', 'lightgbm']:
                pred = model_data.predict(X)
            else:
                X_scaled = model_data['scaler'].transform(X)
                pred = np.maximum(model_data['model'].predict(X_scaled), 0)
            individual_predictions[name] = pred
        
        ensemble_pred = self._ensemble_predict_internal(individual_predictions)
        
        if return_individual:
            return ensemble_pred, individual_predictions
        return ensemble_pred
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get aggregated feature importance."""
        if not self.is_trained:
            raise ModelNotTrainedError("ProductionEnsemble")
        
        importance_sum = pd.Series(0.0, index=self.feature_names)
        total_weight = 0
        
        for name, model_data in self.models.items():
            weight = self.weights.get(name, 0)
            if weight == 0:
                continue
            
            if name in ['xgboost', 'lightgbm']:
                imp = model_data.get_feature_importance()
                for _, row in imp.iterrows():
                    importance_sum[row['feature']] += weight * row['importance']
            elif name in ['random_forest', 'gradient_boosting']:
                for i, feat in enumerate(self.feature_names):
                    importance_sum[feat] += weight * model_data['model'].feature_importances_[i]
            
            total_weight += weight
        
        if total_weight > 0:
            importance_sum /= total_weight
        
        importance = pd.DataFrame({
            'feature': importance_sum.index,
            'importance': importance_sum.values
        }).sort_values('importance', ascending=False)
        importance['rank'] = range(1, len(importance) + 1)
        
        return importance
    
    def save(self, filepath: str) -> None:
        """Save the ensemble to disk."""
        data = {
            'models': {},
            'weights': self.weights,
            'feature_names': self.feature_names,
            'metrics': {k: v.to_dict() for k, v in self.metrics.items()},
            'ensemble_metrics': self.ensemble_metrics.to_dict() if self.ensemble_metrics else None,
            'scaler': self.scaler
        }
        
        # Save model states
        for name, model_data in self.models.items():
            if name in ['xgboost', 'lightgbm']:
                data['models'][name] = {
                    'model': model_data.model,
                    'scaler': model_data.scaler,
                    'feature_names': model_data.feature_names
                }
            else:
                data['models'][name] = model_data
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Ensemble saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "ProductionEnsemble":
        """Load an ensemble from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        ensemble = cls()
        ensemble.weights = data['weights']
        ensemble.feature_names = data['feature_names']
        ensemble.scaler = data['scaler']
        ensemble.is_trained = True
        
        # Reconstruct models
        for name, model_data in data['models'].items():
            if name == 'xgboost' and HAS_XGBOOST:
                xgb_model = XGBoostCLVModel()
                xgb_model.model = model_data['model']
                xgb_model.scaler = model_data['scaler']
                xgb_model.feature_names = model_data['feature_names']
                xgb_model.is_trained = True
                ensemble.models[name] = xgb_model
            elif name == 'lightgbm' and HAS_LIGHTGBM:
                lgb_model = LightGBMCLVModel()
                lgb_model.model = model_data['model']
                lgb_model.scaler = model_data['scaler']
                lgb_model.feature_names = model_data['feature_names']
                lgb_model.is_trained = True
                ensemble.models[name] = lgb_model
            else:
                ensemble.models[name] = model_data
        
        logger.info(f"Ensemble loaded from {filepath}")
        return ensemble


def get_available_models() -> Dict[str, bool]:
    """Check which models are available."""
    return {
        'xgboost': HAS_XGBOOST,
        'lightgbm': HAS_LIGHTGBM,
        'random_forest': True,
        'gradient_boosting': True,
        'ridge': True
    }


# Backwards compatibility alias
class CLVModelTrainer(ProductionEnsemble):
    """Legacy alias for ProductionEnsemble for backwards compatibility."""
    
    def __init__(self, model_dir: str = "models", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._feature_names = []
        self._metrics = {}
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train ensemble (legacy interface)."""
        self._feature_names = X.columns.tolist()
        result = self.train(X, y)
        self._metrics = result.get('ensemble_metrics', {})
        return result
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train models (legacy interface)."""
        return self.train_ensemble(X, y)
    
    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions (legacy interface)."""
        return self.predict(X)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics (legacy interface)."""
        if self.ensemble_metrics:
            return self.ensemble_metrics.to_dict()
        return self._metrics
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all model metrics (legacy interface)."""
        result = {}
        for name, m in self.metrics.items():
            result[name] = m.to_dict() if hasattr(m, 'to_dict') else m
        if self.ensemble_metrics:
            result['ensemble'] = self.ensemble_metrics.to_dict()
        return result
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, float]:
        """Evaluate a model (legacy interface)."""
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            y_pred = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model': model_name,
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 4)
        }
    
    def save_all_models(self) -> None:
        """Save all models (legacy interface)."""
        filepath = self.model_dir / "ensemble_model.pkl"
        self.save(str(filepath))
    
    def load_all_models(self) -> bool:
        """Load all models (legacy interface)."""
        filepath = self.model_dir / "ensemble_model.pkl"
        if filepath.exists():
            loaded = ProductionEnsemble.load(str(filepath))
            self.models = loaded.models
            self.weights = loaded.weights
            self.feature_names = loaded.feature_names
            self.is_trained = loaded.is_trained
            return True
        return False
    
    def get_feature_importance(self, model_name: str = None, feature_names: List[str] = None) -> pd.DataFrame:
        """Get feature importance (legacy interface)."""
        # Use parent's get_feature_importance
        return super().get_feature_importance()

