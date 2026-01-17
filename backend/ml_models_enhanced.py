"""
Enhanced ML Models for CLV Prediction System
Production-ready model training with versioning, validation, and advanced features.
"""

import pickle
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings

from .config import get_config, ModelConfig
from .logging_config import get_logger, LogMetrics
from .exceptions import (
    ModelNotTrainedError, ModelTrainingError, ModelPredictionError,
    FeatureMismatchError, ModelNotFoundError, ModelError
)
from .cache import cached, get_cache

logger = get_logger(__name__)
metrics_logger = LogMetrics(logger)


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    model_name: str
    version: str
    trained_at: datetime
    samples_trained: int
    feature_names: List[str]
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    config_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['trained_at'] = self.trained_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        data['trained_at'] = datetime.fromisoformat(data['trained_at'])
        return cls(**data)


@dataclass
class ModelVersion:
    """Represents a model version with its artifacts."""
    version: str
    model_path: Path
    metadata_path: Path
    is_active: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


class BaseModelWrapper:
    """Base wrapper for ML models with common functionality."""
    
    def __init__(self, model_name: str, config: Optional[ModelConfig] = None):
        self.model_name = model_name
        self.config = config or get_config().model
        self.model = None
        self.scaler = None
        self.metadata: Optional[ModelMetadata] = None
        self.is_trained = False
        self._feature_names: List[str] = []
        
    def _validate_features(self, X: pd.DataFrame) -> None:
        """Validate input features match expected features."""
        if not self._feature_names:
            return
            
        missing = set(self._feature_names) - set(X.columns)
        extra = set(X.columns) - set(self._feature_names)
        
        if missing:
            raise FeatureMismatchError(
                expected_features=len(self._feature_names),
                received_features=len(X.columns),
                missing=list(missing)
            )
    
    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare and validate features for prediction."""
        self._validate_features(X)
        
        # Ensure column order matches training
        if self._feature_names:
            X = X[self._feature_names]
        
        # Scale features
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X.values
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate model evaluation metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (handle zero values)
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = None
        
        metrics = {
            'mae': round(mae, 4),
            'mse': round(mse, 4),
            'rmse': round(rmse, 4),
            'r2': round(r2, 4),
        }
        
        if mape is not None:
            metrics['mape'] = round(mape, 4)
        
        return metrics
    
    def _generate_version(self) -> str:
        """Generate a version string based on current timestamp."""
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    def _generate_config_hash(self) -> str:
        """Generate hash of current configuration."""
        config_str = json.dumps(asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else {}, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class EnhancedRandomForest(BaseModelWrapper):
    """Enhanced Random Forest model with hyperparameter tuning."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__("RandomForest", config)
        
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune_hyperparameters: bool = False
    ) -> Dict[str, float]:
        """Train the Random Forest model."""
        start_time = datetime.utcnow()
        
        try:
            self._feature_names = X.columns.tolist()
            
            # Initialize scaler
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            if tune_hyperparameters:
                self.model = self._tune_hyperparameters(X_scaled, y)
            else:
                self.model = RandomForestRegressor(
                    n_estimators=self.config.random_forest_estimators,
                    max_depth=self.config.random_forest_max_depth,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=self.config.random_state
                )
                self.model.fit(X_scaled, y)
            
            # Calculate metrics using cross-validation
            cv_scores = cross_val_score(
                self.model, X_scaled, y,
                cv=5, scoring='neg_mean_absolute_error'
            )
            
            # Get predictions for final metrics
            y_pred = self.model.predict(X_scaled)
            metrics = self._calculate_metrics(y.values, y_pred)
            metrics['cv_mae_mean'] = round(-cv_scores.mean(), 4)
            metrics['cv_mae_std'] = round(cv_scores.std(), 4)
            
            # Create metadata
            self.metadata = ModelMetadata(
                model_name=self.model_name,
                version=self._generate_version(),
                trained_at=start_time,
                samples_trained=len(X),
                feature_names=self._feature_names,
                hyperparameters=self.model.get_params(),
                metrics=metrics,
                config_hash=self._generate_config_hash()
            )
            
            self.is_trained = True
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics_logger.log_model_training(
                self.model_name, len(X), metrics, duration
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise ModelTrainingError(self.model_name, cause=e)
    
    def _tune_hyperparameters(
        self,
        X: np.ndarray,
        y: pd.Series
    ) -> RandomForestRegressor:
        """Tune hyperparameters using GridSearchCV."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        base_model = RandomForestRegressor(
            n_jobs=-1,
            random_state=self.config.random_state
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            grid_search.fit(X, y)
        
        logger.info(f"Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_trained:
            raise ModelNotTrainedError(self.model_name)
        
        try:
            X_prepared = self._prepare_features(X)
            predictions = self.model.predict(X_prepared)
            return np.maximum(predictions, 0)  # CLV cannot be negative
        except Exception as e:
            raise ModelPredictionError(self.model_name, cause=e)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance rankings."""
        if not self.is_trained:
            raise ModelNotTrainedError(self.model_name)
        
        importance = pd.DataFrame({
            'feature': self._feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance['rank'] = range(1, len(importance) + 1)
        return importance


class EnhancedGradientBoosting(BaseModelWrapper):
    """Enhanced Gradient Boosting model with early stopping."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__("GradientBoosting", config)
        
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_fraction: float = 0.1
    ) -> Dict[str, float]:
        """Train the Gradient Boosting model with early stopping."""
        start_time = datetime.utcnow()
        
        try:
            self._feature_names = X.columns.tolist()
            
            # Initialize scaler
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.gradient_boosting_estimators,
                learning_rate=self.config.gradient_boosting_learning_rate,
                max_depth=self.config.gradient_boosting_max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                validation_fraction=validation_fraction,
                n_iter_no_change=10,
                random_state=self.config.random_state
            )
            
            self.model.fit(X_scaled, y)
            
            # Get predictions and metrics
            y_pred = self.model.predict(X_scaled)
            metrics = self._calculate_metrics(y.values, y_pred)
            metrics['n_estimators_used'] = self.model.n_estimators_
            
            # Create metadata
            self.metadata = ModelMetadata(
                model_name=self.model_name,
                version=self._generate_version(),
                trained_at=start_time,
                samples_trained=len(X),
                feature_names=self._feature_names,
                hyperparameters=self.model.get_params(),
                metrics=metrics,
                config_hash=self._generate_config_hash()
            )
            
            self.is_trained = True
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics_logger.log_model_training(
                self.model_name, len(X), metrics, duration
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise ModelTrainingError(self.model_name, cause=e)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_trained:
            raise ModelNotTrainedError(self.model_name)
        
        try:
            X_prepared = self._prepare_features(X)
            predictions = self.model.predict(X_prepared)
            return np.maximum(predictions, 0)
        except Exception as e:
            raise ModelPredictionError(self.model_name, cause=e)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance rankings."""
        if not self.is_trained:
            raise ModelNotTrainedError(self.model_name)
        
        importance = pd.DataFrame({
            'feature': self._feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance['rank'] = range(1, len(importance) + 1)
        return importance


class EnhancedEnsemble(BaseModelWrapper):
    """Production ensemble model combining multiple base models."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__("Ensemble", config)
        self.models: Dict[str, BaseModelWrapper] = {}
        self.weights = self.config.ensemble_weights
        
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune_hyperparameters: bool = False
    ) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        start_time = datetime.utcnow()
        
        try:
            self._feature_names = X.columns.tolist()
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            all_metrics = {}
            
            # Train Random Forest
            rf_model = EnhancedRandomForest(self.config)
            rf_metrics = rf_model.train(X_train, y_train, tune_hyperparameters)
            self.models['random_forest'] = rf_model
            all_metrics['random_forest'] = rf_metrics
            
            # Train Gradient Boosting
            gb_model = EnhancedGradientBoosting(self.config)
            gb_metrics = gb_model.train(X_train, y_train)
            self.models['gradient_boosting'] = gb_model
            all_metrics['gradient_boosting'] = gb_metrics
            
            # Train Ridge Regression (linear baseline)
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X_train)
            
            linear_model = Ridge(alpha=1.0)
            linear_model.fit(X_scaled, y_train)
            
            # Wrap linear model
            class LinearWrapper(BaseModelWrapper):
                def __init__(self, model, scaler, feature_names):
                    super().__init__("Linear", None)
                    self.model = model
                    self.scaler = scaler
                    self._feature_names = feature_names
                    self.is_trained = True
                
                def predict(self, X):
                    X_prepared = self.scaler.transform(X[self._feature_names])
                    return np.maximum(self.model.predict(X_prepared), 0)
            
            self.models['linear'] = LinearWrapper(linear_model, self.scaler, self._feature_names)
            
            # Validate on held-out data
            val_predictions = self.predict(X_val)
            val_metrics = self._calculate_metrics(y_val.values, val_predictions)
            all_metrics['ensemble_validation'] = val_metrics
            
            # Create metadata
            self.metadata = ModelMetadata(
                model_name=self.model_name,
                version=self._generate_version(),
                trained_at=start_time,
                samples_trained=len(X),
                feature_names=self._feature_names,
                hyperparameters={'weights': self.weights},
                metrics=val_metrics,
                config_hash=self._generate_config_hash()
            )
            
            self.is_trained = True
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            metrics_logger.log_model_training(
                self.model_name, len(X), val_metrics, duration
            )
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            raise ModelTrainingError(self.model_name, cause=e)
    
    def predict(
        self,
        X: pd.DataFrame,
        return_individual: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ModelNotTrainedError(self.model_name)
        
        try:
            individual_predictions = {}
            weighted_sum = np.zeros(len(X))
            
            for model_name, model in self.models.items():
                weight = self.weights.get(model_name, 0)
                if weight > 0:
                    preds = model.predict(X)
                    individual_predictions[model_name] = preds
                    weighted_sum += weight * preds
            
            ensemble_predictions = weighted_sum / sum(self.weights.values())
            ensemble_predictions = np.maximum(ensemble_predictions, 0)
            
            if return_individual:
                return ensemble_predictions, individual_predictions
            return ensemble_predictions
            
        except Exception as e:
            raise ModelPredictionError(self.model_name, cause=e)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get aggregated feature importance from all models."""
        if not self.is_trained:
            raise ModelNotTrainedError(self.model_name)
        
        importance_sum = pd.Series(0, index=self._feature_names, dtype=float)
        total_weight = 0
        
        for model_name, model in self.models.items():
            weight = self.weights.get(model_name, 0)
            if weight > 0 and hasattr(model, 'get_feature_importance'):
                try:
                    model_importance = model.get_feature_importance()
                    for _, row in model_importance.iterrows():
                        importance_sum[row['feature']] += weight * row['importance']
                    total_weight += weight
                except:
                    pass
        
        if total_weight > 0:
            importance_sum /= total_weight
        
        importance = pd.DataFrame({
            'feature': importance_sum.index,
            'importance': importance_sum.values
        }).sort_values('importance', ascending=False)
        
        importance['rank'] = range(1, len(importance) + 1)
        return importance


class ModelRegistry:
    """Registry for managing model versions."""
    
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or get_config().model.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._versions: Dict[str, List[ModelVersion]] = {}
        
    def save_model(self, model: BaseModelWrapper) -> str:
        """Save a trained model and return version string."""
        if not model.is_trained or model.metadata is None:
            raise ModelNotTrainedError(model.model_name)
        
        version = model.metadata.version
        model_name = model.model_name.lower()
        
        # Create version directory
        version_dir = self.model_dir / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = version_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model.model,
                'scaler': model.scaler,
                'feature_names': model._feature_names,
                'models': getattr(model, 'models', None),
                'weights': getattr(model, 'weights', None)
            }, f)
        
        # Save metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model.metadata.to_dict(), f, indent=2)
        
        # Register version
        if model_name not in self._versions:
            self._versions[model_name] = []
        
        self._versions[model_name].append(ModelVersion(
            version=version,
            model_path=model_path,
            metadata_path=metadata_path,
            is_active=True,
            created_at=model.metadata.trained_at
        ))
        
        logger.info(f"Saved {model_name} version {version}")
        return version
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> BaseModelWrapper:
        """Load a model by name and optional version."""
        model_name = model_name.lower()
        model_dir = self.model_dir / model_name
        
        if not model_dir.exists():
            raise ModelNotFoundError(str(model_dir))
        
        # Find version
        if version is None:
            versions = sorted(model_dir.iterdir(), reverse=True)
            if not versions:
                raise ModelNotFoundError(str(model_dir))
            version = versions[0].name
        
        version_dir = model_dir / version
        model_path = version_dir / "model.pkl"
        metadata_path = version_dir / "metadata.json"
        
        if not model_path.exists():
            raise ModelNotFoundError(str(model_path))
        
        # Load model artifacts
        with open(model_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # Reconstruct model wrapper
        if 'ensemble' in model_name:
            model = EnhancedEnsemble()
            model.models = artifacts.get('models', {})
            model.weights = artifacts.get('weights', {})
        elif 'random' in model_name or 'forest' in model_name:
            model = EnhancedRandomForest()
        elif 'gradient' in model_name or 'boosting' in model_name:
            model = EnhancedGradientBoosting()
        else:
            model = BaseModelWrapper(model_name)
        
        model.model = artifacts['model']
        model.scaler = artifacts['scaler']
        model._feature_names = artifacts['feature_names']
        model.metadata = ModelMetadata.from_dict(metadata_dict)
        model.is_trained = True
        
        logger.info(f"Loaded {model_name} version {version}")
        return model
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model."""
        model_name = model_name.lower()
        model_dir = self.model_dir / model_name
        
        if not model_dir.exists():
            return []
        
        versions = []
        for version_dir in sorted(model_dir.iterdir(), reverse=True):
            metadata_path = version_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                versions.append({
                    'version': version_dir.name,
                    'trained_at': metadata.get('trained_at'),
                    'metrics': metadata.get('metrics', {})
                })
        
        return versions
