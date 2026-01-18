"""
MLflow Configuration for CLV Prediction System
Provides experiment tracking, model versioning, and model registry.
"""
import os
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class MLflowConfig:
    """MLflow configuration and utilities."""
    
    # Default experiment name
    EXPERIMENT_NAME = "clv-prediction"
    
    # Model names
    MODEL_RANDOM_FOREST = "clv-random-forest"
    MODEL_XGBOOST = "clv-xgboost"
    MODEL_LIGHTGBM = "clv-lightgbm"
    MODEL_ENSEMBLE = "clv-ensemble"
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow configuration.
        
        Args:
            tracking_uri: MLflow tracking server URI (e.g., http://mlflow-server:5000)
            experiment_name: Name of the experiment
            artifact_location: S3/GCS path for artifacts
        """
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", 
            "http://localhost:5000"
        )
        self.experiment_name = experiment_name or os.getenv(
            "MLFLOW_EXPERIMENT_NAME",
            self.EXPERIMENT_NAME
        )
        self.artifact_location = artifact_location or os.getenv(
            "MLFLOW_ARTIFACT_LOCATION",
            None
        )
        self._client: Optional[MlflowClient] = None
        self._experiment_id: Optional[str] = None
    
    def setup(self) -> None:
        """Set up MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.info(f"MLflow tracking URI: {self.tracking_uri}")
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self._experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.artifact_location
                )
                logger.info(f"Created experiment: {self.experiment_name}")
            else:
                self._experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            self._client = MlflowClient()
            
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Tracking disabled.")
            self._experiment_id = None
    
    @property
    def is_enabled(self) -> bool:
        """Check if MLflow tracking is enabled."""
        return self._experiment_id is not None
    
    @property
    def client(self) -> Optional[MlflowClient]:
        """Get MLflow client."""
        return self._client
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Additional tags for the run
            description: Run description
            
        Returns:
            MLflow run context
        """
        if not self.is_enabled:
            return None
        
        run_tags = {"project": "clv-prediction"}
        if tags:
            run_tags.update(tags)
        
        return mlflow.start_run(
            run_name=run_name,
            tags=run_tags,
            description=description
        )
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if self.is_enabled:
            mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        if self.is_enabled:
            mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model,
        model_name: str,
        registered_name: Optional[str] = None,
        signature=None,
        input_example=None
    ) -> Optional[str]:
        """
        Log and optionally register a model.
        
        Args:
            model: The model to log
            model_name: Artifact path name
            registered_name: Model registry name (if registering)
            signature: Model signature
            input_example: Example input data
            
        Returns:
            Model URI if logged, None otherwise
        """
        if not self.is_enabled:
            return None
        
        try:
            # Determine model type and log appropriately
            model_type = type(model).__module__.split('.')[0]
            
            if model_type == 'sklearn':
                mlflow.sklearn.log_model(
                    model, 
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_name
                )
            elif model_type == 'xgboost':
                mlflow.xgboost.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_name
                )
            elif model_type == 'lightgbm':
                mlflow.lightgbm.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_name
                )
            else:
                mlflow.sklearn.log_model(
                    model,
                    model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_name
                )
            
            run = mlflow.active_run()
            model_uri = f"runs:/{run.info.run_id}/{model_name}"
            logger.info(f"Logged model: {model_name} -> {model_uri}")
            return model_uri
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            return None
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact file."""
        if self.is_enabled:
            mlflow.log_artifact(local_path, artifact_path)
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as JSON artifact."""
        if self.is_enabled:
            mlflow.log_dict(dictionary, artifact_file)
    
    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run."""
        if self.is_enabled:
            mlflow.set_tag(key, value)
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        if self.is_enabled:
            mlflow.end_run(status=status)
    
    def get_best_model(
        self,
        model_name: str,
        metric: str = "rmse",
        ascending: bool = True
    ) -> Optional[str]:
        """
        Get the best model version by metric.
        
        Args:
            model_name: Registered model name
            metric: Metric to compare
            ascending: True for min metric, False for max
            
        Returns:
            Model URI or None
        """
        if not self._client:
            return None
        
        try:
            versions = self._client.search_model_versions(f"name='{model_name}'")
            if not versions:
                return None
            
            best_version = None
            best_metric = None
            
            for version in versions:
                run = self._client.get_run(version.run_id)
                value = run.data.metrics.get(metric)
                
                if value is not None:
                    if best_metric is None:
                        best_metric = value
                        best_version = version
                    elif ascending and value < best_metric:
                        best_metric = value
                        best_version = version
                    elif not ascending and value > best_metric:
                        best_metric = value
                        best_version = version
            
            if best_version:
                return f"models:/{model_name}/{best_version.version}"
            
        except Exception as e:
            logger.error(f"Error getting best model: {e}")
        
        return None
    
    def transition_model_stage(
        self,
        model_name: str,
        version: int,
        stage: str = "Production"
    ) -> bool:
        """
        Transition a model version to a new stage.
        
        Args:
            model_name: Registered model name
            version: Model version number
            stage: Target stage (Staging, Production, Archived)
            
        Returns:
            True if successful
        """
        if not self._client:
            return False
        
        try:
            self._client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            return True
        except Exception as e:
            logger.error(f"Failed to transition model: {e}")
            return False


def mlflow_track(experiment_name: Optional[str] = None):
    """
    Decorator to automatically track function execution with MLflow.
    
    Usage:
        @mlflow_track("my-experiment")
        def train_model(X, y):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = MLflowConfig(experiment_name=experiment_name)
            config.setup()
            
            with config.start_run(run_name=func.__name__):
                result = func(*args, **kwargs)
                
                # If result is a dict with metrics, log them
                if isinstance(result, dict):
                    metrics = {k: v for k, v in result.items() if isinstance(v, (int, float))}
                    if metrics:
                        config.log_metrics(metrics)
                
                return result
        return wrapper
    return decorator


# Global instance
_mlflow_config: Optional[MLflowConfig] = None


def get_mlflow_config() -> MLflowConfig:
    """Get or create global MLflow configuration."""
    global _mlflow_config
    if _mlflow_config is None:
        _mlflow_config = MLflowConfig()
        _mlflow_config.setup()
    return _mlflow_config


def init_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None
) -> MLflowConfig:
    """
    Initialize MLflow with custom settings.
    
    Args:
        tracking_uri: MLflow tracking server URI
        experiment_name: Experiment name
        
    Returns:
        MLflowConfig instance
    """
    global _mlflow_config
    _mlflow_config = MLflowConfig(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )
    _mlflow_config.setup()
    return _mlflow_config
