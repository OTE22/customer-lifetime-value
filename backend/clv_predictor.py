"""
CLV Predictor Module
Main interface for customer lifetime value predictions.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import logging

from .data_processor import DataProcessor
from .feature_engineering import AdvancedFeatureEngineer as FeatureEngineer
from .ml_models import CLVModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLVPredictor:
    """
    Main class for CLV prediction pipeline.
    Handles data processing, feature engineering, and prediction.
    """
    
    def __init__(
        self,
        model_dir: str = "models",
        data_path: Optional[str] = None
    ):
        """
        Initialize CLV Predictor.
        
        Args:
            model_dir: Directory containing trained models.
            data_path: Path to customer data CSV.
        """
        self.model_dir = Path(model_dir)
        self.data_path = data_path
        
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = CLVModelTrainer(str(self.model_dir))
        
        self.feature_columns = None
        self.is_trained = False
        
    def train(
        self,
        data_path: Optional[str] = None,
        test_size: float = 0.2,
        save_models: bool = True
    ) -> Dict[str, Any]:
        """
        Train CLV prediction models.
        
        Args:
            data_path: Path to training data CSV.
            test_size: Fraction for test set.
            save_models: Whether to save trained models.
            
        Returns:
            Dictionary with training results and metrics.
        """
        data_path = data_path or self.data_path
        if not data_path:
            raise ValueError("No data path provided")
        
        logger.info("Starting CLV model training pipeline...")
        
        # Load and process data
        df = self.data_processor.load_data(data_path)
        df = self.data_processor.clean_data(df)
        
        # Engineer features
        df_features = self.feature_engineer.fit_transform(df)
        
        # Prepare training data
        X, y = self.data_processor.preprocess_for_training(df_features)
        self.feature_columns = X.columns.tolist()
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train ensemble
        self.model_trainer.train_ensemble(X_train, y_train)
        
        # Evaluate all models
        metrics = {}
        for model_name in ['random_forest', 'gradient_boosting', 'linear', 'ensemble']:
            model = self.model_trainer.models.get(model_name)
            if model_name == 'ensemble':
                metrics[model_name] = self.model_trainer.evaluate_model(
                    model, X_test, y_test, model_name
                )
            elif model:
                metrics[model_name] = self.model_trainer.evaluate_model(
                    model, X_test, y_test, model_name
                )
        
        # Get feature importance
        feature_importance = self.model_trainer.get_feature_importance(
            'random_forest',
            self.feature_columns
        )
        
        # Save models if requested
        if save_models:
            self.model_trainer.save_all_models()
        
        self.is_trained = True
        
        results = {
            'metrics': metrics,
            'feature_importance': feature_importance.head(15).to_dict('records'),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': len(self.feature_columns)
        }
        
        logger.info("Training complete!")
        return results
    
    def predict_single(self, customer_data: Dict) -> Dict[str, Any]:
        """
        Predict CLV for a single customer.
        
        Args:
            customer_data: Dictionary with customer attributes.
            
        Returns:
            Dictionary with prediction and confidence.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Engineer features
        df_features = self.feature_engineer.fit_transform(df)
        
        # Preprocess
        X, _ = self.data_processor.preprocess_for_training(
            df_features,
            target_column='actual_clv' if 'actual_clv' in df_features.columns else 'total_spent'
        )
        
        # Align columns with training features
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[self.feature_columns]
        
        # Predict
        prediction = self.model_trainer.predict_ensemble(X)[0]
        
        # Determine segment
        segment = self._determine_segment(prediction)
        
        return {
            'predicted_clv': round(float(prediction), 2),
            'segment': segment,
            'confidence': self._calculate_confidence(prediction),
            'recommended_cac': round(float(prediction * 0.3), 2)
        }
    
    def predict_batch(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict CLV for multiple customers.
        
        Args:
            df: DataFrame with customer data.
            
        Returns:
            DataFrame with predictions added.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"Predicting CLV for {len(df)} customers...")
        
        # Engineer features
        df_features = self.feature_engineer.fit_transform(df.copy())
        
        # Preprocess
        target_col = 'actual_clv' if 'actual_clv' in df_features.columns else 'total_spent'
        X, _ = self.data_processor.preprocess_for_training(df_features, target_column=target_col)
        
        # Align columns
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        extra_cols = set(X.columns) - set(self.feature_columns)
        X = X.drop(columns=list(extra_cols), errors='ignore')
        X = X[self.feature_columns]
        
        # Predict
        predictions = self.model_trainer.predict_ensemble(X)
        
        # Add predictions to original DataFrame
        result = df.copy()
        result['predicted_clv'] = predictions.round(2)
        result['predicted_segment'] = result['predicted_clv'].apply(self._determine_segment)
        result['recommended_cac'] = (result['predicted_clv'] * 0.3).round(2)
        
        logger.info("Batch prediction complete")
        return result
    
    def segment_customers(
        self,
        df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Segment customers based on predicted CLV.
        
        Args:
            df: DataFrame with predictions.
            
        Returns:
            Dictionary mapping segment names to customer DataFrames.
        """
        if 'predicted_clv' not in df.columns:
            df = self.predict_batch(df)
        
        segments = {
            'High-CLV': df[df['predicted_segment'] == 'High-CLV'],
            'Growth-Potential': df[df['predicted_segment'] == 'Growth-Potential'],
            'Low-CLV': df[df['predicted_segment'] == 'Low-CLV']
        }
        
        logger.info(f"Segmented customers - High: {len(segments['High-CLV'])}, "
                   f"Growth: {len(segments['Growth-Potential'])}, "
                   f"Low: {len(segments['Low-CLV'])}")
        
        return segments
    
    def get_segment_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics by customer segment.
        
        Args:
            df: DataFrame with predictions.
            
        Returns:
            Dictionary with segment statistics.
        """
        if 'predicted_clv' not in df.columns:
            df = self.predict_batch(df)
        
        summary = {}
        for segment in ['High-CLV', 'Growth-Potential', 'Low-CLV']:
            segment_df = df[df['predicted_segment'] == segment]
            summary[segment] = {
                'count': len(segment_df),
                'percentage': round(len(segment_df) / len(df) * 100, 2),
                'avg_predicted_clv': round(segment_df['predicted_clv'].mean(), 2),
                'total_predicted_value': round(segment_df['predicted_clv'].sum(), 2),
                'avg_recommended_cac': round(segment_df['recommended_cac'].mean(), 2)
            }
        
        return summary
    
    def get_all_predictions(self) -> pd.DataFrame:
        """
        Get all customer predictions.
        
        Returns:
            DataFrame with all customers and their predictions.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Load data if not already loaded
        if self.data_processor.raw_data is None:
            if self.data_path:
                self.data_processor.load_data(self.data_path)
            else:
                raise ValueError("No data loaded. Provide data_path.")
        
        return self.predict_batch(self.data_processor.raw_data)
    
    def get_segment_analysis(self) -> Dict[str, Any]:
        """
        Get segment analysis with statistics.
        
        Returns:
            Dictionary with segment statistics.
        """
        df = self.get_all_predictions()
        summary = self.get_segment_summary(df)
        
        return {
            'segments': summary,
            'total_customers': len(df),
            'total_predicted_value': df['predicted_clv'].sum()
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        importance = self.model_trainer.get_feature_importance(
            'random_forest',
            self.feature_columns
        )
        
        return importance
    
    def _determine_segment(self, clv: float) -> str:
        """Determine customer segment based on predicted CLV."""
        if clv >= 500:
            return 'High-CLV'
        elif clv >= 150:
            return 'Growth-Potential'
        else:
            return 'Low-CLV'
    
    def _calculate_confidence(self, prediction: float) -> str:
        """Calculate prediction confidence level."""
        if prediction > 1000:
            return 'High'
        elif prediction > 300:
            return 'Medium'
        else:
            return 'Low'
    
    def predict_with_confidence(
        self,
        customer_data: Dict[str, Any],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Predict CLV with confidence intervals and uncertainty estimation.
        
        Uses ensemble variance to estimate prediction uncertainty.
        
        Args:
            customer_data: Dictionary with customer attributes.
            confidence_level: Confidence level for intervals (0.90, 0.95, 0.99)
            
        Returns:
            Dictionary with prediction, intervals, and uncertainty metrics.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get base prediction
        base_result = self.predict_single(customer_data)
        prediction = base_result['predicted_clv']
        
        # Get individual model predictions for variance estimation
        df = pd.DataFrame([customer_data])
        df_features = self.feature_engineer.fit_transform(df)
        
        target_col = 'actual_clv' if 'actual_clv' in df_features.columns else 'total_spent'
        X, _ = self.data_processor.preprocess_for_training(df_features, target_column=target_col)
        
        # Align columns
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[self.feature_columns]
        
        # Get predictions from each model
        model_predictions = []
        for model_name in ['random_forest', 'gradient_boosting', 'linear']:
            model = self.model_trainer.models.get(model_name)
            if model is not None:
                try:
                    pred = model.predict(X)[0]
                    model_predictions.append(pred)
                except Exception:
                    pass
        
        # Calculate uncertainty metrics
        if len(model_predictions) >= 2:
            std_dev = np.std(model_predictions)
            variance = np.var(model_predictions)
            
            # Confidence interval based on model disagreement
            # Using t-distribution multiplier approximation
            z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            z = z_scores.get(confidence_level, 1.96)
            
            margin = z * std_dev
            lower_bound = max(0, prediction - margin)
            upper_bound = prediction + margin
            
            # Uncertainty score (0-1, lower is more certain)
            cv = std_dev / abs(prediction) if prediction != 0 else 1
            uncertainty_score = min(cv, 1.0)
            
            # Confidence category based on model agreement
            if cv < 0.1:
                confidence_category = 'Very High'
            elif cv < 0.2:
                confidence_category = 'High'
            elif cv < 0.4:
                confidence_category = 'Medium'
            else:
                confidence_category = 'Low'
            
        else:
            # Fallback: estimate based on prediction magnitude
            std_dev = prediction * 0.25  # 25% default uncertainty
            lower_bound = max(0, prediction * 0.75)
            upper_bound = prediction * 1.25
            uncertainty_score = 0.5
            confidence_category = 'Medium'
            variance = std_dev ** 2
        
        return {
            **base_result,
            'confidence_interval': {
                'level': confidence_level,
                'lower': round(float(lower_bound), 2),
                'upper': round(float(upper_bound), 2),
                'margin': round(float(upper_bound - prediction), 2)
            },
            'uncertainty': {
                'score': round(float(uncertainty_score), 4),
                'category': confidence_category,
                'std_dev': round(float(std_dev), 2),
                'variance': round(float(variance), 2)
            },
            'model_predictions': {
                'count': len(model_predictions),
                'min': round(float(min(model_predictions)), 2) if model_predictions else None,
                'max': round(float(max(model_predictions)), 2) if model_predictions else None,
                'spread': round(float(max(model_predictions) - min(model_predictions)), 2) if len(model_predictions) >= 2 else None
            }
        }
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get all model performance metrics."""
        return self.model_trainer.get_all_metrics()


def create_predictor_from_data(data_path: str) -> CLVPredictor:
    """
    Convenience function to create and train a predictor.
    
    Args:
        data_path: Path to customer data CSV.
        
    Returns:
        Trained CLVPredictor instance.
    """
    predictor = CLVPredictor(data_path=data_path)
    predictor.train()
    return predictor
