"""
Unit Tests for CLV Prediction System
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data_processor import DataProcessor
from backend.feature_engineering import AdvancedFeatureEngineer as FeatureEngineer
from backend.meta_ads_integration import MetaAdsIntegration
from backend.ml_models import CLVModelTrainer


class TestDataProcessor:
    """Tests for DataProcessor module."""

    @pytest.fixture
    def sample_data(self):
        """Create sample customer data."""
        return pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003", "C004", "C005"],
                "total_orders": [5, 2, 10, 1, 7],
                "total_spent": [500.0, 100.0, 1500.0, 50.0, 800.0],
                "avg_order_value": [100.0, 50.0, 150.0, 50.0, 114.28],
                "days_since_first_purchase": [180, 30, 365, 7, 200],
                "days_since_last_purchase": [10, 20, 5, 7, 15],
                "acquisition_source": ["Meta Ads", "Google Ads", "Email", "Direct", "Meta Ads"],
                "campaign_type": ["Prospecting", "Retargeting", "None", "None", "Brand"],
                "email_engagement_rate": [0.7, 0.3, 0.9, 0.1, 0.6],
                "return_rate": [0.05, 0.15, 0.02, 0.30, 0.08],
                "num_categories": [3, 1, 5, 1, 4],
                "acquisition_cost": [45.0, 50.0, 5.0, 0.0, 40.0],
                "customer_segment": [
                    "High-CLV",
                    "Low-CLV",
                    "High-CLV",
                    "Low-CLV",
                    "Growth-Potential",
                ],
                "actual_clv": [850.0, 120.0, 2500.0, 55.0, 1100.0],
            }
        )

    def test_clean_data(self, sample_data):
        """Test data cleaning."""
        processor = DataProcessor()
        processor.raw_data = sample_data
        cleaned = processor.clean_data()

        assert len(cleaned) == 5
        assert cleaned["return_rate"].max() <= 1.0
        assert cleaned["return_rate"].min() >= 0.0

    def test_preprocess_for_training(self, sample_data):
        """Test preprocessing for ML training."""
        processor = DataProcessor()
        processor.raw_data = sample_data
        processor.clean_data()

        X, y = processor.preprocess_for_training()

        assert len(X) == len(y)
        assert "customer_id" not in X.columns
        assert "actual_clv" not in X.columns


class TestFeatureEngineering:
    """Tests for FeatureEngineer module."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for feature engineering."""
        return pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003"],
                "total_orders": [5, 2, 10],
                "total_spent": [500.0, 100.0, 1500.0],
                "avg_order_value": [100.0, 50.0, 150.0],
                "days_since_first_purchase": [180, 30, 365],
                "days_since_last_purchase": [10, 60, 5],
                "email_engagement_rate": [0.7, 0.3, 0.9],
                "return_rate": [0.05, 0.15, 0.02],
                "num_categories": [3, 1, 5],
                "acquisition_source": ["Meta Ads", "Google Ads", "Email"],
                "campaign_type": ["Prospecting", "Retargeting", "None"],
            }
        )

    def test_calculate_rfm_scores(self, sample_data):
        """Test RFM score calculation."""
        engineer = FeatureEngineer()
        result = engineer.calculate_rfm_scores(sample_data)

        assert "recency_score" in result.columns
        assert "frequency_score" in result.columns
        assert "monetary_score" in result.columns
        assert "rfm_score" in result.columns

    def test_create_behavioral_features(self, sample_data):
        """Test behavioral feature creation."""
        engineer = FeatureEngineer()
        result = engineer.create_behavioral_features(sample_data)

        assert "purchase_velocity" in result.columns
        assert "engagement_score" in result.columns

    def test_fit_transform(self, sample_data):
        """Test full feature matrix preparation."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_data)

        assert len(result) == 3
        assert "rfm_score" in result.columns


class TestMLModels:
    """Tests for ML model training."""

    @pytest.fixture
    def training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame(
            {
                "total_orders": np.random.randint(1, 20, n_samples),
                "total_spent": np.random.uniform(50, 2000, n_samples),
                "avg_order_value": np.random.uniform(25, 200, n_samples),
                "days_since_last_purchase": np.random.randint(1, 180, n_samples),
                "email_engagement_rate": np.random.uniform(0, 1, n_samples),
                "return_rate": np.random.uniform(0, 0.3, n_samples),
            }
        )

        y = X["total_spent"] * 1.5 + np.random.normal(0, 100, n_samples)
        y = y.clip(lower=0)

        return X, pd.Series(y)

    def test_train_ensemble_creates_models(self, training_data):
        """Test that ensemble training creates the expected models."""
        X, y = training_data
        trainer = CLVModelTrainer()

        trainer.train_ensemble(X, y)

        # Check models are created
        assert "random_forest" in trainer.models
        assert "gradient_boosting" in trainer.models

    def test_train_ensemble(self, training_data):
        """Test ensemble model training."""
        X, y = training_data
        trainer = CLVModelTrainer()

        ensemble = trainer.train_ensemble(X, y)

        assert ensemble is not None
        assert "random_forest" in trainer.models
        assert "gradient_boosting" in trainer.models
        assert "ridge" in trainer.models

    def test_predict_ensemble(self, training_data):
        """Test ensemble predictions."""
        X, y = training_data
        trainer = CLVModelTrainer()
        trainer.train_ensemble(X, y)

        predictions = trainer.predict_ensemble(X)

        assert len(predictions) == len(X)
        assert all(p >= 0 for p in predictions)

    def test_feature_importance(self, training_data):
        """Test feature importance extraction."""
        X, y = training_data
        trainer = CLVModelTrainer()
        trainer.train_ensemble(X, y)

        importance = trainer.get_feature_importance()

        assert importance is not None
        assert len(importance) > 0
        # Check it's a DataFrame or dict with feature importances
        if hasattr(importance, "columns"):
            assert "feature" in importance.columns or len(importance.columns) > 0


class TestMetaAdsIntegration:
    """Tests for Meta Ads integration."""

    @pytest.fixture
    def predictions_data(self):
        """Create sample predictions data."""
        return pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003", "C004", "C005"],
                "predicted_clv": [850.0, 120.0, 500.0, 75.0, 300.0],
                "predicted_segment": [
                    "High-CLV",
                    "Low-CLV",
                    "Growth-Potential",
                    "Low-CLV",
                    "Growth-Potential",
                ],
            }
        )

    def test_generate_audience_segments(self, predictions_data):
        """Test audience segment generation."""
        integration = MetaAdsIntegration(predictions_data)
        audiences = integration.generate_audience_segments()

        assert "High-CLV" in audiences
        assert "Growth-Potential" in audiences
        assert "Low-CLV" in audiences

    def test_calculate_optimal_cac(self):
        """Test optimal CAC calculation."""
        integration = MetaAdsIntegration()

        result = integration.calculate_optimal_cac(1000)

        assert result["predicted_clv"] == 1000
        assert result["optimal_cac"] == 300  # 30% of CLV
        assert result["expected_roas"] == 3.33

    def test_generate_budget_allocation(self, predictions_data):
        """Test budget allocation generation."""
        integration = MetaAdsIntegration(predictions_data)

        allocation = integration.generate_budget_allocation(10000)

        assert allocation["total_budget"] == 10000
        assert "High-CLV" in allocation["allocation"]
        assert allocation["allocation"]["High-CLV"]["budget"] == 5000  # 50%
        assert allocation["allocation"]["Growth-Potential"]["budget"] == 3500  # 35%
        assert allocation["allocation"]["Low-CLV"]["budget"] == 1500  # 15%


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self):
        """Test complete prediction pipeline."""
        # Create sample data
        np.random.seed(42)
        n_samples = 200

        data = pd.DataFrame(
            {
                "customer_id": [f"C{i:04d}" for i in range(n_samples)],
                "total_orders": np.random.randint(1, 20, n_samples),
                "total_spent": np.random.uniform(50, 2000, n_samples),
                "avg_order_value": np.random.uniform(25, 200, n_samples),
                "days_since_first_purchase": np.random.randint(30, 365, n_samples),
                "days_since_last_purchase": np.random.randint(1, 180, n_samples),
                "num_categories": np.random.randint(1, 6, n_samples),
                "acquisition_source": np.random.choice(
                    ["Meta Ads", "Google Ads", "Email", "Direct"], n_samples
                ),
                "campaign_type": np.random.choice(
                    ["Prospecting", "Retargeting", "None"], n_samples
                ),
                "acquisition_cost": np.random.uniform(0, 70, n_samples),
                "email_engagement_rate": np.random.uniform(0, 1, n_samples),
                "return_rate": np.random.uniform(0, 0.3, n_samples),
                "actual_clv": np.random.uniform(50, 2500, n_samples),
            }
        )

        # Process data
        processor = DataProcessor()
        processor.raw_data = data
        processor.clean_data()

        # Engineer features
        engineer = FeatureEngineer()
        featured_data = engineer.fit_transform(processor.processed_data)

        # Prepare for training
        X, y = processor.preprocess_for_training(featured_data)

        # Train models
        trainer = CLVModelTrainer()
        trainer.train_ensemble(X, y)

        # Make predictions
        predictions = trainer.predict_ensemble(X)

        assert len(predictions) == n_samples
        assert all(p >= 0 for p in predictions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
