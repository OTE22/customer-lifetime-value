"""
Meta Ads Integration Module
Generates audience segments, budget allocation, and Meta Ads optimization recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaAdsIntegration:
    """
    Handles Meta Ads integration for CLV-optimized campaigns.
    """
    
    # Budget allocation rule (3:2:1 for High:Growth:Low CLV)
    BUDGET_WEIGHTS = {
        'High-CLV': 0.50,
        'Growth-Potential': 0.35,
        'Low-CLV': 0.15
    }
    
    # Target CAC as percentage of predicted CLV
    TARGET_CAC_RATIO = 0.30
    
    def __init__(self, predictions_df: Optional[pd.DataFrame] = None):
        """
        Initialize Meta Ads Integration.
        
        Args:
            predictions_df: DataFrame with CLV predictions.
        """
        self.predictions_df = predictions_df
        self.audiences = {}
        
    def set_predictions(self, df: pd.DataFrame) -> None:
        """Set predictions DataFrame."""
        self.predictions_df = df
    
    def generate_audience_segments(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate audience segments for Meta Ads targeting.
        
        Args:
            df: DataFrame with predictions (uses stored df if not provided).
            
        Returns:
            Dictionary mapping segment names to customer DataFrames.
        """
        df = df if df is not None else self.predictions_df
        
        if df is None:
            raise ValueError("No predictions data available")
        
        logger.info("Generating audience segments for Meta Ads...")
        
        # Create segments
        if 'predicted_segment' in df.columns:
            segment_col = 'predicted_segment'
        elif 'customer_segment' in df.columns:
            segment_col = 'customer_segment'
        else:
            raise ValueError("No segment column found in data")
        
        self.audiences = {
            'High-CLV': df[df[segment_col] == 'High-CLV'].copy(),
            'Growth-Potential': df[df[segment_col] == 'Growth-Potential'].copy(),
            'Low-CLV': df[df[segment_col] == 'Low-CLV'].copy()
        }
        
        # Add targeting recommendations
        for segment_name, audience_df in self.audiences.items():
            audience_df['targeting_priority'] = self._get_targeting_priority(segment_name)
            audience_df['lookalike_recommendation'] = self._get_lookalike_recommendation(segment_name)
            audience_df['bidding_strategy'] = self._get_bidding_strategy(segment_name)
        
        logger.info(f"Generated {len(self.audiences)} audience segments")
        return self.audiences
    
    def _get_targeting_priority(self, segment: str) -> str:
        """Get targeting priority for segment."""
        priorities = {
            'High-CLV': 'Primary',
            'Growth-Potential': 'Secondary',
            'Low-CLV': 'Tertiary'
        }
        return priorities.get(segment, 'Unknown')
    
    def _get_lookalike_recommendation(self, segment: str) -> str:
        """Get lookalike audience recommendation."""
        recommendations = {
            'High-CLV': '1% Lookalike - Best for prospecting',
            'Growth-Potential': '2-3% Lookalike - Broader reach',
            'Low-CLV': '5%+ Lookalike - Testing only'
        }
        return recommendations.get(segment, 'Unknown')
    
    def _get_bidding_strategy(self, segment: str) -> str:
        """Get bidding strategy for segment."""
        strategies = {
            'High-CLV': 'Target Cost / Value Optimization',
            'Growth-Potential': 'Lowest Cost with Conversion Optimization',
            'Low-CLV': 'Strict Cost Caps'
        }
        return strategies.get(segment, 'Unknown')
    
    def calculate_optimal_cac(
        self,
        predicted_clv: float,
        ratio: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate optimal Customer Acquisition Cost based on predicted CLV.
        
        Args:
            predicted_clv: Predicted customer lifetime value.
            ratio: Target CAC/CLV ratio (default 0.30).
            
        Returns:
            Dictionary with CAC recommendations.
        """
        ratio = ratio or self.TARGET_CAC_RATIO
        
        optimal_cac = predicted_clv * ratio
        
        return {
            'predicted_clv': round(predicted_clv, 2),
            'optimal_cac': round(optimal_cac, 2),
            'cac_ratio': ratio,
            'max_cac': round(predicted_clv * 0.40, 2),  # Absolute maximum
            'min_cac': round(predicted_clv * 0.15, 2),  # Minimum viable
            'expected_roas': round(1 / ratio, 2)
        }
    
    def generate_budget_allocation(
        self,
        total_budget: float,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate budget allocation recommendations based on 3:2:1 rule.
        
        Args:
            total_budget: Total advertising budget.
            df: DataFrame with predictions.
            
        Returns:
            Dictionary with budget allocation recommendations.
        """
        df = df if df is not None else self.predictions_df
        
        if df is None:
            raise ValueError("No predictions data available")
        
        logger.info(f"Generating budget allocation for ${total_budget:,.2f}...")
        
        # Generate segments if not already done
        if not self.audiences:
            self.generate_audience_segments(df)
        
        allocation = {}
        for segment_name, weight in self.BUDGET_WEIGHTS.items():
            segment_budget = total_budget * weight
            segment_df = self.audiences.get(segment_name, pd.DataFrame())
            
            customer_count = len(segment_df)
            
            if 'predicted_clv' in segment_df.columns:
                avg_clv = segment_df['predicted_clv'].mean()
            elif 'actual_clv' in segment_df.columns:
                avg_clv = segment_df['actual_clv'].mean()
            else:
                avg_clv = 0
            
            optimal_cac = avg_clv * self.TARGET_CAC_RATIO if avg_clv > 0 else 0
            expected_acquisitions = int(segment_budget / optimal_cac) if optimal_cac > 0 else 0
            
            allocation[segment_name] = {
                'budget': round(segment_budget, 2),
                'budget_percentage': round(weight * 100, 1),
                'customer_count': customer_count,
                'avg_predicted_clv': round(avg_clv, 2),
                'optimal_cac': round(optimal_cac, 2),
                'expected_acquisitions': expected_acquisitions,
                'expected_revenue': round(expected_acquisitions * avg_clv, 2),
                'expected_roas': round((expected_acquisitions * avg_clv) / segment_budget, 2) if segment_budget > 0 else 0
            }
        
        # Calculate totals
        total_expected_acquisitions = sum(a['expected_acquisitions'] for a in allocation.values())
        total_expected_revenue = sum(a['expected_revenue'] for a in allocation.values())
        
        return {
            'total_budget': total_budget,
            'allocation': allocation,
            'summary': {
                'total_expected_acquisitions': total_expected_acquisitions,
                'total_expected_revenue': round(total_expected_revenue, 2),
                'blended_roas': round(total_expected_revenue / total_budget, 2) if total_budget > 0 else 0
            }
        }
    
    def export_for_meta_ads(
        self,
        segment: str,
        output_path: Optional[str] = None,
        include_fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Export audience for Meta Ads upload.
        
        Args:
            segment: Segment name to export.
            output_path: Optional path to save CSV.
            include_fields: Fields to include in export.
            
        Returns:
            DataFrame formatted for Meta Ads.
        """
        if segment not in self.audiences:
            raise ValueError(f"Segment '{segment}' not found. Generate segments first.")
        
        audience_df = self.audiences[segment]
        
        # Default fields for Meta Ads
        if include_fields is None:
            include_fields = ['customer_id', 'acquisition_source', 'predicted_clv']
        
        # Filter to available fields
        available_fields = [f for f in include_fields if f in audience_df.columns]
        export_df = audience_df[available_fields].copy()
        
        # Add export metadata
        export_df['export_date'] = datetime.now().strftime('%Y-%m-%d')
        export_df['segment'] = segment
        
        if output_path:
            export_df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(export_df)} customers to {output_path}")
        
        return export_df
    
    def create_lookalike_recommendations(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate lookalike audience recommendations.
        
        Args:
            df: DataFrame with predictions.
            
        Returns:
            Dictionary with lookalike recommendations.
        """
        df = df if df is not None else self.predictions_df
        
        if df is None:
            raise ValueError("No predictions data available")
        
        if not self.audiences:
            self.generate_audience_segments(df)
        
        recommendations = {
            'prospecting': {
                'source_audience': 'High-CLV',
                'source_size': len(self.audiences.get('High-CLV', [])),
                'lookalike_size': '1%',
                'estimated_reach': '500K - 1M',
                'priority': 'Highest',
                'strategy': 'Use for new customer acquisition with value optimization'
            },
            'expansion': {
                'source_audience': 'High-CLV + Growth-Potential',
                'source_size': len(self.audiences.get('High-CLV', [])) + len(self.audiences.get('Growth-Potential', [])),
                'lookalike_size': '2-3%',
                'estimated_reach': '1M - 2M',
                'priority': 'Medium',
                'strategy': 'Use for broader reach with conversion optimization'
            },
            'testing': {
                'source_audience': 'Growth-Potential',
                'source_size': len(self.audiences.get('Growth-Potential', [])),
                'lookalike_size': '3-5%',
                'estimated_reach': '2M - 5M',
                'priority': 'Lower',
                'strategy': 'Use for testing new markets with lowest cost bidding'
            }
        }
        
        return recommendations
    
    def get_campaign_recommendations(
        self,
        segment: str
    ) -> Dict[str, Any]:
        """
        Get campaign setup recommendations for a segment.
        
        Args:
            segment: Customer segment name.
            
        Returns:
            Dictionary with campaign recommendations.
        """
        recommendations = {
            'High-CLV': {
                'campaign_objective': 'Conversions',
                'optimization_event': 'Purchase',
                'bidding_strategy': 'Highest Value or Target ROAS',
                'budget_type': 'CBO (Campaign Budget Optimization)',
                'ad_format': 'Video (60+ seconds), Carousel',
                'messaging': 'Premium positioning, quality focus, lifestyle',
                'retargeting_window': '180 days',
                'frequency_cap': 'No cap / High tolerance'
            },
            'Growth-Potential': {
                'campaign_objective': 'Conversions',
                'optimization_event': 'Purchase',
                'bidding_strategy': 'Lowest Cost',
                'budget_type': 'CBO',
                'ad_format': 'Carousel, Single Image',
                'messaging': 'Value proposition, benefits, social proof',
                'retargeting_window': '90 days',
                'frequency_cap': '3-5 per week'
            },
            'Low-CLV': {
                'campaign_objective': 'Conversions',
                'optimization_event': 'Add to Cart or Purchase',
                'bidding_strategy': 'Cost Cap',
                'budget_type': 'ABO (Ad Set Budget)',
                'ad_format': 'Single Image',
                'messaging': 'Price focus, urgency, discounts',
                'retargeting_window': '30 days',
                'frequency_cap': '2 per week'
            }
        }
        
        return recommendations.get(segment, {})
    
    def get_full_strategy(
        self,
        total_budget: float,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Get complete Meta Ads strategy with all recommendations.
        
        Args:
            total_budget: Total advertising budget.
            df: DataFrame with predictions.
            
        Returns:
            Complete strategy dictionary.
        """
        self.generate_audience_segments(df)
        
        budget_allocation = self.generate_budget_allocation(total_budget)
        lookalike_recs = self.create_lookalike_recommendations()
        
        campaign_recs = {}
        for segment in ['High-CLV', 'Growth-Potential', 'Low-CLV']:
            campaign_recs[segment] = self.get_campaign_recommendations(segment)
        
        return {
            'budget_allocation': budget_allocation,
            'lookalike_recommendations': lookalike_recs,
            'campaign_recommendations': campaign_recs,
            'audience_sizes': {
                segment: len(audience) 
                for segment, audience in self.audiences.items()
            }
        }


def create_meta_strategy(
    predictions_df: pd.DataFrame,
    total_budget: float
) -> Dict[str, Any]:
    """
    Convenience function to create complete Meta Ads strategy.
    
    Args:
        predictions_df: DataFrame with CLV predictions.
        total_budget: Total advertising budget.
        
    Returns:
        Complete strategy dictionary.
    """
    integration = MetaAdsIntegration(predictions_df)
    return integration.get_full_strategy(total_budget)
