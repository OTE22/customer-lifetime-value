"""
Generate fake e-commerce customer data for CLV prediction.
Creates 1000 rows of realistic customer data suitable for ML training.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_customer_data(n_customers: int = 1000) -> pd.DataFrame:
    """Generate realistic e-commerce customer data."""
    
    # Date range for purchases (last 2 years)
    end_date = datetime(2026, 1, 15)
    start_date = end_date - timedelta(days=730)
    
    # Acquisition sources with realistic distribution
    acquisition_sources = ['Meta Ads', 'Google Ads', 'Email', 'Direct', 'Organic', 'Referral']
    source_weights = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
    
    # Campaign types
    campaign_types = ['Prospecting', 'Retargeting', 'Brand', 'None']
    
    # Product categories
    product_categories_list = [
        'Electronics', 'Fashion', 'Home & Garden', 'Beauty', 
        'Sports', 'Books', 'Toys', 'Food & Beverage'
    ]
    
    customers = []
    
    for i in range(n_customers):
        customer_id = f"CUST_{i+1:05d}"
        
        # Determine customer segment first (to influence other attributes)
        segment_roll = random.random()
        if segment_roll < 0.20:
            segment = 'High-CLV'
            base_orders = random.randint(5, 25)
            base_aov = random.uniform(80, 250)
            engagement = random.uniform(0.5, 0.95)
            return_rate = random.uniform(0.01, 0.08)
        elif segment_roll < 0.80:
            segment = 'Growth-Potential'
            base_orders = random.randint(2, 8)
            base_aov = random.uniform(40, 120)
            engagement = random.uniform(0.2, 0.6)
            return_rate = random.uniform(0.05, 0.15)
        else:
            segment = 'Low-CLV'
            base_orders = random.randint(1, 3)
            base_aov = random.uniform(20, 60)
            engagement = random.uniform(0.0, 0.3)
            return_rate = random.uniform(0.10, 0.30)
        
        # First purchase date
        first_purchase = start_date + timedelta(days=random.randint(0, 700))
        
        # Last purchase based on segment
        if segment == 'High-CLV':
            days_since_last = random.randint(1, 30)
        elif segment == 'Growth-Potential':
            days_since_last = random.randint(15, 90)
        else:
            days_since_last = random.randint(60, 365)
        
        last_purchase = min(end_date, first_purchase + timedelta(days=random.randint(0, 700)))
        last_purchase = max(first_purchase, end_date - timedelta(days=days_since_last))
        
        # Calculate metrics
        total_orders = base_orders
        avg_order_value = round(base_aov * random.uniform(0.8, 1.2), 2)
        total_spent = round(total_orders * avg_order_value * random.uniform(0.9, 1.1), 2)
        
        days_since_first = (end_date - first_purchase).days
        days_since_last_purchase = (end_date - last_purchase).days
        
        # Acquisition source
        acq_source = random.choices(acquisition_sources, weights=source_weights)[0]
        
        # Campaign type based on source
        if acq_source in ['Meta Ads', 'Google Ads']:
            camp_type = random.choice(['Prospecting', 'Retargeting', 'Brand'])
        else:
            camp_type = 'None'
        
        # Product categories purchased
        n_categories = min(len(product_categories_list), random.randint(1, min(5, total_orders)))
        categories = random.sample(product_categories_list, n_categories)
        
        # Calculate actual CLV (with noise for realism)
        base_clv = total_spent
        if segment == 'High-CLV':
            future_multiplier = random.uniform(2.0, 4.0)
        elif segment == 'Growth-Potential':
            future_multiplier = random.uniform(1.2, 2.0)
        else:
            future_multiplier = random.uniform(0.8, 1.2)
        
        actual_clv = round(base_clv * future_multiplier, 2)
        
        # Acquisition cost
        if acq_source == 'Meta Ads':
            acquisition_cost = round(random.uniform(15, 60), 2)
        elif acq_source == 'Google Ads':
            acquisition_cost = round(random.uniform(20, 70), 2)
        elif acq_source == 'Email':
            acquisition_cost = round(random.uniform(2, 10), 2)
        else:
            acquisition_cost = round(random.uniform(0, 5), 2)
        
        customer = {
            'customer_id': customer_id,
            'first_purchase_date': first_purchase.strftime('%Y-%m-%d'),
            'last_purchase_date': last_purchase.strftime('%Y-%m-%d'),
            'total_orders': total_orders,
            'total_spent': total_spent,
            'avg_order_value': avg_order_value,
            'days_since_first_purchase': days_since_first,
            'days_since_last_purchase': days_since_last_purchase,
            'product_categories': json.dumps(categories),
            'num_categories': len(categories),
            'acquisition_source': acq_source,
            'campaign_type': camp_type,
            'acquisition_cost': acquisition_cost,
            'email_engagement_rate': round(engagement, 3),
            'return_rate': round(return_rate, 3),
            'customer_segment': segment,
            'actual_clv': actual_clv
        }
        
        customers.append(customer)
    
    return pd.DataFrame(customers)


def main():
    """Generate and save customer data."""
    print("Generating 1000 customer records...")
    df = generate_customer_data(1000)
    
    # Save to CSV
    output_path = 'customers.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total customers: {len(df)}")
    print(f"\nSegment distribution:")
    print(df['customer_segment'].value_counts())
    print(f"\nAcquisition source distribution:")
    print(df['acquisition_source'].value_counts())
    print(f"\nCLV Statistics:")
    print(df['actual_clv'].describe())
    
    return df


if __name__ == "__main__":
    main()
