import pandas as pd
import numpy as np
from typing import Dict, List

class FeatureEngineer:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def create_time_features(self) -> pd.DataFrame:
        df = self.df.copy()
        
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['quarter'] = df['order_date'].dt.quarter
        df['day_of_week'] = df['order_date'].dt.dayofweek
        df['day_of_month'] = df['order_date'].dt.day
        df['week_of_year'] = df['order_date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        df['month_name'] = df['order_date'].dt.month_name()
        df['day_name'] = df['order_date'].dt.day_name()
        
        df['year_month'] = df['order_date'].dt.to_period('M')
        
        print(f"✓ Created time features: year, month, quarter, day_of_week, etc.")
        return df
    
    def create_transaction_features(self) -> pd.DataFrame:
        df = self.df.copy()
        
        if 'discount' in df.columns:
            df['effective_price'] = df['unit_price'] * (1 - df['discount'])
        else:
            df['effective_price'] = df['unit_price']
        
        df['revenue_per_item'] = df['sales'] / df['quantity']
        
        if 'profit' in df.columns and df['profit'].notna().any():
            df['profit_margin'] = (df['profit'] / df['sales']).replace([np.inf, -np.inf], 0)
            df['profit_margin'] = df['profit_margin'].fillna(0)
        
        df['quantity_category'] = pd.cut(
            df['quantity'],
            bins=[0, 1, 5, 10, 50, np.inf],
            labels=['Single', 'Small', 'Medium', 'Large', 'Bulk']
        )
        
        df['revenue_category'] = pd.qcut(
            df['sales'],
            q=4,
            labels=['Low', 'Medium', 'High', 'Very High'],
            duplicates='drop'
        )
        
        print(f"✓ Created transaction features: effective_price, revenue_per_item, etc.")
        return df
    
    def create_customer_features(self) -> pd.DataFrame:
        df = self.df.copy()
        
        reference_date = df['order_date'].max()
        
        customer_stats = df.groupby('customer_id').agg({
            'order_date': ['min', 'max', 'count'],
            'order_id': 'nunique',
            'sales': ['sum', 'mean'],
            'quantity': 'sum',
            'product_id': 'nunique'
        }).reset_index()
        
        customer_stats.columns = ['customer_id', 'first_purchase_date', 'last_purchase_date',
                                  'transaction_count', 'order_count', 'total_revenue',
                                  'avg_order_value', 'total_quantity', 'unique_products']
        
        customer_stats['recency_days'] = (reference_date - customer_stats['last_purchase_date']).dt.days
        customer_stats['customer_lifetime_days'] = (
            customer_stats['last_purchase_date'] - customer_stats['first_purchase_date']
        ).dt.days
        
        customer_stats['purchase_frequency'] = (
            customer_stats['transaction_count'] / 
            (customer_stats['customer_lifetime_days'] + 1)
        )
        
        customer_stats['recency_score'] = pd.qcut(
            customer_stats['recency_days'],
            q=4,
            labels=['Very Recent', 'Recent', 'Old', 'Very Old'],
            duplicates='drop'
        )
        
        customer_stats['monetary_score'] = pd.qcut(
            customer_stats['total_revenue'],
            q=4,
            labels=['Low Value', 'Medium Value', 'High Value', 'VIP'],
            duplicates='drop'
        )
        
        customer_stats['frequency_score'] = pd.qcut(
            customer_stats['transaction_count'],
            q=4,
            labels=['Rare', 'Occasional', 'Regular', 'Frequent'],
            duplicates='drop'
        )
        
        df = df.merge(customer_stats, on='customer_id', how='left')
        
        print(f"✓ Created customer features: RFM metrics, customer segments, etc.")
        return df
    
    def create_product_features(self) -> pd.DataFrame:
        df = self.df.copy()
        
        product_stats = df.groupby('product_id').agg({
            'order_date': ['min', 'max', 'count'],
            'sales': ['sum', 'mean'],
            'quantity': 'sum',
            'customer_id': 'nunique'
        }).reset_index()
        
        product_stats.columns = ['product_id', 'first_sold_date', 'last_sold_date',
                                'times_sold', 'total_product_revenue', 'avg_product_price',
                                'total_units_sold', 'unique_customers']
        
        product_stats['popularity_score'] = pd.qcut(
            product_stats['times_sold'],
            q=4,
            labels=['Niche', 'Regular', 'Popular', 'Bestseller'],
            duplicates='drop'
        )
        
        df = df.merge(product_stats[['product_id', 'total_product_revenue', 
                                     'popularity_score', 'unique_customers']], 
                     on='product_id', how='left')
        
        print(f"✓ Created product features: popularity, total revenue, etc.")
        return df
    
    def create_aggregated_views(self) -> Dict[str, pd.DataFrame]:
        df = self.df.copy()
        
        views = {}
        
        views['daily_sales'] = df.groupby('order_date').agg({
            'sales': 'sum',
            'quantity': 'sum',
            'order_id': 'nunique',
            'customer_id': 'nunique'
        }).reset_index()
        views['daily_sales'].columns = ['date', 'revenue', 'units_sold', 
                                        'orders', 'unique_customers']
        
        monthly = df.groupby(df['order_date'].dt.to_period('M')).agg({
            'sales': 'sum',
            'quantity': 'sum',
            'order_id': 'nunique',
            'customer_id': 'nunique',
            'product_id': 'nunique'
        }).reset_index()
        monthly.columns = ['month', 'revenue', 'units_sold', 'orders', 
                          'unique_customers', 'unique_products']
        monthly['month'] = monthly['month'].dt.to_timestamp()
        views['monthly_sales'] = monthly
        
        if 'category' in df.columns:
            views['category_sales'] = df.groupby('category').agg({
                'sales': 'sum',
                'quantity': 'sum',
                'order_id': 'nunique',
                'profit': 'sum' if 'profit' in df.columns else 'count'
            }).reset_index().sort_values('sales', ascending=False)
        
        if 'region' in df.columns and df['region'].notna().any():
            views['regional_sales'] = df.groupby('region').agg({
                'sales': 'sum',
                'quantity': 'sum',
                'order_id': 'nunique'
            }).reset_index().sort_values('sales', ascending=False)
        
        if 'segment' in df.columns:
            views['segment_sales'] = df.groupby('segment').agg({
                'sales': 'sum',
                'customer_id': 'nunique',
                'order_id': 'nunique'
            }).reset_index().sort_values('sales', ascending=False)
        
        print(f"✓ Created {len(views)} aggregated views")
        return views
    
    def run_all_engineering(self) -> pd.DataFrame:
        print("\n=== Running Feature Engineering ===")
        
        df = self.create_time_features()
        self.df = df
        
        df = self.create_transaction_features()
        self.df = df
        
        df = self.create_customer_features()
        self.df = df
        
        df = self.create_product_features()
        self.df = df
        
        print(f"\nFeature engineering complete: {len(df.columns)} total features")
        return df