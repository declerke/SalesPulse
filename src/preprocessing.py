import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    
    def __init__(self):
        self.cleaning_log = []
        
    def log_action(self, action: str):
        self.cleaning_log.append(action)
        print(f"✓ {action}")
    
    def load_superstore_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, encoding='utf-8')
        self.log_action(f"Loaded Superstore data: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def load_online_retail_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
        self.log_action(f"Loaded Online Retail data: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def clean_superstore(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
        
        date_columns = ['Order_Date', 'Ship_Date']
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        if 'Order_Date' in df_clean.columns:
            df_clean = df_clean[df_clean['Order_Date'].notna()]
            self.log_action(f"Removed {initial_rows - len(df_clean)} rows with missing Order_Date")
        
        numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        if 'Quantity' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[df_clean['Quantity'] > 0]
            self.log_action(f"Removed {before - len(df_clean)} rows with non-positive Quantity")
        
        if 'Sales' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[df_clean['Sales'] >= 0]
            self.log_action(f"Removed {before - len(df_clean)} rows with negative Sales")
        
        cat_cols = ['Ship_Mode', 'Segment', 'Category', 'Sub_Category', 'Region']
        for col in cat_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].str.strip()
        
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        self.log_action(f"Removed {before - len(df_clean)} duplicate rows")
        
        self.log_action(f"Superstore cleaning complete: {len(df_clean)} rows remaining")
        return df_clean
    
    def clean_online_retail(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '_')
        
        if 'InvoiceDate' in df_clean.columns:
            df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')
            df_clean = df_clean[df_clean['InvoiceDate'].notna()]
        
        if 'CustomerID' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[df_clean['CustomerID'].notna()]
            self.log_action(f"Removed {before - len(df_clean)} rows with missing CustomerID")
        
        if 'StockCode' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[df_clean['StockCode'].notna()]
            self.log_action(f"Removed {before - len(df_clean)} rows with missing StockCode")
        
        if 'Quantity' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[df_clean['Quantity'] > 0]
            self.log_action(f"Removed {before - len(df_clean)} rows with non-positive Quantity")
        
        if 'UnitPrice' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[df_clean['UnitPrice'] > 0]
            self.log_action(f"Removed {before - len(df_clean)} rows with non-positive UnitPrice")
        
        if 'Quantity' in df_clean.columns and 'UnitPrice' in df_clean.columns:
            df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
        
        if 'Quantity' in df_clean.columns:
            before = len(df_clean)
            q99 = df_clean['Quantity'].quantile(0.99)
            df_clean = df_clean[df_clean['Quantity'] <= q99 * 3]
            self.log_action(f"Removed {before - len(df_clean)} extreme quantity outliers")
        
        if 'UnitPrice' in df_clean.columns:
            before = len(df_clean)
            q99 = df_clean['UnitPrice'].quantile(0.99)
            df_clean = df_clean[df_clean['UnitPrice'] <= q99 * 3]
            self.log_action(f"Removed {before - len(df_clean)} extreme price outliers")
        
        if 'Description' in df_clean.columns:
            df_clean['Description'] = df_clean['Description'].str.strip()
        
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        self.log_action(f"Removed {before - len(df_clean)} duplicate rows")
        
        self.log_action(f"Online Retail cleaning complete: {len(df_clean)} rows remaining")
        return df_clean
    
    def standardize_schema(self, df_superstore: pd.DataFrame, 
                          df_retail: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        superstore_mapped = pd.DataFrame({
            'order_id': df_superstore.get('Order_ID', df_superstore.get('Row_ID')),
            'order_date': df_superstore.get('Order_Date'),
            'customer_id': df_superstore.get('Customer_ID'),
            'customer_name': df_superstore.get('Customer_Name'),
            'segment': df_superstore.get('Segment'),
            'country': df_superstore.get('Country', 'USA'),
            'region': df_superstore.get('Region'),
            'product_id': df_superstore.get('Product_ID'),
            'product_name': df_superstore.get('Product_Name'),
            'category': df_superstore.get('Category'),
            'sub_category': df_superstore.get('Sub_Category'),
            'quantity': df_superstore.get('Quantity'),
            'unit_price': df_superstore.get('Sales', 0) / df_superstore.get('Quantity', 1),
            'sales': df_superstore.get('Sales'),
            'discount': df_superstore.get('Discount', 0),
            'profit': df_superstore.get('Profit'),
            'source': 'Superstore'
        })
        
        retail_mapped = pd.DataFrame({
            'order_id': df_retail.get('InvoiceNo'),
            'order_date': df_retail.get('InvoiceDate'),
            'customer_id': df_retail.get('CustomerID').astype(str),
            'customer_name': None,
            'segment': 'Online',
            'country': df_retail.get('Country'),
            'region': None,
            'product_id': df_retail.get('StockCode'),
            'product_name': df_retail.get('Description'),
            'category': 'Retail',
            'sub_category': None,
            'quantity': df_retail.get('Quantity'),
            'unit_price': df_retail.get('UnitPrice'),
            'sales': df_retail.get('TotalPrice'),
            'discount': 0,
            'profit': None,
            'source': 'OnlineRetail'
        })
        
        self.log_action("Standardized schemas for both datasets")
        return superstore_mapped, retail_mapped
    
    def merge_datasets(self, df_superstore: pd.DataFrame, 
                      df_retail: pd.DataFrame) -> pd.DataFrame:
        
        df_combined = pd.concat([df_superstore, df_retail], ignore_index=True)
        
        df_combined['order_date'] = pd.to_datetime(df_combined['order_date'])
        df_combined['quantity'] = pd.to_numeric(df_combined['quantity'], errors='coerce')
        df_combined['sales'] = pd.to_numeric(df_combined['sales'], errors='coerce')
        
        df_combined = df_combined.sort_values('order_date').reset_index(drop=True)
        
        self.log_action(f"Merged datasets: {len(df_combined)} total transactions")
        return df_combined
    
    def get_cleaning_summary(self) -> str:
        return "\n".join(self.cleaning_log)