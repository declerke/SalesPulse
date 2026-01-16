import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class DataQualityChecker:
    
    def __init__(self, df: pd.DataFrame, dataset_name: str = "Dataset"):
        self.df = df.copy()
        self.dataset_name = dataset_name
        self.report = {}
        
    def check_missing_values(self) -> pd.DataFrame:
        missing_stats = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum().values,
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df) * 100).values,
            'Data_Type': self.df.dtypes.values
        })
        missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        )
        self.report['missing_values'] = missing_stats
        return missing_stats
    
    def check_duplicates(self) -> Dict:
        total_duplicates = self.df.duplicated().sum()
        duplicate_percentage = (total_duplicates / len(self.df)) * 100
        
        duplicate_info = {
            'total_duplicates': total_duplicates,
            'duplicate_percentage': duplicate_percentage,
            'unique_rows': len(self.df) - total_duplicates
        }
        self.report['duplicates'] = duplicate_info
        return duplicate_info
    
    def check_outliers_iqr(self, numeric_columns: List[str] = None) -> Dict:
        if numeric_columns is None:
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_info = {}
        for col in numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min_outlier': outliers[col].min() if len(outliers) > 0 else None,
                'max_outlier': outliers[col].max() if len(outliers) > 0 else None
            }
        
        self.report['outliers'] = outlier_info
        return outlier_info
    
    def check_data_types(self) -> pd.DataFrame:
        type_info = pd.DataFrame({
            'Column': self.df.columns,
            'Current_Type': self.df.dtypes.values,
            'Unique_Values': [self.df[col].nunique() for col in self.df.columns],
            'Sample_Value': [self.df[col].iloc[0] if len(self.df) > 0 else None 
                           for col in self.df.columns]
        })
        self.report['data_types'] = type_info
        return type_info
    
    def check_cardinality(self) -> pd.DataFrame:
        cardinality = pd.DataFrame({
            'Column': self.df.columns,
            'Unique_Count': [self.df[col].nunique() for col in self.df.columns],
            'Unique_Percentage': [(self.df[col].nunique() / len(self.df)) * 100 
                                 for col in self.df.columns]
        }).sort_values('Unique_Percentage', ascending=False)
        
        self.report['cardinality'] = cardinality
        return cardinality
    
    def check_negative_values(self, columns: List[str]) -> Dict:
        negative_info = {}
        for col in columns:
            if col in self.df.columns:
                negative_count = (self.df[col] < 0).sum()
                negative_info[col] = {
                    'count': negative_count,
                    'percentage': (negative_count / len(self.df)) * 100
                }
        self.report['negative_values'] = negative_info
        return negative_info
    
    def generate_summary_stats(self) -> pd.DataFrame:
        numeric_df = self.df.select_dtypes(include=[np.number])
        summary = numeric_df.describe().T
        summary['skewness'] = numeric_df.skew()
        summary['kurtosis'] = numeric_df.kurt()
        self.report['summary_stats'] = summary
        return summary
    
    def run_full_quality_check(self) -> Dict:
        self.report['basic_info'] = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        self.check_missing_values()
        self.check_duplicates()
        self.check_data_types()
        self.check_cardinality()
        self.generate_summary_stats()
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            self.check_outliers_iqr(numeric_cols)
        
        return self.report
    
    def save_report(self, output_path: str):
        with open(output_path, 'w') as f:
            f.write(f"# Data Quality Report: {self.dataset_name}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Basic Information\n\n")
            f.write(f"- Total Rows: {self.report['basic_info']['total_rows']:,}\n")
            f.write(f"- Total Columns: {self.report['basic_info']['total_columns']}\n")
            f.write(f"- Memory Usage: {self.report['basic_info']['memory_usage_mb']:.2f} MB\n\n")
            
            f.write("## Missing Values\n\n")
            if len(self.report['missing_values']) > 0:
                f.write(self.report['missing_values'].to_markdown(index=False))
            else:
                f.write("No missing values detected.\n")
            f.write("\n\n")
            
            f.write("## Duplicate Rows\n\n")
            dup = self.report['duplicates']
            f.write(f"- Total Duplicates: {dup['total_duplicates']:,}\n")
            f.write(f"- Percentage: {dup['duplicate_percentage']:.2f}%\n\n")
            
            if 'outliers' in self.report:
                f.write("## Outliers (IQR Method)\n\n")
                for col, info in self.report['outliers'].items():
                    if info['count'] > 0:
                        f.write(f"### {col}\n")
                        f.write(f"- Count: {info['count']:,} ({info['percentage']:.2f}%)\n")
                        f.write(f"- Range: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]\n\n")
    
    def plot_missing_values(self, figsize=(10, 6)):
        if len(self.report['missing_values']) > 0:
            fig, ax = plt.subplots(figsize=figsize)
            missing_df = self.report['missing_values'].sort_values('Missing_Percentage')
            ax.barh(missing_df['Column'], missing_df['Missing_Percentage'])
            ax.set_xlabel('Missing Percentage (%)')
            ax.set_title(f'Missing Values - {self.dataset_name}')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            return fig
        return None
    
    def plot_outliers_boxplot(self, columns: List[str] = None, figsize=(15, 10)):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, col in enumerate(columns):
            if idx < len(axes):
                axes[idx].boxplot(self.df[col].dropna())
                axes[idx].set_title(col)
                axes[idx].grid(alpha=0.3)
        
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig