import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

from prophet import Prophet

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import warnings
warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    
    def __init__(self, df: pd.DataFrame, date_col: str = 'date', value_col: str = 'revenue'):
        self.df = df.copy()
        self.date_col = date_col
        self.value_col = value_col
        
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values(date_col).reset_index(drop=True)
        
        self.ts = self.df.set_index(date_col)[value_col]
        
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        
    def check_stationarity(self, significance_level: float = 0.05) -> Dict:
        result = adfuller(self.ts.dropna())
        
        is_stationary = result[1] < significance_level
        
        adf_results = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'used_lag': result[2],
            'n_observations': result[3],
            'critical_values': result[4],
            'is_stationary': is_stationary
        }
        
        print(f"\n=== Stationarity Test (ADF) ===")
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"P-value: {result[1]:.4f}")
        print(f"Stationary: {'Yes' if is_stationary else 'No'}")
        
        return adf_results
    
    def decompose_series(self, model: str = 'additive', period: int = None) -> Dict:
        if period is None:
            if self.ts.index.freq == 'MS' or len(self.ts) > 24:
                period = 12
            else:
                period = 7
        
        decomposition = seasonal_decompose(self.ts, model=model, period=period)
        
        print(f"\n=== Time Series Decomposition ===")
        print(f"Model: {model}")
        print(f"Period: {period}")
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'observed': decomposition.observed
        }
    
    def plot_decomposition(self, decomposition: Dict, figsize=(15, 10)):
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        decomposition['observed'].plot(ax=axes[0], title='Observed')
        axes[0].set_ylabel('Observed')
        
        decomposition['trend'].plot(ax=axes[1], title='Trend')
        axes[1].set_ylabel('Trend')
        
        decomposition['seasonal'].plot(ax=axes[2], title='Seasonal')
        axes[2].set_ylabel('Seasonal')
        
        decomposition['residual'].plot(ax=axes[3], title='Residual')
        axes[3].set_ylabel('Residual')
        
        plt.tight_layout()
        return fig
    
    def plot_acf_pacf(self, lags: int = 40, figsize=(15, 6)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        plot_acf(self.ts.dropna(), lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')
        
        plot_pacf(self.ts.dropna(), lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        return fig
    
    def split_train_test(self, test_size: int = 12) -> Tuple[pd.Series, pd.Series]:
        train = self.ts[:-test_size]
        test = self.ts[-test_size:]
        
        print(f"\n=== Train/Test Split ===")
        print(f"Train size: {len(train)} observations")
        print(f"Test size: {len(test)} observations")
        print(f"Train period: {train.index[0]} to {train.index[-1]}")
        print(f"Test period: {test.index[0]} to {test.index[-1]}")
        
        return train, test
    
    def fit_arima_auto(self, train: pd.Series, seasonal: bool = True, 
                        m: int = 12) -> Dict:
        print(f"\n=== Fitting Auto ARIMA ===")
        
        model = pm.auto_arima(
            train,
            seasonal=seasonal,
            m=m,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False,
            max_order=5
        )
        
        print(f"Best model: {model.order} x {model.seasonal_order if seasonal else 'non-seasonal'}")
        print(f"AIC: {model.aic():.2f}")
        
        self.models['arima'] = model
        
        return {
            'model': model,
            'order': model.order,
            'seasonal_order': model.seasonal_order if seasonal else None,
            'aic': model.aic()
        }
    
    def fit_prophet(self, train_df: pd.DataFrame = None) -> Dict:
        print(f"\n=== Fitting Prophet ===")
        
        if train_df is None:
            train_df = self.df.copy()
            train_df = train_df.rename(columns={self.date_col: 'ds', self.value_col: 'y'})
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        
        model.fit(train_df)
        self.models['prophet'] = model
        
        print(f"Prophet model fitted successfully")
        
        return {'model': model}
    
    def fit_baseline_ml(self, train: pd.Series, lags: int = 12) -> Dict:
        print(f"\n=== Fitting ML Baseline (Random Forest) ===")
        
        df_ml = pd.DataFrame({'y': train})
        for i in range(1, lags + 1):
            df_ml[f'lag_{i}'] = df_ml['y'].shift(i)
        
        df_ml = df_ml.dropna()
        
        X = df_ml.drop('y', axis=1)
        y = df_ml['y']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        
        self.models['ml_baseline'] = {
            'model': model,
            'lags': lags,
            'last_values': train.tail(lags).values
        }
        
        print(f"Random Forest fitted with {lags} lag features")
        
        return self.models['ml_baseline']
    
    def forecast_arima(self, steps: int = 12) -> pd.DataFrame:
        if 'arima' not in self.models:
            raise ValueError("ARIMA model not fitted. Call fit_arima_auto first.")
        
        model = self.models['arima']
        forecast_result = model.predict(n_periods=steps, return_conf_int=True)
        
        last_date = self.ts.index[-1]
        
        if isinstance(last_date, pd.Timestamp):
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=steps,
                freq='MS'
            )
        else:
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=steps,
                freq='D'
            )
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'arima_forecast': forecast_result[0],
            'arima_lower': forecast_result[1][:, 0],
            'arima_upper': forecast_result[1][:, 1]
        })
        
        self.forecasts['arima'] = forecast_df
        
        return forecast_df
    
    def forecast_prophet(self, steps: int = 12) -> pd.DataFrame:
        if 'prophet' not in self.models:
            raise ValueError("Prophet model not fitted. Call fit_prophet first.")
        
        model = self.models['prophet']
        
        last_date = self.df[self.date_col].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=steps,
            freq='MS'
        )
        
        future = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future)
        
        forecast_df = pd.DataFrame({
            'date': forecast['ds'],
            'prophet_forecast': forecast['yhat'],
            'prophet_lower': forecast['yhat_lower'],
            'prophet_upper': forecast['yhat_upper']
        })
        
        self.forecasts['prophet'] = forecast_df
        
        return forecast_df
    
    def forecast_ml_baseline(self, steps: int = 12) -> pd.DataFrame:
        if 'ml_baseline' not in self.models:
            raise ValueError("ML baseline not fitted. Call fit_baseline_ml first.")
        
        model_info = self.models['ml_baseline']
        model = model_info['model']
        lags = model_info['lags']
        
        last_values = list(model_info['last_values'])
        predictions = []
        
        for _ in range(steps):
            X_pred = np.array(last_values[-lags:]).reshape(1, -1)
            pred = model.predict(X_pred)[0]
            predictions.append(pred)
            last_values.append(pred)
        
        last_date = self.ts.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=steps,
            freq='MS'
        )
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'ml_forecast': predictions
        })
        
        self.forecasts['ml_baseline'] = forecast_df
        
        return forecast_df
    
    def evaluate_model(self, actual: pd.Series, predicted: np.ndarray, 
                      model_name: str) -> Dict:
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        
        metrics = {
            'model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        self.metrics[model_name] = metrics
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        if not self.metrics:
            print("No models evaluated yet.")
            return None
        
        comparison = pd.DataFrame(self.metrics).T
        comparison = comparison.sort_values('MAPE')
        
        print("\n=== Model Comparison ===")
        print(comparison.to_string())
        
        return comparison
    
    def plot_forecast_comparison(self, test: pd.Series = None, figsize=(15, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        
        self.ts.plot(ax=ax, label='Historical', color='black', linewidth=2)
        
        if test is not None:
            test.plot(ax=ax, label='Actual (Test)', color='blue', linewidth=2, linestyle='--')
        
        if 'arima' in self.forecasts:
            arima_fc = self.forecasts['arima']
            ax.plot(arima_fc['date'], arima_fc['arima_forecast'], 
                   label='ARIMA', color='red', linewidth=2)
            ax.fill_between(arima_fc['date'], arima_fc['arima_lower'], 
                           arima_fc['arima_upper'], alpha=0.2, color='red')
        
        if 'prophet' in self.forecasts:
            prophet_fc = self.forecasts['prophet']
            ax.plot(prophet_fc['date'], prophet_fc['prophet_forecast'], 
                   label='Prophet', color='green', linewidth=2)
            ax.fill_between(prophet_fc['date'], prophet_fc['prophet_lower'], 
                           prophet_fc['prophet_upper'], alpha=0.2, color='green')
        
        if 'ml_baseline' in self.forecasts:
            ml_fc = self.forecasts['ml_baseline']
            ax.plot(ml_fc['date'], ml_fc['ml_forecast'], 
                   label='ML Baseline', color='orange', linewidth=2)
        
        ax.set_xlabel('Date')
        ax.set_ylabel(self.value_col.capitalize())
        ax.set_title('Forecast Comparison')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig