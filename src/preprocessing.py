import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from config import RISK_FREE_RATE
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

def clean_data(data):
    """Handle missing values and calculate returns"""
    # Convert to float to avoid downcasting warnings
    data = data.astype(float)
    
    # Forward fill then backfill
    data = data.ffill().bfill()
    
    # Ensure no NaNs remain
    if data.isnull().any().any():
        logger.warning("NaNs detected after cleaning. Filling with last valid value")
        data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Handle any remaining NaNs
    returns = returns.fillna(0)
    
    return data, returns

def calculate_volatility(returns, window=30):
    """Calculate rolling volatility"""
    return returns.rolling(window=window).std() * np.sqrt(252)

def adf_test(series):
    """Perform Augmented Dickey-Fuller test for stationarity"""
    if series.dropna().empty:
        return {
            'adf_statistic': np.nan,
            'p_value': np.nan,
            'critical_values': {}
        }
    
    result = adfuller(series.dropna())
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4]
    }

def calculate_risk_metrics(returns):
    """Calculate VaR and Sharpe Ratio with robust handling"""
    try:
        if returns.dropna().empty:
            return {
                'var_95': np.nan,
                'annualized_return': np.nan,
                'annualized_vol': np.nan,
                'sharpe_ratio': np.nan
            }
        
        var_95 = np.percentile(returns, 5)
        annualized_return = returns.mean() * 252
        annualized_vol = returns.std() * np.sqrt(252)
        
        if annualized_vol > 0:
            sharpe = (annualized_return - RISK_FREE_RATE) / annualized_vol
        else:
            sharpe = np.nan
        
        return {
            'var_95': var_95,
            'annualized_return': annualized_return,
            'annualized_vol': annualized_vol,
            'sharpe_ratio': sharpe
        }
    except Exception as e:
        logger.error(f"Risk metrics calculation failed: {str(e)}")
        return {
            'var_95': np.nan,
            'annualized_return': np.nan,
            'annualized_vol': np.nan,
            'sharpe_ratio': np.nan
        }

def generate_eda_plots(data, returns, save_dir='reports/figures'):
    """Generate EDA visualizations for interim report"""
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 1. Normalized Prices Plot
        plt.figure(figsize=(12, 6))
        (data / data.iloc[0] * 100).plot()
        plt.title('Normalized Asset Prices (Base 100)')
        plt.ylabel('Price (Index)')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'normalized_prices.png'))
        plt.close()
        
        # 2. Volatility Analysis (TSLA 30-day rolling volatility)
        plt.figure(figsize=(12, 6))
        tsla_volatility = calculate_volatility(returns['TSLA'])
        tsla_volatility.plot()
        plt.title('TSLA 30-Day Rolling Volatility')
        plt.ylabel('Annualized Volatility')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'volatility_analysis.png'))
        plt.close()
        
        # 3. Returns Distribution
        plt.figure(figsize=(12, 6))
        sns.kdeplot(returns['TSLA'], label='TSLA', fill=True)
        sns.kdeplot(returns['SPY'], label='SPY', fill=True)
        sns.kdeplot(returns['BND'], label='BND', fill=True)
        plt.title('Distribution of Daily Returns')
        plt.xlabel('Daily Return')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'returns_distribution.png'))
        plt.close()
        
        logger.info("Generated EDA visualizations for interim report")
    except Exception as e:
        logger.error(f"Failed to generate EDA plots: {str(e)}")