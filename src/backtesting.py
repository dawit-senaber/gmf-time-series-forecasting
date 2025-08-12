import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import BACKTEST_START, BACKTEST_END, RISK_FREE_RATE
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backtest_strategy(data, strategy_weights_dict, benchmark_weights_dict):
    """Backtest strategy against benchmark with fixed weight calculation"""
    try:
        bt_data = data.loc[BACKTEST_START:BACKTEST_END]
        returns = bt_data.pct_change().dropna()
        
        # Handle NaNs in returns
        returns = returns.fillna(0)
        
        # Convert weights to arrays
        tickers = list(strategy_weights_dict.keys())
        strategy_weights = np.array([strategy_weights_dict[t] for t in tickers])
        benchmark_weights = np.array([benchmark_weights_dict[t] for t in tickers])
        
        # Initialize portfolio values
        strategy_value = 1.0
        benchmark_value = 1.0
        strategy_values = [1.0]
        benchmark_values = [1.0]
        
        # Simulate daily trading
        for i in range(len(returns)):
            # Get daily returns as array
            daily_returns = returns.iloc[i].values
            
            # Strategy performance
            strategy_return = np.dot(strategy_weights, daily_returns)
            strategy_value *= (1 + strategy_return)
            
            # Benchmark performance
            benchmark_return = np.dot(benchmark_weights, daily_returns)
            benchmark_value *= (1 + benchmark_return)
            
            strategy_values.append(strategy_value)
            benchmark_values.append(benchmark_value)
        
        # Create results series
        dates = bt_data.index[1:]  # Skip first date (no return)
        cumulative_strategy = pd.Series(strategy_values[1:], index=dates)
        cumulative_benchmark = pd.Series(benchmark_values[1:], index=dates)
        
        # Calculate performance metrics
        def calculate_performance(values_series):
            if len(values_series) < 2:
                return 0.0, 0.0, 0.0
                
            daily_returns = values_series.pct_change().dropna()
            total_return = values_series.iloc[-1] - 1
            volatility = daily_returns.std() * np.sqrt(252)
            
            if volatility > 0:
                sharpe = (daily_returns.mean() * 252 - RISK_FREE_RATE) / volatility
            else:
                sharpe = 0.0
                
            return total_return, volatility, sharpe
        
        strategy_perf = calculate_performance(cumulative_strategy)
        benchmark_perf = calculate_performance(cumulative_benchmark)
        
        return {
            'strategy': cumulative_strategy,
            'benchmark': cumulative_benchmark,
            'strategy_perf': strategy_perf,
            'benchmark_perf': benchmark_perf
        }
    except Exception as e:
        logger.error(f"Backtesting failed: {str(e)}")
        return {
            'strategy': pd.Series(),
            'benchmark': pd.Series(),
            'strategy_perf': (0.0, 0.0, 0.0),
            'benchmark_perf': (0.0, 0.0, 0.0)
        }

def plot_backtest_results(results, file_path):
    """Plot backtesting performance"""
    try:
        plt.figure(figsize=(12, 6))
        
        if not results['strategy'].empty:
            plt.plot(results['strategy'], label='Optimized Strategy')
        if not results['benchmark'].empty:
            plt.plot(results['benchmark'], label='60/40 Benchmark')
            
        plt.title('Backtesting Performance')
        plt.ylabel('Portfolio Value (Base 1.0)')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()
    except Exception as e:
        logger.error(f"Backtest plotting failed: {str(e)}")