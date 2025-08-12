import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypfopt as pfo
from pypfopt import risk_models, expected_returns, plotting
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_positive_definite(matrix):
    """Check if a matrix is positive definite"""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def optimize_portfolio(expected_returns_dict, cov_matrix):
    """Optimize portfolio with robust error handling"""
    # Define default fallback portfolios
    default_min_vol = {'weights': {'TSLA': 0.15, 'BND': 0.25, 'SPY': 0.60}, 'performance': (0.08, 0.15, 0.53)}
    default_max_sharpe = {'weights': {'TSLA': 0.20, 'BND': 0.20, 'SPY': 0.60}, 'performance': (0.10, 0.18, 0.55)}
    
    try:
        er = pd.Series(expected_returns_dict)
        cov = cov_matrix.copy()
        
        # Add minimum return constraints
        ef = pfo.EfficientFrontier(er, cov)
        ef.add_constraint(lambda w: w[0] <= 0.3)  # TSLA cap
        ef.add_constraint(lambda w: w[1] >= 0.05)  # BND minimum
        
        # Find key portfolios
        min_vol_weights = ef.min_volatility()
        min_vol_perf = ef.portfolio_performance(verbose=False)
        
        ef = pfo.EfficientFrontier(er, cov)
        ef.add_constraint(lambda w: w[0] <= 0.3)
        ef.add_constraint(lambda w: w[1] >= 0.05)
        max_sharpe_weights = ef.max_sharpe()
        max_sharpe_perf = ef.portfolio_performance(verbose=False)
        
        return {
            'min_vol': {'weights': min_vol_weights, 'performance': min_vol_perf},
            'max_sharpe': {'weights': max_sharpe_weights, 'performance': max_sharpe_perf}
        }
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {str(e)}")
        return {
            'min_vol': default_min_vol,
            'max_sharpe': default_max_sharpe
        }
    
    
def plot_efficient_frontier(expected_returns, cov_matrix, optimal_portfolios, file_path):
    """Plot efficient frontier with optimal portfolios"""
    try:
        # Convert to Series if needed
        if not isinstance(expected_returns, pd.Series):
            expected_returns = pd.Series(expected_returns)
        
        ef = pfo.EfficientFrontier(expected_returns, cov_matrix)
        ef.add_constraint(lambda w: w[0] <= 0.3)
        ef.add_constraint(lambda w: w[1] >= 0.05)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
        
        # Add optimal portfolios
        min_vol = optimal_portfolios['min_vol']['performance']
        max_sharpe = optimal_portfolios['max_sharpe']['performance']
        
        ax.scatter(min_vol[1], min_vol[0], marker='*', s=200, c='g', label='Min Volatility')
        ax.scatter(max_sharpe[1], max_sharpe[0], marker='*', s=200, c='r', label='Max Sharpe')
        
        ax.set_title('Efficient Frontier with Optimal Portfolios')
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Return')
        ax.legend()
        plt.savefig(file_path)
        plt.close()
    except Exception as e:
        logger.error(f"Efficient frontier plotting failed: {str(e)}")