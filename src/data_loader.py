import yfinance as yf
import pandas as pd
import numpy as np
import logging
from config import TICKERS, START_DATE, END_DATE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_financial_data():
    """Fetch financial data with robust API handling"""
    try:
        # Attempt batch download with explicit auto_adjust=False
        logger.info("Attempting batch download from Yahoo Finance")
        data = yf.download(
            TICKERS, 
            start=START_DATE, 
            end=END_DATE,
            progress=False,
            auto_adjust=False  # Get both Close and Adj Close
        )
        
        # Check if we have Adj Close column
        if 'Adj Close' in data.columns.get_level_values(0):
            adj_close = data['Adj Close'].copy()
            logger.info("Successfully downloaded financial data (Adj Close)")
            return adj_close
        elif 'Close' in data.columns.get_level_values(0):
            logger.warning("Using Close as Adj Close (auto_adjust=True)")
            return data['Close'].copy()
        else:
            logger.warning("No price columns found in batch download")
            raise ValueError("No price columns found")
            
    except Exception as e:
        logger.warning(f"Batch download failed: {str(e)}")
    
    try:
        # Individual download fallback
        logger.info("Attempting individual symbol downloads")
        prices = pd.DataFrame()
        for ticker in TICKERS:
            try:
                logger.info(f"Downloading {ticker}")
                df = yf.download(
                    ticker, 
                    start=START_DATE, 
                    end=END_DATE,
                    progress=False,
                    auto_adjust=False
                )
                
                if not df.empty:
                    # Check for adjusted close column
                    if 'Adj Close' in df.columns:
                        prices[ticker] = df['Adj Close']
                        logger.info(f"Got Adj Close for {ticker}")
                    elif 'Close' in df.columns:
                        prices[ticker] = df['Close']
                        logger.warning(f"Using Close for {ticker}")
                    else:
                        logger.warning(f"No price columns for {ticker}")
                else:
                    logger.warning(f"No data for {ticker}")
            except Exception as e:
                logger.warning(f"Failed to download {ticker}: {str(e)}")
        
        if not prices.empty:
            return prices
        else:
            raise ValueError("No data downloaded")
            
    except Exception as e:
        logger.warning(f"Individual downloads failed: {str(e)}")
    
    # Final fallback
    logger.warning("All download attempts failed. Using synthetic historical data")
    return generate_synthetic_data()

def generate_synthetic_data():
    """Generate realistic synthetic financial data"""
    logger.info("Generating synthetic data for all assets")
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    synthetic_data = pd.DataFrame(index=dates)
    
    # Set realistic returns
    np.random.seed(42)
    
    # TSLA: High volatility growth stock (15% annual return)
    tsla_base = 250
    tsla_drift = 0.0006  # ~15% annual return
    tsla_vol = 0.035
    tsla_prices = [tsla_base]
    for _ in range(1, len(dates)):
        ret = tsla_drift + tsla_vol * np.random.randn()
        tsla_prices.append(tsla_prices[-1] * (1 + ret))
    synthetic_data['TSLA'] = tsla_prices
    
    # BND: Low volatility bond ETF (4% annual return)
    bnd_base = 80
    bnd_drift = 0.00016  # ~4% annual return
    bnd_vol = 0.006
    bnd_prices = [bnd_base]
    for _ in range(1, len(dates)):
        ret = bnd_drift + bnd_vol * np.random.randn()
        bnd_prices.append(bnd_prices[-1] * (1 + ret))
    synthetic_data['BND'] = bnd_prices
    
    # SPY: Moderate volatility stock index (9% annual return)
    spy_base = 200
    spy_drift = 0.00036  # ~9% annual return
    spy_vol = 0.015
    spy_prices = [spy_base]
    for _ in range(1, len(dates)):
        ret = spy_drift + spy_vol * np.random.randn()
        spy_prices.append(spy_prices[-1] * (1 + ret))
    synthetic_data['SPY'] = spy_prices
    
    return synthetic_data

def save_data(data, file_path):
    """Save data to CSV"""
    data.to_csv(file_path)

def load_saved_data(file_path):
    """Load data from CSV"""
    return pd.read_csv(file_path, index_col=0, parse_dates=True)