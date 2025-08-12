import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import SEQ_LENGTH, FORECAST_STEPS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sequences(data, seq_length):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_arima(train_data):
    """Train ARIMA model"""
    # Remove any NaN values
    train_clean = train_data.dropna()
    
    if len(train_clean) < 10:
        logger.error("Not enough data for ARIMA training")
        return None
    
    try:
        auto_model = auto_arima(
            train_clean, seasonal=False, 
            suppress_warnings=True, stepwise=True,
            error_action='ignore'
        )
        arima_model = ARIMA(train_clean, order=auto_model.order)
        arima_fit = arima_model.fit()
        return arima_fit
    except Exception as e:
        logger.error(f"ARIMA training failed: {str(e)}")
        return None

def train_lstm(train_data):
    """Train LSTM model with improved stability"""
    # Ensure data is properly formatted
    if train_data.isnull().any().any():
        train_data = train_data.fillna(method='ffill').fillna(method='bfill')
    
    train_clean = train_data.values
    
    if len(train_clean) < SEQ_LENGTH * 2:
        logger.error(f"Not enough data for LSTM training (need at least {SEQ_LENGTH*2} points)")
        return None, None
    
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(train_clean)
        
        # Create sequences
        X_train, y_train = create_sequences(scaled_data, SEQ_LENGTH)
        
        # Reshape for LSTM [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Build model with gradient clipping
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(1)
        ])
        
        # Use Adam with gradient clipping
        optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
        model.compile(optimizer=optimizer, loss='mse')
        
        # Add early stopping
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True
        )
        
        model.fit(
            X_train, y_train, 
            epochs=100, 
            batch_size=32, 
            verbose=0,
            callbacks=[callback]
        )
        
        return model, scaler
    except Exception as e:
        logger.error(f"LSTM training failed: {str(e)}")
        return None, None

def generate_forecast(model, scaler, history):
    """Generate forecast using LSTM model - FIXED INDEX HANDLING"""
    if model is None or scaler is None or len(history) < SEQ_LENGTH:
        return np.full(FORECAST_STEPS, np.nan)
    
    try:
        # Prepare input sequence - ensure 2D array
        last_sequence = history[-SEQ_LENGTH:].values.reshape(-1, 1)
        scaled_sequence = scaler.transform(last_sequence)
        forecast = []
        
        for _ in range(FORECAST_STEPS):
            # Ensure proper input shape [1, SEQ_LENGTH, 1]
            x_input = scaled_sequence[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
            pred = model.predict(x_input, verbose=0)[0,0]
            forecast.append(pred)
            
            # Update sequence with new prediction (as 2D array)
            new_input = np.array([[pred]])
            scaled_sequence = np.vstack([scaled_sequence, new_input])
        
        price_forecast = scaler.inverse_transform(
            np.array(forecast).reshape(-1, 1)
        ).flatten()
        
        return price_forecast
    except Exception as e:
        logger.error(f"Forecast generation failed: {str(e)}")
        return np.full(FORECAST_STEPS, np.nan)

def evaluate_forecast(actual, forecast):
    """Calculate evaluation metrics with robust alignment"""
    try:
        # Ensure both are Series with proper indexes
        if not isinstance(actual, pd.Series):
            actual = pd.Series(actual)
        if not isinstance(forecast, pd.Series):
            forecast = pd.Series(forecast, index=actual.index[:len(forecast)])
        
        # Align lengths by trimming forecast
        min_len = min(len(actual), len(forecast))
        actual = actual.iloc[:min_len]
        forecast = forecast.iloc[:min_len]
        
        # Filter out NaN values
        valid_mask = ~np.isnan(actual) & ~np.isnan(forecast)
        actual = actual[valid_mask]
        forecast = forecast[valid_mask]
        
        if len(actual) == 0:
            return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}
            
        mae = mean_absolute_error(actual, forecast)
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        
        return {'mae': mae, 'rmse': rmse}
    except Exception as e:
        logger.error(f"Forecast evaluation failed: {str(e)}")
        return {'mae': np.nan, 'rmse': np.nan}