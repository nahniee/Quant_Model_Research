# workflow.py
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configuration for model training and prediction
CONFIG = {
    'quarterly': {
        'seq_length': 252,
        'forecast_horizon': 65,
        'interval': '1d',
        'model_path': 'quarterly_model.h5',
        'scaler_path': 'quarterly_scaler.pkl',
        'plot_history_points': 120 # Plot last 120 days
    },
    'hourly': {
        'seq_length': 140,
        'forecast_horizon': 7,
        'interval': '1h',
        'model_path': 'hourly_model.h5',
        'scaler_path': 'hourly_scaler.pkl',
        'plot_history_points': 35 # Plot last 35 hours (5 trading days)
    }
}
FEATURE_COUNT = 5    # Open, High, Low, Close, Volume

# Attention and directional_accuracy are redefined to load exactly the same as the trained model
@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_w', shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=False)
        self.b = self.add_weight(name='att_b', shape=(input_shape[1], 1), initializer='zeros', trainable=False)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        alpha = K.softmax(K.squeeze(e, axis=-1))
        context = K.sum(x * K.expand_dims(alpha, axis=-1), axis=1)
        return context

def directional_accuracy(y_true, y_pred):
    true_direction = K.sign(y_true[:, :, 3]) 
    pred_direction = K.sign(y_pred[:, :, 3])
    correct_direction = K.equal(true_direction, pred_direction)
    return K.mean(tf.cast(correct_direction, tf.float32))

def load_prediction_tools(config):
    """Loads the correct model and scaler based on the configuration."""
    try:
        custom_objects = {'Attention': Attention, 'directional_accuracy': directional_accuracy}
        model = load_model(config['model_path'], custom_objects=custom_objects, compile=False)
        scaler = joblib.load(config['scaler_path'])
        print(f"Successfully loaded model ({config['model_path']}) and scaler.")
        return model, scaler
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        return None, None

# (Modify)
def predict_single_ticker(ticker, model, scaler, config, prediction_date=None):
    """Dynamically fetches data and predicts for a single ticker."""
    try:
        seq_length = config['seq_length']
        interval = config['interval']

        # Determine the end date for fetching historical data
        end_date = pd.to_datetime(prediction_date) if prediction_date else pd.Timestamp.now()

        # Calculate how many calendar days of data to download
        if interval == '1d':
            days_to_download = int(seq_length * 1.6) + 10
            start_date = end_date - pd.DateOffset(days=days_to_download)
        elif interval == '1h':
            days_to_download = int(seq_length / 7 * 1.5) + 5
            start_date = end_date - pd.DateOffset(days=days_to_download)

        raw_df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if len(raw_df) < seq_length: 
            return None

        # Save last actual closing price for later reconstruction of absolute predictions
        last_real_close = float(raw_df['Close'].iloc[-1])
        
        processed_df = pd.DataFrame(index=raw_df.index)
        processed_df['Open'] = np.log(raw_df['Open']).diff()
        processed_df['High'] = np.log(raw_df['High']).diff()
        processed_df['Low'] = np.log(raw_df['Low']).diff()
        processed_df['Close'] = np.log(raw_df['Close']).diff()
        processed_df['Volume'] = np.log1p(raw_df['Volume']).diff()
        processed_df.dropna(inplace=True)

        # Prepare last sequence for prediction
        last_sequence_processed = processed_df.iloc[-seq_length:].values
        last_sequence_scaled = scaler.transform(last_sequence_processed)

        # Model input shape: (samples, sequence_length, features)
        X_pred = np.expand_dims(last_sequence_scaled, axis=0)

        # Run prediction
        predicted_scaled = model.predict(X_pred, verbose=0)

        # Reverse scaling
        predicted_processed = scaler.inverse_transform(predicted_scaled[0])
        
        # Reverse log-diff transformation
        predicted_closes = []
        current_log_close = np.log(last_real_close)
        for log_diff in predicted_processed[:, 3]:  # Column index 3 = Close
            current_log_close += log_diff
            predicted_closes.append(np.exp(current_log_close))
        
        # Compute expected growth
        final_predicted_close = predicted_closes[-1]
        growth_rate = (final_predicted_close - last_real_close) / last_real_close

        # Get actual future close price
        actual_future_close = None
        forecast_horizon = config['forecast_horizon']

        # Get actual future close price
        if interval == '1d':
            # Get future data for 65 days
            future_start_date = end_date + pd.Timedelta(days=1)
            future_end_date = future_start_date + pd.DateOffset(days=forecast_horizon * 1.6)
            future_df = yf.download(ticker, 
                                    start=future_start_date, 
                                    end=future_end_date, 
                                    interval='1d', 
                                    progress=False, 
                                    auto_adjust=False)           
            if len(future_df) >= forecast_horizon:
                actual_future_close = float(future_df['Close'].iloc[forecast_horizon - 1])

        else: # '1h'
            # Get future data for the next trading day
            future_start_date = (end_date + pd.Timedelta(days=1)).normalize()
            future_end_date = future_start_date + pd.Timedelta(days=4) # Consider weekends
            future_df = yf.download(ticker, 
                                    start=future_start_date, 
                                    end=future_end_date, 
                                    interval='1h', 
                                    progress=False, 
                                    auto_adjust=False)
            if not future_df.empty:
                # Get the last close price at the 7th hour of the day
                if len(future_df) >= forecast_horizon:
                    actual_future_close = float(future_df['Close'].iloc[forecast_horizon - 1])

        return {'ticker': ticker, 
                'growth_rate': growth_rate, 
                'last_real_close': last_real_close,
                'historical_data': raw_df, 
                'predicted_closes': predicted_closes,
                'actual_future_close': actual_future_close} 
    
    except Exception as e:
        return None

# (Modify)
def plot_prediction(result, config):
    """Dynamically plots results for either daily or hourly predictions."""
    historical_df = result['historical_data']

    # Create date index for forecast horizon
    last_timestamp = historical_df.index[-1]
    forecast_horizon = config['forecast_horizon']
    
    # Create future timestamps
    if config['interval'] == '1d':
        future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(days=1), periods=forecast_horizon, freq='B') # B for Business Day
        time_unit = "Days"
        date_format = '%Y-%m-%d'
    else: # '1h'
        # Find the next trading day and create hourly timestamps
        next_day = last_timestamp.normalize() + pd.Timedelta(days=1)
        if next_day.weekday() == 5: # Saturday
            next_day += pd.Timedelta(days=2)
        elif next_day.weekday() == 6: # Sunday
            next_day += pd.Timedelta(days=1)


        # Assume US market hours 9:30 -> 16:00
        future_timestamps = pd.to_datetime([next_day.replace(hour=h) for h in range(10, 10 + forecast_horizon)])
        time_unit = "Hours"
        date_format = '%Y-%m-%d %H:%M'

    plt.figure(figsize=(14, 7))
    
    # Plot historical and predicted data
    plt.plot(historical_df.index[-config['plot_history_points']:], historical_df['Close'][-config['plot_history_points']:], label='Historical Close Price', color='royalblue')
    plt.plot(future_timestamps, result['predicted_closes'], label='Predicted Close Price', color='darkorange', linestyle='--')
    plt.plot([last_timestamp, future_timestamps[0]], [result['last_real_close'], result['predicted_closes'][0]], color='darkorange', linestyle=':')
    
    # Annotations
    current_price = result['last_real_close']
    plt.scatter(last_timestamp, current_price, color='red', zorder=5)
    plt.annotate(f'Current: ${current_price:.2f}\n({last_timestamp.strftime(date_format)})', (last_timestamp, current_price), textcoords="offset points", xytext=(-10, -30), ha='right', fontweight='bold')
    
    predicted_price = result['predicted_closes'][-1]
    plt.scatter(future_timestamps[-1], predicted_price, color='green', zorder=5)
    plt.annotate(f'Predicted: ${predicted_price:.2f}\n({future_timestamps[-1].strftime(date_format)})', (future_timestamps[-1], predicted_price), textcoords="offset points", xytext=(10, 20), ha='left', fontweight='bold')

    # Actual price
    actual_price = result.get('actual_future_close')
    if actual_price:
        plt.scatter(future_timestamps[-1], actual_price, color='blue', zorder=5, marker='X', s=100)
        plt.annotate(f'Actual: ${actual_price:.2f}', (future_timestamps[-1], actual_price), textcoords="offset points", xytext=(10, -30), ha='left', fontweight='bold', color='blue')

    plt.title(f"{result['ticker']} Stock Price Prediction (Next {forecast_horizon} {time_unit})", fontsize=16)
    plt.xlabel('Date / Time')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# (Modify)
def main(model_type, prediction_date=None):
    config = CONFIG[model_type]
    
    # List of tickers to analyze
    analyze_tickers = [
        'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 
        'NFLX', 'AMD', 'RKLB', 'IONQ', 'SMCI', 'PLTR', 'SMR', 'LCID', 'RIVN',
        'JPM', 'BAC', 'GS', 'WFC', 'V', 'MA', 'TSM', 'ORCL', 'AVGO', 'SOFI',
        'JNJ', 'PFE', 'MRNA', 'UNH', 'LLY', 'ABBV', 'XOM', 'CVX', 'SHEL',
        'WMT', 'COST', 'TGT', 'KO', 'PEP', 'MCD', 'BA', 'CAT', 'LMT', 'DE', 'UNP',
        'RUN', 'WULF'
    ]

    model, scaler = load_prediction_tools(config)

    # Check if model and scaler are loaded successfully
    if model and scaler:
        all_results = []
        print(f"\nStarting model prediction {model_type} (reference date: {prediction_date or 'today'}).")
        for ticker in tqdm(analyze_tickers, desc=f"Predicting ({model_type})"):
            result = predict_single_ticker(ticker, model, scaler, config, prediction_date)
            if result:
                all_results.append(result)
        
        if all_results:
            ranked_results = sorted(all_results, key=lambda x: x['predicted_closes'][-1] - x['last_real_close'], reverse=True)
            top_10_results = ranked_results[:10]

            print(f"\nTop {len(top_10_results)} Predictions for Next {config['forecast_horizon']} { 'Days' if config['interval'] == '1d' else 'Hours'}")
            
            df_data = {
                'Rank': range(1, len(top_10_results) + 1),
                'Ticker': [r['ticker'] for r in top_10_results],
                'Current Close': [f"${r['last_real_close']:.2f}" for r in top_10_results],
                'Predicted Close': [f"${r['predicted_closes'][-1]:.2f}" for r in top_10_results],
                'Actual': [],
                'Accuracy (%)': [],
                'Expected Growth': [f"{r['growth_rate']:.2%}" for r in top_10_results],
                'Predicted - Current': [f"${r['predicted_closes'][-1] - r['last_real_close']:.2f}" for r in top_10_results]
            }
            for r in top_10_results:
                actual = r.get('actual_future_close')
                predicted = r['predicted_closes'][-1]
                if actual:
                    df_data['Actual'].append(f"${actual:.2f}")
                    accuracy = (1 - abs(actual - predicted) / actual) * 100
                    df_data['Accuracy (%)'].append(f"{accuracy:.2f}%")
                else:
                    df_data['Actual'].append("N/A")
                    df_data['Accuracy (%)'].append("N/A")

            df = pd.DataFrame(df_data)

            print(df.to_string(index=False))

            for result in top_10_results:
                plot_prediction(result, config)
    
if __name__ == '__main__':
    # Options: 'quarterly' or 'hourly'
    model_to_run = 'quarterly'
    historical_date = '2025-04-01'
    
    main(model_to_run, prediction_date=historical_date)