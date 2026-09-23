# train_model.py
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, Dropout, Layer, RepeatVector, TimeDistributed, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tqdm import tqdm
import datetime
import joblib

"""  
Model configuration guidelines:

* Long term (Quarterly): 
  - Training Frequency: Weekly (e.g., every Sunday night)
  - SEQ_LENGTH = 252 (about 1 trading year of input)
  - FORECAST_DAYS = 65 (approx. 3 months prediction horizon)
  - CNN: 3 layers, 128 filters, kernel_size = 5
  - LSTM: 3 layers, 256 units

* Short term (Hourly): 
  - Training Frequency: Daily (e.g., every night after market close)
  - SEQ_LENGTH = 140 (about 140 hours (20 days) of input)
  - FORECAST_STEPS = 7 (next 7 hours prediction)
  - CNN: 2 layers, 64 filters, kernel_size = 3
  - LSTM: 2 layers, 128 units
"""
# Configuration for model training and prediction
CONFIG = {
    'quarterly': {
        'seq_length': 252,
        'forecast_horizon': 65,
        'interval': '1d', # Daily data
        'cnn_layers': [
            {'filters': 128, 'kernel_size': 5},
            {'filters': 128, 'kernel_size': 5},
            {'filters': 128, 'kernel_size': 5}
        ],
        'lstm_layers': [
            {'units': 256},
            {'units': 256},
            {'units': 256}
        ],
        'data_start_date': '2010-01-01',
        'validation_period': {'years': 2}
    },
    'hourly': {
        'seq_length': 140,
        'forecast_horizon': 7,
        'interval': '1h', # Hourly data
        'cnn_layers': [
            {'filters': 64, 'kernel_size': 3},
            {'filters': 64, 'kernel_size': 3}
        ],
        'lstm_layers': [
            {'units': 128},
            {'units': 128}
        ],
        # yfinance provides max 730 days of hourly data
        'data_start_date': (datetime.date.today() - datetime.timedelta(days=729)).strftime('%Y-%m-%d'),
        'validation_period': {'months': 3}
    }
}
FEATURE_COUNT = 5 # Open, High, Low, Close, Volume

# Custom Attention layer and custom metric (Directional Accuracy)
@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs): super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weight matrix for attention scoring
        self.W = self.add_weight(name='att_w', shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True) 
        # Bias term
        self.b = self.add_weight(name='att_b', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # Attention scores
        e = K.tanh(K.dot(x, self.W) + self.b)
        alpha = K.softmax(K.squeeze(e, axis=-1))
        # Weighted sum to generate context vector
        context = K.sum(x * K.expand_dims(alpha, axis=-1), axis=1)
        return context

# Measures accuracy of predicted price movement direction
def directional_accuracy(y_true, y_pred):
    # Kept so old model files still load. The sign of a scaled value isn't the return's direction.
    true_direction = K.sign(y_true[:, :, 3]) # Close is the 4th feature (index 3)
    pred_direction = K.sign(y_pred[:, :, 3])
    correct_direction = K.equal(true_direction, pred_direction)
    return K.mean(tf.cast(correct_direction, tf.float32))

@tf.keras.utils.register_keras_serializable(package="clam")
class ReturnDirectionalAccuracy(tf.keras.metrics.Metric):
    """Direction in original return units, using the training scaler's image of zero."""
    def __init__(self, scaled_zero, name="return_directional_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.scaled_zero = float(scaled_zero)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        correct = tf.cast(tf.equal(tf.sign(y_true[:, :, 3] - self.scaled_zero),
                                   tf.sign(y_pred[:, :, 3] - self.scaled_zero)), self.dtype)
        weights = tf.ones_like(correct)
        if sample_weight is not None:
            sw = tf.cast(sample_weight, self.dtype)
            if sw.shape.rank == 1:
                sw = sw[:, None]
            weights *= sw
        self.correct.assign_add(tf.reduce_sum(correct * weights))
        self.count.assign_add(tf.reduce_sum(weights))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.count)

    def reset_state(self):
        self.correct.assign(0); self.count.assign(0)

    def get_config(self):
        return {**super().get_config(), "scaled_zero": self.scaled_zero}


# Model creation function
def create_model(config, scaler=None, legacy_metric=False):
    if not legacy_metric and scaler is None:
        raise ValueError("A fitted scaler is required for return-direction accuracy")
    seq_len = config['seq_length']
    forecast_horizon = config['forecast_horizon']
    
    encoder_inputs = Input(shape=(seq_len, FEATURE_COUNT))
    x = encoder_inputs

    # Build CNN layers dynamically
    for layer_params in config['cnn_layers']:
        x = Conv1D(filters=layer_params['filters'], kernel_size=layer_params['kernel_size'], padding='causal', activation='relu')(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)

    # Build LSTM layers dynamically
    for layer_params in config['lstm_layers']:
        x = LSTM(units=layer_params['units'], return_sequences=True)(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
    
    # Attention context vector
    context_vector = Attention()(x)

    # Decoder for sequence forecasting
    decoder_lstm_units = config['lstm_layers'][-1]['units']
    decoder_inputs = RepeatVector(forecast_horizon)(context_vector)
    decoder_lstm = LSTM(decoder_lstm_units, return_sequences=True)(decoder_inputs)
    outputs = TimeDistributed(Dense(FEATURE_COUNT))(decoder_lstm)
    
    model = Model(encoder_inputs, outputs)

    # Compile with custom metric
    # Huber loss and AdamW hold up well on volatile stock data with frequent outliers
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4), 
                  loss='huber', 
                  metrics=[directional_accuracy] if legacy_metric else
                  [ReturnDirectionalAccuracy(scaler.min_[3])])
    return model

# Sliding window sequence generation
def create_sequences(data, seq_len, forecast_len):
    X, y = [], []
    for i in range(len(data) - seq_len - forecast_len + 1):
        X.append(data[i:(i + seq_len)])
        y.append(data[(i + seq_len):(i + seq_len + forecast_len)])
    return np.array(X), np.array(y)


# Main training loop (Modify)
def main(model_type, training_end_date=None, legacy_metric=False):
    print(f"Starting training for [{model_type.upper()}] model")
    
    # Load the correct configuration
    config = CONFIG[model_type]
    seq_length = config['seq_length']
    forecast_horizon = config['forecast_horizon']
    
    # Training tickers (94 stocks across several sectors)
    TRAINING_TICKERS = [
        # Technology
        'AAPL',  # Apple
        'MSFT',  # Microsoft
        'GOOGL', # Alphabet (Google)
        'NVDA',  # NVIDIA
        'AVGO',  # Broadcom
        'ORCL',  # Oracle
        'ADBE',  # Adobe
        'CRM',   # Salesforce
        'CSCO',  # Cisco Systems
        'AMD',   # Advanced Micro Devices
        'INTC',  # Intel
        'QCOM',  # Qualcomm
        'IBM',   # IBM
        'TXN',   # Texas Instruments
        'AMAT',  # Applied Materials
        'MU',    # Micron Technology
        'LRCX',  # Lam Research
        'ADI',   # Analog Devices
        'SNPS',  # Synopsys
        'CDNS',  # Cadence Design Systems

        # Financials
        'JPM',   # JPMorgan Chase
        'BAC',   # Bank of America
        'WFC',   # Wells Fargo
        'GS',    # Goldman Sachs
        'MS',    # Morgan Stanley
        'C',     # Citigroup
        'BLK',   # BlackRock
        'SPGI',  # S&P Global
        'AXP',   # American Express
        'V',     # Visa
        'MA',    # Mastercard
        'PYPL',  # PayPal
        'SCHW',  # Charles Schwab
        'PNC',   # PNC Financial Services
        'USB',   # U.S. Bancorp

        # Healthcare
        'JNJ',   # Johnson & Johnson
        'UNH',   # UnitedHealth Group
        'LLY',   # Eli Lilly
        'PFE',   # Pfizer
        'MRK',   # Merck & Co.
        'ABBV',  # AbbVie
        'TMO',   # Thermo Fisher Scientific
        'DHR',   # Danaher
        'AMGN',  # Amgen
        'GILD',  # Gilead Sciences
        'MDT',   # Medtronic
        'ISRG',  # Intuitive Surgical
        'SYK',   # Stryker
        'BSX',   # Boston Scientific
        'CI',    # Cigna

        # Consumer
        'AMZN',  # Amazon
        'TSLA',  # Tesla
        'WMT',   # Walmart
        'COST',  # Costco
        'HD',    # Home Depot
        'MCD',   # McDonald's
        'NKE',   # Nike
        'SBUX',  # Starbucks
        'TGT',   # Target
        'LOW',   # Lowe's
        'KO',    # Coca-Cola
        'PEP',   # PepsiCo
        'PG',    # Procter & Gamble
        'CL',    # Colgate-Palmolive
        'KMB',   # Kimberly-Clark
        'GIS',   # General Mills
        'F',     # Ford Motor
        'GM',    # General Motors
        'DIS',   # Disney
        'NFLX',  # Netflix

        # Industrials 
        'CAT',   # Caterpillar
        'BA',    # Boeing
        'LMT',   # Lockheed Martin
        'RTX',   # RTX Corporation (Raytheon)
        'HON',   # Honeywell
        'UNP',   # Union Pacific
        'UPS',   # United Parcel Service
        'FDX',   # FedEx
        'DE',    # Deere & Company
        'GE',    # General Electric
        'MMM',   # 3M
        'GD',    # General Dynamics
        'NOC',   # Northrop Grumman
        'WM',    # Waste Management
        'ETN',   # Eaton Corporation

        # Energy
        'XOM',   # Exxon Mobil
        'CVX',   # Chevron
        'SHEL',  # Shell
        'COP',   # ConocoPhillips
        'SLB',   # Schlumberger
        'EOG',   # EOG Resources
        'MPC',   # Marathon Petroleum
        'VLO',   # Valero Energy
        'PSX',   # Phillips 66
        'OXY',   # Occidental Petroleum
        'HAL',   # Halliburton
        'KMI',   # Kinder Morgan
        'WMB',   # Williams Companies
        'DVN'    # Devon Energy
    ]

    # Data download and preprocessing
    all_processed_dfs = []
    failed_tickers = []
    
    # Modify
    end_date = pd.to_datetime(training_end_date) if training_end_date else pd.Timestamp.now()
    
    for ticker in tqdm(TRAINING_TICKERS, desc="Processing Tickers"):
        try:
            raw_df = yf.download(ticker, 
                                 start=config['data_start_date'], 
                                 end=end_date, 
                                 interval=config['interval'], 
                                 progress=False,
                                 auto_adjust=False)
            if raw_df.empty: raise ValueError("No data from yfinance.")


            processed_df = pd.DataFrame(index=raw_df.index)
            processed_df['Open'] = np.log(raw_df['Open']).diff()
            processed_df['High'] = np.log(raw_df['High']).diff()
            processed_df['Low'] = np.log(raw_df['Low']).diff()
            processed_df['Close'] = np.log(raw_df['Close']).diff()
            processed_df['Volume'] = np.log1p(raw_df['Volume']).diff()

            # Drop initial NaN rows caused by diff()
            processed_df.dropna(inplace=True)
            all_processed_dfs.append(processed_df)
        except Exception as e:
            failed_tickers.append(ticker)

    if not failed_tickers: 
        print("All tickers processed successfully!")
    else: 
        print(f"Failed to process: {failed_tickers}")

    # Merge all tickers and sort by date
    full_processed_df = pd.concat(all_processed_dfs).sort_index()

    # Data splitting
    last_date = full_processed_df.index.max()
    split_date = last_date - pd.DateOffset(**config['validation_period'])
    train_df = full_processed_df[full_processed_df.index < split_date]
    val_df = full_processed_df[full_processed_df.index >= split_date]
    
    print(f"Training data period: {train_df.index.min()} ~ {train_df.index.max()}")
    print(f"Validation data period: {val_df.index.min()} ~ {val_df.index.max()}")

    # Scaling and Sequencing
    scaler = MinMaxScaler(feature_range=(-1, 1))    # Use (-1,1) since returns can be negative
    scaler.fit(train_df)
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    
    # Generate sequences
    X_train, y_train = create_sequences(train_scaled, seq_length, forecast_horizon)
    X_val, y_val = create_sequences(val_scaled, seq_length, forecast_horizon)

    # Build and train model
    model = create_model(config, scaler=scaler, legacy_metric=legacy_metric)
    model.summary()
    
    callbacks = [
        # Save the best model based on directional accuracy
        EarlyStopping(monitor='val_directional_accuracy' if legacy_metric else 'val_return_directional_accuracy', mode='max', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6)
    ]
    
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks)

    # Save Artifacts
    model_filename = f"{model_type}_model.h5"
    scaler_filename = f"{model_type}_scaler.pkl"
    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)
    import json, hashlib
    from pathlib import Path
    metadata = {"requested_training_end": training_end_date,
                "data_first_date": str(full_processed_df.index.min()),
                "data_last_date": str(full_processed_df.index.max()),
                "train_last_date": str(train_df.index.max()),
                "validation_first_date": str(val_df.index.min()),
                "legacy_metric": legacy_metric,
                "sequence_construction": "legacy cross-ticker date-sorted rows; known defect",
                "model_sha256": hashlib.sha256(Path(model_filename).read_bytes()).hexdigest()}
    Path(f"{model_type}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nTraining complete. {model_filename} and {scaler_filename} have been saved.")

if __name__ == '__main__':
    historical_training_date = '2025-04-01'
    # To train a different model, options are 'quarterly' or 'hourly'
    model_to_train = 'quarterly' 
    print(f"Starting notebook training for the '{model_to_train}' model.")
    main(model_to_train, training_end_date=historical_training_date)