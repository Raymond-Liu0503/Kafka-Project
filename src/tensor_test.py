import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.timeseries import TimeSeries
import pandas_ta as ta  # Library for technical analysis

# Your Alpha Vantage API key
api_key = "7HX5TK1UEV1V9TT5"

# Create a TimeSeries object
ts = TimeSeries(key=api_key, output_format='pandas')

# Get historical daily data for a specific symbol (e.g., Apple Inc. - "AAPL")
ticker_symbol = "AAPL"
data, meta_data = ts.get_daily(symbol=ticker_symbol, outputsize="full")

# Rename columns to match expected names
data.rename(columns={
    "1. open": "Open",
    "2. high": "High",
    "3. low": "Low",
    "4. close": "Close",
    "5. volume": "Volume"
}, inplace=True)

# Convert index to datetime
data.index = pd.to_datetime(data.index)

# Sort data by date
data = data.sort_index()

# Calculate technical indicators using pandas_ta
# Simple Moving Average (SMA)
data['SMA'] = ta.sma(data['Close'], timeperiod=14)

# Moving Average Convergence Divergence (MACD)
macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)

# Assign the MACD values to the main DataFrame
data['MACD'] = macd['MACD_12_26_9']
data['MACD_Signal'] = macd['MACDs_12_26_9']
data['MACD_Hist'] = macd['MACDh_12_26_9']


# Relative Strength Index (RSI)
data['RSI'] = ta.rsi(data['Close'], length=14)

# Drop rows with NaN values (due to indicator calculations)
data = data.dropna()

# Reset index to prepare the DataFrame
df = data.reset_index()

# Drop the date column (optional, as it's already in the index)
df = df.drop(columns=['date'])

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI']])

# Function to create sequences of data for LSTM
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])  # Input sequence
        y.append(data[i + seq_length, 3])  # Target: the 'Close' price (column index 3)
    return np.array(X), np.array(y)

seq_length = 50
X_train, y_train = create_sequences(scaled_features, seq_length)

# Define the LSTM model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.LSTM(units=100, return_sequences=False),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1)  # Predict the 'Close' price
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

# Create and train the model
model = create_model()
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the trained model
model.save(f"{ticker_symbol}_lstm_model.h5")
print(f"Model trained and saved as '{ticker_symbol}_lstm_model.h5'")

# Make predictions
last_sequence = X_train[-1:]
predicted_scaled = model.predict(last_sequence)

# Append zeros for other columns to inverse transform correctly
predicted_close = np.zeros((predicted_scaled.shape[0], scaled_features.shape[1]))
predicted_close[:, 3] = predicted_scaled[:, 0]

# Reverse scaling
predicted_close = scaler.inverse_transform(predicted_close)[:, 3]

print(f"Predicted Close price for the next time step: {predicted_close[0]}")
