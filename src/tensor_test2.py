import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from alpha_vantage.timeseries import TimeSeries
import pandas_ta as ta  # Library for technical analysis
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import matplotlib.pyplot as plt

# Your Alpha Vantage API key
api_key = "7HX5TK1UEV1V9TT5"

# Create a TimeSeries object
ts = TimeSeries(key=api_key, output_format='pandas')

# Get historical daily data for a specific symbol
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

# Calculate technical indicators
data['SMA'] = ta.sma(data['Close'], length=14)
macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
data['MACD'] = macd['MACD_12_26_9']
data['MACD_Signal'] = macd['MACDs_12_26_9']
data['MACD_Hist'] = macd['MACDh_12_26_9']
data['RSI'] = ta.rsi(data['Close'], length=14)

# Drop rows with NaN values
data = data.dropna()

# Automatically determine ARIMA order using auto_arima
stepwise_fit = auto_arima(data['Close'], seasonal=False, trace=True, suppress_warnings=True)
optimal_p, optimal_d, optimal_q = stepwise_fit.order
print(f"Optimal ARIMA order: ({optimal_p}, {optimal_d}, {optimal_q})")

# Fit ARIMA model with optimal parameters
arima_model = ARIMA(data['Close'], order=(optimal_p, optimal_d, optimal_q))
arima_fit = arima_model.fit()

# Get ARIMA predictions (for the linear component)
arima_predictions = arima_fit.predict(start=0, end=len(data)-1)

# Calculate residuals (ARIMA prediction - actual data)
residuals = data['Close'] - arima_predictions

# Normalize residuals
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_residuals = scaler.fit_transform(residuals.values.reshape(-1, 1))

# Function to create sequences of data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 50
X_train, y_train = create_sequences(scaled_residuals, seq_length)

# Define and train LSTM model
def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(units=50, return_sequences=False),
        tf.keras.layers.Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

lstm_model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict residuals using the LSTM model
scaled_residuals_pred = lstm_model.predict(X_train)

# Inverse transform the scaled residuals to original scale
residuals_pred = scaler.inverse_transform(scaled_residuals_pred.reshape(-1, 1)).flatten()

# Combine ARIMA predictions and LSTM-predicted residuals
final_predictions = arima_predictions[seq_length:] + residuals_pred

# Actual values for comparison
actual_values = data['Close'].values[seq_length:]

# Calculate evaluation metrics
mae = mean_absolute_error(actual_values, final_predictions)
rmse = np.sqrt(mean_squared_error(actual_values, final_predictions))

# Calculate accuracy percentage
mean_actual = np.mean(actual_values)
mae_accuracy = 100 - (mae / mean_actual * 100)
rmse_accuracy = 100 - (rmse / mean_actual * 100)

# Print results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"MAE-Based Accuracy: {mae_accuracy:.2f}%")
print(f"RMSE-Based Accuracy: {rmse_accuracy:.2f}%")

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(data.index[seq_length:], actual_values, label="Actual Values", color="blue")
plt.plot(data.index[seq_length:], final_predictions, label="ARIMA+LSTM Predictions", color="red")
plt.title("Actual vs Predicted Values")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.savefig("arima_lstm_predictions.png")

# Choose num of days to forecast
days_to_forecast = 7
arima_forecast = arima_fit.forecast(steps=days_to_forecast)

# Prepare residuals for LSTM forecasting
last_sequence = scaled_residuals[-seq_length:]
lstm_forecast = []

for _ in range(days_to_forecast):
    input_sequence = last_sequence.reshape(1, seq_length, 1)
    next_residual = lstm_model.predict(input_sequence, verbose=0)
    lstm_forecast.append(next_residual[0, 0])
    last_sequence = np.append(last_sequence[1:], next_residual)

lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()

final_forecast = arima_forecast + lstm_forecast

# Step 4: Create a DataFrame for the forecast
last_date = data.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_forecast)
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Forecast': final_forecast
})

print(forecast_df)

# Plot the forecast
recent_days = 100
recent_data = data.iloc[-recent_days:]

# Plot recent historical data and forecast
plt.figure(figsize=(12, 6))
plt.plot(recent_data.index, recent_data['Close'], label="Recent Historical Data", color="blue")
plt.plot(forecast_dates, final_forecast, label=f"{days_to_forecast}-Day Forecast", color="red", linestyle="--")

# Limit range
start_date = recent_data.index[-50]  # Last 50 days of recent data
end_date = forecast_dates[-1]  
plt.xlim(start_date, end_date)

plt.title(f"{days_to_forecast}-Day ARIMA+LSTM Forecast (Recent Data)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()

# Save and show the plot
plt.savefig(f"recent_{days_to_forecast}_day_forecast.png")