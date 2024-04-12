"""
Cryptocurrency Trading Strategies and Predictions

This script contains various trading strategies and predictive models for cryptocurrency price movements, implemented using Python and several libraries including pandas, numpy, ta, sklearn, and matplotlib. Here's a breakdown of what each section does:

1. Data Loading and Preparation:
    - Loads historical cryptocurrency data from a CSV file.
    - Converts timestamps to datetime format and sets 'timestamp' column as index.
    - Defines the market index using the 'marketCap' column.

2. Bullish and Bearish Engulfing Pattern Detection:
    - Defines functions to detect bullish and bearish engulfing patterns in the data.

3. Strategy Returns and Metrics:
    - Computes percentage variation and strategy returns based on detected patterns.
    - Calculates Sortino Ratio, Beta Ratio, CAPM Metric, and Maximum Drawdown.

4. Support and Resistance Trading Strategy:
    - Implements a trading strategy based on RSI and SMA indicators.
    - Generates trading signals for buying and selling.

5. Scalping Trading Strategy:
    - Defines a scalping trading strategy based on RSI and SMA indicators.

6. Predictive Modeling:
    - Predicts average price for the next 24 hours.
    - Sets support and resistance levels.
    - Implements momentum-based prediction strategy and utilizes Random Forest Regressor for advanced price prediction.

7. Visualization:
    - Plots closing prices, predicted prices, support and resistance levels, and the next 24 hours prediction window.

Please note that this script provides a comprehensive approach to cryptocurrency trading and prediction and can be further customized or extended based on specific requirements or preferences.

"""

import pandas as pd
import numpy as np
from ta.momentum import rsi
from ta.trend import sma_indicator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('WIF_1M_graph_coinmarketcap.csv', sep=';')

# Convert 'timestamp' column to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Set the 'timestamp' column as the index
data = data.set_index('timestamp')

# Use 'marketCap' column as the market index
market_index = data['marketCap']

# Function to detect bullish and bearish engulfing patterns
def detect_engulfing(data):
    data['bullish_engulfing'] = (data['open'] > data['close'].shift(1)) & (data['close'] > data['open'].shift(1)) & (data['open'] < data['open'].shift(1)) & (data['close'] > data['close'].shift(1))
    data['bearish_engulfing'] = (data['open'] < data['close'].shift(1)) & (data['close'] < data['open'].shift(1)) & (data['open'] > data['open'].shift(1)) & (data['close'] < data['close'].shift(1))

detect_engulfing(data)

# Compute percentage variation
data['pct_change'] = data['close'].pct_change()

# Compute strategy returns
data['strategy_returns'] = np.where(data['bullish_engulfing'], data['pct_change'], np.where(data['bearish_engulfing'], -data['pct_change'], 0))

# Sortino Ratio
def sortino_ratio(returns, target_return=0, periods=252):
    downside_returns = np.minimum(returns - target_return, 0)
    volatility = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(periods)
    return np.mean(returns) / volatility

# Beta Ratio
def beta_ratio(data, market_index):
    X = market_index.values.reshape(-1, 1)
    y = data['close'].values
    model = LinearRegression().fit(X, y)
    return model.coef_[0]

# CAPM
def capm(data, market_index, risk_free_rate=0.02):
    beta = beta_ratio(data, market_index)
    market_returns = market_index.pct_change().mean() * 252
    return risk_free_rate + beta * (market_returns - risk_free_rate)

# DrawDown
def drawdown(data):
    cumulative = data['close'].cumsum()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

# Support and Resistance Trading Strategy
data['rsi'] = rsi(data['close'], window=14)
data['sma'] = sma_indicator(data['close'], window=20)

# Trading Signals
data['signal'] = 0
data.loc[data['rsi'] < 30, 'signal'] = 1  # Buy
data.loc[data['rsi'] > 70, 'signal'] = -1  # Sell

# Scalping Trading Strategy
data['scalp_signal'] = np.where((data['close'] > data['sma']) & (data['rsi'] > 50), 1, np.where((data['close'] < data['sma']) & (data['rsi'] < 50), -1, 0))

# Predict average price for next 24 hours
last_24_hours = data.tail(24)
min_price = last_24_hours['low'].min()
max_price = last_24_hours['high'].max()

print(f"Minimum predicted price for the next 24 hours: {min_price}")
print(f"Maximum predicted price for the next 24 hours: {max_price}")

# Set target levels (support and resistance)
data['support'] = data['close'].rolling(window=20).min()
data['resistance'] = data['close'].rolling(window=20).max()

# Momentum-based prediction strategy
data['momentum'] = data['close'].pct_change(periods=10)
data['signal_momentum'] = np.where(data['momentum'] > 0, 1, -1)

# Advanced price prediction using Random Forest Regressor
features = ['open', 'high', 'low', 'volume']
X = data[features].shift(1).dropna()
y = data['close'].shift(1).dropna()

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

predicted_prices = model.predict(X)
data['predicted_price'] = np.concatenate([data['close'].iloc[:1], predicted_prices])

# Compute other metrics
sortino = sortino_ratio(data['pct_change'])
market_index = data['marketCap']
beta = beta_ratio(data, market_index)
capm_metric = capm(data, market_index)
max_drawdown = drawdown(data)

print(f"Sortino Ratio: {sortino}")
print(f"Beta Ratio: {beta}")
print(f"CAPM Metric: {capm_metric}")
print(f"Maximum Drawdown: {max_drawdown}")

# Plot the closing prices, predicted prices, and target levels
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['close'], label='Closing Prices')
plt.plot(data.index, data['predicted_price'], label='Predicted Prices')
plt.plot(data.index, data['support'], label='Support Levels')
plt.plot(data.index, data['resistance'], label='Resistance Levels')
plt.axvspan(data.index[-24], data.index[-1], alpha=0.3, color='green', label='Next 24 hours')
plt.legend()
plt.title('Cryptocurrency Prices and Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()