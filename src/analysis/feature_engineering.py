import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Directory for processed data
processed_data_dir = 'src/data/processed'

# List of tickers
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "BRK-B", "NVDA", "META", "V", "JNJ"
]

def load_processed_data():
    yesterday = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
    data_frames = []

    for ticker in tickers:
        processed_file = os.path.join(processed_data_dir, f"{ticker}_{yesterday}.csv")
        if os.path.exists(processed_file):
            df = pd.read_csv(processed_file)
            df['Ticker'] = ticker
            data_frames.append(df)
        else:
            print(f"No processed data found for {ticker} on {yesterday}")

    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        print("No data loaded.")
        return None

def add_moving_average(df, window):
    """
    Adds a moving average (MA) column to the DataFrame.
    """
    df[f"MA_{window}"] = df['Close'].rolling(window=window).mean()
    return df

def add_exponential_moving_average(df, span):
    """
    Adds an exponential moving average (EMA) column to the DataFrame.
    """
    df[f"EMA_{span}"] = df['Close'].ewm(span=span, adjust=False).mean()
    return df

def add_rsi(df, window=14):
    """
    Adds Relative Strength Index (RSI) column to the DataFrame.
    """
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return df

def add_volatility_features(df):
    # High-Low range as a volatility measure
    df['High_Low_Range'] = df['High'] - df['Low']

    # Daily percentage price change
    df['Daily_Change'] = df['Close'].pct_change() * 100

    return df

def add_trend_indicators(df):
    # Moving Average Convergence Divergence (MACD)
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema

    # Bollinger Bands (20-day moving average with 2 standard deviations)
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()

    return df

def add_volume_based_features(df):
    # Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    # On-Balance Volume (OBV)
    df['Daily_Change'] = df['Close'].diff()
    df['OBV'] = np.where(df['Daily_Change'] > 0, df['Volume'], -df['Volume']).cumsum()

    return df

def add_time_features(df):
    # Extract day of the week and month
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df['Day_of_Week'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['Month'] = pd.to_datetime(df['Date']).dt.month

    return df

def add_dividend_yield(df):
    # Dividend Yield
    df['Dividend_Yield'] = (df['Dividends'] / df['Close']) * 100
    return df

def add_rolling_std(df, window=20):
    # Rolling standard deviation of closing prices (volatility indicator)
    df[f'Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
    return df

def add_all_features(df):
    df = add_moving_average(df=df, window=14)
    df = add_exponential_moving_average(df=df, span=14)
    df = add_rsi(df=df, window=14)
    df = add_volatility_features(df)
    df = add_trend_indicators(df)
    df = add_volume_based_features(df)
    df = add_time_features(df)
    df = add_dividend_yield(df)
    df = add_rolling_std(df)
    # Drop rows with NaN values introduced by rolling calculations
    df.dropna(inplace=True)
    return df


def visualize_stock_prices(combined_df, tickers):
    for ticker in tickers:
        ticker_df = combined_df[combined_df['Ticker'] == ticker].copy()
        ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], utc=True).dt.tz_convert('UTC')

        yearly_ticks = pd.date_range(start=ticker_df['Date'].min(), end=ticker_df['Date'].max(), freq='2YS', tz='UTC')

        plt.figure(figsize=(14, 7))
        plt.plot(ticker_df['Date'], ticker_df['Close'], label=f'{ticker} Close', color='blue')
        plt.plot(ticker_df['Date'], ticker_df['MA_20'], label=f'{ticker} MA_20', color='orange')

        plt.xticks(yearly_ticks, [date.strftime('%Y-%m-%d') for date in yearly_ticks])
        plt.title(f"Stock Prices and 20-Day Moving Averages for {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    # Load processed data
    combined_df = load_processed_data()

    if combined_df is not None:
        # Apply feature engineering
        combined_df = add_all_features(combined_df)

        # Visualize the stock prices
        #visualize_stock_prices(combined_df, tickers)
    else:
        print("No data available for visualization.")


if __name__ == "__main__":
    main()
