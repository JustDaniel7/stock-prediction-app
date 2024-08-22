import os
import pandas as pd
from datetime import datetime, timedelta
import schedule
import time

# Directory for raw and processed data
raw_data_dir = '../data/raw'
processed_data_dir = '../data/processed'

# Create necessary directories if they don't exist
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

def remove_previous_day_files(directory, date_str):
    """
    Remove all files in the specified directory that match the given date.

    Args:
        directory (str): The directory to remove files from.
        date_str (str): Date string in 'YYYY-MM-DD' format to match the files.
    """
    for filename in os.listdir(directory):
        if date_str in filename:
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Failed to remove {file_path}. Reason: {e}")

def preprocess_data(ticker, date_str):
    """
    Process the raw data for a specific ticker and date and save the processed data.

    Args:
        ticker (str): The ticker symbol of the stock.
        date_str (str): The date string in 'YYYY-MM-DD' format.
    """
    raw_file = os.path.join(raw_data_dir, f"{ticker}_{date_str}.csv")

    if os.path.exists(raw_file):
        # Load the raw data
        df = pd.read_csv(raw_file)

        # Example preprocessing: calculate moving averages and other features
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()

        # Save the processed data
        processed_file = os.path.join(processed_data_dir, f"{ticker}_{date_str}.csv")
        df.to_csv(processed_file, index=False)

        print(f"Data processed for {ticker} and saved to {processed_file}")
    else:
        print(f"No raw data found for {ticker} on {date_str}")

def get_yesterday():
    """
    Get yesterday's date as a string in 'YYYY-MM-DD' format.

    Returns:
        str: Yesterday's date.
    """
    return (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

def preprocess_data_now():
    """
    Function to preprocess data immediately for all tickers.
    """
    tickers = [
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Alphabet (Google)
        "AMZN",  # Amazon
        "TSLA",  # Tesla
        "BRK-B", # Berkshire Hathaway
        "NVDA",  # NVIDIA
        "META",  # Meta Platforms (Facebook)
        "V",     # Visa
        "JNJ"    # Johnson & Johnson
    ]

    end_date = get_yesterday()  # Process data from yesterday

    # Remove previous day's processed files
    remove_previous_day_files(processed_data_dir, end_date)

    # Preprocess data for each ticker and save it to the processed data directory
    for ticker in tickers:
        preprocess_data(ticker, end_date)

def daily_preprocessing():
    """
    Function to preprocess data daily for all tickers.
    """
    preprocess_data_now()

if __name__ == "__main__":
    # Preprocess data right now
    preprocess_data_now()

    # Schedule the script to run every day at 9:00 AM starting tomorrow
    schedule.every().day.at("09:00").do(daily_preprocessing)

    # Run the schedule loop
    while True:
        schedule.run_pending()
        time.sleep(60)  # Wait one minute between checks
