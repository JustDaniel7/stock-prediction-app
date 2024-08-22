import yfinance as yf
import os
from datetime import datetime, timedelta
import schedule
import time

# Directory for raw data
raw_data_dir = '../data/raw'

# Create the raw data directory if it doesn't exist
os.makedirs(raw_data_dir, exist_ok=True)

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

def download_stock_data(ticker, start_date, end_date, save_dir):
    """
    Download historical stock data from Yahoo Finance for a specific ticker.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date for the data in 'YYYY-MM-DD' format.
        end_date (str): End date for the data in 'YYYY-MM-DD' format.
        save_dir (str): Directory where the data should be saved.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker} between {start_date} and {end_date}")

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the data as a CSV file with the date in the filename
    output_file = os.path.join(save_dir, f"{ticker}_{end_date}.csv")
    df.to_csv(output_file)

    print(f"Data downloaded for {ticker} and saved to {output_file}")

def get_yesterday():
    """
    Get yesterday's date as a string in 'YYYY-MM-DD' format.

    Returns:
        str: Yesterday's date.
    """
    return (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

def download_data_now():
    """
    Function to download data immediately for all tickers.
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

    start_date = "2020-01-01"
    end_date = get_yesterday()  # Fetch data until yesterday

    # Remove previous day's files
    remove_previous_day_files(raw_data_dir, end_date)

    # Download data for each ticker and save it to the raw data directory
    for ticker in tickers:
        try:
            download_stock_data(ticker, start_date, end_date, raw_data_dir)
        except ValueError as e:
            print(e)

def daily_data_download():
    """
    Function to download data daily for all tickers.
    """
    download_data_now()

if __name__ == "__main__":
    # Download data right now
    download_data_now()

    # Schedule the script to run every day at 8:00 AM starting tomorrow
    schedule.every().day.at("08:00").do(daily_data_download)

    # Run the schedule loop
    while True:
        schedule.run_pending()
        time.sleep(60)  # Wait one minute between checks
