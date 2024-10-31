import yfinance as yf
from datetime import datetime, timedelta
import boto3
from io import StringIO

# S3 bucket and folder configuration
S3_BUCKET_NAME = 'stock-prediction-app-bucket-1'
S3_RAW_DATA_FOLDER = 'data/raw/'  # Folder path in S3 where raw data will be stored

# Initialize the S3 client
s3 = boto3.client('s3', region_name='eu-central-1')

def upload_to_s3(bucket_name, key, df):
    """
    Upload a pandas DataFrame to S3 as a CSV.

    Args:
        bucket_name (str): The S3 bucket name.
        key (str): The key (path) in the S3 bucket.
        df (pd.DataFrame): The DataFrame to upload.
    """
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=key, Body=csv_buffer.getvalue())
    print(f"Uploaded data to S3 at {key}")

def delete_previous_day_s3_files(bucket_name, folder_prefix):
    """
    Delete all files in the S3 bucket from the previous day.

    Args:
        bucket_name (str): The S3 bucket name.
        folder_prefix (str): The folder prefix to list the files.
    """
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)
    if 'Contents' in response:
        for obj in response['Contents']:
            print(f"Deleting {obj['Key']}")
            s3.delete_object(Bucket=bucket_name, Key=obj['Key'])

def download_stock_data(ticker, start_date, end_date):
    """
    Download historical stock data from Yahoo Finance for a specific ticker.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date for the data in 'YYYY-MM-DD' format.
        end_date (str): End date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing the stock data.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker} between {start_date} and {end_date}")

    return df

def get_yesterday():
    """
    Get yesterday's date as a string in 'YYYY-MM-DD' format.

    Returns:
        str: Yesterday's date.
    """
    return (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

def download_and_upload_data(event, context):
    """
    Function to download data and upload it to S3.
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

    start_date = "2000-01-01"
    end_date = get_yesterday()  # Fetch data until yesterday

    # Remove previous day's files in S3
    delete_previous_day_s3_files(S3_BUCKET_NAME, S3_RAW_DATA_FOLDER)

    # Download and upload data for each ticker
    for ticker in tickers:
        try:
            # Download stock data
            df = download_stock_data(ticker, start_date, end_date)

            # Define S3 key (path) for the file
            s3_key = f"{S3_RAW_DATA_FOLDER}{ticker}_{end_date}.csv"

            # Upload the data to S3
            upload_to_s3(S3_BUCKET_NAME, s3_key, df)

        except ValueError as e:
            print(e)

# Lambda handler
def lambda_handler(event, context):
    download_and_upload_data(event, context)
