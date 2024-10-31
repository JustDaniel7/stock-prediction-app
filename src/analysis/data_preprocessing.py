import os
import pandas as pd
import boto3
from datetime import datetime, timedelta
from io import StringIO

# S3 bucket and folder configuration
S3_BUCKET_NAME = 'your-bucket-name'
S3_RAW_DATA_FOLDER = 'data/raw/'
S3_PROCESSED_DATA_FOLDER = 'data/processed/'

# Initialize the S3 client
s3 = boto3.client('s3', region_name='eu-central-1')

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

def preprocess_data_from_s3(event, context):
    """
    Lambda function to preprocess stock data fetched from S3.

    Args:
        event: The event data passed by the invoking Lambda function.
        context: The runtime information of the Lambda function.
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

    # Remove previous day's processed files in S3
    delete_previous_day_s3_files(S3_BUCKET_NAME, S3_PROCESSED_DATA_FOLDER)

    # Preprocess data for each ticker
    for ticker in tickers:
        try:
            raw_key = f"{S3_RAW_DATA_FOLDER}{ticker}_{end_date}.csv"
            response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=raw_key)
            df = pd.read_csv(response['Body'])

            # Example preprocessing: calculate moving averages and other features
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['MA_200'] = df['Close'].rolling(window=200).mean()

            # Save the processed data back to S3
            processed_key = f"{S3_PROCESSED_DATA_FOLDER}{ticker}_{end_date}.csv"
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            s3.put_object(Bucket=S3_BUCKET_NAME, Key=processed_key, Body=csv_buffer.getvalue())

            print(f"Data processed for {ticker} and saved to {processed_key}")
        except Exception as e:
            print(f"Failed to process data for {ticker}. Reason: {str(e)}")

def get_yesterday():
    """
    Get yesterday's date as a string in 'YYYY-MM-DD' format.

    Returns:
        str: Yesterday's date.
    """
    return (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

# Lambda handler
def lambda_handler(event, context):
    preprocess_data_from_s3(event, context)
