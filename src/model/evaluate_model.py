import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.model.model import MyLSTMModel

from src.analysis.pca_preparation import scale_and_reduce, create_sequences
from src.analysis import feature_engineering
import logging
import matplotlib.pyplot as plt
import glob

# Configuration
BASE_DIR = '/app'  # This should be the root directory in your Docker container
model_save_dir = os.path.join(BASE_DIR, 'models', 'saved_models')
data_dir = os.path.join(BASE_DIR, 'data', 'processed')
eval_plots_dir = os.path.join(BASE_DIR, 'models', 'eval_plots')
eval_logs_dir = os.path.join(BASE_DIR, 'models', 'logs', 'evaluation_logs')
sequence_length = 30
pca_components = 10

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_features_to_df(df):
    return feature_engineering.add_all_features(df)

def load_model(model_path):
    checkpoint = torch.load(model_path)

    input_size = pca_components
    hidden_size = 128
    output_size = 1
    dropout = 0.3

    model = MyLSTMModel(input_size, hidden_size, output_size, dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    target_scaler = checkpoint['target_scaler']

    return model, target_scaler

def evaluate_model(company_code):
    logger.info(f"Evaluating model for {company_code}...")

    # Load the trained model
    model_path = os.path.join(model_save_dir, f"{company_code}_model.pt")
    model, target_scaler = load_model(model_path)

    # Search for the appropriate CSV file using a wildcard pattern
    csv_pattern = os.path.join(data_dir, f"{company_code}_*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        logger.error(f"No CSV file found for {company_code} with pattern {csv_pattern}")
        return

    # Assume we're using the first matching file
    test_file_path = csv_files[0]
    df = pd.read_csv(test_file_path)
    df = add_features_to_df(df)

    # Scaling and PCA
    reduced_features, close_prices = scale_and_reduce(df, pca_components)

    # Scale the target variable (close prices)
    scaled_close_prices = target_scaler.transform(close_prices.reshape(-1, 1)).flatten()

    # Create sequences
    X, y = create_sequences(reduced_features, scaled_close_prices, sequence_length)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X).squeeze()

    # Inverse transform predictions and actual values
    predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actual = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)

    logger.info(f"Mean Squared Error: {mse:.4f}")
    logger.info(f"Root Mean Squared Error: {rmse:.4f}")
    logger.info(f"Mean Absolute Error: {mae:.4f}")
    logger.info(f"R-squared Score: {r2:.4f}")

    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(f"{company_code} Stock Price - Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.savefig(f"models/eval_plots/{company_code}_evaluation_plot.png")
    plt.close()

    return mse, rmse, mae, r2

def main():
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Contents of {BASE_DIR}:")
    for root, dirs, files in os.walk(BASE_DIR):
        level = root.replace(BASE_DIR, '').count(os.sep)
        indent = ' ' * 4 * (level)
        logger.info(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            logger.info(f"{subindent}{f}")

    logger.info(f"Checking if {model_save_dir} exists: {os.path.exists(model_save_dir)}")

    if not os.path.exists(model_save_dir):
        logger.error(f"Directory {model_save_dir} does not exist.")
        return

    model_files = [f for f in os.listdir(model_save_dir) if f.endswith("_model.pt")]

    if not model_files:
        logger.error(f"No model files found in {model_save_dir}")
        return

    for model_file in model_files:
        company_code = model_file.split("_model.pt")[0]
        try:
            mse, rmse, mae, r2 = evaluate_model(company_code)

            # Ensure the evaluation logs directory exists
            os.makedirs(eval_logs_dir, exist_ok=True)

            # Save the evaluation metrics to a log file for further analysis
            with open(os.path.join(eval_logs_dir, f"{company_code}_evaluation_results.log"), "a") as f:
                f.write(f"MSE: {mse:.4f}\n")
                f.write(f"RMSE: {rmse:.4f}\n")
                f.write(f"MAE: {mae:.4f}\n")
                f.write(f"R-squared: {r2:.4f}\n")

        except Exception as e:
            logger.error(f"Error evaluating model for {company_code}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
