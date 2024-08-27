import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Calculate the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# Add it to sys.path
sys.path.append(project_root)

from src.analysis.pca_preparation import scale_and_reduce, create_sequences
from src.analysis import feature_engineering
import logging
import matplotlib.pyplot as plt
import glob

# Configuration
model_save_dir = '../../models/saved_models/'
data_dir = '../../data/processed/'
sequence_length = 30
pca_components = 5

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_features_to_df(df):
    return feature_engineering.add_all_features(df)

def load_model(model_path):
    checkpoint = torch.load(model_path)

    # Assuming we're using the simplified LSTM model from the training script
    input_size = pca_components
    hidden_size = 64
    num_layers = 2
    dropout = 0.2

    model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
    fc = nn.Linear(hidden_size, 1)

    model.load_state_dict(checkpoint['lstm_state_dict'])
    fc.load_state_dict(checkpoint['fc_state_dict'])
    target_scaler = checkpoint['target_scaler']

    return model, fc, target_scaler

def evaluate_model(company_code):
    logger.info(f"Evaluating model for {company_code}...")

    # Load the trained model
    model_path = os.path.join(model_save_dir, f"{company_code}_model.pt")
    model, fc, target_scaler = load_model(model_path)

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
        _, (hidden, _) = model(X)
        predictions = fc(hidden[-1]).squeeze()

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
    plt.savefig(f"../../models/logs/{company_code}_evaluation_plot.png")
    plt.close()

    # Predict the next day's closing price
    last_sequence = X[-1].unsqueeze(0)
    with torch.no_grad():
        _, (hidden, _) = model(last_sequence)
        next_day_prediction = fc(hidden[-1]).squeeze()

    next_day_prediction = target_scaler.inverse_transform(next_day_prediction.reshape(-1, 1)).item()
    logger.info(f"Predicted closing price for the next day: ${next_day_prediction:.2f}")

    return mse, rmse, mae, r2, next_day_prediction

def main():
    # Iterate over all trained models and evaluate them
    for model_file in os.listdir(model_save_dir):
        if model_file.endswith("_model.pt"):
            company_code = model_file.split("_model.pt")[0]
            try:
                mse, rmse, mae, r2, next_day_prediction = evaluate_model(company_code)

                # You can save these metrics to a file or database for further analysis
                with open(f"../../models/logs/{company_code}_evaluation_results.txt", "w") as f:
                    f.write(f"MSE: {mse:.4f}\n")
                    f.write(f"RMSE: {rmse:.4f}\n")
                    f.write(f"MAE: {mae:.4f}\n")
                    f.write(f"R-squared: {r2:.4f}\n")
                    f.write(f"Next day prediction: ${next_day_prediction:.2f}\n")

            except Exception as e:
                logger.error(f"Error evaluating model for {company_code}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
