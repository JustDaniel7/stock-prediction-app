import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import logging
import glob

# Calculate the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.analysis.pca_preparation import scale_and_reduce, create_sequences
from src.analysis import feature_engineering
import matplotlib.pyplot as plt

# Configuration
model_save_dir = os.path.join(project_root, 'models/saved_models/')
data_dir = os.path.join(project_root, 'data/processed/')
sequence_length = 30
pca_components = 10

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_features_to_df(df):
    return feature_engineering.add_all_features(df)

def load_model(model_path):
    checkpoint = torch.load(model_path)

    # Assuming we're using the simplified LSTM model from the training script
    input_size = pca_components
    hidden_size = 128
    num_layers = 2
    dropout = 0.2

    model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
    fc = nn.Linear(hidden_size, 1)

    model.load_state_dict(checkpoint['lstm_state_dict'])
    fc.load_state_dict(checkpoint['fc_state_dict'])
    target_scaler = checkpoint['target_scaler']

    return model, fc, target_scaler

def predict_next_day(company_code):
    logger.info(f"Predicting next day's closing price for {company_code}...")

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

    # Create sequences
    X, _ = create_sequences(reduced_features, close_prices, sequence_length)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)

    # Make prediction for the next day (today's closing price)
    model.eval()
    with torch.no_grad():
        last_sequence = X[-1].unsqueeze(0)  # Get the last sequence to predict the next value
        _, (hidden, _) = model(last_sequence)
        next_day_prediction = fc(hidden[-1]).squeeze()

    # Inverse transform the prediction
    next_day_prediction = target_scaler.inverse_transform(next_day_prediction.reshape(-1, 1)).item()
    logger.info(f"Predicted closing price for the next day: ${next_day_prediction:.2f}")

    return next_day_prediction

def main():
    # Iterate over all trained models and predict the next day's closing price
    for model_file in os.listdir(model_save_dir):
        if model_file.endswith("_model.pt"):
            company_code = model_file.split("_model.pt")[0]
            try:
                next_day_prediction = predict_next_day(company_code)

                # Save the prediction result to a file or database for further analysis
                with open(f"../../models/logs/prediction_logs/{company_code}_next_day_prediction.log", "a") as f:
                    f.write(f"Next day prediction: ${next_day_prediction:.2f}\n")

            except Exception as e:
                logger.error(f"Error predicting next day price for {company_code}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
