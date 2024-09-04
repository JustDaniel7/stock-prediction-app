import glob
import logging
import os

import pandas as pd
import torch

from src.model.model import MyLSTMModel
from src.analysis.pca_preparation import scale_and_reduce, create_sequences
from src.analysis import feature_engineering

# Configuration
BASE_DIR = '/app'  # This should be the root directory in your Docker container
model_save_dir = os.path.join(BASE_DIR, 'models', 'saved_models')
data_dir = os.path.join(BASE_DIR, 'data', 'processed')
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

def predict_next_day(company_code):
    logger.info(f"Predicting next day's closing price for {company_code}...")

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

    # Create sequences
    X, _ = create_sequences(reduced_features, close_prices, sequence_length)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)

    # Make prediction for the next day (today's closing price)
    model.eval()
    with torch.no_grad():
        last_sequence = X[-1].unsqueeze(0)  # Get the last sequence to predict the next value
        next_day_prediction = model(last_sequence).squeeze()

    # Inverse transform the prediction
    next_day_prediction = target_scaler.inverse_transform(next_day_prediction.reshape(-1, 1)).item()
    logger.info(f"Predicted closing price for the next day: ${next_day_prediction:.2f}")

    return next_day_prediction

def main():
    if not os.path.exists(model_save_dir):
        logger.error(f"Directory {model_save_dir} does not exist.")
        return

    model_files = [f for f in os.listdir(model_save_dir) if f.endswith("_model.pt")]

    if not model_files:
        logger.error(f"No model files found in {model_save_dir}")
        return

    for model_file in model_files:
        if model_file.endswith("_model.pt"):
            company_code = model_file.split("_model.pt")[0]
            try:
                next_day_prediction = predict_next_day(company_code)

                # Save the prediction result to a file or database for further analysis
                log_dir = os.path.join(BASE_DIR, 'models', 'logs', 'prediction_logs')
                os.makedirs(log_dir, exist_ok=True)
                with open(os.path.join(log_dir, f"{company_code}_next_day_prediction.log"), "a") as f:
                    f.write(f"Next day prediction: ${next_day_prediction:.2f}\n")

            except Exception as e:
                logger.error(f"Error predicting next day price for {company_code}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()