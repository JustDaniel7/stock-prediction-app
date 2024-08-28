import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Calculate the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# Add it to sys.path
sys.path.append(project_root)

from src.analysis.pca_preparation import scale_and_reduce, create_sequences
from src.analysis import feature_engineering
import logging

# Configuration
log_dir = '../../models/logs/training_logs/'
model_save_dir = '../../models/saved_models/'
sequence_length = 30
batch_size = 32
pca_components = 10  # This will be the input size for the LSTM model

# Ensure directories exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_features_to_df(df):
    return feature_engineering.add_all_features(df)

def train_model_for_company(file_path):
    company_code = os.path.basename(file_path).split('_')[0]
    logger.info(f"Training model for {company_code}...")

    # Load and preprocess data
    df = pd.read_csv(file_path)
    df = add_features_to_df(df)

    # Scaling and PCA
    reduced_features, close_prices = scale_and_reduce(df, pca_components)

    # Scale the target variable (close prices)
    target_scaler = MinMaxScaler()
    scaled_close_prices = target_scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()

    # Create sequences
    X, y = create_sequences(reduced_features, scaled_close_prices, sequence_length)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model configuration
    input_size = pca_components  # The number of PCA components becomes the input size
    hidden_size = 128
    num_layers = 2
    output_size = 1
    dropout = 0.2

    # Simplified LSTM model
    model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
    fc = nn.Linear(hidden_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam([*model.parameters(), *fc.parameters()], lr=0.0001)  # Reduced learning rate

    # Training loop
    epochs = 50  # Increased epochs for better training
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            _, (hidden, _) = model(X_batch)
            outputs = fc(hidden[-1])
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        avg_epoch_loss = epoch_loss / len(dataloader.dataset)

        # Log training progress
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.6f}")

        # Write training progress to log file
        log_file_path = os.path.join(log_dir, f"{company_code}_training.log")
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.6f}\n")

    # Save the trained model
    model_save_path = os.path.join(model_save_dir, f"{company_code}_model.pt")
    torch.save({
        'lstm_state_dict': model.state_dict(),
        'fc_state_dict': fc.state_dict(),
        'target_scaler': target_scaler
    }, model_save_path)
    logger.info(f"Model saved to {model_save_path}")

def main():
    import glob

    # Iterate over processed data files and train models
    for file_path in glob.glob('../../data/processed/*.csv'):
        try:
            train_model_for_company(file_path)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
