#!/bin/bash

# Exit on error
set -e

echo "Step 1: Downloading data..."
python3 download_data.py
echo "Data download complete."

echo "Step 2: Preprocessing data..."
python3 data_preprocessing.py
echo "Data preprocessing complete."

echo "Step 3: Feature engineering..."
python3 ../analysis/feature_engineering.py
echo "Feature engineering complete."

echo "Step 4: Applying PCA and preparing data..."
python3 ../analysis/pca_preparation.py
echo "PCA preparation complete."

echo "Step 5: Training the model..."
python3 ../model/train_model.py
echo "Model training complete."
