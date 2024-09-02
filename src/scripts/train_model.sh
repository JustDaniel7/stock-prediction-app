#!/bin/bash

# Exit on error
set -e

echo "Step 1: Downloading data..."
python3 ../src/analysis/download_data.py
echo "Data download complete."

echo "Step 2: Preprocessing data..."
python3 ../src/analysis/data_preprocessing.py
echo "Data preprocessing complete."

echo "Step 3: Feature engineering..."
python3 ../src/analysis/feature_engineering.py
echo "Feature engineering complete."

echo "Step 4: Applying PCA and preparing data..."
python3 ../src/analysis/pca_preparation.py
echo "PCA preparation complete."

echo "Step 5: Training the model..."
python3 ../src/model/train_model.py
echo "Model training complete."
