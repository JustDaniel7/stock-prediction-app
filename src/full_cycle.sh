#!/bin/bash

# Exit on error
set -e

echo "Step 1: Downloading data..."
python3 analysis/download_data.py
echo "Data download complete."

echo "Step 2: Preprocessing data..."
python3 analysis/data_preprocessing.py
echo "Data preprocessing complete."

echo "Step 3: Feature engineering..."
python3 analysis/feature_engineering.py
echo "Feature engineering complete."

echo "Step 4: Applying PCA and preparing data..."
python3 analysis/pca_preparation.py
echo "PCA preparation complete."

echo "Step 5: Training the model..."
python3 model/train_model.py
echo "Model training complete."

echo "Step 6: Evaluating the model..."
python3 model/evaluate_model.py
echo "Model evaluation complete."

echo "Step 7: Predicting next day's closing prices..."
python3 model/predict.py
echo "Prediction complete."
