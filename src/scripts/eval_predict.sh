#!/bin/bash

# Exit on error
set -e

echo "Step 6: Evaluating the model..."
python3 ../model/evaluate_model.py
echo "Model evaluation complete."

echo "Step 7: Predicting next day's closing prices..."
python3 ../model/predict.py
echo "Prediction complete."