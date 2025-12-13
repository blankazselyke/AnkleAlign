#!/bin/bash
set -e

cd /app/src

echo "-----------------------------------"
echo "1. Running Preprocessing..."
python 01_data_preprocessing.py

echo "-----------------------------------"
echo "2a. Running BASELINE Model..."
python 02_baseline.py

echo "-----------------------------------"
echo "2b. Running MAIN Model (ResNet)..."
python 02_training.py

echo "-----------------------------------"
echo "3. Running Evaluation..."
python 03_evaluation.py

echo "-----------------------------------"
echo "4. Running Inference..."
python 04_inference.py

echo "Pipeline finished successfully."