#!/bin/bash

# Print commands and exit on errors
set -e
set -x

echo "=== Bitcoin Time Series Analysis Run Script ==="

# Check if virtual environment exists
if [ ! -d "btc_venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source btc_venv/bin/activate

# Create necessary directories (in case they don't exist)
mkdir -p data
mkdir -p figures
mkdir -p models

# Run the Bitcoin data downloader
echo "Downloading Bitcoin data..."
python btc_data_downloader.py

# Run the Bitcoin analysis
echo "Analyzing Bitcoin data..."
python btc_analysis.py

# Display the generated plots
echo "Analysis complete! Opening plots..."

# Open plots on macOS
open figures/btc_price_history.png
open figures/historic_price.png
open figures/returns_distribution.png
open figures/volatility.png
open figures/decomposition.png

echo "=== Analysis complete! ===" 