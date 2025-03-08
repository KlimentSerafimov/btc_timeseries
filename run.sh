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
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source btc_venv/Scripts/activate
else
    source btc_venv/bin/activate
fi

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

# Try to open the plots with the appropriate command based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open figures/btc_price_history.png
    open figures/historic_price.png
    open figures/returns_distribution.png
    open figures/volatility.png
    open figures/decomposition.png
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    start figures/btc_price_history.png
    start figures/historic_price.png
    start figures/returns_distribution.png
    start figures/volatility.png
    start figures/decomposition.png
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open figures/btc_price_history.png
    xdg-open figures/historic_price.png
    xdg-open figures/returns_distribution.png
    xdg-open figures/volatility.png
    xdg-open figures/decomposition.png
else
    echo "Plots saved in the 'figures' directory."
fi

echo "=== Analysis complete! ===" 