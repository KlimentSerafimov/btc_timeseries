#!/bin/bash

# Print commands and exit on errors
set -e
set -x

echo "=== Bitcoin Futures Trading Simulator ==="

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

# Check if Bitcoin data exists, download if needed
if [ ! -f "data/btc_price_data.csv" ]; then
    echo "Bitcoin price data not found. Downloading data..."
    python btc_data_downloader.py
fi

# Run the Bitcoin futures simulator
echo "Running Bitcoin futures trading simulation..."
cd "$(dirname "$0")"  # Change to the script's directory to ensure relative imports work
python simulator/btc_futures_simulator.py

# Display the generated performance plot
echo "Simulation complete! Opening performance plot..."

echo "=== Simulation complete! ==="