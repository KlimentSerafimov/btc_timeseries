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
python btc_futures_simulator.py

# Display the generated performance plot
echo "Simulation complete! Opening performance plot..."

# Detect OS and open the plot accordingly
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open figures/bot_performance.png
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v xdg-open &> /dev/null; then
        xdg-open figures/bot_performance.png
    else
        echo "Cannot open the plot automatically. Please check figures/bot_performance.png"
    fi
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    start figures/bot_performance.png
else
    echo "Cannot open the plot automatically. Please check figures/bot_performance.png"
fi

echo "=== Simulation complete! ==="
echo "You can find the performance plot in figures/bot_performance.png" 