#!/bin/bash

# Print commands and exit on errors
set -e
set -x

echo "=== Bitcoin Time Series Analysis Setup and Run Script ==="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv btc_venv

# Activate virtual environment (different for Windows vs Unix-based systems)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source btc_venv/Scripts/activate
else
    source btc_venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Install development packages (type stubs for linting)
if [ -f requirements-dev.txt ]; then
    echo "Installing development packages..."
    # Install each package separately to continue if one fails
    while read -r package || [[ -n "$package" ]]; do
        # Skip comments and empty lines
        [[ "$package" =~ ^#.*$ || -z "$package" ]] && continue
        pip install "$package" || echo "Warning: Could not install $package, continuing anyway..."
    done < requirements-dev.txt
fi

# Create necessary directories
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
    open figures/returns_distribution.png
    open figures/volatility.png
    open figures/decomposition.png
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    start figures/btc_price_history.png
    start figures/returns_distribution.png
    start figures/volatility.png
    start figures/decomposition.png
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open figures/btc_price_history.png
    xdg-open figures/returns_distribution.png
    xdg-open figures/volatility.png
    xdg-open figures/decomposition.png
else
    echo "Plots saved in the 'figures' directory."
fi

echo "=== Setup and analysis complete! ==="
echo "To activate this environment in the future, run:"
echo "source btc_venv/bin/activate  # On Unix/macOS"
echo "source btc_venv/Scripts/activate  # On Windows" 