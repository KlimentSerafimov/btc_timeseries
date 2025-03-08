#!/bin/bash

# Print commands and exit on errors
set -e
set -x

echo "=== Bitcoin Time Series Analysis Setup Script ==="

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

echo "=== Setup complete! ==="
echo "To activate this environment in the future, run:"
echo "source btc_venv/bin/activate  # On Unix/macOS"
echo "source btc_venv/Scripts/activate  # On Windows"
echo ""
echo "To run the analysis, use: ./run.sh" 