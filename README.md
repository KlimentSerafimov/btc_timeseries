# Bitcoin Time Series Analysis

This project analyzes Bitcoin price data to identify patterns, trends, and potential predictive models.

## Project Structure

- `btc_data_downloader.py`: Script to download and save Bitcoin price data
- `btc_analysis.py`: Script to analyze Bitcoin price data and generate visualizations
- `data/`: Directory containing the downloaded data
- `figures/`: Directory containing generated visualizations
- `models/`: Directory for saved predictive models (to be implemented)

## Setup and Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Download the Bitcoin price data:
   ```
   python btc_data_downloader.py
   ```

2. Run the analysis script:
   ```
   python btc_analysis.py
   ```

## Data Sources

The project uses Bitcoin price data from Yahoo Finance, which provides daily OHLCV (Open, High, Low, Close, Volume) data.

## Analysis Features

- Price trend visualization
- Return distribution analysis
- Volatility analysis
- Time series decomposition
- (More to be added)

## Future Enhancements

- Implement predictive models (ARIMA, LSTM, etc.)
- Add trading strategy backtesting
- Include sentiment analysis from news and social media 