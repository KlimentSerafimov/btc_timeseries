import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import os

def download_btc_data(start_date='2018-01-01', end_date=None, interval='1d'):
    """
    Download Bitcoin price data using Yahoo Finance
    
    Parameters:
    - start_date: Start date for data collection (format: 'YYYY-MM-DD')
    - end_date: End date for data collection (format: 'YYYY-MM-DD'), defaults to today
    - interval: Data frequency ('1d', '1wk', '1mo', etc.)
    
    Returns:
    - DataFrame with Bitcoin price data
    """
    print(f"Downloading BTC data from {start_date} to {end_date or 'today'} with {interval} interval...")
    btc_data = yf.download('BTC-USD', start=start_date, end=end_date, interval=interval)
    
    # Ensure all numeric columns are float
    for col in btc_data.columns:
        btc_data[col] = pd.to_numeric(btc_data[col], errors='coerce')
    
    # Ensure index is DatetimeIndex
    if not isinstance(btc_data.index, pd.DatetimeIndex):
        btc_data.index = pd.to_datetime(btc_data.index)
    
    print(f"Downloaded {len(btc_data)} records.")
    print(f"Index type: {type(btc_data.index)}")
    return btc_data

def save_data(data, filename='data/btc_price_data.csv'):
    """Save DataFrame to CSV file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Reset index to make sure the date is saved as a column
    data_to_save = data.reset_index()
    
    # Rename the index column to 'Date' if it's not already named
    if 'index' in data_to_save.columns:
        data_to_save.rename(columns={'index': 'Date'}, inplace=True)
    
    # Save with float_format to ensure numeric values are saved properly
    data_to_save.to_csv(filename, float_format='%.6f', index=False)
    print(f"Data saved to {filename}")
    return filename

def plot_price_history(data, title='Bitcoin Price History', save_path='figures/btc_price_history.png'):
    """Create and save a plot of Bitcoin price history"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Price history plot saved to {save_path}")

if __name__ == "__main__":
    # Download data
    btc_data = download_btc_data(start_date='2018-01-01')
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(btc_data.describe())
    
    # Print data types to debug
    print("\nData types:")
    print(btc_data.dtypes)
    
    # Save data to CSV
    save_data(btc_data)
    
    # Plot and save price history
    plot_price_history(btc_data)
    
    print("\nData download and initial processing complete!") 