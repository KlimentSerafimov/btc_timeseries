import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import os
import matplotlib.dates as mdates

def download_btc_data(start_date, end_date=None, interval='1d'):
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
    
    # Set dark style for plots
    plt.style.use('dark_background')
    
    # Define colors for better visibility on dark background
    line_color = '#00a8ff'  # Bright blue
    title_color = '#e0e0e0'  # Light gray
    grid_color = '#555555'   # Medium gray
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1e1e1e')
    ax.set_facecolor('#2d2d2d')
    
    # Plot the data
    ax.plot(data.index, data['Close'], color=line_color, linewidth=2)
    
    # Set title and labels
    ax.set_title(title, color=title_color, fontsize=16)
    ax.set_xlabel('Date', color=title_color, fontsize=12)
    ax.set_ylabel('Price (USD)', color=title_color, fontsize=12)
    
    # Configure grid and spines
    ax.grid(True, alpha=0.2, color=grid_color)
    ax.tick_params(colors=title_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)
    
    # Set date ticks - using a different approach for approximately 20 ticks
    date_range = [data.index[0], data.index[-1]]
    ax.set_xlim(date_range)
    
    # Calculate tick positions for approximately 20 ticks
    date_interval = (date_range[1] - date_range[0]) / 20
    locator = mdates.DayLocator(interval=max(1, int(date_interval.days)))
    ax.xaxis.set_major_locator(locator)
    
    # Format the tick labels
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color=title_color)
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1e1e1e')
    plt.close()
    
    # Reset to default style
    plt.style.use('default')
    
    print(f"Price history plot saved to {save_path}")

if __name__ == "__main__":
    # Download data
    btc_data = download_btc_data(start_date='2013-01-01')
    
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