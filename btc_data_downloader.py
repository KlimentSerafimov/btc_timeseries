import pandas as pd
import yfinance as yf  # type: ignore
import matplotlib.pyplot as plt
from datetime import datetime
import os, matplotlib.dates as mdates

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
    
    # Ensure all numeric columns are float and index is DatetimeIndex
    for col in btc_data.columns:
        btc_data[col] = pd.to_numeric(btc_data[col], errors='coerce')
    
    if not isinstance(btc_data.index, pd.DatetimeIndex):
        btc_data.index = pd.to_datetime(btc_data.index)
    
    print(f"Downloaded {len(btc_data)} records.\nIndex type: {type(btc_data.index)}")
    return btc_data

def save_data(data, filename='data/btc_price_data.csv'):
    """Save DataFrame to CSV file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Reset index to make sure the date is saved as a column
    data_to_save = data.reset_index()
    if 'index' in data_to_save.columns:
        data_to_save.rename(columns={'index': 'Date'}, inplace=True)
    
    data_to_save.to_csv(filename, float_format='%.6f', index=False)
    print(f"Data saved to {filename}")
    return filename

def plot_price_history(data, title='Bitcoin Price History', save_path='figures/btc_price_history.png'):
    """Create and save a plot of Bitcoin price history"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Set dark style and define colors
    plt.style.use('dark_background')
    colors = {
        'line': '#00a8ff',   # Bright blue
        'title': '#e0e0e0',  # Light gray
        'grid': '#555555'    # Medium gray
    }
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1e1e1e')
    ax.set_facecolor('#2d2d2d')
    
    # Plot data and set styling
    ax.plot(data.index, data['Close'], color=colors['line'], linewidth=2)
    ax.set_title(title, color=colors['title'], fontsize=16)
    ax.set_xlabel('Date', color=colors['title'], fontsize=12)
    ax.set_ylabel('Price (USD)', color=colors['title'], fontsize=12)
    
    # Configure grid and spines
    ax.grid(True, alpha=0.2, color=colors['grid'])
    ax.tick_params(colors=colors['title'])
    for spine in ax.spines.values():
        spine.set_color(colors['grid'])
    
    # Set date ticks
    date_range = [data.index[0], data.index[-1]]
    ax.set_xlim(date_range)
    date_interval = (date_range[1] - date_range[0]) / 20
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(date_interval.days))))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color=colors['title'])
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1e1e1e')
    plt.close()
    plt.style.use('default')  # Reset to default style
    
    print(f"Price history plot saved to {save_path}")

if __name__ == "__main__":
    # Download, analyze and save data
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