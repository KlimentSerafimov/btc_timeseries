import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

def load_data(filepath='data/btc_price_data.csv'):
    """Load Bitcoin data from CSV file"""
    # Read the CSV file with explicit date parsing
    data = pd.read_csv(filepath)
    
    # Convert the date column to datetime and set as index
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    else:
        # If the first column is the date but not named 'Date'
        data.index = pd.to_datetime(data.index)
    
    # Ensure index is DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        # Try to convert the first column to datetime and use as index
        first_col = data.columns[0]
        data[first_col] = pd.to_datetime(data[first_col])
        data.set_index(first_col, inplace=True)
    
    # Convert numeric columns to float
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    print(f"Loaded data with shape: {data.shape}")
    print(f"Index type: {type(data.index)}")
    return data

def calculate_returns(data):
    """Calculate daily and monthly returns"""
    # Daily returns
    data['Daily_Return'] = data['Close'].pct_change(fill_method=None)
    
    # Monthly returns (resample to month end and calculate returns)
    monthly_data = data['Close'].resample('ME').last()  # Using 'ME' instead of deprecated 'M'
    monthly_returns = monthly_data.pct_change(fill_method=None)
    
    return data, monthly_returns

def plot_returns_distribution(returns, save_path='figures/returns_distribution.png'):
    """Plot the distribution of returns"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Plot histogram with kernel density estimate
    sns.histplot(returns.dropna(), kde=True, bins=50)
    
    plt.title('Distribution of Bitcoin Daily Returns')
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"Returns distribution plot saved to {save_path}")

def plot_volatility(data, window=20, save_path='figures/volatility.png'):
    """Plot rolling volatility"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Calculate rolling standard deviation
    data['Volatility'] = data['Daily_Return'].rolling(window=window).std() * np.sqrt(window)
    
    plt.figure(figsize=(12, 6))
    plt.plot(data['Volatility'])
    plt.title(f'Bitcoin {window}-Day Rolling Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Volatility plot saved to {save_path}")

def decompose_time_series(data, save_path='figures/decomposition.png'):
    """Decompose time series into trend, seasonal, and residual components"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Make a copy of the data to avoid modifying the original
    close_prices = data['Close'].copy()
    
    # Fill missing values using forward fill and then backward fill to ensure no NaNs
    close_prices = close_prices.ffill().bfill()
    
    # Check if there are still any NaN values
    if close_prices.isna().any():
        print("Warning: Data still contains NaN values after filling. Creating simple trend plot instead.")
        plt.figure(figsize=(12, 6))
        plt.plot(close_prices.dropna())
        plt.title('Bitcoin Price Trend (Seasonal Decomposition Failed)')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"Simple trend plot saved to {save_path}")
        return
    
    # Ensure we have enough data points and adjust period if necessary
    data_length = len(close_prices)
    if data_length < 730:  # Less than 2 years of data
        # Use a smaller period for seasonal decomposition
        period = min(365, data_length // 4)
        if period < 7:
            print("Warning: Not enough data for meaningful seasonal decomposition.")
            return
        print(f"Using period={period} for seasonal decomposition.")
    else:
        period = 365  # Use annual seasonality for longer data
    
    # Resample to daily frequency to ensure regular time series
    # This helps with irregular trading days (weekends, holidays)
    close_prices = close_prices.resample('D').interpolate(method='linear')
    
    try:
        # Try multiplicative decomposition first
        try:
            decomposition = seasonal_decompose(close_prices, model='multiplicative', period=period)
        except Exception as e:
            print(f"Multiplicative decomposition failed: {e}")
            print("Trying additive decomposition instead...")
            decomposition = seasonal_decompose(close_prices, model='additive', period=period)
        
        # Plot decomposition without the seasonal component
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        decomposition.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        ax1.grid(True)
        
        decomposition.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        ax2.grid(True)
        
        decomposition.resid.plot(ax=ax3)
        ax3.set_title('Residual')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Time series decomposition saved to {save_path}")
        
    except Exception as e:
        print(f"Error in seasonal decomposition: {e}")
        print("Creating alternative visualization...")
        
        # Create an alternative visualization with rolling statistics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot original data with trend (rolling mean)
        close_prices.plot(ax=ax1, label='Original')
        close_prices.rolling(window=period//3).mean().plot(ax=ax1, label=f'Trend ({period//3}-day MA)', 
                                                         color='red')
        ax1.set_title('Bitcoin Price with Trend')
        ax1.legend()
        ax1.grid(True)
        
        # Plot rolling standard deviation as a proxy for volatility/seasonality
        close_prices.rolling(window=20).std().plot(ax=ax2, label='20-day Rolling Std Dev', color='green')
        ax2.set_title('Rolling Volatility (Proxy for Seasonality)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Alternative time series analysis saved to {save_path}")

def plot_historic_price(data, save_path='figures/historic_price.png'):
    """Create a detailed plot of Bitcoin price history with volume"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create figure with two subplots (price and volume)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price on the first subplot
    ax1.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=2)
    
    # Add moving averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    ax1.plot(data.index, data['MA50'], label='50-Day MA', color='orange', linewidth=1.5)
    ax1.plot(data.index, data['MA200'], label='200-Day MA', color='red', linewidth=1.5)
    
    # Set labels and title for price subplot
    ax1.set_title('Bitcoin Price History with Moving Averages', fontsize=16)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format y-axis with commas for thousands
    ax1.get_yaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    
    # Plot volume on the second subplot
    volume_data = data['Volume']
    ax2.bar(data.index, volume_data, color='gray', alpha=0.7, width=2)
    
    # Set labels for volume subplot
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format y-axis with commas for thousands and scientific notation for large numbers
    ax2.get_yaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:.1e}'))
    
    # Add annotations for key events
    key_events = [
        ('2020-03-12', 'COVID-19 Crash', 5000),
        ('2021-04-14', 'Coinbase IPO', 64000),
        ('2021-11-10', 'All-Time High', 69000),
        ('2022-11-11', 'FTX Collapse', 17000)
    ]
    
    for date, label, price in key_events:
        try:
            event_date = pd.to_datetime(date)
            if event_date in data.index or (data.index[0] <= event_date <= data.index[-1]):
                # Find the closest date in the index
                closest_date = data.index[data.index.get_indexer([event_date], method='nearest')[0]]
                ax1.annotate(label, 
                            xy=(closest_date, price),
                            xytext=(10, 0),
                            textcoords="offset points",
                            arrowprops=dict(arrowstyle="->", color='green'),
                            fontsize=10)
        except Exception as e:
            print(f"Could not add annotation for {date}: {e}")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Historic price plot saved to {save_path}")

if __name__ == "__main__":
    # Load data
    btc_data = load_data()
    
    # Print data types to debug
    print("Data types before conversion:")
    print(btc_data.dtypes)
    
    # Plot historic price with volume
    plot_historic_price(btc_data)
    
    # Calculate returns
    btc_data, monthly_returns = calculate_returns(btc_data)
    
    # Plot returns distribution
    plot_returns_distribution(btc_data['Daily_Return'])
    
    # Plot volatility
    plot_volatility(btc_data)
    
    # Decompose time series
    try:
        decompose_time_series(btc_data)
    except Exception as e:
        print(f"Error in time series decomposition: {e}")
        print("Continuing with analysis...")
    
    print("\nAnalysis complete!") 