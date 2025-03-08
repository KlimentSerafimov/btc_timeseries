import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import matplotlib.dates as mdates

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
    
    # Set dark style for plots
    plt.style.use('dark_background')
    
    # Define colors for better visibility on dark background
    hist_color = '#00a8ff'  # Bright blue
    kde_color = '#00ff7f'   # Green
    title_color = '#e0e0e0' # Light gray
    grid_color = '#555555'  # Medium gray
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1e1e1e')
    ax.set_facecolor('#2d2d2d')
    
    # Plot histogram with kernel density estimate
    sns.histplot(returns.dropna(), kde=True, bins=50, ax=ax, color=hist_color, 
                 line_kws={'color': kde_color, 'linewidth': 2})
    
    # Set title and labels
    ax.set_title('Distribution of Bitcoin Daily Returns', color=title_color, fontsize=16)
    ax.set_xlabel('Daily Returns', color=title_color, fontsize=12)
    ax.set_ylabel('Frequency', color=title_color, fontsize=12)
    
    # Configure grid and spines
    ax.grid(True, alpha=0.2, color=grid_color)
    ax.tick_params(colors=title_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1e1e1e')
    plt.close()
    
    # Reset to default style
    plt.style.use('default')
    
    print(f"Returns distribution plot saved to {save_path}")

def plot_volatility(data, window=20, save_path='figures/volatility.png'):
    """Plot rolling volatility"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Calculate rolling standard deviation
    data['Volatility'] = data['Daily_Return'].rolling(window=window).std() * np.sqrt(window)
    
    # Set dark style for plots
    plt.style.use('dark_background')
    
    # Define colors for better visibility on dark background
    line_color = '#00a8ff'  # Bright blue
    title_color = '#e0e0e0' # Light gray
    grid_color = '#555555'  # Medium gray
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1e1e1e')
    ax.set_facecolor('#2d2d2d')
    
    # Plot the data
    ax.plot(data['Volatility'], color=line_color, linewidth=2)
    
    # Set title and labels
    ax.set_title(f'Bitcoin {window}-Day Rolling Volatility', color=title_color, fontsize=16)
    ax.set_xlabel('Date', color=title_color, fontsize=12)
    ax.set_ylabel('Volatility', color=title_color, fontsize=12)
    
    # Configure grid and spines
    ax.grid(True, alpha=0.2, color=grid_color)
    ax.tick_params(colors=title_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)
    
    # Set date ticks
    set_date_ticks(ax, 20, tick_color=title_color)
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1e1e1e')
    plt.close()
    
    # Reset to default style
    plt.style.use('default')
    
    print(f"Volatility plot saved to {save_path}")

def decompose_time_series(data, save_path='figures/decomposition.png'):
    """Decompose time series into trend, seasonal, and residual components"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Make a copy of the data to avoid modifying the original
    close_prices = data['Close'].copy()
    
    # Fill missing values using forward fill and then backward fill to ensure no NaNs
    close_prices = close_prices.ffill().bfill()
    
    # Set dark style for plots
    plt.style.use('dark_background')
    
    # Check if there are still any NaN values
    if close_prices.isna().any():
        print("Warning: Data still contains NaN values after filling. Creating simple trend plot instead.")
        plt.figure(figsize=(12, 6), facecolor='#1e1e1e')
        plt.plot(close_prices.dropna(), color='#00a8ff')
        plt.title('Bitcoin Price Trend (Seasonal Decomposition Failed)', color='#e0e0e0')
        plt.xlabel('Date', color='#e0e0e0')
        plt.ylabel('Price (USD)', color='#e0e0e0')
        plt.grid(True, alpha=0.2, color='#555555')
        
        # Set 20 x-ticks
        set_date_ticks(plt.gca(), 20)
        
        plt.savefig(save_path, facecolor='#1e1e1e')
        plt.close()
        print(f"Simple trend plot saved to {save_path}")
        # Reset to default style
        plt.style.use('default')
        return
    
    # Ensure we have enough data points and adjust period if necessary
    data_length = len(close_prices)
    if data_length < 730:  # Less than 2 years of data
        # Use a smaller period for seasonal decomposition
        period = min(365, data_length // 4)
        if period < 7:
            print("Warning: Not enough data for meaningful seasonal decomposition.")
            # Reset to default style
            plt.style.use('default')
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
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), facecolor='#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        
        # Define colors for better visibility on dark background
        line_color = '#00a8ff'  # Bright blue
        title_color = '#e0e0e0'  # Light gray
        grid_color = '#555555'   # Medium gray
        
        # Original price plot
        decomposition.observed.plot(ax=ax1, color=line_color)
        ax1.set_title('Observed', color=title_color)
        ax1.set_facecolor('#2d2d2d')
        ax1.grid(True, alpha=0.2, color=grid_color)
        ax1.tick_params(colors=title_color)
        ax1.spines['bottom'].set_color(grid_color)
        ax1.spines['top'].set_color(grid_color)
        ax1.spines['left'].set_color(grid_color)
        ax1.spines['right'].set_color(grid_color)
        set_date_ticks(ax1, 20, tick_color=title_color)
        
        # Log scale plot
        decomposition.observed.plot(ax=ax2, color=line_color)
        ax2.set_yscale('log')
        ax2.set_title('Observed (Log Scale)', color=title_color)
        ax2.set_facecolor('#2d2d2d')
        ax2.grid(True, alpha=0.2, color=grid_color)
        ax2.tick_params(colors=title_color)
        ax2.spines['bottom'].set_color(grid_color)
        ax2.spines['top'].set_color(grid_color)
        ax2.spines['left'].set_color(grid_color)
        ax2.spines['right'].set_color(grid_color)
        set_date_ticks(ax2, 20, tick_color=title_color)
        
        # Trend plot
        decomposition.trend.plot(ax=ax3, color=line_color)
        ax3.set_title('Trend', color=title_color)
        ax3.set_facecolor('#2d2d2d')
        ax3.grid(True, alpha=0.2, color=grid_color)
        ax3.tick_params(colors=title_color)
        ax3.spines['bottom'].set_color(grid_color)
        ax3.spines['top'].set_color(grid_color)
        ax3.spines['left'].set_color(grid_color)
        ax3.spines['right'].set_color(grid_color)
        set_date_ticks(ax3, 20, tick_color=title_color)
        
        # Residual plot
        decomposition.resid.plot(ax=ax4, color=line_color)
        ax4.set_title('Residual', color=title_color)
        ax4.set_facecolor('#2d2d2d')
        ax4.grid(True, alpha=0.2, color=grid_color)
        ax4.tick_params(colors=title_color)
        ax4.spines['bottom'].set_color(grid_color)
        ax4.spines['top'].set_color(grid_color)
        ax4.spines['left'].set_color(grid_color)
        ax4.spines['right'].set_color(grid_color)
        set_date_ticks(ax4, 20, tick_color=title_color)
        
        plt.tight_layout()
        plt.savefig(save_path, facecolor='#1e1e1e')
        plt.close()
        print(f"Time series decomposition saved to {save_path}")
        
    except Exception as e:
        print(f"Error in seasonal decomposition: {e}")
        print("Creating alternative visualization...")
        
        # Create an alternative visualization with rolling statistics
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), facecolor='#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        
        # Define colors
        line_color = '#00a8ff'  # Bright blue
        line_color2 = '#00ff7f'  # Green
        title_color = '#e0e0e0'  # Light gray
        grid_color = '#555555'   # Medium gray
        
        # Plot original data
        close_prices.plot(ax=ax1, label='Original', color=line_color)
        ax1.set_title('Bitcoin Price', color=title_color)
        ax1.set_facecolor('#2d2d2d')
        ax1.legend(facecolor='#2d2d2d', edgecolor=grid_color, labelcolor=title_color)
        ax1.grid(True, alpha=0.2, color=grid_color)
        ax1.tick_params(colors=title_color)
        ax1.spines['bottom'].set_color(grid_color)
        ax1.spines['top'].set_color(grid_color)
        ax1.spines['left'].set_color(grid_color)
        ax1.spines['right'].set_color(grid_color)
        set_date_ticks(ax1, 20, tick_color=title_color)
        
        # Plot log scale
        close_prices.plot(ax=ax2, label='Original (Log Scale)', color=line_color)
        ax2.set_yscale('log')
        ax2.set_title('Bitcoin Price (Log Scale)', color=title_color)
        ax2.set_facecolor('#2d2d2d')
        ax2.legend(facecolor='#2d2d2d', edgecolor=grid_color, labelcolor=title_color)
        ax2.grid(True, alpha=0.2, color=grid_color)
        ax2.tick_params(colors=title_color)
        ax2.spines['bottom'].set_color(grid_color)
        ax2.spines['top'].set_color(grid_color)
        ax2.spines['left'].set_color(grid_color)
        ax2.spines['right'].set_color(grid_color)
        set_date_ticks(ax2, 20, tick_color=title_color)
        
        # Plot rolling standard deviation as a proxy for volatility
        close_prices.rolling(window=20).std().plot(ax=ax3, label='20-day Rolling Std Dev', color=line_color2)
        ax3.set_title('Rolling Volatility', color=title_color)
        ax3.set_facecolor('#2d2d2d')
        ax3.legend(facecolor='#2d2d2d', edgecolor=grid_color, labelcolor=title_color)
        ax3.grid(True, alpha=0.2, color=grid_color)
        ax3.tick_params(colors=title_color)
        ax3.spines['bottom'].set_color(grid_color)
        ax3.spines['top'].set_color(grid_color)
        ax3.spines['left'].set_color(grid_color)
        ax3.spines['right'].set_color(grid_color)
        set_date_ticks(ax3, 20, tick_color=title_color)
        
        plt.tight_layout()
        plt.savefig(save_path, facecolor='#1e1e1e')
        plt.close()
        print(f"Alternative time series analysis saved to {save_path}")
    
    # Reset to default style
    plt.style.use('default')

def set_date_ticks(ax, num_ticks=20, tick_color='#e0e0e0'):
    """Set approximately num_ticks date ticks on the x-axis"""
    # Get the date range
    dates = ax.get_xlim()
    # Convert to matplotlib dates if not already
    if not isinstance(dates[0], float):
        dates = mdates.date2num(dates)
    
    # Calculate tick positions
    tick_positions = np.linspace(dates[0], dates[1], num_ticks)
    
    # Set the tick positions
    ax.set_xticks(tick_positions)
    
    # Format the tick labels
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)
    
    # Rotate the tick labels for better readability and set color
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color=tick_color)

def plot_historic_price(data, save_path='figures/historic_price.png'):
    """Create a detailed plot of Bitcoin price history with volume"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Set dark style for plots
    plt.style.use('dark_background')
    
    # Define colors for better visibility on dark background
    line_color = '#00a8ff'      # Bright blue
    ma50_color = '#ff9500'      # Orange
    ma200_color = '#ff3b30'     # Red
    volume_color = '#8e8e93'    # Gray
    title_color = '#e0e0e0'     # Light gray
    grid_color = '#555555'      # Medium gray
    annotation_color = '#34c759'# Green
    
    # Create figure with two subplots (price and volume)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   facecolor='#1e1e1e')
    ax1.set_facecolor('#2d2d2d')
    ax2.set_facecolor('#2d2d2d')
    
    # Plot price on the first subplot
    ax1.plot(data.index, data['Close'], label='Close Price', color=line_color, linewidth=2)
    
    # Add moving averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    ax1.plot(data.index, data['MA50'], label='50-Day MA', color=ma50_color, linewidth=1.5)
    ax1.plot(data.index, data['MA200'], label='200-Day MA', color=ma200_color, linewidth=1.5)
    
    # Set labels and title for price subplot
    ax1.set_title('Bitcoin Price History with Moving Averages', fontsize=16, color=title_color)
    ax1.set_ylabel('Price (USD)', fontsize=12, color=title_color)
    ax1.grid(True, alpha=0.2, color=grid_color)
    
    # Configure legend and ticks
    legend = ax1.legend(facecolor='#2d2d2d', edgecolor=grid_color, labelcolor=title_color)
    ax1.tick_params(colors=title_color)
    for spine in ax1.spines.values():
        spine.set_color(grid_color)
    
    # Format y-axis with commas for thousands
    ax1.get_yaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    
    # Plot volume on the second subplot
    volume_data = data['Volume']
    ax2.bar(data.index, volume_data, color=volume_color, alpha=0.7, width=2)
    
    # Set labels for volume subplot
    ax2.set_xlabel('Date', fontsize=12, color=title_color)
    ax2.set_ylabel('Volume', fontsize=12, color=title_color)
    ax2.grid(True, alpha=0.2, color=grid_color)
    ax2.tick_params(colors=title_color)
    for spine in ax2.spines.values():
        spine.set_color(grid_color)
    
    # Format y-axis with commas for thousands and scientific notation for large numbers
    ax2.get_yaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:.1e}'))
    
    # Set date ticks for both subplots
    set_date_ticks(ax1, 20, tick_color=title_color)
    set_date_ticks(ax2, 20, tick_color=title_color)
    
    # Add annotations for key events
    key_events = [
        ('2020-03-12', 'COVID-19 Crash', 5000),
        ('2021-04-14', 'Coinbase IPO', 64000),
        ('2021-11-10', 'All-Time High', 69000),
        ('2022-11-11', 'FTX Collapse', 17000)
    ]
    
    # Fix: Ensure the data index is sorted before finding closest dates
    sorted_data = data.sort_index()
    
    for date, label, price in key_events:
        try:
            event_date = pd.to_datetime(date)
            if event_date >= sorted_data.index[0] and event_date <= sorted_data.index[-1]:
                # Find the closest date in the index
                closest_idx = sorted_data.index.get_indexer([event_date], method='nearest')[0]
                closest_date = sorted_data.index[closest_idx]
                ax1.annotate(label, 
                            xy=(closest_date, price),
                            xytext=(10, 0),
                            textcoords="offset points",
                            arrowprops=dict(arrowstyle="->", color=annotation_color),
                            fontsize=10,
                            color=title_color)
        except Exception as e:
            print(f"Could not add annotation for {date}: {e}")
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1e1e1e')
    plt.close()
    
    # Reset to default style
    plt.style.use('default')
    
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