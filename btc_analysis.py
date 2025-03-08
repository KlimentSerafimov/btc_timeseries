import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from statsmodels.tsa.seasonal import seasonal_decompose  # type: ignore
import os
import matplotlib.dates as mdates 

def load_data(filepath='data/btc_price_data.csv'):
    """Load Bitcoin data from CSV file"""
    data = pd.read_csv(filepath)
    
    # Handle date column and set as index
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    elif not isinstance(data.index, pd.DatetimeIndex):
        # Try to convert the first column to datetime and use as index
        first_col = data.columns[0]
        data[first_col] = pd.to_datetime(data[first_col])
        data.set_index(first_col, inplace=True)
    
    # Convert numeric columns to float
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    print(f"Loaded data with shape: {data.shape}\nIndex type: {type(data.index)}")
    return data

def calculate_returns(data):
    """Calculate daily and monthly returns"""
    data['Daily_Return'] = data['Close'].pct_change(fill_method=None)
    monthly_data = data['Close'].resample('ME').last()
    monthly_returns = monthly_data.pct_change(fill_method=None)
    return data, monthly_returns

def set_date_ticks(ax, num_ticks=20, tick_color='#e0e0e0'):
    """Set approximately num_ticks date ticks on the x-axis"""
    dates = ax.get_xlim()
    if not isinstance(dates[0], float):
        dates = mdates.date2num(dates)
    
    ax.set_xticks(np.linspace(dates[0], dates[1], num_ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color=tick_color)

def plot_returns_distribution(returns, save_path='figures/returns_distribution.png'):
    """Plot the distribution of returns"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.style.use('dark_background')
    
    # Define colors
    colors = {
        'hist': '#00a8ff',   # Bright blue
        'kde': '#00ff7f',    # Green
        'title': '#e0e0e0',  # Light gray
        'grid': '#555555'    # Medium gray
    }
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1e1e1e')
    ax.set_facecolor('#2d2d2d')
    
    # Plot histogram with kernel density estimate
    sns.histplot(returns.dropna(), kde=True, bins=50, ax=ax, color=colors['hist'], 
                 line_kws={'color': colors['kde'], 'linewidth': 2})
    
    # Set styling
    ax.set_title('Distribution of Bitcoin Daily Returns', color=colors['title'], fontsize=16)
    ax.set_xlabel('Daily Returns', color=colors['title'], fontsize=12)
    ax.set_ylabel('Frequency', color=colors['title'], fontsize=12)
    ax.grid(True, alpha=0.2, color=colors['grid'])
    ax.tick_params(colors=colors['title'])
    for spine in ax.spines.values():
        spine.set_color(colors['grid'])
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1e1e1e')
    plt.close()
    plt.style.use('default')
    
    print(f"Returns distribution plot saved to {save_path}")

def plot_volatility(data, window=20, save_path='figures/volatility.png'):
    """Plot rolling volatility"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data['Volatility'] = data['Daily_Return'].rolling(window=window).std() * np.sqrt(window)
    
    plt.style.use('dark_background')
    colors = {'line': '#00a8ff', 'title': '#e0e0e0', 'grid': '#555555'}
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1e1e1e')
    ax.set_facecolor('#2d2d2d')
    
    # Plot and style
    ax.plot(data['Volatility'], color=colors['line'], linewidth=2)
    ax.set_title(f'Bitcoin {window}-Day Rolling Volatility', color=colors['title'], fontsize=16)
    ax.set_xlabel('Date', color=colors['title'], fontsize=12)
    ax.set_ylabel('Volatility', color=colors['title'], fontsize=12)
    ax.grid(True, alpha=0.2, color=colors['grid'])
    ax.tick_params(colors=colors['title'])
    for spine in ax.spines.values():
        spine.set_color(colors['grid'])
    
    set_date_ticks(ax, 20, tick_color=colors['title'])
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1e1e1e')
    plt.close()
    plt.style.use('default')
    
    print(f"Volatility plot saved to {save_path}")

def decompose_time_series(data, save_path='figures/decomposition.png'):
    """Decompose time series into trend, seasonal, and residual components"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    close_prices = data['Close'].copy().ffill().bfill()
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
        set_date_ticks(plt.gca(), 20)
        plt.savefig(save_path, facecolor='#1e1e1e')
        plt.close()
        print(f"Simple trend plot saved to {save_path}")
        plt.style.use('default')
        return
    
    # Determine appropriate period for decomposition
    data_length = len(close_prices)
    period = min(365, data_length // 4) if data_length < 730 else 365
    if period < 7:
        print("Warning: Not enough data for meaningful seasonal decomposition.")
        plt.style.use('default')
        return
    
    # Resample to daily frequency for regular time series
    close_prices = close_prices.resample('D').interpolate(method='linear')
    
    try:
        # Try decomposition
        try:
            decomposition = seasonal_decompose(close_prices, model='multiplicative', period=period)
        except Exception as e:
            print(f"Multiplicative decomposition failed: {e}\nTrying additive decomposition instead...")
            decomposition = seasonal_decompose(close_prices, model='additive', period=period)
        
        # Define colors
        colors = {'line': '#00a8ff', 'title': '#e0e0e0', 'grid': '#555555'}
        
        # Plot decomposition without the seasonal component
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), facecolor='#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        
        # Style each subplot
        for ax, title, data in [
            (ax1, 'Observed', decomposition.observed),
            (ax2, 'Observed (Log Scale)', decomposition.observed),
            (ax3, 'Trend', decomposition.trend),
            (ax4, 'Residual', decomposition.resid)
        ]:
            data.plot(ax=ax, color=colors['line'])
            if ax == ax2:
                ax.set_yscale('log')
                title += ' (Log Scale)'
            ax.set_title(title, color=colors['title'])
            ax.set_facecolor('#2d2d2d')
            ax.grid(True, alpha=0.2, color=colors['grid'])
            ax.tick_params(colors=colors['title'])
            for spine in ax.spines.values():
                spine.set_color(colors['grid'])
            set_date_ticks(ax, 20, tick_color=colors['title'])
        
        plt.tight_layout()
        plt.savefig(save_path, facecolor='#1e1e1e')
        plt.close()
        print(f"Time series decomposition saved to {save_path}")
        
    except Exception as e:
        print(f"Error in seasonal decomposition: {e}\nCreating alternative visualization...")
        
        # Create an alternative visualization with rolling statistics
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), facecolor='#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        
        # Define colors
        colors = {'line1': '#00a8ff', 'line2': '#00ff7f', 'title': '#e0e0e0', 'grid': '#555555'}
        
        # Plot and style each subplot
        close_prices.plot(ax=ax1, label='Original', color=colors['line1'])
        ax1.set_title('Bitcoin Price', color=colors['title'])
        ax1.set_facecolor('#2d2d2d')
        
        close_prices.plot(ax=ax2, label='Original (Log Scale)', color=colors['line1'])
        ax2.set_yscale('log')
        ax2.set_title('Bitcoin Price (Log Scale)', color=colors['title'])
        ax2.set_facecolor('#2d2d2d')
        
        close_prices.rolling(window=20).std().plot(ax=ax3, label='20-day Rolling Std Dev', color=colors['line2'])
        ax3.set_title('Rolling Volatility', color=colors['title'])
        ax3.set_facecolor('#2d2d2d')
        
        # Apply common styling to all subplots
        for ax in [ax1, ax2, ax3]:
            ax.legend(facecolor='#2d2d2d', edgecolor=colors['grid'], labelcolor=colors['title'])
            ax.grid(True, alpha=0.2, color=colors['grid'])
            ax.tick_params(colors=colors['title'])
            for spine in ax.spines.values():
                spine.set_color(colors['grid'])
            set_date_ticks(ax, 20, tick_color=colors['title'])
        
        plt.tight_layout()
        plt.savefig(save_path, facecolor='#1e1e1e')
        plt.close()
        print(f"Alternative time series analysis saved to {save_path}")
    
    plt.style.use('default')

def plot_historic_price(data, save_path='figures/historic_price.png'):
    """Create a detailed plot of Bitcoin price history with volume"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.style.use('dark_background')
    
    # Define colors
    colors = {
        'line': '#00a8ff',      # Bright blue
        'ma50': '#ff9500',      # Orange
        'ma200': '#ff3b30',     # Red
        'volume': '#8e8e93',    # Gray
        'title': '#e0e0e0',     # Light gray
        'grid': '#555555',      # Medium gray
        'annotation': '#34c759' # Green
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   facecolor='#1e1e1e')
    ax1.set_facecolor('#2d2d2d')
    ax2.set_facecolor('#2d2d2d')
    
    # Calculate moving averages
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # Plot price and moving averages
    ax1.plot(data.index, data['Close'], label='Close Price', color=colors['line'], linewidth=2)
    ax1.plot(data.index, data['MA50'], label='50-Day MA', color=colors['ma50'], linewidth=1.5)
    ax1.plot(data.index, data['MA200'], label='200-Day MA', color=colors['ma200'], linewidth=1.5)
    
    # Style price subplot
    ax1.set_title('Bitcoin Price History with Moving Averages', fontsize=16, color=colors['title'])
    ax1.set_ylabel('Price (USD)', fontsize=12, color=colors['title'])
    ax1.grid(True, alpha=0.2, color=colors['grid'])
    ax1.legend(facecolor='#2d2d2d', edgecolor=colors['grid'], labelcolor=colors['title'])
    ax1.tick_params(colors=colors['title'])
    ax1.get_yaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    
    # Plot volume
    ax2.bar(data.index, data['Volume'], color=colors['volume'], alpha=0.7, width=2)
    
    # Style volume subplot
    ax2.set_xlabel('Date', fontsize=12, color=colors['title'])
    ax2.set_ylabel('Volume', fontsize=12, color=colors['title'])
    ax2.grid(True, alpha=0.2, color=colors['grid'])
    ax2.tick_params(colors=colors['title'])
    ax2.get_yaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:.1e}'))
    
    # Style spines for both subplots
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_color(colors['grid'])
        set_date_ticks(ax, 20, tick_color=colors['title'])
    
    # Add annotations for key events
    key_events = [
        ('2020-03-12', 'COVID-19 Crash', 5000),
        ('2021-04-14', 'Coinbase IPO', 64000),
        ('2021-11-10', 'All-Time High', 69000),
        ('2022-11-11', 'FTX Collapse', 17000)
    ]
    
    # Ensure the data index is sorted before finding closest dates
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
                            arrowprops=dict(arrowstyle="->", color=colors['annotation']),
                            fontsize=10,
                            color=colors['title'])
        except Exception as e:
            print(f"Could not add annotation for {date}: {e}")
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1e1e1e')
    plt.close()
    plt.style.use('default')
    
    print(f"Historic price plot saved to {save_path}")

if __name__ == "__main__":
    # Load data and run analysis
    btc_data = load_data()
    print("Data types before conversion:")
    print(btc_data.dtypes)
    
    plot_historic_price(btc_data)
    btc_data, monthly_returns = calculate_returns(btc_data)
    plot_returns_distribution(btc_data['Daily_Return'])
    plot_volatility(btc_data)
    
    try:
        decompose_time_series(btc_data)
    except Exception as e:
        print(f"Error in time series decomposition: {e}\nContinuing with analysis...")
    
    print("\nAnalysis complete!") 