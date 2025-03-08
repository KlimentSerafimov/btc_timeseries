import pandas as pd
from typing import Dict, Any
from .BaseStrategy import BaseStrategy

class MovingAverageCrossover(BaseStrategy):
    """Moving Average Crossover trading strategy"""
    
    def __init__(self, params: Dict[str, Any] = {}):
        """
        Initialize the strategy with parameters
        
        Parameters:
        - params: Strategy parameters
        """
        # Default parameters
        default_params = {
            'short_window': 10,
            'long_window': 30,
            'position_size': 0.1,  # BTC
            'leverage': 3,
            'take_profit_pct': 0.2,
            'stop_loss_pct': 0.07
        }
        
        # Merge default parameters with provided parameters
        merged_params = {**default_params, **params}
        super().__init__(merged_params)
    
    def calculate_signal(self, price_data: pd.DataFrame) -> int:
        """
        Calculate trading signal based on moving average crossover
        
        Parameters:
        - price_data: DataFrame with price data
        
        Returns:
        - signal: 1 for buy, -1 for sell, 0 for neutral
        """
        # Create a copy to avoid SettingWithCopyWarning
        df = price_data.copy()
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Return signal only if we have valid data
        if df.empty or df['MA_short'].isna().iloc[-1] or df['MA_long'].isna().iloc[-1]:
            return 0
        
        # Update market regime
        self._update_market_regime(df)
        
        # Return buy/sell signal based on crossover
        return 1 if df['MA_short'].iloc[-1] > df['MA_long'].iloc[-1] else -1
    
    def calculate_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy
        
        Parameters:
        - price_data: DataFrame with price data
        
        Returns:
        - DataFrame with added indicator columns
        """
        # Create a copy to avoid SettingWithCopyWarning
        df = price_data.copy()
        
        # Calculate indicators
        short_window = int(self.params['short_window'])
        long_window = int(self.params['long_window'])
        df['MA_short'] = df['Close'].rolling(window=short_window).mean()
        df['MA_long'] = df['Close'].rolling(window=long_window).mean()
        
        # Update volatility
        self._update_volatility(df)
        
        # Calculate signal based on crossover (for indicator calculation)
        if 'Signal' not in df.columns:
            df['Signal'] = 0
        df.loc[df['MA_short'] > df['MA_long'], 'Signal'] = 1
        df.loc[df['MA_short'] < df['MA_long'], 'Signal'] = -1
        
        return df 