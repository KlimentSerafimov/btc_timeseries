import pandas as pd
from typing import Dict, Any
from .BaseStrategy import BaseStrategy

class BollingerBands(BaseStrategy):
    """Bollinger Bands trading strategy"""
    
    def __init__(self, params: Dict[str, Any] = {}):
        """
        Initialize the strategy with parameters
        
        Parameters:
        - params: Strategy parameters
        """
        # Default parameters
        default_params = {
            'window': 20,
            'num_std': 2.5,
            'position_size': 0.1,
            'leverage': 2,
            'take_profit_pct': 0.15,
            'stop_loss_pct': 0.07
        }
        
        # Merge default parameters with provided parameters
        merged_params = {**default_params, **params}
        super().__init__(merged_params)
    
    def calculate_signal(self, price_data: pd.DataFrame) -> int:
        """
        Calculate trading signal based on Bollinger Bands
        
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
        if df.empty or df['Close'].isna().iloc[-1] or df['Upper_Band'].isna().iloc[-1] or df['Lower_Band'].isna().iloc[-1]:
            return 0
        
        # Update market regime
        self._update_market_regime(df)
        
        # Add trend filter to reduce false signals
        trend_direction = 0
        if len(df) > 10:
            short_ma = df['Close'].rolling(window=10).mean()
            long_ma = df['Close'].rolling(window=30).mean()
            if not short_ma.isna().iloc[-1] and not long_ma.isna().iloc[-1]:
                trend_direction = 1 if short_ma.iloc[-1] > long_ma.iloc[-1] else -1
        
        # Return buy/sell signal based on price position relative to bands
        # Only take signals that align with the trend
        if df['Close'].iloc[-1] < df['Lower_Band'].iloc[-1] and (trend_direction >= 0 or self.market_regime == 'bear'):
            return 1  # Buy signal when price below lower band
        elif df['Close'].iloc[-1] > df['Upper_Band'].iloc[-1] and (trend_direction <= 0 or self.market_regime == 'bull'):
            return -1  # Sell signal when price above upper band
        return 0
    
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
        window = int(self.params['window'])
        num_std = self.params['num_std']
        df['MA'] = df['Close'].rolling(window=window).mean()
        df['STD'] = df['Close'].rolling(window=window).std()
        df['Upper_Band'] = df['MA'] + (df['STD'] * num_std)
        df['Lower_Band'] = df['MA'] - (df['STD'] * num_std)
        
        # Update volatility
        self._update_volatility(df)
        
        # Calculate signals (for indicator calculation)
        if 'Signal' not in df.columns:
            df['Signal'] = 0
        df.loc[df['Close'] < df['Lower_Band'], 'Signal'] = 1
        df.loc[df['Close'] > df['Upper_Band'], 'Signal'] = -1
        
        return df 