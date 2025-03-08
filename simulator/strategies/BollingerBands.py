import pandas as pd
import numpy as np
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
        # Default parameters - improved based on simulation results
        default_params = {
            'window': 20,
            'num_std': 2.5,  # Increased from 2.0 to reduce false signals
            'position_size': 0.1,
            'leverage': 2,  # Kept conservative
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
        
        # IMPROVEMENT: Add trend filter to reduce false signals
        trend_direction = 0
        if len(df) > 10:
            short_ma = df['Close'].rolling(window=10).mean()
            long_ma = df['Close'].rolling(window=30).mean()
            if not short_ma.isna().iloc[-1] and not long_ma.isna().iloc[-1]:
                trend_direction = 1 if short_ma.iloc[-1] > long_ma.iloc[-1] else -1
        
        # IMPROVEMENT: Check for Bollinger Band squeeze (volatility contraction)
        band_width = (df['Upper_Band'] - df['Lower_Band']) / df['MA']
        is_squeeze = band_width.iloc[-1] < band_width.iloc[-20:].mean() * 0.85 if len(band_width) >= 20 else False
        
        # IMPROVEMENT: Check for Bollinger Band bounce vs. breakout
        # Bounce: Price touches band and reverses
        # Breakout: Price crosses band and continues
        bounce_threshold = 0.02  # 2% from band
        
        # Check for bounce or breakout conditions
        lower_band_touch = df['Close'].iloc[-1] < df['Lower_Band'].iloc[-1] * (1 + bounce_threshold)
        upper_band_touch = df['Close'].iloc[-1] > df['Upper_Band'].iloc[-1] * (1 - bounce_threshold)
        
        # Check previous candles for direction
        price_direction = 1 if df['Close'].iloc[-1] > df['Close'].iloc[-2] else -1
        
        # Return buy/sell signal based on price position relative to bands
        signal = 0
        
        # Buy signals
        if df['Close'].iloc[-1] < df['Lower_Band'].iloc[-1]:
            # Strong buy signal: Price below lower band
            if trend_direction >= 0 or self.market_regime == 'bull':
                signal = 1
        elif lower_band_touch and price_direction > 0:
            # Bounce off lower band
            if trend_direction >= 0:
                signal = 1
        elif is_squeeze and trend_direction > 0:
            # Volatility contraction with uptrend - potential breakout
            signal = 1
            
        # Sell signals
        if df['Close'].iloc[-1] > df['Upper_Band'].iloc[-1]:
            # Strong sell signal: Price above upper band
            if trend_direction <= 0 or self.market_regime == 'bear':
                signal = -1
        elif upper_band_touch and price_direction < 0:
            # Bounce off upper band
            if trend_direction <= 0:
                signal = -1
        elif is_squeeze and trend_direction < 0:
            # Volatility contraction with downtrend - potential breakdown
            signal = -1
            
        return signal
    
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
        
        # IMPROVEMENT: Calculate Bollinger Band Width
        df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA']
        
        # IMPROVEMENT: Calculate %B (position within bands)
        df['%B'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
        
        # Update volatility
        self._update_volatility(df)
        
        # Calculate signals (for indicator calculation)
        if 'Signal' not in df.columns:
            df['Signal'] = 0
        df.loc[df['Close'] < df['Lower_Band'], 'Signal'] = 1
        df.loc[df['Close'] > df['Upper_Band'], 'Signal'] = -1
        
        return df 