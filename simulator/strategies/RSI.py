import pandas as pd
from typing import Dict, Any
from .BaseStrategy import BaseStrategy

class RSI(BaseStrategy):
    """Relative Strength Index (RSI) trading strategy"""
    
    def __init__(self, params: Dict[str, Any] = {}):
        """
        Initialize the strategy with parameters
        
        Parameters:
        - params: Strategy parameters
        """
        # Default parameters
        default_params = {
            'window': 14,
            'overbought': 75,
            'oversold': 25,
            'position_size': 0.05,
            'leverage': 2,
            'take_profit_pct': 0.15,
            'stop_loss_pct': 0.07
        }
        
        # Merge default parameters with provided parameters
        merged_params = {**default_params, **params}
        super().__init__(merged_params)
    
    def calculate_signal(self, price_data: pd.DataFrame) -> int:
        """
        Calculate trading signal based on RSI
        
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
        if df.empty or df['RSI'].isna().iloc[-1]:
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
        
        # Return buy/sell signal based on RSI thresholds with trend filter
        if df['RSI'].iloc[-1] < self.params['oversold']:
            # Only take buy signals in bull market or when trend is up
            if self.market_regime == 'bull' or trend_direction > 0:
                return 1  # Buy signal
        elif df['RSI'].iloc[-1] > self.params['overbought']:
            # Only take sell signals in bear market or when trend is down
            if self.market_regime == 'bear' or trend_direction < 0:
                return -1  # Sell signal
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
        window = int(self.params['window'])
        
        # Calculate RSI more efficiently
        close_prices = df['Close']
        delta = close_prices.diff()
        
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Update volatility
        self._update_volatility(df)
        
        # Calculate signals (for indicator calculation)
        if 'Signal' not in df.columns:
            df['Signal'] = 0
        df.loc[df['RSI'] < self.params['oversold'], 'Signal'] = 1
        df.loc[df['RSI'] > self.params['overbought'], 'Signal'] = -1
        
        return df 