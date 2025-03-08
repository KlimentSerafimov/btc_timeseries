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
        # Default parameters - improved based on simulation results
        default_params = {
            'window': 14,
            'overbought': 70,  # Unchanged
            'oversold': 30,    # Unchanged
            'position_size': 0.1,  # Increased from 0.05
            'leverage': 3,     # Increased from 2
            'take_profit_pct': 0.15,  # Unchanged
            'stop_loss_pct': 0.07     # Unchanged
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
        
        # IMPROVEMENT: Add RSI divergence detection
        divergence = self._check_divergence(df)
        
        # IMPROVEMENT: Add RSI trend line breakout detection
        breakout = self._check_rsi_breakout(df)
        
        # Return buy/sell signal based on RSI thresholds with trend filter
        if df['RSI'].iloc[-1] < self.params['oversold']:
            # Only take buy signals in bull market or when trend is up
            if self.market_regime == 'bull' or trend_direction > 0 or divergence == 1 or breakout == 1:
                return 1  # Buy signal
        elif df['RSI'].iloc[-1] > self.params['overbought']:
            # Only take sell signals in bear market or when trend is down
            if self.market_regime == 'bear' or trend_direction < 0 or divergence == -1 or breakout == -1:
                return -1  # Sell signal
        return 0
    
    def _check_divergence(self, df: pd.DataFrame) -> int:
        """
        Check for RSI divergence
        
        Parameters:
        - df: DataFrame with price and RSI data
        
        Returns:
        - 1 for bullish divergence, -1 for bearish divergence, 0 for no divergence
        """
        if len(df) < 20:
            return 0
            
        # Get last 20 periods
        recent_df = df.iloc[-20:].copy()
        
        # Find local price lows and highs
        price_lows = []
        price_highs = []
        rsi_lows = []
        rsi_highs = []
        
        for i in range(2, len(recent_df) - 2):
            # Price low
            if (recent_df['Close'].iloc[i] < recent_df['Close'].iloc[i-1] and 
                recent_df['Close'].iloc[i] < recent_df['Close'].iloc[i-2] and
                recent_df['Close'].iloc[i] < recent_df['Close'].iloc[i+1] and
                recent_df['Close'].iloc[i] < recent_df['Close'].iloc[i+2]):
                price_lows.append((i, recent_df['Close'].iloc[i]))
                
            # Price high
            if (recent_df['Close'].iloc[i] > recent_df['Close'].iloc[i-1] and 
                recent_df['Close'].iloc[i] > recent_df['Close'].iloc[i-2] and
                recent_df['Close'].iloc[i] > recent_df['Close'].iloc[i+1] and
                recent_df['Close'].iloc[i] > recent_df['Close'].iloc[i+2]):
                price_highs.append((i, recent_df['Close'].iloc[i]))
                
            # RSI low
            if (recent_df['RSI'].iloc[i] < recent_df['RSI'].iloc[i-1] and 
                recent_df['RSI'].iloc[i] < recent_df['RSI'].iloc[i-2] and
                recent_df['RSI'].iloc[i] < recent_df['RSI'].iloc[i+1] and
                recent_df['RSI'].iloc[i] < recent_df['RSI'].iloc[i+2]):
                rsi_lows.append((i, recent_df['RSI'].iloc[i]))
                
            # RSI high
            if (recent_df['RSI'].iloc[i] > recent_df['RSI'].iloc[i-1] and 
                recent_df['RSI'].iloc[i] > recent_df['RSI'].iloc[i-2] and
                recent_df['RSI'].iloc[i] > recent_df['RSI'].iloc[i+1] and
                recent_df['RSI'].iloc[i] > recent_df['RSI'].iloc[i+2]):
                rsi_highs.append((i, recent_df['RSI'].iloc[i]))
        
        # Check for bullish divergence (price making lower lows, RSI making higher lows)
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            if (price_lows[-1][1] < price_lows[-2][1] and 
                rsi_lows[-1][1] > rsi_lows[-2][1]):
                return 1
                
        # Check for bearish divergence (price making higher highs, RSI making lower highs)
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            if (price_highs[-1][1] > price_highs[-2][1] and 
                rsi_highs[-1][1] < rsi_highs[-2][1]):
                return -1
                
        return 0
    
    def _check_rsi_breakout(self, df: pd.DataFrame) -> int:
        """
        Check for RSI trend line breakout
        
        Parameters:
        - df: DataFrame with RSI data
        
        Returns:
        - 1 for bullish breakout, -1 for bearish breakout, 0 for no breakout
        """
        if len(df) < 14:
            return 0
            
        # Get RSI values
        rsi_values = df['RSI'].iloc[-14:].values
        
        # Simple trend line breakout detection
        if len(rsi_values) >= 5:
            # Bullish breakout: RSI crosses above 50 from below
            if rsi_values[-2] < 50 and rsi_values[-1] > 50:
                return 1
                
            # Bearish breakout: RSI crosses below 50 from above
            if rsi_values[-2] > 50 and rsi_values[-1] < 50:
                return -1
                
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
        
        # IMPROVEMENT: Add RSI EMA for trend detection
        df['RSI_EMA'] = df['RSI'].ewm(span=9, adjust=False).mean()
        
        # Update volatility
        self._update_volatility(df)
        
        # Calculate signals (for indicator calculation)
        if 'Signal' not in df.columns:
            df['Signal'] = 0
        df.loc[df['RSI'] < self.params['oversold'], 'Signal'] = 1
        df.loc[df['RSI'] > self.params['overbought'], 'Signal'] = -1
        
        return df 