import pandas as pd
import numpy as np
from typing import Dict, Any
from .BaseStrategy import BaseStrategy

class AdaptiveMomentum(BaseStrategy):
    """
    Adaptive Momentum Strategy that combines multiple indicators and adapts to market conditions.
    
    Features:
    - Uses MACD for trend direction
    - Incorporates RSI for overbought/oversold conditions
    - Adapts parameters based on volatility
    - Uses volume as confirmation
    - Implements dynamic stop-loss and take-profit levels
    """
    
    def __init__(self, params: Dict[str, Any] = {}):
        """
        Initialize the strategy with parameters
        
        Parameters:
        - params: Strategy parameters
        """
        # Default parameters
        default_params = {
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'rsi_window': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'atr_window': 14,
            'position_size': 0.1,
            'leverage': 3,
            'base_take_profit_pct': 0.15,
            'base_stop_loss_pct': 0.07,
            'trend_strength_window': 50,
            'take_profit_pct': 0.15,  # For compatibility with TradingBot
            'stop_loss_pct': 0.07     # For compatibility with TradingBot
        }
        
        # Merge default parameters with provided parameters
        merged_params = {**default_params, **params}
        super().__init__(merged_params)
        
        # Additional state variables
        self.trend_strength = 0
        self.volatility_ratio = 1.0
        self.last_signal = 0
        self.signal_counter = 0
    
    def calculate_signal(self, price_data: pd.DataFrame) -> int:
        """
        Calculate trading signal based on adaptive momentum strategy
        
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
        if df.empty or df['MACD'].isna().iloc[-1] or df['RSI'].isna().iloc[-1]:
            return 0
        
        # Update market regime and volatility
        self._update_market_regime(df)
        self._update_volatility(df)
        
        # Calculate trend strength (how strong the current trend is)
        self._calculate_trend_strength(df)
        
        # Get current values
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['MACD_Signal'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        
        # Generate base signal
        signal = 0
        
        # MACD crossover (primary signal)
        if current_macd > current_signal:
            # Bullish MACD crossover
            if self.last_signal <= 0:  # Only change if previous signal was not bullish
                signal = 1
        elif current_macd < current_signal:
            # Bearish MACD crossover
            if self.last_signal >= 0:  # Only change if previous signal was not bearish
                signal = -1
        
        # Filter signals based on RSI (avoid overbought/oversold conditions)
        if signal == 1 and current_rsi > self.params['rsi_overbought']:
            # Don't buy if RSI is overbought
            signal = 0
        elif signal == -1 and current_rsi < self.params['rsi_oversold']:
            # Don't sell if RSI is oversold
            signal = 0
        
        # Filter signals based on market regime
        if signal == 1 and self.market_regime == 'bear' and self.trend_strength > 0.7:
            # Only take long signals in bear market if trend strength is weak
            signal = 0
        elif signal == -1 and self.market_regime == 'bull' and self.trend_strength > 0.7:
            # Only take short signals in bull market if trend strength is weak
            signal = 0
        
        # Implement signal persistence (avoid frequent switching)
        if signal == 0 and self.signal_counter < 3:
            signal = self.last_signal
            self.signal_counter += 1
        else:
            self.signal_counter = 0
            
        # Update last signal
        if signal != 0:
            self.last_signal = signal
        
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
        
        # Calculate MACD
        fast = int(self.params['macd_fast'])
        slow = int(self.params['macd_slow'])
        signal_period = int(self.params['macd_signal'])
        
        df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Calculate RSI
        rsi_window = int(self.params['rsi_window'])
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=rsi_window).mean()
        avg_loss = loss.rolling(window=rsi_window).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate ATR for volatility measurement
        atr_window = int(self.params['atr_window'])
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(window=atr_window).mean()
        
        # Calculate percentage ATR (ATR relative to price)
        df['ATR_Pct'] = df['ATR'] / df['Close'] * 100
        
        # Update volatility ratio for position sizing and risk management
        if not df['ATR_Pct'].isna().iloc[-1]:
            recent_volatility = df['ATR_Pct'].iloc[-1]
            avg_volatility = df['ATR_Pct'].rolling(window=50).mean().iloc[-1]
            
            if not pd.isna(avg_volatility) and avg_volatility > 0:
                self.volatility_ratio = recent_volatility / avg_volatility
        
        # Calculate signals (for indicator calculation)
        if 'Signal' not in df.columns:
            df['Signal'] = 0
        df.loc[(df['MACD'] > df['MACD_Signal']) & (df['RSI'] < self.params['rsi_overbought']), 'Signal'] = 1
        df.loc[(df['MACD'] < df['MACD_Signal']) & (df['RSI'] > self.params['rsi_oversold']), 'Signal'] = -1
        
        return df
    
    def _calculate_trend_strength(self, price_data: pd.DataFrame) -> None:
        """
        Calculate the strength of the current trend
        
        Parameters:
        - price_data: DataFrame with price data
        """
        if len(price_data) < self.params['trend_strength_window']:
            return
        
        # Use linear regression slope as trend strength indicator
        window = int(self.params['trend_strength_window'])
        y = price_data['Close'].iloc[-window:].values
        x = np.arange(len(y))
        
        # Calculate slope and R-squared
        if len(y) > 1:
            # Convert to numpy array with float values to ensure compatibility
            y_array = np.array(y, dtype=float)
            slope, intercept = np.polyfit(x, y_array, 1)
            y_pred = slope * x + intercept
            # Ensure y_array is used for calculations
            y_mean = np.mean(y_array)
            ss_total = np.sum((y_array - y_mean)**2)
            ss_residual = np.sum((y_array - y_pred)**2)
            
            if ss_total > 0:
                r_squared = 1 - (ss_residual / ss_total)
                # Normalize trend strength between 0 and 1
                self.trend_strength = min(1.0, max(0.0, abs(r_squared)))
            else:
                self.trend_strength = 0
    
    def get_dynamic_position_size(self, base_size: float) -> float:
        """
        Calculate position size dynamically based on volatility and trend strength
        
        Parameters:
        - base_size: Base position size
        
        Returns:
        - Adjusted position size
        """
        # Start with base size
        adjusted_size = base_size
        
        # Adjust based on volatility ratio
        if self.volatility_ratio > 1.5:
            # High volatility - reduce position size
            adjusted_size *= 0.7
        elif self.volatility_ratio < 0.7:
            # Low volatility - increase position size
            adjusted_size *= 1.2
        
        # Adjust based on trend strength
        if self.trend_strength > 0.7:
            # Strong trend - increase position size
            adjusted_size *= 1.2
        elif self.trend_strength < 0.3:
            # Weak trend - reduce position size
            adjusted_size *= 0.8
        
        # Adjust based on market regime
        if (self.last_signal == 1 and self.market_regime == 'bull') or \
           (self.last_signal == -1 and self.market_regime == 'bear'):
            # Trading with the market regime - increase position size
            adjusted_size *= 1.2
        
        return adjusted_size
    
    def get_dynamic_take_profit(self) -> float:
        """
        Calculate dynamic take-profit level based on market conditions
        
        Returns:
        - Take-profit percentage
        """
        base_tp = self.params['base_take_profit_pct']
        
        # Adjust based on volatility
        if self.volatility_ratio > 1.5:
            # Higher volatility - increase take profit
            return base_tp * 1.3
        elif self.volatility_ratio < 0.7:
            # Lower volatility - decrease take profit
            return base_tp * 0.8
        
        return base_tp
    
    def get_dynamic_stop_loss(self) -> float:
        """
        Calculate dynamic stop-loss level based on market conditions
        
        Returns:
        - Stop-loss percentage
        """
        base_sl = self.params['base_stop_loss_pct']
        
        # Adjust based on volatility
        if self.volatility_ratio > 1.5:
            # Higher volatility - increase stop loss
            return base_sl * 1.3
        elif self.volatility_ratio < 0.7:
            # Lower volatility - decrease stop loss
            return base_sl * 0.8
        
        return base_sl 