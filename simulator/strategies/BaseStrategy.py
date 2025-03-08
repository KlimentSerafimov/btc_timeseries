import pandas as pd
from typing import Dict, Any, List, Optional

class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self, params: Dict[str, Any] = {}):
        """
        Initialize the strategy with parameters
        
        Parameters:
        - params: Strategy parameters
        """
        self.params = params
        self.volatility_history: List[float] = []
        self.market_regime = 'unknown'
    
    def calculate_signal(self, price_data: pd.DataFrame) -> int:
        """
        Calculate trading signal based on the strategy
        
        Parameters:
        - price_data: DataFrame with price data
        
        Returns:
        - signal: 1 for buy, -1 for sell, 0 for neutral
        """
        raise NotImplementedError("Subclasses must implement calculate_signal method")
    
    def calculate_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy
        
        Parameters:
        - price_data: DataFrame with price data
        
        Returns:
        - DataFrame with added indicator columns
        """
        raise NotImplementedError("Subclasses must implement calculate_indicators method")
    
    def _update_market_regime(self, price_data: pd.DataFrame) -> None:
        """
        Detect market regime (bull/bear) based on price action
        """
        if len(price_data) < 50:
            return
            
        # Calculate 50-day moving average
        ma_50 = price_data['Close'].rolling(window=50).mean()
        
        # Calculate 200-day moving average
        ma_200 = price_data['Close'].rolling(window=200).mean()
        
        if not ma_50.isna().iloc[-1] and not ma_200.isna().iloc[-1]:
            # Bull market if 50-day MA > 200-day MA
            if ma_50.iloc[-1] > ma_200.iloc[-1]:
                self.market_regime = 'bull'
            else:
                self.market_regime = 'bear'
    
    def _update_volatility(self, price_data: pd.DataFrame) -> None:
        """
        Update volatility history based on price data
        """
        # Calculate volatility
        df = price_data.copy()
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Store latest volatility for position sizing
        if not df['volatility'].isna().iloc[-1]:
            self.volatility_history.append(df['volatility'].iloc[-1])
            if len(self.volatility_history) > 20:
                self.volatility_history.pop(0)
    
    def get_dynamic_position_size(self, base_size: float) -> float:
        """
        Calculate position size dynamically based on volatility
        """
        # If we don't have enough volatility data, use base size
        if len(self.volatility_history) < 5:
            return base_size
            
        # Calculate average volatility
        avg_volatility = sum(self.volatility_history) / len(self.volatility_history)
        
        # If volatility is high, reduce position size
        if avg_volatility > 0.05:  # 5% daily volatility is high
            return base_size * 0.5
        # If volatility is very low, increase position size slightly
        elif avg_volatility < 0.02:  # 2% daily volatility is low
            return base_size * 1.2
            
        return base_size