from datetime import datetime
from typing import Optional

class FuturesPosition:
    """Represents a futures position with leverage"""

    def __init__(self,
                 entry_price: float,
                 size: float,
                 leverage: float,
                 is_long: bool,
                 liquidation_price: float,
                 timestamp: datetime):
        """
        Initialize a futures position

        Parameters:
        - entry_price: Entry price in USD
        - size: Position size in BTC
        - leverage: Leverage multiplier
        - is_long: True for long positions, False for short
        - liquidation_price: Price at which position gets liquidated
        - timestamp: Time when position was opened
        """
        self.entry_price = entry_price
        self.size = size
        self.leverage = leverage
        self.is_long = is_long
        self.liquidation_price = liquidation_price
        self.timestamp = timestamp
        self.pnl = 0.0
        self.is_liquidated = False
        self.exit_price: Optional[float] = None
        self.exit_timestamp: Optional[datetime] = None

    def calculate_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized PnL for inverse perpetual futures in USD

        For inverse futures (denominated in BTC):
        - Long position: PnL = position_value * (current_price/entry_price - 1)
        - Short position: PnL = position_value * (1 - current_price/entry_price)
        """
        position_value = self.size * self.entry_price  # Value in USD

        # Calculate PnL based on position type
        multiplier = (current_price/self.entry_price - 1) if self.is_long else (1 - current_price/self.entry_price)
        self.pnl = position_value * multiplier * self.leverage

        return self.pnl

    def close_position(self, exit_price: float, timestamp: datetime) -> float:
        """Close the position and realize PnL"""
        self.exit_price = exit_price
        self.exit_timestamp = timestamp
        self.calculate_pnl(exit_price)
        return self.pnl

    def is_liquidated_at_price(self, price: float) -> bool:
        """Check if position would be liquidated at given price"""
        return (self.is_long and price <= self.liquidation_price) or \
               (not self.is_long and price >= self.liquidation_price)

    def __str__(self) -> str:
        position_type = "LONG" if self.is_long else "SHORT"
        status = "LIQUIDATED" if self.is_liquidated else "CLOSED" if self.exit_price else "OPEN"

        return (f"{position_type} {self.size:.6f} BTC @ {self.entry_price:.2f} USD "
                f"(Leverage: {self.leverage}x, {status})")