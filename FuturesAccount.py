from FuturesPosition import FuturesPosition


from datetime import datetime
from typing import Any, Dict, List, Optional


class FuturesAccount:
    """Represents a trading account for futures trading"""

    def __init__(self, initial_balance: float = 10000.0):
        """
        Initialize a futures trading account

        Parameters:
        - initial_balance: Initial account balance in USD
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.available_balance = initial_balance
        self.positions: List[FuturesPosition] = []
        self.active_positions: List[FuturesPosition] = []
        self.closed_positions: List[FuturesPosition] = []
        self.transaction_history: List[Dict[str, Any]] = []

    def open_position(self,
                     price: float,
                     size: float,
                     leverage: float,
                     is_long: bool,
                     liquidation_price: float,
                     timestamp: datetime) -> Optional[FuturesPosition]:
        """Open a new position"""
        # Check if we have enough available balance
        margin_required = size * price / leverage

        if margin_required > self.available_balance:
            print(f"Insufficient balance. Required: {margin_required}, Available: {self.available_balance}")
            return None

        # Create the position
        position = FuturesPosition(
            entry_price=price,
            size=size,
            leverage=leverage,
            is_long=is_long,
            liquidation_price=liquidation_price,
            timestamp=timestamp
        )

        # Add to active positions
        self.active_positions.append(position)

        # Record transaction
        self.transaction_history.append({
            'type': 'OPEN',
            'position_type': 'LONG' if is_long else 'SHORT',
            'price': price,
            'size': size,
            'leverage': leverage,
            'timestamp': timestamp
        })

        return position

    def close_position(self,
                      position: FuturesPosition,
                      price: float,
                      timestamp: datetime,
                      fee_rate: float = 0.0005) -> float:
        """
        Close an open position

        Parameters:
        - position: The position to close
        - price: Current market price
        - timestamp: Current time
        - fee_rate: Trading fee rate (default 0.05%)

        Returns:
        - Realized PnL
        """
        if position not in self.active_positions:
            print("Position not found or already closed")
            return 0.0

        # Calculate PnL
        pnl = position.close_position(price, timestamp)

        # Calculate fee
        position_value = position.size * price
        fee = position_value * fee_rate

        # Update account balance
        position_margin = position.size * position.entry_price / position.leverage
        self.available_balance += position_margin + pnl - fee
        self.balance += pnl - fee

        # Record transaction
        self.transaction_history.append({
            'type': 'CLOSE',
            'position_type': 'LONG' if position.is_long else 'SHORT',
            'entry_price': position.entry_price,
            'exit_price': price,
            'size': position.size,
            'leverage': position.leverage,
            'pnl': pnl,
            'fee': fee,
            'timestamp': timestamp
        })

        # Move from active to closed positions
        self.active_positions.remove(position)
        self.closed_positions.append(position)

        return pnl

    def liquidate_position(self,
                          position: FuturesPosition,
                          price: float,
                          timestamp: datetime) -> float:
        """Liquidate a position that hit its liquidation price"""
        if position not in self.active_positions:
            return 0.0

        position.is_liquidated = True
        pnl = position.close_position(position.liquidation_price, timestamp)

        # In liquidation, the entire margin is lost
        position_margin = position.size * position.entry_price / position.leverage

        # Record transaction
        self.transaction_history.append({
            'type': 'LIQUIDATION',
            'position_type': 'LONG' if position.is_long else 'SHORT',
            'entry_price': position.entry_price,
            'liquidation_price': position.liquidation_price,
            'size': position.size,
            'leverage': position.leverage,
            'margin_lost': position_margin,
            'timestamp': timestamp
        })

        # Move from active to closed positions
        self.active_positions.remove(position)
        self.closed_positions.append(position)

        return pnl

    def update_account(self, current_price: float, timestamp: datetime) -> None:
        """Update account status based on current price"""
        # Check for liquidations
        for position in list(self.active_positions):
            if position.is_liquidated_at_price(current_price):
                self.liquidate_position(position, current_price, timestamp)

        # Update unrealized PnL for remaining positions
        total_unrealized_pnl = sum(position.calculate_pnl(current_price) for position in self.active_positions)

        # Calculate used margin
        used_margin = sum(pos.size * pos.entry_price / pos.leverage for pos in self.active_positions)

        # Update balance with unrealized PnL
        realized_pnl = self.balance - self.initial_balance - total_unrealized_pnl
        self.balance = self.initial_balance + realized_pnl + total_unrealized_pnl

        # Update available balance
        self.available_balance = self.balance - used_margin

    def get_account_summary(self) -> Dict[str, Any]:
        """Get a summary of the account status"""
        return {
            'balance': self.balance,
            'available_balance': self.available_balance,
            'active_positions': len(self.active_positions),
            'closed_positions': len(self.closed_positions),
            'total_pnl': self.balance - self.initial_balance,
            'pnl_percentage': ((self.balance / self.initial_balance) - 1) * 100
        }