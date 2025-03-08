from FuturesAccount import FuturesAccount
from FuturesPosition import FuturesPosition


import numpy as np
import pandas as pd


from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


class FuturesExchange:
    """Simulates a Bitcoin futures exchange"""

    def __init__(self, price_data: pd.DataFrame):
        """
        Initialize the exchange with historical price data

        Parameters:
        - price_data: DataFrame with historical price data (must have 'Close' column and DatetimeIndex)
        """
        self.price_data = price_data
        self.current_index = 0
        self.current_price = price_data['Close'].iloc[0]
        self.current_timestamp = price_data.index[0]
        self.accounts: Dict[str, FuturesAccount] = {}
        self.order_book: List[Dict[str, Any]] = []
        self.executed_orders: List[Dict[str, Any]] = []

    def register_account(self, account_id: str, initial_balance: float = 10000.0) -> FuturesAccount:
        """Register a new trading account"""
        account = FuturesAccount(initial_balance)
        self.accounts[account_id] = account
        return account

    def get_current_price(self) -> float:
        """Get the current market price"""
        return self.current_price

    def get_current_timestamp(self) -> datetime:
        """Get the current timestamp"""
        return self.current_timestamp

    def _record_order(self, account_id: str, order_type: str, **kwargs) -> None:
        """Record an executed order with common fields"""
        order = {
            'account_id': account_id,
            'type': 'MARKET',
            'action': order_type,
            'timestamp': self.current_timestamp,
            **kwargs
        }
        self.executed_orders.append(order)

    def place_market_order(self,
                          account_id: str,
                          size: float,
                          leverage: float,
                          is_long: bool) -> Optional[FuturesPosition]:
        """Place a market order to open a position"""
        if account_id not in self.accounts:
            print(f"Account {account_id} not found")
            return None

        account = self.accounts[account_id]
        position = account.open_position(
            price=self.current_price,
            size=size,
            leverage=leverage,
            is_long=is_long,
            liquidation_price=self.current_price,
            timestamp=self.current_timestamp
        )

        if position:
            self._record_order(
                account_id=account_id,
                order_type='OPEN',
                position_type='LONG' if is_long else 'SHORT',
                price=self.current_price,
                size=size,
                leverage=leverage
            )

        return position

    def close_position(self, account_id: str, position: FuturesPosition) -> float:
        """Close an open position at current market price"""
        if account_id not in self.accounts:
            print(f"Account {account_id} not found")
            return 0.0

        account = self.accounts[account_id]
        pnl = account.close_position(
            position=position,
            price=self.current_price,
            timestamp=self.current_timestamp
        )

        self._record_order(
            account_id=account_id,
            order_type='CLOSE',
            position_type='LONG' if position.is_long else 'SHORT',
            entry_price=position.entry_price,
            exit_price=self.current_price,
            size=position.size,
            leverage=position.leverage,
            pnl=pnl
        )

        return pnl

    def advance_time(self) -> bool:
        """Advance to the next time step in the price data"""
        if self.current_index >= len(self.price_data) - 1:
            return False

        self.current_index += 1
        self.current_price = self.price_data['Close'].iloc[self.current_index]
        self.current_timestamp = self.price_data.index[self.current_index]

        # Update all accounts
        for account in self.accounts.values():
            account.update_account(self.current_price, self.current_timestamp)

        return True

    def run_simulation(self, days: int = 0) -> None:
        """Run the simulation for a specified number of days or until the end of data"""
        start_index = self.current_index
        end_index = len(self.price_data) - 1

        if days:
            # Calculate the index that corresponds to 'days' days from current
            target_date = self.current_timestamp + timedelta(days=days)
            future_indices = self.price_data.index.searchsorted(target_date)
            # Handle both scalar and array return types
            if isinstance(future_indices, (list, np.ndarray)):
                future_indices = future_indices[0]
            end_index = min(future_indices, end_index)

        steps = end_index - start_index
        print(f"Running simulation for {steps} time steps...")

        for _ in range(steps):
            if not self.advance_time():
                break

    def get_price_history(self, lookback_days: int = 30) -> pd.DataFrame:
        """Get price history for the last N days from current time"""
        start_idx = max(0, self.current_index - lookback_days)
        return self.price_data.iloc[start_idx:self.current_index + 1]

    def open_position(self, account_id: str, is_long: bool, size: float, leverage: float) -> Optional[FuturesPosition]:
        """Open a new position at current market price"""
        if account_id not in self.accounts:
            print(f"Account {account_id} not found")
            return None

        account = self.accounts[account_id]

        # Calculate liquidation price
        entry_price = self.current_price
        liquidation_price = entry_price * (1 - (1 / leverage) if is_long else 1 + (1 / leverage))

        # Create and add the position
        position = account.open_position(
            price=entry_price,
            size=size,
            leverage=leverage,
            is_long=is_long,
            liquidation_price=liquidation_price,
            timestamp=self.current_timestamp
        )

        if position:
            self._record_order(
                account_id=account_id,
                order_type='OPEN',
                position_type='LONG' if is_long else 'SHORT',
                price=entry_price,
                size=size,
                leverage=leverage
            )

        return position