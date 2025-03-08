import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Any

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
        
        if self.is_long:
            self.pnl = position_value * (current_price/self.entry_price - 1)
        else:
            self.pnl = position_value * (1 - current_price/self.entry_price)
        
        # Apply leverage
        self.pnl = self.pnl * self.leverage
        
        return self.pnl
    
    def close_position(self, exit_price: float, timestamp: datetime) -> float:
        """Close the position and realize PnL"""
        self.exit_price = exit_price
        self.exit_timestamp = timestamp
        self.calculate_pnl(exit_price)
        return self.pnl
    
    def is_liquidated_at_price(self, price: float) -> bool:
        """Check if position would be liquidated at given price"""
        if self.is_long and price <= self.liquidation_price:
            return True
        elif not self.is_long and price >= self.liquidation_price:
            return True
        return False
    
    def __str__(self) -> str:
        position_type = "LONG" if self.is_long else "SHORT"
        status = "OPEN" if self.exit_price is None else "CLOSED"
        if self.is_liquidated:
            status = "LIQUIDATED"
            
        return (f"{position_type} {self.size:.6f} BTC @ {self.entry_price:.2f} USD "
                f"(Leverage: {self.leverage}x, {status})")


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
        total_unrealized_pnl = 0.0
        for position in self.active_positions:
            pnl = position.calculate_pnl(current_price)
            total_unrealized_pnl += pnl
        
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
            # Record the executed order
            self.executed_orders.append({
                'account_id': account_id,
                'type': 'MARKET',
                'action': 'OPEN',
                'position_type': 'LONG' if is_long else 'SHORT',
                'price': self.current_price,
                'size': size,
                'leverage': leverage,
                'timestamp': self.current_timestamp
            })
        
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
        
        # Record the executed order
        self.executed_orders.append({
            'account_id': account_id,
            'type': 'MARKET',
            'action': 'CLOSE',
            'position_type': 'LONG' if position.is_long else 'SHORT',
            'entry_price': position.entry_price,
            'exit_price': self.current_price,
            'size': position.size,
            'leverage': position.leverage,
            'pnl': pnl,
            'timestamp': self.current_timestamp
        })
        
        return pnl
    
    def advance_time(self) -> bool:
        """Advance to the next time step in the price data"""
        if self.current_index >= len(self.price_data) - 1:
            return False
        
        self.current_index += 1
        self.current_price = self.price_data['Close'].iloc[self.current_index]
        self.current_timestamp = self.price_data.index[self.current_index]
        
        # Update all accounts
        for account_id, account in self.accounts.items():
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
        if self.current_index < lookback_days:
            start_idx = 0
        else:
            start_idx = self.current_index - lookback_days
        
        return self.price_data.iloc[start_idx:self.current_index + 1]
    
    def open_position(self, account_id: str, is_long: bool, size: float, leverage: float) -> Optional[FuturesPosition]:
        """Open a new position at current market price"""
        if account_id not in self.accounts:
            print(f"Account {account_id} not found")
            return None
        
        account = self.accounts[account_id]
        
        # Calculate liquidation price
        entry_price = self.current_price
        if is_long:
            liquidation_price = entry_price * (1 - (1 / leverage))
        else:
            liquidation_price = entry_price * (1 + (1 / leverage))
        
        # Create and add the position
        position = account.open_position(
            price=entry_price,
            size=size,
            leverage=leverage,
            is_long=is_long,
            liquidation_price=liquidation_price,
            timestamp=self.current_timestamp
        )
        
        # Record the executed order
        self.executed_orders.append({
            'account_id': account_id,
            'type': 'MARKET',
            'action': 'OPEN',
            'position_type': 'LONG' if is_long else 'SHORT',
            'price': entry_price,
            'size': size,
            'leverage': leverage,
            'timestamp': self.current_timestamp
        })
        
        return position


class TradingBot:
    """A simple trading bot for the futures exchange"""
    
    def __init__(self, 
                exchange: FuturesExchange, 
                account_id: str,
                strategy: str = 'moving_average_crossover',
                params: Dict[str, Any] = {}):
        """
        Initialize the trading bot
        
        Parameters:
        - exchange: The futures exchange to trade on
        - account_id: The account ID to use for trading
        - strategy: Trading strategy to use
        - params: Strategy parameters
        """
        self.exchange = exchange
        self.account_id = account_id
        self.strategy = strategy
        
        # Default parameters
        self.default_params = {
            'moving_average_crossover': {
                'short_window': 5,
                'long_window': 20,
                'position_size': 0.1,  # BTC
                'leverage': 5,
                'take_profit_pct': 0.1,  # 10%
                'stop_loss_pct': 0.05    # 5%
            },
            'bollinger_bands': {
                'window': 20,
                'num_std': 2,
                'position_size': 0.1,
                'leverage': 3
            },
            'rsi': {
                'window': 14,
                'overbought': 70,
                'oversold': 30,
                'position_size': 0.1,
                'leverage': 3
            }
        }
        
        # Set parameters
        if not params:
            self.params = self.default_params[strategy].copy()
        else:
            self.params = {**self.default_params[strategy], **params}
        
        # Trading state
        self.active_position: Optional[FuturesPosition] = None
        self.trade_signals: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []  # New: Track detailed trade history
    
    def calculate_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators based on the strategy"""
        df = price_data.copy()
        
        if self.strategy == 'moving_average_crossover':
            # Convert window sizes to integers
            short_window = int(self.params['short_window'])
            long_window = int(self.params['long_window'])
            
            df['MA_short'] = df['Close'].rolling(window=short_window).mean()
            df['MA_long'] = df['Close'].rolling(window=long_window).mean()
            
            # Calculate crossover signal
            df['Signal'] = 0
            df.loc[df['MA_short'] > df['MA_long'], 'Signal'] = 1  # Buy signal
            df.loc[df['MA_short'] < df['MA_long'], 'Signal'] = -1  # Sell signal
            
        elif self.strategy == 'bollinger_bands':
            # Convert window size to integer
            window = int(self.params['window'])
            num_std = self.params['num_std']
            
            df['MA'] = df['Close'].rolling(window=window).mean()
            df['STD'] = df['Close'].rolling(window=window).std()
            df['Upper_Band'] = df['MA'] + (df['STD'] * num_std)
            df['Lower_Band'] = df['MA'] - (df['STD'] * num_std)
            
            # Calculate signals
            df['Signal'] = 0
            df.loc[df['Close'] < df['Lower_Band'], 'Signal'] = 1  # Buy signal when price below lower band
            df.loc[df['Close'] > df['Upper_Band'], 'Signal'] = -1  # Sell signal when price above upper band
            
        elif self.strategy == 'rsi':
            # Convert window size to integer
            window = int(self.params['window'])
            
            # Convert to numpy array for calculations
            close_values = pd.to_numeric(df['Close']).to_numpy(dtype=float)
            delta_values = np.diff(close_values, prepend=close_values[0])
            
            # Calculate gains and losses using numpy
            gains_array = np.where(delta_values > 0, delta_values, 0)
            losses_array = np.where(delta_values < 0, -delta_values, 0)
            
            # Convert back to Series for rolling calculations
            gains = pd.Series(gains_array, index=df.index)
            losses = pd.Series(losses_array, index=df.index)
            
            # Calculate rolling means
            avg_gain = gains.rolling(window=window).mean()
            avg_loss = losses.rolling(window=window).mean()
            
            # Calculate RSI
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate signals
            df['Signal'] = 0
            df.loc[df['RSI'] < self.params['oversold'], 'Signal'] = 1
            df.loc[df['RSI'] > self.params['overbought'], 'Signal'] = -1
        
        return df
    
    def check_take_profit_stop_loss(self) -> bool:
        """Check if take profit or stop loss conditions are met for the active position"""
        if not self.active_position:
            return False
        
        current_price = self.exchange.get_current_price()
        entry_price = self.active_position.entry_price
        
        # Calculate unrealized PnL percentage
        if self.active_position.is_long:
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check take profit
        if pnl_pct >= self.params.get('take_profit_pct', float('inf')):
            print(f"Take profit triggered at {pnl_pct:.2%}")
            return True
        
        # Check stop loss (negative value)
        if pnl_pct <= -self.params.get('stop_loss_pct', float('inf')):
            print(f"Stop loss triggered at {pnl_pct:.2%}")
            return True
        
        return False
    
    def execute_trade(self, signal: int) -> None:
        """Execute a trade based on the signal"""
        # We don't need to store the account variable if we're not using it
        # Just check if the account exists
        if self.account_id not in self.exchange.accounts:
            print(f"Account {self.account_id} not found")
            return
        
        current_price = self.exchange.get_current_price()
        current_time = self.exchange.get_current_timestamp()
        
        # If we have an active position
        if self.active_position:
            # Close position if signal is opposite to current position or neutral signal
            if (self.active_position.is_long and signal <= 0) or \
               (not self.active_position.is_long and signal >= 0):
                pnl = self.exchange.close_position(self.account_id, self.active_position)
                
                # Record the trade in our history
                self.trade_history.append({
                    'type': 'LONG' if self.active_position.is_long else 'SHORT',
                    'entry_price': self.active_position.entry_price,
                    'entry_time': self.active_position.timestamp,
                    'exit_price': current_price,
                    'exit_time': current_time,
                    'size': self.active_position.size,
                    'leverage': self.active_position.leverage,
                    'pnl': pnl,
                    'pnl_percent': (pnl / (self.active_position.size * self.active_position.entry_price / self.active_position.leverage)) * 100
                })
                
                print(f"CLOSED {'LONG' if self.active_position.is_long else 'SHORT'} position at ${current_price:.2f} | "
                      f"PnL: ${pnl:.2f} | Time: {current_time}")
                
                self.active_position = None
        
        # Open new position if we don't have one and signal is not neutral
        if self.active_position is None and signal != 0:
            is_long = signal == 1
            size = self.params['position_size']
            leverage = self.params['leverage']
            
            self.active_position = self.exchange.open_position(
                account_id=self.account_id,
                is_long=is_long,
                size=size,
                leverage=leverage
            )
            
            if self.active_position:
                print(f"OPENED {'LONG' if is_long else 'SHORT'} position at ${current_price:.2f} | "
                      f"Size: {size} BTC | Leverage: {leverage}x | Time: {current_time}")
    
    def run(self, lookback_days: int = 30) -> None:
        """Run the trading bot for one time step"""
        # Get price history
        price_history = self.exchange.get_price_history(lookback_days)
        
        # Calculate indicators
        indicators = self.calculate_indicators(price_history)
        
        # Get the latest signal
        if not indicators.empty and not indicators['Signal'].isna().all():
            latest_signal = indicators['Signal'].iloc[-1]
            
            # Record the signal
            self.trade_signals.append({
                'timestamp': self.exchange.get_current_timestamp(),
                'price': self.exchange.get_current_price(),
                'signal': latest_signal
            })
            
            # Execute the trade
            self.execute_trade(latest_signal)
        
        # Check for take profit / stop loss if we have an active position
        if self.active_position and self.check_take_profit_stop_loss():
            self.exchange.close_position(self.account_id, self.active_position)
            
            current_price = self.exchange.get_current_price()
            current_time = self.exchange.get_current_timestamp()
            
            # Calculate PnL
            if self.active_position.is_long:
                pnl = self.active_position.size * (current_price - self.active_position.entry_price) * self.active_position.leverage
            else:
                pnl = self.active_position.size * (self.active_position.entry_price - current_price) * self.active_position.leverage
            
            # Record the trade in our history
            self.trade_history.append({
                'type': 'LONG' if self.active_position.is_long else 'SHORT',
                'entry_price': self.active_position.entry_price,
                'entry_time': self.active_position.timestamp,
                'exit_price': current_price,
                'exit_time': current_time,
                'size': self.active_position.size,
                'leverage': self.active_position.leverage,
                'pnl': pnl,
                'pnl_percent': (pnl / (self.active_position.size * self.active_position.entry_price / self.active_position.leverage)) * 100,
                'exit_reason': 'Take Profit/Stop Loss'
            })
            
            print(f"TP/SL CLOSED {'LONG' if self.active_position.is_long else 'SHORT'} position at ${current_price:.2f} | "
                  f"PnL: ${pnl:.2f} | Time: {current_time}")
            
            self.active_position = None
        
        # Record performance
        account = self.exchange.accounts[self.account_id]
        summary = account.get_account_summary()
        
        self.performance_history.append({
            'timestamp': self.exchange.get_current_timestamp(),
            'price': self.exchange.get_current_price(),
            'balance': summary['balance'],
            'pnl_percentage': summary['pnl_percentage'],
            'position_status': 'LONG' if self.active_position and self.active_position.is_long else 
                              'SHORT' if self.active_position and not self.active_position.is_long else 'NEUTRAL'
        })
    
    def print_trade_summary(self) -> None:
        """Print a summary of all trades"""
        if not self.trade_history:
            print("No trades executed.")
            return
        
        print("\n===== TRADE SUMMARY =====")
        print(f"{'TYPE':<6} {'ENTRY PRICE':<12} {'EXIT PRICE':<12} {'SIZE':<8} {'LEV':<5} {'PNL':<10} {'PNL %':<8}")
        print("-" * 70)
        
        total_pnl = 0
        winning_trades = 0
        
        for trade in self.trade_history:
            print(f"{trade['type']:<6} ${trade['entry_price']:<10.2f} ${trade['exit_price']:<10.2f} "
                  f"{trade['size']:<6.3f} {trade['leverage']:<3.0f}x ${trade['pnl']:<8.2f} {trade['pnl_percent']:<6.2f}%")
            
            total_pnl += trade['pnl']
            if trade['pnl'] > 0:
                winning_trades += 1
        
        print("-" * 70)
        win_rate = (winning_trades / len(self.trade_history)) * 100 if self.trade_history else 0
        print(f"Total Trades: {len(self.trade_history)} | Win Rate: {win_rate:.2f}% | Total PnL: ${total_pnl:.2f}")
        print("========================\n")
    
    def plot_performance(self, save_path: str = 'figures/bot_performance.png') -> None:
        """Plot the bot's performance"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert performance history to DataFrame
        performance_df = pd.DataFrame(self.performance_history)
        
        # Check if we have any performance data
        if performance_df.empty:
            print("No performance data to plot.")
            return
        
        performance_df.set_index('timestamp', inplace=True)
        
        # Convert trade signals to DataFrame
        signals_df = pd.DataFrame(self.trade_signals)
        if not signals_df.empty:
            signals_df.set_index('timestamp', inplace=True)
        
        # Convert trade history to DataFrame for plotting entries and exits
        trades_df = pd.DataFrame(self.trade_history)
        
        # Set dark style for plots
        plt.style.use('dark_background')
        
        # Define colors
        colors = {
            'price': '#00a8ff',      # Blue
            'balance': '#00ff7f',    # Green
            'buy': '#00ff7f',        # Green
            'sell': '#ff3b30',       # Red
            'title': '#e0e0e0',      # Light gray
            'grid': '#555555',       # Medium gray
            'entry': '#ffcc00',      # Yellow
            'exit': '#ff9500',       # Orange
            'neutral': '#999999'     # Gray
        }
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                      gridspec_kw={'height_ratios': [2, 1]},
                                      facecolor='#1e1e1e')
        
        # Set background color
        ax1.set_facecolor('#2d2d2d')
        ax2.set_facecolor('#2d2d2d')
        
        # Plot price
        ax1.plot(performance_df.index, performance_df['price'], 
                label='BTC Price', color=colors['price'], linewidth=2)
        
        # Plot trade entries and exits if we have trade history
        if not trades_df.empty:
            try:
                # Plot long entries
                long_entries = trades_df[trades_df['type'] == 'LONG']
                if not long_entries.empty:
                    ax1.scatter(long_entries['entry_time'], long_entries['entry_price'], 
                              marker='^', s=120, color=colors['buy'], label='Long Entry')
                    ax1.scatter(long_entries['exit_time'], long_entries['exit_price'], 
                              marker='o', s=100, color=colors['exit'], label='Long Exit')
                
                # Plot short entries
                short_entries = trades_df[trades_df['type'] == 'SHORT']
                if not short_entries.empty:
                    ax1.scatter(short_entries['entry_time'], short_entries['entry_price'], 
                              marker='v', s=120, color=colors['sell'], label='Short Entry')
                    ax1.scatter(short_entries['exit_time'], short_entries['exit_price'], 
                              marker='o', s=100, color=colors['exit'], label='Short Exit')
            except Exception as e:
                print(f"Warning: Could not plot trade entries/exits: {e}")
        
        # Style price subplot
        ax1.set_title('Bitcoin Price and Trading Activity', fontsize=16, color=colors['title'])
        ax1.set_ylabel('Price (USD)', fontsize=12, color=colors['title'])
        ax1.grid(True, alpha=0.2, color=colors['grid'])
        ax1.legend(facecolor='#2d2d2d', edgecolor=colors['grid'], labelcolor=colors['title'])
        ax1.tick_params(colors=colors['title'])
        
        # Plot account balance
        ax2.plot(performance_df.index, performance_df['balance'], 
                label='Account Balance', color=colors['balance'], linewidth=2)
        
        # Plot position status as background color
        if 'position_status' in performance_df.columns:
            try:
                # Create arrays for each position type
                long_mask = performance_df['position_status'] == 'LONG'
                short_mask = performance_df['position_status'] == 'SHORT'
                neutral_mask = performance_df['position_status'] == 'NEUTRAL'
                
                # Plot colored background for position status
                dates = performance_df.index
                
                # Make sure we have valid dates
                valid_dates = pd.notna(dates)
                if valid_dates.all() and len(dates) > 1:
                    for i in range(len(dates)-1):
                        # Skip any NaT values
                        if pd.isna(dates[i]) or pd.isna(dates[i+1]):
                            continue
                            
                        if long_mask.iloc[i]:
                            ax2.axvspan(dates[i], dates[i+1], alpha=0.2, color=colors['buy'])
                        elif short_mask.iloc[i]:
                            ax2.axvspan(dates[i], dates[i+1], alpha=0.2, color=colors['sell'])
                        elif neutral_mask.iloc[i]:
                            ax2.axvspan(dates[i], dates[i+1], alpha=0.1, color=colors['neutral'])
            except Exception as e:
                print(f"Warning: Could not plot position status background: {e}")
        
        # Style balance subplot
        ax2.set_title('Account Balance and Position Status', fontsize=16, color=colors['title'])
        ax2.set_xlabel('Date', fontsize=12, color=colors['title'])
        ax2.set_ylabel('Balance (USD)', fontsize=12, color=colors['title'])
        ax2.grid(True, alpha=0.2, color=colors['grid'])
        ax2.legend(facecolor='#2d2d2d', edgecolor=colors['grid'], labelcolor=colors['title'])
        ax2.tick_params(colors=colors['title'])
        
        # Style spines for both subplots
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_color(colors['grid'])
            
            # Set date ticks safely
            if not performance_df.empty and len(performance_df.index) > 1:
                try:
                    # Make sure we have valid dates for the x-axis limits
                    valid_dates = pd.notna(performance_df.index)
                    if valid_dates.any():
                        first_date = performance_df.index[valid_dates].min()
                        last_date = performance_df.index[valid_dates].max()
                        
                        if pd.notna(first_date) and pd.notna(last_date):
                            ax.set_xlim([first_date, last_date])
                            
                            # Format x-axis dates
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color=colors['title'])
                except Exception as e:
                    print(f"Warning: Could not set date limits: {e}")
        
        plt.tight_layout()
        plt.savefig(save_path, facecolor='#1e1e1e')
        plt.close()
        plt.style.use('default')  # Reset to default style
        
        print(f"Performance plot saved to {save_path}")


def run_simulation():
    """Run a complete simulation with the exchange and trading bot"""
    # Load Bitcoin price data
    try:
        # Use relative imports instead of package imports
        from btc_analysis import load_data
        
        btc_data = load_data()
    except FileNotFoundError:
        print("Price data file not found. Downloading data...")
        from btc_data_downloader import download_btc_data, save_data
        btc_data = download_btc_data(start_date='2020-01-01')
        save_data(btc_data)
    
    # Initialize the exchange with the price data
    exchange = FuturesExchange(btc_data)
    
    # Register a trading account
    account_id = "bot_account"
    exchange.register_account(account_id, initial_balance=10000.0)
    
    # Create a trading bot
    bot_params = {
        'short_window': 10,
        'long_window': 30,
        'position_size': 0.1,
        'leverage': 5,
        'take_profit_pct': 0.15,
        'stop_loss_pct': 0.07
    }
    
    bot = TradingBot(
        exchange=exchange,
        account_id=account_id,
        strategy='moving_average_crossover',
        params=bot_params
    )
    
    # Run the simulation
    print("Starting simulation...")
    simulation_days = 365  # Simulate one year of trading
    
    # Calculate how many steps we need to run
    total_steps = min(simulation_days, len(btc_data) - 1)
    
    for i in range(total_steps):
        # Run the bot
        bot.run()
        
        # Advance time
        exchange.advance_time()
        
        # Print progress every 10%
        if i % (total_steps // 10) == 0 or i == total_steps - 1:
            progress = (i + 1) / total_steps * 100
            account = exchange.accounts[account_id]
            summary = account.get_account_summary()
            print(f"Progress: {progress:.1f}% | Date: {exchange.get_current_timestamp().date()} | "
                  f"Balance: ${summary['balance']:.2f} | PnL: {summary['pnl_percentage']:.2f}%")
    
    # Print trade summary
    bot.print_trade_summary()
    
    # Plot final performance
    bot.plot_performance()
    
    # Print final results
    account = exchange.accounts[account_id]
    summary = account.get_account_summary()
    print("\nSimulation Complete!")
    print(f"Initial Balance: ${account.initial_balance:.2f}")
    print(f"Final Balance: ${summary['balance']:.2f}")
    print(f"Total PnL: ${summary['total_pnl']:.2f} ({summary['pnl_percentage']:.2f}%)")
    print(f"Total Trades: {len(account.closed_positions)}")
    
    # Calculate win rate
    if account.closed_positions:
        winning_trades = sum(1 for pos in account.closed_positions if pos.pnl > 0)
        win_rate = winning_trades / len(account.closed_positions) * 100
        print(f"Win Rate: {win_rate:.2f}%")
    
    return bot, exchange

if __name__ == "__main__":
    bot, exchange = run_simulation()