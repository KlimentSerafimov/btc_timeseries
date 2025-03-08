from FuturesExchange import FuturesExchange
from FuturesPosition import FuturesPosition


import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import os
from typing import Any, Dict, List, Optional


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
        self.params = {**self.default_params[strategy], **(params or {})}

        # Trading state
        self.active_position: Optional[FuturesPosition] = None
        self.trade_signals: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []

        # Add a dictionary to map strategy names to their functions
        self.strategy_functions = {
            'moving_average_crossover': self._moving_average_crossover_strategy,
            'bollinger_bands': self._bollinger_bands_strategy,
            'rsi': self._rsi_strategy
        }

    def _moving_average_crossover_strategy(self, price_data: pd.DataFrame) -> int:
        """
        Moving Average Crossover strategy implementation
        Returns: 1 for buy signal, -1 for sell signal, 0 for neutral
        """
        df = price_data.copy()
        
        # Convert window sizes to integers
        short_window = int(self.params['short_window'])
        long_window = int(self.params['long_window'])

        df['MA_short'] = df['Close'].rolling(window=short_window).mean()
        df['MA_long'] = df['Close'].rolling(window=long_window).mean()

        # Calculate crossover signal
        signal = 0
        if not df.empty and not df['MA_short'].isna().iloc[-1] and not df['MA_long'].isna().iloc[-1]:
            if df['MA_short'].iloc[-1] > df['MA_long'].iloc[-1]:
                signal = 1  # Buy signal
            elif df['MA_short'].iloc[-1] < df['MA_long'].iloc[-1]:
                signal = -1  # Sell signal
                
        return signal

    def _bollinger_bands_strategy(self, price_data: pd.DataFrame) -> int:
        """
        Bollinger Bands strategy implementation
        Returns: 1 for buy signal, -1 for sell signal, 0 for neutral
        """
        df = price_data.copy()
        
        # Convert window size to integer
        window = int(self.params['window'])
        num_std = self.params['num_std']

        df['MA'] = df['Close'].rolling(window=window).mean()
        df['STD'] = df['Close'].rolling(window=window).std()
        df['Upper_Band'] = df['MA'] + (df['STD'] * num_std)
        df['Lower_Band'] = df['MA'] - (df['STD'] * num_std)

        # Calculate signal
        signal = 0
        if not df.empty and not df['Close'].isna().iloc[-1] and not df['Upper_Band'].isna().iloc[-1] and not df['Lower_Band'].isna().iloc[-1]:
            if df['Close'].iloc[-1] < df['Lower_Band'].iloc[-1]:
                signal = 1  # Buy signal when price below lower band
            elif df['Close'].iloc[-1] > df['Upper_Band'].iloc[-1]:
                signal = -1  # Sell signal when price above upper band
                
        return signal

    def _rsi_strategy(self, price_data: pd.DataFrame) -> int:
        """
        RSI strategy implementation
        Returns: 1 for buy signal, -1 for sell signal, 0 for neutral
        """
        df = price_data.copy()
        
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

        # Calculate signal
        signal = 0
        if not df.empty and not df['RSI'].isna().iloc[-1]:
            if df['RSI'].iloc[-1] < self.params['oversold']:
                signal = 1  # Buy signal
            elif df['RSI'].iloc[-1] > self.params['overbought']:
                signal = -1  # Sell signal
                
        return signal

    def get_trading_signal(self, price_data: pd.DataFrame) -> int:
        """
        Get the current trading signal based on the selected strategy
        
        Parameters:
        - price_data: DataFrame with price data up to current time
        
        Returns:
        - signal: 1 for buy, -1 for sell, 0 for neutral
        """
        # Check if we have a strategy function for the selected strategy
        if self.strategy not in self.strategy_functions:
            print(f"Strategy '{self.strategy}' not implemented")
            return 0
        
        # Call the appropriate strategy function
        return self.strategy_functions[self.strategy](price_data)


    def check_take_profit_stop_loss(self) -> bool:
        """Check if take profit or stop loss conditions are met for the active position"""
        if not self.active_position:
            return False

        current_price = self.exchange.get_current_price()
        entry_price = self.active_position.entry_price

        # Calculate unrealized PnL percentage
        pnl_pct = ((current_price - entry_price) / entry_price) if self.active_position.is_long else \
                  ((entry_price - current_price) / entry_price)

        # Check take profit or stop loss
        if pnl_pct >= self.params.get('take_profit_pct', float('inf')):
            print(f"Take profit triggered at {pnl_pct:.2%}")
            return True

        if pnl_pct <= -self.params.get('stop_loss_pct', float('inf')):
            print(f"Stop loss triggered at {pnl_pct:.2%}")
            return True

        return False

    def _record_trade(self, is_close: bool) -> None:
        """Record a trade in the trade history"""
        if not self.active_position:
            return

        current_price = self.exchange.get_current_price()
        current_time = self.exchange.get_current_timestamp()

        # Calculate PnL for recording purposes
        if is_close:
            if self.active_position.is_long:
                pnl = self.active_position.size * (current_price - self.active_position.entry_price) * self.active_position.leverage
            else:
                pnl = self.active_position.size * (self.active_position.entry_price - current_price) * self.active_position.leverage

            # Calculate PnL percentage
            margin = self.active_position.size * self.active_position.entry_price / self.active_position.leverage
            pnl_percent = (pnl / margin) * 100

            trade_record = {
                'type': 'LONG' if self.active_position.is_long else 'SHORT',
                'entry_price': self.active_position.entry_price,
                'entry_time': self.active_position.timestamp,
                'exit_price': current_price,
                'exit_time': current_time,
                'size': self.active_position.size,
                'leverage': self.active_position.leverage,
                'pnl': pnl,
                'pnl_percent': pnl_percent
            }

            self.trade_history.append(trade_record)

    def execute_trade(self, signal: int) -> None:
        """Execute a trade based on the signal"""
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

                # Record the trade
                self._record_trade(is_close=True)

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
        # Get price history up to current time
        price_history = self.exchange.get_price_history(lookback_days)
        
        # Get the latest signal using the new method
        latest_signal = self.get_trading_signal(price_history)
        
        # Execute the trade
        self.execute_trade(latest_signal)
        
        # Check for take profit / stop loss if we have an active position
        if self.active_position and self.check_take_profit_stop_loss():
            self.exchange.close_position(self.account_id, self.active_position)
            
            # Record the trade
            self._record_trade(is_close=True)
            
            current_price = self.exchange.get_current_price()
            current_time = self.exchange.get_current_timestamp()
            
            print(f"TP/SL CLOSED {'LONG' if self.active_position.is_long else 'SHORT'} position at ${current_price:.2f} | "
                  f"Time: {current_time}")
            
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

    def calculate_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators based on the strategy"""
        df = price_data.copy()

        if self.strategy == 'moving_average_crossover':
            # Convert window sizes to integers
            short_window = int(self.params['short_window'])
            long_window = int(self.params['long_window'])

            df['MA_short'] = df['Close'].rolling(window=short_window).mean()
            df['MA_long'] = df['Close'].rolling(window=long_window).mean()
            
            # Calculate signal based on crossover
            df['Signal'] = 0
            df.loc[df['MA_short'] > df['MA_long'], 'Signal'] = 1
            df.loc[df['MA_short'] < df['MA_long'], 'Signal'] = -1

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
            df.loc[df['Close'] < df['Lower_Band'], 'Signal'] = 1
            df.loc[df['Close'] > df['Upper_Band'], 'Signal'] = -1

        elif self.strategy == 'rsi':
            # Convert window size to integer
            window = int(self.params['window'])

            # Calculate RSI
            close_values = pd.to_numeric(df['Close']).to_numpy(dtype=float)
            delta_values = np.diff(close_values, prepend=close_values[0])
            
            gains_array = np.where(delta_values > 0, delta_values, 0)
            losses_array = np.where(delta_values < 0, -delta_values, 0)
            
            gains = pd.Series(gains_array, index=df.index)
            losses = pd.Series(losses_array, index=df.index)
            
            avg_gain = gains.rolling(window=window).mean()
            avg_loss = losses.rolling(window=window).mean()
            
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate signals
            df['Signal'] = 0
            df.loc[df['RSI'] < self.params['oversold'], 'Signal'] = 1
            df.loc[df['RSI'] > self.params['overbought'], 'Signal'] = -1

        return df