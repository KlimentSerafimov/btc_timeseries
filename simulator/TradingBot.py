from FuturesExchange import FuturesExchange
from FuturesPosition import FuturesPosition
from strategies import create_strategy

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import os
from typing import Any, Dict, List, Optional


class TradingBot:
    """A simple trading bot for the futures exchange"""

    def __init__(self,
                exchange: FuturesExchange,
                account_id: str,
                strategy: str = 'moving_average_crossover',
                params: Dict[str, Any] = {},
                verbose: bool = False):
        """
        Initialize the trading bot

        Parameters:
        - exchange: The futures exchange to trade on
        - account_id: The account ID to use for trading
        - strategy: Trading strategy to use
        - params: Strategy parameters
        - verbose: Whether to print trading activity to console
        """
        self.exchange = exchange
        self.account_id = account_id
        self.strategy_name = strategy
        self.verbose = verbose

        # Create strategy instance
        self.strategy = create_strategy(strategy, params)

        # Trading state
        self.active_position: Optional[FuturesPosition] = None
        self.trade_signals: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []

    def get_trading_signal(self, price_data: pd.DataFrame) -> int:
        """
        Get the current trading signal based on the selected strategy
        
        Parameters:
        - price_data: DataFrame with price data up to current time
        
        Returns:
        - signal: 1 for buy, -1 for sell, 0 for neutral
        """
        return self.strategy.calculate_signal(price_data)

    def calculate_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators based on the strategy"""
        return self.strategy.calculate_indicators(price_data)

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
        if pnl_pct >= self.strategy.params.get('take_profit_pct', float('inf')):
            if self.verbose:
                print(f"Take profit triggered at {pnl_pct:.2%}")
            return True

        if pnl_pct <= -self.strategy.params.get('stop_loss_pct', float('inf')):
            if self.verbose:
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
        """Execute a trade based on the signal with improved risk management"""
        if self.account_id not in self.exchange.accounts:
            if self.verbose:
                print(f"Account {self.account_id} not found")
            return

        # Get account and check if it has sufficient funds
        account = self.exchange.accounts[self.account_id]
        if account.balance <= 0:
            # Cannot trade with zero or negative balance
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

                if self.verbose:
                    print(f"CLOSED {'LONG' if self.active_position.is_long else 'SHORT'} position at ${current_price:.2f} | "
                          f"PnL: ${pnl:.2f} | Time: {current_time}")

                self.active_position = None

        # Open new position if we don't have one and signal is not neutral
        if self.active_position is None and signal != 0:
            is_long = signal == 1
            
            # Use dynamic position sizing based on volatility and market regime
            base_size = self.strategy.params['position_size']
            size = self.strategy.get_dynamic_position_size(base_size)
            
            # Adjust position size based on market regime
            if (is_long and self.strategy.market_regime == 'bear') or (not is_long and self.strategy.market_regime == 'bull'):
                # Reduce position size when trading against the market regime
                size *= 0.7
            
            leverage = self.strategy.params['leverage']

            # Check if account has sufficient margin for this trade
            required_margin = size * current_price / leverage
            if account.available_balance < required_margin:
                # Adjust position size based on available balance
                adjusted_size = (account.available_balance * leverage) / current_price
                if adjusted_size < size * 0.1:  # If less than 10% of intended size, don't trade
                    return
                size = adjusted_size

            self.active_position = self.exchange.open_position(
                account_id=self.account_id,
                is_long=is_long,
                size=size,
                leverage=leverage
            )

            if self.active_position and self.verbose:
                print(f"OPENED {'LONG' if is_long else 'SHORT'} position at ${current_price:.2f} | "
                      f"Size: {size} BTC | Leverage: {leverage}x | Time: {current_time} | Regime: {self.strategy.market_regime}")

    def run(self, lookback_days: int = 200) -> None:
        """Run the trading bot for one time step"""
        # Get price history and latest signal
        price_history = self.exchange.get_price_history(lookback_days)
        latest_signal = self.get_trading_signal(price_history)
        
        # Execute trade and check for TP/SL
        self.execute_trade(latest_signal)
        
        # Check and close position if TP/SL is triggered
        if self.active_position and self.check_take_profit_stop_loss():
            self.exchange.close_position(self.account_id, self.active_position)
            self._record_trade(is_close=True)
            
            current_price = self.exchange.get_current_price()
            current_time = self.exchange.get_current_timestamp()
            
            if self.verbose:
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
        if performance_df.empty:
            print("No performance data to plot.")
            return

        performance_df.set_index('timestamp', inplace=True)
        trades_df = pd.DataFrame(self.trade_history)

        # Set plot style and colors
        plt.style.use('dark_background')
        colors = {
            'price': '#00a8ff', 'balance': '#00ff7f', 'buy': '#00ff7f', 
            'sell': '#ff3b30', 'title': '#e0e0e0', 'grid': '#555555',
            'entry': '#ffcc00', 'exit': '#ff9500', 'neutral': '#999999'
        }

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                      gridspec_kw={'height_ratios': [2, 1]},
                                      facecolor='#1e1e1e')
        ax1.set_facecolor('#2d2d2d')
        ax2.set_facecolor('#2d2d2d')

        # Plot price and trades
        ax1.plot(performance_df.index, performance_df['price'], 
                label='BTC Price', color=colors['price'], linewidth=2)
        
        # Plot trade markers
        if not trades_df.empty:
            try:
                # Plot long entries/exits
                long_entries = trades_df[trades_df['type'] == 'LONG']
                if not long_entries.empty:
                    ax1.scatter(long_entries['entry_time'], long_entries['entry_price'],
                              marker='^', s=120, color=colors['buy'], label='Long Entry')
                    ax1.scatter(long_entries['exit_time'], long_entries['exit_price'],
                              marker='o', s=100, color=colors['exit'], label='Long Exit')

                # Plot short entries/exits
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
        
        for spine in ax1.spines.values():
            spine.set_color(colors['grid'])

        # Plot account balance
        ax2.plot(performance_df.index, performance_df['balance'],
                label='Account Balance', color=colors['balance'], linewidth=2)
        
        # Plot position status as background
        if 'position_status' in performance_df.columns:
            try:
                # Create arrays for each position type
                long_mask = performance_df['position_status'] == 'LONG'
                short_mask = performance_df['position_status'] == 'SHORT'
                neutral_mask = performance_df['position_status'] == 'NEUTRAL'

                # Plot colored background for position status
                dates = performance_df.index
                valid_dates = pd.notna(dates)
                
                if valid_dates.all() and len(dates) > 1:
                    for i in range(len(dates)-1):
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
        ax2.set_ylabel('Balance (USD)', fontsize=12, color=colors['title'])
        ax2.set_xlabel('Date', fontsize=12, color=colors['title'])
        ax2.grid(True, alpha=0.2, color=colors['grid'])
        ax2.legend(facecolor='#2d2d2d', edgecolor=colors['grid'], labelcolor=colors['title'])
        ax2.tick_params(colors=colors['title'])
        
        for spine in ax2.spines.values():
            spine.set_color(colors['grid'])

        # Format date axis
        try:
            valid_dates = pd.notna(performance_df.index)
            if valid_dates.any():
                first_date = performance_df.index[valid_dates].min()
                last_date = performance_df.index[valid_dates].max()

                if pd.notna(first_date) and pd.notna(last_date):
                    for ax in [ax1, ax2]:
                        ax.set_xlim([first_date, last_date])
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color=colors['title'])
        except Exception as e:
            print(f"Warning: Could not set date limits: {e}")

        plt.tight_layout()
        plt.savefig(save_path, facecolor='#1e1e1e')
        plt.close()
        plt.style.use('default')  # Reset to default style
        print(f"Performance plot saved to {save_path}")