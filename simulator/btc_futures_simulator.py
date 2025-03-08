from typing import Tuple, Dict, Any
import os
import sys

# Add parent directory to path to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.FuturesExchange import FuturesExchange
from simulator.TradingBot import TradingBot

def run_simulation() -> Tuple[TradingBot, FuturesExchange]:
    """Run a complete simulation with the exchange and trading bot"""
    # Load Bitcoin price data
    try:
        from btc_analysis import load_data
        btc_data = load_data()
    except (ImportError, FileNotFoundError):
        print("Price data file not found or btc_analysis module not found. Downloading data...")
        from btc_data_downloader import download_btc_data, save_data
        btc_data = download_btc_data(start_date='2020-01-01')
        save_data(btc_data)
    
    # Initialize exchange and register account
    exchange = FuturesExchange(btc_data)
    account_id = "bot_account"
    exchange.register_account(account_id, initial_balance=10000.0)
    
    # Create trading bot with strategy parameters
    bot = TradingBot(
        exchange=exchange,
        account_id=account_id,
        strategy='moving_average_crossover',
        params={
            'short_window': 10,
            'long_window': 30,
            'position_size': 0.1,
            'leverage': 5,
            'take_profit_pct': 0.15,
            'stop_loss_pct': 0.07
        }
    )
    
    # Run the simulation
    print("Starting simulation...")
    simulation_days = 365  # Simulate one year of trading
    
    # Calculate how many steps we need to run
    total_steps = min(simulation_days, len(btc_data) - 1)
    
    for i in range(total_steps):
        # Important: First let the bot make decisions with current data,
        # then advance time to prevent look-ahead bias
        bot.run()
        exchange.advance_time()
        
        # Print progress every 10%
        if i % (total_steps // 10) == 0 or i == total_steps - 1:
            progress = (i + 1) / total_steps * 100
            account = exchange.accounts[account_id]
            summary = account.get_account_summary()
            
            # Get the current trading signal based on data up to current time
            price_history = exchange.get_price_history(lookback_days=30)
            current_signal = bot.get_trading_signal(price_history)
            signal_text = "BUY" if current_signal == 1 else "SELL" if current_signal == -1 else "NEUTRAL"
            
            print(f"Progress: {progress:.1f}% | Date: {exchange.get_current_timestamp().date()} | "
                  f"Balance: ${summary['balance']:.2f} | PnL: {summary['pnl_percentage']:.2f}% | Signal: {signal_text}")
    
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

def get_bot_prediction(bot: TradingBot, lookback_days: int = 30) -> Dict[str, Any]:
    """
    Get the bot's current prediction and analysis
    
    Parameters:
    - bot: The trading bot instance
    - lookback_days: Number of days to look back for analysis
    
    Returns:
    - Dictionary with prediction details
    """
    # Get price history for analysis (only up to current time)
    price_history = bot.exchange.get_price_history(lookback_days)
    
    # Get the current trading signal
    signal = bot.get_trading_signal(price_history)
    
    # Calculate indicators
    indicators = bot.calculate_indicators(price_history)
    
    # Get current account status
    account = bot.exchange.accounts[bot.account_id]
    summary = account.get_account_summary()
    
    # Create prediction result
    result = {
        'timestamp': bot.exchange.get_current_timestamp(),
        'current_price': bot.exchange.get_current_price(),
        'signal': signal,
        'signal_text': "BUY" if signal == 1 else "SELL" if signal == -1 else "NEUTRAL",
        'strategy': bot.strategy,
        'account_balance': summary['balance'],
        'pnl_percentage': summary['pnl_percentage'],
        'active_position': bool(bot.active_position),
        'position_type': 'LONG' if bot.active_position and bot.active_position.is_long else 
                         'SHORT' if bot.active_position and not bot.active_position.is_long else 'NONE',
        'indicators': {}
    }
    
    # Add strategy-specific indicators
    if bot.strategy == 'moving_average_crossover':
        result['indicators'] = {
            'short_ma': indicators['MA_short'].iloc[-1] if 'MA_short' in indicators else None,
            'long_ma': indicators['MA_long'].iloc[-1] if 'MA_long' in indicators else None
        }
    elif bot.strategy == 'bollinger_bands':
        result['indicators'] = {
            'upper_band': indicators['Upper_Band'].iloc[-1] if 'Upper_Band' in indicators else None,
            'lower_band': indicators['Lower_Band'].iloc[-1] if 'Lower_Band' in indicators else None,
            'middle_band': indicators['MA'].iloc[-1] if 'MA' in indicators else None
        }
    elif bot.strategy == 'rsi':
        result['indicators'] = {
            'rsi': indicators['RSI'].iloc[-1] if 'RSI' in indicators else None,
            'overbought': bot.params['overbought'],
            'oversold': bot.params['oversold']
        }
    
    return result

if __name__ == "__main__":
    bot, exchange = run_simulation()
    
    # Example of using the get_bot_prediction function
    prediction = get_bot_prediction(bot)
    print("\nCurrent Bot Prediction:")
    print(f"Strategy: {prediction['strategy']}")
    print(f"Signal: {prediction['signal_text']}")
    print(f"Current Price: ${prediction['current_price']:.2f}")
    
    # Print strategy-specific indicators
    print("\nIndicators:")
    for key, value in prediction['indicators'].items():
        if value is not None:
            print(f"  {key}: {value:.2f}")