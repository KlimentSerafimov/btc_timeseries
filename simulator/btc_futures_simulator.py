from typing import Tuple, Dict, Any, List
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.FuturesExchange import FuturesExchange
from simulator.TradingBot import TradingBot

def run_simulation(strategy: str = 'moving_average_crossover', params: Dict[str, Any] = None) -> Tuple[TradingBot, FuturesExchange]:
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
    account_id = f"{strategy}_account"
    exchange.register_account(account_id, initial_balance=10000.0)
    
    # Default parameters for each strategy
    default_params = {
        'moving_average_crossover': {
            'short_window': 10,
            'long_window': 30,
            'position_size': 0.1,
            'leverage': 5,
            'take_profit_pct': 0.15,
            'stop_loss_pct': 0.07
        },
        'bollinger_bands': {
            'window': 20,
            'num_std': 2,
            'position_size': 0.1,
            'leverage': 3,
            'take_profit_pct': 0.15,
            'stop_loss_pct': 0.07
        },
        'rsi': {
            'window': 14,
            'overbought': 70,
            'oversold': 30,
            'position_size': 0.1,
            'leverage': 3,
            'take_profit_pct': 0.15,
            'stop_loss_pct': 0.07
        }
    }
    
    # Use provided params or default ones
    strategy_params = params if params else default_params[strategy]
    
    # Create trading bot with strategy parameters
    bot = TradingBot(
        exchange=exchange,
        account_id=account_id,
        strategy=strategy,
        params=strategy_params
    )
    
    # Run the simulation
    print(f"\nStarting simulation for {strategy} strategy...")
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
    
    # Plot final performance with updated path
    os.makedirs('figures/strategies', exist_ok=True)
    bot.plot_performance(save_path=f'figures/strategies/{strategy}_performance.png')
    
    # Print final results
    account = exchange.accounts[account_id]
    summary = account.get_account_summary()
    print(f"\n{strategy.upper()} Strategy Simulation Complete!")
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

def run_all_strategies() -> Dict[str, Dict[str, Any]]:
    """Run simulations for all available strategies and compare results"""
    strategies = ['moving_average_crossover', 'bollinger_bands', 'rsi']
    results = {}
    
    for strategy in strategies:
        bot, exchange = run_simulation(strategy)
        
        # Store results
        account = exchange.accounts[f"{strategy}_account"]
        summary = account.get_account_summary()
        
        # Calculate win rate
        win_rate = 0
        if account.closed_positions:
            winning_trades = sum(1 for pos in account.closed_positions if pos.pnl > 0)
            win_rate = winning_trades / len(account.closed_positions) * 100
        
        results[strategy] = {
            'initial_balance': account.initial_balance,
            'final_balance': summary['balance'],
            'total_pnl': summary['total_pnl'],
            'pnl_percentage': summary['pnl_percentage'],
            'total_trades': len(account.closed_positions),
            'win_rate': win_rate,
            'performance_history': bot.performance_history
        }
    
    # Compare and display results
    compare_strategies(results)
    
    return results

def compare_strategies(results: Dict[str, Dict[str, Any]]) -> None:
    """Compare and visualize the performance of different strategies"""
    # Print comparison table
    print("\n===== STRATEGY COMPARISON =====")
    print(f"{'STRATEGY':<25} {'FINAL BALANCE':<15} {'PNL %':<10} {'TRADES':<10} {'WIN RATE':<10}")
    print("-" * 70)
    
    for strategy, data in results.items():
        strategy_name = strategy.replace('_', ' ').title()
        print(f"{strategy_name:<25} ${data['final_balance']:<13.2f} {data['pnl_percentage']:<8.2f}% {data['total_trades']:<8} {data['win_rate']:<8.2f}%")
    
    print("-" * 70)
    
    # Create comparison chart
    plt.figure(figsize=(14, 8))
    plt.style.use('dark_background')
    
    # Plot balance over time for each strategy
    for strategy, data in results.items():
        strategy_name = strategy.replace('_', ' ').title()
        
        # Convert performance history to DataFrame
        df = pd.DataFrame(data['performance_history'])
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            plt.plot(df.index, df['balance'], label=strategy_name, linewidth=2)
    
    plt.title('Strategy Performance Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Account Balance (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the comparison chart in the main figures directory
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/strategy_comparison.png', facecolor='#1e1e1e')
    plt.close()
    
    print("\nStrategy comparison chart saved to figures/strategy_comparison.png")
    print("Individual strategy charts saved to figures/strategies/ directory")

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
    # Run all strategies and compare results
    results = run_all_strategies()
    
    # Alternatively, you can run a single strategy
    # bot, exchange = run_simulation('moving_average_crossover')
    
    # Example of using the get_bot_prediction function
    # prediction = get_bot_prediction(bot)
    # print("\nCurrent Bot Prediction:")
    # print(f"Strategy: {prediction['strategy']}")
    # print(f"Signal: {prediction['signal_text']}")
    # print(f"Current Price: ${prediction['current_price']:.2f}")
    
    # Print strategy-specific indicators
    # print("\nIndicators:")
    # for key, value in prediction['indicators'].items():
    #     if value is not None:
    #         print(f"  {key}: {value:.2f}")