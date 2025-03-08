from typing import Tuple, Dict, Any
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
    
    # Create trading bot with strategy parameters
    # The strategy classes will handle their own default parameters
    bot = TradingBot(
        exchange=exchange,
        account_id=account_id,
        strategy=strategy,
        params=params or {},  # Use empty dict if params is None
        verbose=False  # Set verbose to False by default
    )
    
    # Run the simulation
    print(f"\nStarting simulation for {strategy} strategy...")
    simulation_days = 365*8  # Simulate one year of trading
    
    # Get start date for directory naming - safely handle potential NaT value
    try:
        start_date = exchange.get_current_timestamp().strftime('%Y-%m-%d')
    except (ValueError, AttributeError):
        # If timestamp is NaT, use a safe fallback approach
        try:
            # Try to find the first non-NaT timestamp in the index
            for idx in btc_data.index:
                if pd.notna(idx):
                    start_date = idx.strftime('%Y-%m-%d')
                    break
            else:
                # If all timestamps are NaT, use a default value
                start_date = "unknown_start"
        except Exception as e:
            # Fallback if any other error occurs
            start_date = "unknown_start"
    
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
            
            # Check if account is bankrupt and stop simulation if needed
            if summary['balance'] <= 0:
                print(f"Account bankrupt! Stopping simulation at {progress:.1f}%")
                break
    
    # Print trade summary
    bot.print_trade_summary()
    
    # Print final results
    account = exchange.accounts[account_id]
    summary = account.get_account_summary()
    print(f"\n{strategy.upper()} Strategy Simulation Complete!")
    print(f"Initial Balance: ${account.initial_balance:.2f}")
    print(f"Final Balance: ${summary['balance']:.2f}")
    print(f"Total PnL: ${summary['total_pnl']:.2f} ({summary['pnl_percentage']:.2f}%)")
    print(f"Total Trades: {len(account.closed_positions)}")
    
    # Calculate win rate
    win_rate = 0
    if account.closed_positions:
        winning_trades = sum(1 for pos in account.closed_positions if pos.pnl > 0)
        win_rate = winning_trades / len(account.closed_positions) * 100
        print(f"Win Rate: {win_rate:.2f}%")
    
    # Create a more organized directory structure for results
    # Main results directory with timestamp to avoid overwriting
    import datetime
    
    # Check if this is the first strategy in a run (create new directory)
    # or if we should use an existing directory (passed from run_all_strategies)
    if 'base_results_dir' in globals() and globals()['base_results_dir']:
        base_results_dir = globals()['base_results_dir']
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_results_dir = f"results/run_{timestamp}"
        # Store in global variable for subsequent runs
        globals()['base_results_dir'] = base_results_dir
    
    # Strategy-specific subdirectory within the common base directory
    results_dir = f"{base_results_dir}/{strategy}_{start_date}_{total_steps}days"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save summary as TXT only
    with open(f"{results_dir}/summary.txt", "w") as f:
        f.write(f"{strategy.upper()} Strategy Simulation Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Simulation Period: {start_date} to {exchange.get_current_timestamp().strftime('%Y-%m-%d')} ({total_steps} days)\n\n")
        f.write(f"Initial Balance: ${account.initial_balance:.2f}\n")
        f.write(f"Final Balance: ${summary['balance']:.2f}\n")
        f.write(f"Total PnL: ${summary['total_pnl']:.2f} ({summary['pnl_percentage']:.2f}%)\n")
        f.write(f"Total Trades: {len(account.closed_positions)}\n")
        f.write(f"Win Rate: {win_rate:.2f}%\n\n")
        
        # Add strategy parameters - handle None case
        f.write("Strategy Parameters:\n")
        if params:
            for param, value in params.items():
                f.write(f"  {param}: {value}\n")
        else:
            f.write("  Using default parameters\n")
        
        # Add trade details
        f.write("\nTrade History:\n")
        f.write(f"{'Entry Date':<12} {'Exit Date':<12} {'Type':<6} {'Entry':<10} {'Exit':<10} {'PnL':<10} {'PnL %':<10}\n")
        f.write(f"{'-'*70}\n")
        
        for pos in account.closed_positions:
            # Use timestamp and exit_timestamp from FuturesPosition
            try:
                entry_date = pos.timestamp.strftime('%Y-%m-%d') if hasattr(pos, 'timestamp') else "N/A"
                exit_date = pos.exit_timestamp.strftime('%Y-%m-%d') if hasattr(pos, 'exit_timestamp') else "N/A"
            except (AttributeError, ValueError):
                entry_date = "N/A"
                exit_date = "N/A"
            
            # Calculate pnl_percentage if it doesn't exist
            if not hasattr(pos, 'pnl_percentage'):
                try:
                    # Try to calculate it based on entry value and pnl
                    entry_value = pos.entry_price * pos.size
                    if entry_value != 0:
                        pnl_percentage = (pos.pnl / entry_value) * 100
                    else:
                        pnl_percentage = 0.0
                except (AttributeError, ZeroDivisionError):
                    pnl_percentage = 0.0
            else:
                pnl_percentage = pos.pnl_percentage
            
            pos_type = "LONG" if pos.is_long else "SHORT"
            f.write(f"{entry_date:<12} {exit_date:<12} {pos_type:<6} {pos.entry_price:<10.2f} {pos.exit_price:<10.2f} "
                   f"${pos.pnl:<9.2f} {pnl_percentage:<9.2f}%\n")

    print(f"\nSimulation results saved to {results_dir}/")
    
    # Plot final performance with updated path - save only, don't display
    figures_dir = f"{base_results_dir}/figures"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(f"{figures_dir}/strategies", exist_ok=True)
    
    # Save to strategy-specific results directory
    bot.plot_performance(save_path=f'{results_dir}/{strategy}_performance.png')
    
    # Save to common figures directory
    bot.plot_performance(save_path=f'{figures_dir}/strategies/{strategy}_performance.png')
    
    # Close any open matplotlib figures to prevent display
    plt.close('all')
    
    return bot, exchange, base_results_dir

def run_all_strategies() -> Dict[str, Dict[str, Any]]:
    """Run simulations for all available strategies and compare results"""
    strategies = ['moving_average_crossover', 'bollinger_bands', 'rsi', 'adaptive_momentum']
    results = {}
    
    # Reset the global base_results_dir before starting a new run
    globals()['base_results_dir'] = None
    
    for strategy in strategies:
        # Pass an empty dict instead of None to avoid NoneType errors
        bot, exchange, base_dir = run_simulation(strategy, params={})
        
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
    
    # Compare and display results - save in the common base directory
    compare_strategies(results, globals()['base_results_dir'])
    
    return results

def compare_strategies(results: Dict[str, Dict[str, Any]], output_dir: str = None) -> None:
    """Compare and visualize the performance of different strategies"""
    # Print comparison table
    print("\n===== STRATEGY COMPARISON =====")
    print(f"{'STRATEGY':<25} {'FINAL BALANCE':<15} {'PNL %':<10} {'TRADES':<10} {'WIN RATE':<10}")
    print("-" * 70)
    
    comparison_data = []
    
    for strategy, data in results.items():
        strategy_name = strategy.replace('_', ' ').title()
        print(f"{strategy_name:<25} ${data['final_balance']:<13.2f} {data['pnl_percentage']:<8.2f}% {data['total_trades']:<8} {data['win_rate']:<8.2f}%")
        
        comparison_data.append({
            'strategy': strategy_name,
            'initial_balance': data['initial_balance'],
            'final_balance': data['final_balance'],
            'total_pnl': data['total_pnl'],
            'pnl_percentage': data['pnl_percentage'],
            'total_trades': data['total_trades'],
            'win_rate': data['win_rate']
        })
    
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
    
    # Save the comparison chart in the output directory
    if output_dir:
        figures_dir = f"{output_dir}/figures"
        os.makedirs(figures_dir, exist_ok=True)
        
        # Save comparison chart without displaying
        plt.savefig(f'{figures_dir}/strategy_comparison.png', facecolor='#1e1e1e')
        
        with open(f'{output_dir}/comparison_summary.txt', 'w') as f:
            f.write("STRATEGY COMPARISON\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"{'STRATEGY':<25} {'FINAL BALANCE':<15} {'PNL %':<10} {'TRADES':<10} {'WIN RATE':<10}\n")
            f.write("-" * 70 + "\n")
            
            for data in comparison_data:
                f.write(f"{data['strategy']:<25} ${data['final_balance']:<13.2f} {data['pnl_percentage']:<8.2f}% "
                        f"{data['total_trades']:<8} {data['win_rate']:<8.2f}%\n")
    
    # Close the figure without displaying it
    plt.close('all')
    
    print("\nStrategy comparison results saved to the results directory")
    if output_dir:
        print(f"Comparison results also saved to {output_dir}/")
    print("Individual strategy charts saved to figures/strategies/ directory")

if __name__ == "__main__":
    # Run all strategies and compare results
    results = run_all_strategies()
