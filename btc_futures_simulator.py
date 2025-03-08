from typing import Optional, Tuple

from FuturesExchange import FuturesExchange
from TradingBot import TradingBot

def run_simulation() -> Tuple[TradingBot, FuturesExchange]:
    """Run a complete simulation with the exchange and trading bot"""
    # Load Bitcoin price data
    try:
        from btc_analysis import load_data
        btc_data = load_data()
    except FileNotFoundError:
        print("Price data file not found. Downloading data...")
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