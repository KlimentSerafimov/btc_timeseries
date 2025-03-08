from .BaseStrategy import BaseStrategy
from .MovingAverageCrossover import MovingAverageCrossover
from .BollingerBands import BollingerBands
from .RSI import RSI

# Strategy factory
def create_strategy(strategy_name, params=None):
    """
    Factory function to create a strategy instance
    
    Parameters:
    - strategy_name: Name of the strategy to create
    - params: Parameters for the strategy
    
    Returns:
    - Strategy instance
    """
    if params is None:
        params = {}
        
    strategies = {
        'moving_average_crossover': MovingAverageCrossover,
        'bollinger_bands': BollingerBands,
        'rsi': RSI
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Strategy '{strategy_name}' not implemented")
    
    return strategies[strategy_name](params) 