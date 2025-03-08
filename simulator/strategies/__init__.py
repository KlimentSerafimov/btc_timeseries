from .BaseStrategy import BaseStrategy
from .MovingAverageCrossover import MovingAverageCrossover
from .BollingerBands import BollingerBands
from .RSI import RSI
from .AdaptiveMomentum import AdaptiveMomentum

# Strategy factory
def create_strategy(strategy_name: str, params: dict = None) -> BaseStrategy:
    """
    Factory function to create a strategy instance
    
    Parameters:
    - strategy_name: Name of the strategy to create
    - params: Strategy parameters
    
    Returns:
    - Strategy instance
    """
    if params is None:
        params = {}
    
    if strategy_name == 'moving_average_crossover':
        return MovingAverageCrossover(params)
    elif strategy_name == 'bollinger_bands':
        return BollingerBands(params)
    elif strategy_name == 'rsi':
        return RSI(params)
    elif strategy_name == 'adaptive_momentum':
        return AdaptiveMomentum(params)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}") 