"""Search strategy plugins with auto-discovery."""
import importlib
import inspect
from pathlib import Path
from typing import Dict, Type
from .base import BaseSearchStrategy


# Registry of all available search strategies
SEARCH_STRATEGIES: Dict[str, Type[BaseSearchStrategy]] = {}


def discover_strategies() -> Dict[str, Type[BaseSearchStrategy]]:
    """
    Auto-discover all search strategy plugins in this directory.
    
    Returns:
        Dictionary mapping strategy names to strategy classes
    """
    strategies = {}
    
    # Get the directory of this __init__.py file
    current_dir = Path(__file__).parent
    
    # Find all Python files except __init__.py and base.py
    for py_file in current_dir.glob("*.py"):
        if py_file.name in ["__init__.py", "base.py"]:
            continue
        
        # Import the module
        module_name = py_file.stem
        try:
            module = importlib.import_module(
                f"ris_research_engine.plugins.search.{module_name}"
            )
            
            # Find all classes that inherit from BaseSearchStrategy
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseSearchStrategy) and 
                    obj is not BaseSearchStrategy):
                    # Instantiate to get the strategy name
                    instance = obj()
                    strategies[instance.name] = obj
                    
        except Exception as e:
            print(f"Warning: Failed to load strategy from {module_name}: {e}")
    
    return strategies


def get_strategy(name: str) -> Type[BaseSearchStrategy]:
    """
    Get a strategy class by name.
    
    Args:
        name: Name of the strategy (e.g., 'grid_search', 'random_search')
        
    Returns:
        Strategy class
        
    Raises:
        ValueError: If strategy name is not found
    """
    if name not in SEARCH_STRATEGIES:
        available = ", ".join(SEARCH_STRATEGIES.keys())
        raise ValueError(
            f"Strategy '{name}' not found. Available strategies: {available}"
        )
    
    return SEARCH_STRATEGIES[name]


def list_strategies() -> Dict[str, str]:
    """
    List all available strategies with their descriptions.
    
    Returns:
        Dictionary mapping strategy names to descriptions
    """
    descriptions = {}
    for name, strategy_class in SEARCH_STRATEGIES.items():
        instance = strategy_class()
        descriptions[name] = instance.description
    
    return descriptions


# Auto-discover strategies on module import
SEARCH_STRATEGIES = discover_strategies()


# Export commonly used items
__all__ = [
    'BaseSearchStrategy',
    'SEARCH_STRATEGIES',
    'get_strategy',
    'list_strategies',
    'discover_strategies',
]
