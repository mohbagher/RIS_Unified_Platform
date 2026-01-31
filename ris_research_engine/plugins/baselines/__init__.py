"""Baseline plugins for RIS configuration selection.

This module provides simple heuristic baseline strategies for selecting
RIS configurations based on limited probe measurements. These serve as
benchmarks for more sophisticated learning-based approaches.

Available baselines:
- RandomSelection: Uniform random scores
- BestOfProbed: Select best measured configuration
- ExhaustiveSearch: Placeholder for exhaustive measurement
- StrongestBeam: Score by proximity to strongest probe
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, Type

from .base import BaseBaseline

logger = logging.getLogger(__name__)

# Import all baseline implementations
from .random_selection import RandomSelection
from .best_of_probed import BestOfProbed
from .exhaustive_search import ExhaustiveSearch
from .strongest_beam import StrongestBeam


def discover_baselines() -> Dict[str, Type[BaseBaseline]]:
    """Auto-discover all baseline classes in this module.
    
    Returns:
        Dictionary mapping baseline names to their classes.
    """
    baselines = {}
    
    # Get current module path
    current_dir = Path(__file__).parent
    
    # Iterate through all Python files in the directory
    for file_path in current_dir.glob("*.py"):
        # Skip __init__.py and base.py
        if file_path.name in ["__init__.py", "base.py"]:
            continue
        
        # Import the module
        module_name = f"ris_research_engine.plugins.baselines.{file_path.stem}"
        try:
            module = importlib.import_module(module_name)
            
            # Find all classes that inherit from BaseBaseline
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseBaseline) and obj is not BaseBaseline:
                    baselines[name] = obj
        except ImportError as e:
            logger.warning(f"Could not import {module_name}: {e}")
    
    return baselines


# Auto-discover all baselines
AVAILABLE_BASELINES = discover_baselines()

# Export commonly used items
__all__ = [
    "BaseBaseline",
    "RandomSelection",
    "BestOfProbed",
    "ExhaustiveSearch",
    "StrongestBeam",
    "discover_baselines",
    "AVAILABLE_BASELINES",
]
