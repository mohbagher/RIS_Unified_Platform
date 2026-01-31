"""Base class for all probe plugins."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class BaseProbe(ABC):
    """Base class for probe generation strategies."""
    
    name: str = "base"
    description: str = "Base probe class"
    
    @abstractmethod
    def generate(self, N: int, M: int, **kwargs) -> np.ndarray:
        """
        Generate probe matrix.
        
        Args:
            N: Number of RIS elements
            M: Number of probes
            **kwargs: Additional probe-specific parameters
            
        Returns:
            M × N array of phases in [0, 2π)
        """
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for this probe type.
        
        Returns:
            Dictionary of default parameters
        """
        pass
    
    @abstractmethod
    def theoretical_diversity(self, N: int, M: int) -> float:
        """
        Calculate theoretical diversity measure for this probe type.
        
        Args:
            N: Number of RIS elements
            M: Number of probes
            
        Returns:
            Diversity score (higher is better)
        """
        pass
