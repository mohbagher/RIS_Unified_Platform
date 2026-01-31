"""Base class for all model plugins."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


class BaseModel(ABC):
    """Base class for neural network model architectures."""
    
    name: str = "base"
    description: str = "Base model class"
    
    @abstractmethod
    def build(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        """
        Build and return a neural network model.
        
        Args:
            input_dim: Input dimension (number of features)
            output_dim: Output dimension (number of outputs)
            **kwargs: Model-specific hyperparameters
            
        Returns:
            PyTorch nn.Module instance
        """
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default hyperparameters for this model.
        
        Returns:
            Dictionary of default hyperparameters
        """
        pass
    
    @abstractmethod
    def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
        """
        Get valid ranges for hyperparameters (for hyperparameter search).
        
        Returns:
            Dictionary mapping parameter names to (min, max) tuples or lists of valid values
        """
        pass
