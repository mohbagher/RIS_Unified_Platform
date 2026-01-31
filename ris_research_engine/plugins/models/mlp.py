"""Multi-layer perceptron with batch normalization."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from .base import BaseModel


class MLPModule(nn.Module):
    """Multi-layer perceptron module."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = True
    ):
        super().__init__()
        
        # Select activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLPModel(BaseModel):
    """Multi-layer perceptron model with configurable architecture."""
    
    name = "mlp"
    description = "Multi-layer perceptron with batch normalization and configurable activation"
    
    def build(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        """Build MLP model."""
        params = self.get_default_params()
        params.update(kwargs)
        
        return MLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=params["hidden_dims"],
            dropout=params["dropout"],
            activation=params["activation"],
            batch_norm=params["batch_norm"]
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default MLP hyperparameters."""
        return {
            "hidden_dims": [256, 128, 64],
            "dropout": 0.1,
            "activation": "relu",
            "batch_norm": True
        }
    
    def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
        """Get hyperparameter ranges for search."""
        return {
            "hidden_dims": [
                [512, 256, 128],
                [256, 128, 64],
                [256, 128],
                [512, 256],
                [128, 64],
                [256, 256, 256],
            ],
            "dropout": (0.0, 0.5),
            "activation": ["relu", "gelu", "tanh"],
            "batch_norm": [True, False]
        }
