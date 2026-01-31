"""MLP with residual (skip) connections."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from .base import BaseModel


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, dim: int, dropout: float = 0.0, activation: str = "relu"):
        super().__init__()
        
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = act_fn()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResidualMLPModule(nn.Module):
    """MLP with residual blocks."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float = 0.0,
        activation: str = "relu"
    ):
        super().__init__()
        
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Project to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act_fn()
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout, activation)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class ResidualMLPModel(BaseModel):
    """MLP with residual connections for better gradient flow."""
    
    name = "residual_mlp"
    description = "Multi-layer perceptron with residual skip connections"
    
    def build(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        """Build residual MLP model."""
        params = self.get_default_params()
        params.update(kwargs)
        
        return ResidualMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=params["hidden_dim"],
            num_blocks=params["num_blocks"],
            dropout=params["dropout"],
            activation=params["activation"]
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default residual MLP hyperparameters."""
        return {
            "hidden_dim": 256,
            "num_blocks": 4,
            "dropout": 0.1,
            "activation": "relu"
        }
    
    def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
        """Get hyperparameter ranges for search."""
        return {
            "hidden_dim": [128, 256, 512],
            "num_blocks": [2, 3, 4, 6],
            "dropout": (0.0, 0.5),
            "activation": ["relu", "gelu", "tanh"]
        }
