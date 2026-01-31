"""1D CNN for sequential probe data."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .base import BaseModel


class CNN1DModule(nn.Module):
    """1D CNN module for sequential data."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_filters: int,
        kernel_size: int,
        num_layers: int,
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
        
        # Build convolutional layers
        layers = []
        in_channels = 1  # Single channel input
        
        for i in range(num_layers):
            out_channels = num_filters * (2 ** i) if i < num_layers - 1 else num_filters
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                act_fn(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layer
        self.fc = nn.Linear(in_channels, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_dim)
        # Reshape to (batch, 1, input_dim) for Conv1d
        x = x.unsqueeze(1)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Global pooling: (batch, channels, seq_len) -> (batch, channels, 1)
        x = self.pool(x)
        
        # Flatten: (batch, channels, 1) -> (batch, channels)
        x = x.squeeze(-1)
        
        # Output layer
        return self.fc(x)


class CNN1DModel(BaseModel):
    """1D CNN for processing sequential probe data."""
    
    name = "cnn_1d"
    description = "1D convolutional neural network for sequential RIS probe patterns"
    
    def build(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        """Build 1D CNN model."""
        params = self.get_default_params()
        params.update(kwargs)
        
        return CNN1DModule(
            input_dim=input_dim,
            output_dim=output_dim,
            num_filters=params["num_filters"],
            kernel_size=params["kernel_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            activation=params["activation"]
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default 1D CNN hyperparameters."""
        return {
            "num_filters": 64,
            "kernel_size": 5,
            "num_layers": 3,
            "dropout": 0.1,
            "activation": "relu"
        }
    
    def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
        """Get hyperparameter ranges for search."""
        return {
            "num_filters": [32, 64, 128],
            "kernel_size": [3, 5, 7, 9],
            "num_layers": [2, 3, 4, 5],
            "dropout": (0.0, 0.5),
            "activation": ["relu", "gelu", "tanh"]
        }
