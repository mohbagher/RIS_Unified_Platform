"""2D CNN for spatial RIS structure."""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Tuple
from .base import BaseModel


class CNN2DModule(nn.Module):
    """2D CNN module for spatial RIS data."""
    
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
        
        self.input_dim = input_dim
        
        # Calculate spatial dimensions (assume square-ish grid)
        # For M probes and N elements: input is (M*N) flattened
        # Reshape to (M, sqrt(N), sqrt(N)) approximately
        self.sqrt_n = int(math.sqrt(input_dim))
        
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
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_channels),
                act_fn(),
                nn.Dropout2d(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output layer
        self.fc = nn.Linear(in_channels, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_dim)
        batch_size = x.shape[0]
        
        # Reshape to approximate 2D grid
        # If input_dim is not a perfect square, pad or truncate
        target_size = self.sqrt_n * self.sqrt_n
        if x.shape[1] < target_size:
            # Pad with zeros
            padding = torch.zeros(batch_size, target_size - x.shape[1], device=x.device)
            x = torch.cat([x, padding], dim=1)
        elif x.shape[1] > target_size:
            # Truncate
            x = x[:, :target_size]
        
        # Reshape to 2D: (batch, 1, sqrt_n, sqrt_n)
        x = x.view(batch_size, 1, self.sqrt_n, self.sqrt_n)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Global pooling: (batch, channels, H, W) -> (batch, channels, 1, 1)
        x = self.pool(x)
        
        # Flatten: (batch, channels, 1, 1) -> (batch, channels)
        x = x.view(batch_size, -1)
        
        # Output layer
        return self.fc(x)


class CNN2DModel(BaseModel):
    """2D CNN for processing spatial RIS structure."""
    
    name = "cnn_2d"
    description = "2D convolutional neural network for spatial RIS element arrangement"
    
    def build(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        """Build 2D CNN model."""
        params = self.get_default_params()
        params.update(kwargs)
        
        return CNN2DModule(
            input_dim=input_dim,
            output_dim=output_dim,
            num_filters=params["num_filters"],
            kernel_size=params["kernel_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            activation=params["activation"]
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default 2D CNN hyperparameters."""
        return {
            "num_filters": 64,
            "kernel_size": 3,
            "num_layers": 3,
            "dropout": 0.1,
            "activation": "relu"
        }
    
    def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
        """Get hyperparameter ranges for search."""
        return {
            "num_filters": [32, 64, 128],
            "kernel_size": [3, 5, 7],
            "num_layers": [2, 3, 4, 5],
            "dropout": (0.0, 0.5),
            "activation": ["relu", "gelu", "tanh"]
        }
