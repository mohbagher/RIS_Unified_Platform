"""Bidirectional LSTM for sequential data."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .base import BaseModel


class LSTMModule(nn.Module):
    """Bidirectional LSTM module."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection (from feature dim to 1 feature per timestep)
        # Treat input_dim as sequence length with 1 feature per step
        self.input_feature_dim = 1
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Output projection (from bidirectional hidden to output)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_dim)
        batch_size, seq_len = x.shape
        
        # Reshape to (batch, seq_len, 1) for LSTM
        x = x.unsqueeze(-1)
        
        # Apply LSTM
        # lstm_out shape: (batch, seq_len, hidden_size * 2)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state from both directions
        # h_n shape: (num_layers * 2, batch, hidden_size)
        # Get last layer, both directions
        h_forward = h_n[-2]  # Last layer, forward direction
        h_backward = h_n[-1]  # Last layer, backward direction
        
        # Concatenate forward and backward
        h_last = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_size * 2)
        
        # Output projection
        return self.output_proj(h_last)


class LSTMModel(BaseModel):
    """Bidirectional LSTM for sequential RIS data."""
    
    name = "lstm"
    description = "Bidirectional LSTM for sequential probe patterns"
    
    def build(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        """Build LSTM model."""
        params = self.get_default_params()
        params.update(kwargs)
        
        return LSTMModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"]
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default LSTM hyperparameters."""
        return {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.1
        }
    
    def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
        """Get hyperparameter ranges for search."""
        return {
            "hidden_size": [64, 128, 256, 512],
            "num_layers": [1, 2, 3, 4],
            "dropout": (0.0, 0.5)
        }
