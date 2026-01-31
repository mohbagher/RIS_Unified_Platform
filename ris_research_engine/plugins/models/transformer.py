"""Transformer model with multi-head self-attention."""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Tuple
from .base import BaseModel


class TransformerModule(nn.Module):
    """Transformer encoder module."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=input_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_dim)
        batch_size, seq_len = x.shape
        
        # Reshape to (batch, seq_len, 1) for projection
        x = x.unsqueeze(-1)
        
        # Project to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Output projection
        return self.output_proj(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(BaseModel):
    """Transformer with multi-head self-attention and positional encoding."""
    
    name = "transformer"
    description = "Transformer encoder with multi-head self-attention and positional encoding"
    
    def build(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        """Build transformer model."""
        params = self.get_default_params()
        params.update(kwargs)
        
        return TransformerModule(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=params["d_model"],
            num_heads=params["num_heads"],
            num_layers=params["num_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"]
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default transformer hyperparameters."""
        return {
            "d_model": 128,
            "num_heads": 8,
            "num_layers": 4,
            "dim_feedforward": 512,
            "dropout": 0.1
        }
    
    def get_param_ranges(self) -> Dict[str, Tuple[Any, Any]]:
        """Get hyperparameter ranges for search."""
        return {
            "d_model": [64, 128, 256],
            "num_heads": [4, 8, 16],
            "num_layers": [2, 4, 6, 8],
            "dim_feedforward": [256, 512, 1024],
            "dropout": (0.0, 0.3)
        }
