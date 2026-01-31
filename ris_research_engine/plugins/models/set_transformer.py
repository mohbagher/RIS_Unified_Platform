"""Set Transformer - permutation-invariant transformer without positional encoding."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .base import BaseModel


class SetTransformerModule(nn.Module):
    """Set Transformer module (permutation-invariant)."""
    
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
        
        # Input projection (no positional encoding)
        self.input_proj = nn.Linear(1, d_model)
        
        # Transformer encoder layers without positional encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Pooling layer (permutation-invariant aggregation)
        self.pool_type = "mean"  # Can also use "max" or learnable pooling
        
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
        
        # Project to d_model (no positional encoding)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Permutation-invariant pooling
        if self.pool_type == "mean":
            x = x.mean(dim=1)  # (batch, d_model)
        elif self.pool_type == "max":
            x = x.max(dim=1)[0]  # (batch, d_model)
        
        # Output projection
        return self.output_proj(x)


class SetTransformerModel(BaseModel):
    """Set Transformer - permutation-invariant without positional encoding."""
    
    name = "set_transformer"
    description = "Permutation-invariant transformer without positional encoding"
    
    def build(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        """Build set transformer model."""
        params = self.get_default_params()
        params.update(kwargs)
        
        return SetTransformerModule(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=params["d_model"],
            num_heads=params["num_heads"],
            num_layers=params["num_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"]
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default set transformer hyperparameters."""
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
