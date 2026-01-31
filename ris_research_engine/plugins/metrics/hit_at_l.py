"""Hit@L metric."""

import torch
from typing import Optional
from .base import BaseMetric


class HitAtL(BaseMetric):
    """
    Hit@L metric.
    
    Computes the fraction of samples where the target is in the top-L predictions.
    Similar to top-k accuracy but with configurable L parameter.
    """
    
    name: str = "hit_at_l"
    description: str = "Hit@L metric (is target in top-L predictions?)"
    higher_is_better: bool = True
    
    def __init__(self, L: int = 5):
        """
        Initialize Hit@L metric.
        
        Args:
            L: Number of top predictions to consider (default: 5)
        """
        self.L = L
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None
    ) -> float:
        """
        Compute Hit@L.
        
        Args:
            predictions: (batch_size, K) logits or probabilities
            targets: (batch_size,) class indices or (batch_size, num_targets) for multi-target
            metadata: Optional metadata (not used)
            
        Returns:
            Hit@L score as a float in [0, 1]
        """
        # Ensure predictions and targets are on the same device
        if predictions.device != targets.device:
            targets = targets.to(predictions.device)
        
        # Get top-L predictions
        batch_size = predictions.size(0)
        L = min(self.L, predictions.size(1))  # Handle case where L > num_classes
        
        # Get indices of top-L predictions
        _, top_L_indices = predictions.topk(L, dim=1, largest=True, sorted=True)
        
        # Handle single target or multiple targets
        if targets.dim() == 1:
            # Single target per sample
            targets_expanded = targets.view(-1, 1).expand_as(top_L_indices)
            hit = (top_L_indices == targets_expanded).any(dim=1)
        else:
            # Multiple targets per sample - vectorized approach
            # Expand dimensions for broadcasting: (batch, 1, L) and (batch, num_targets, 1)
            top_L_expanded = top_L_indices.unsqueeze(1)  # (batch, 1, L)
            targets_expanded = targets.unsqueeze(2)  # (batch, num_targets, 1)
            # Check if any target matches any top-L prediction
            hit = (top_L_expanded == targets_expanded).any(dim=2).any(dim=1)
        
        # Compute hit rate
        hit_rate = hit.float().mean().item()
        
        return hit_rate


class HitAt1(HitAtL):
    """Hit@1 metric."""
    
    name: str = "hit_at_1"
    description: str = "Hit@1 metric"
    
    def __init__(self):
        super().__init__(L=1)


class HitAt5(HitAtL):
    """Hit@5 metric."""
    
    name: str = "hit_at_5"
    description: str = "Hit@5 metric"
    
    def __init__(self):
        super().__init__(L=5)


class HitAt10(HitAtL):
    """Hit@10 metric."""
    
    name: str = "hit_at_10"
    description: str = "Hit@10 metric"
    
    def __init__(self):
        super().__init__(L=10)
