"""Top-k accuracy metric."""

import torch
from typing import Optional
from .base import BaseMetric


class TopKAccuracy(BaseMetric):
    """
    Top-k accuracy metric.
    
    Computes the fraction of samples where the true class is among
    the top-k predicted classes.
    """
    
    name: str = "top_k_accuracy"
    description: str = "Top-k accuracy metric (k=1,5,10)"
    higher_is_better: bool = True
    
    def __init__(self, k: int = 1):
        """
        Initialize Top-k accuracy metric.
        
        Args:
            k: Number of top predictions to consider (default: 1)
        """
        self.k = k
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None
    ) -> float:
        """
        Compute top-k accuracy.
        
        Args:
            predictions: (batch_size, K) logits or probabilities
            targets: (batch_size,) class indices
            metadata: Optional metadata (not used)
            
        Returns:
            Top-k accuracy as a float in [0, 1]
        """
        # Ensure predictions and targets are on the same device
        if predictions.device != targets.device:
            targets = targets.to(predictions.device)
        
        # Get top-k predictions
        batch_size = predictions.size(0)
        k = min(self.k, predictions.size(1))  # Handle case where k > num_classes
        
        # Get indices of top-k predictions
        _, top_k_indices = predictions.topk(k, dim=1, largest=True, sorted=True)
        
        # Expand targets to match top_k_indices shape
        targets_expanded = targets.view(-1, 1).expand_as(top_k_indices)
        
        # Check if target is in top-k predictions
        correct = (top_k_indices == targets_expanded).any(dim=1)
        
        # Compute accuracy
        accuracy = correct.float().mean().item()
        
        return accuracy


class Top1Accuracy(TopKAccuracy):
    """Top-1 accuracy (standard accuracy)."""
    
    name: str = "top_1_accuracy"
    description: str = "Top-1 accuracy (standard classification accuracy)"
    
    def __init__(self):
        super().__init__(k=1)


class Top5Accuracy(TopKAccuracy):
    """Top-5 accuracy."""
    
    name: str = "top_5_accuracy"
    description: str = "Top-5 accuracy"
    
    def __init__(self):
        super().__init__(k=5)


class Top10Accuracy(TopKAccuracy):
    """Top-10 accuracy."""
    
    name: str = "top_10_accuracy"
    description: str = "Top-10 accuracy"
    
    def __init__(self):
        super().__init__(k=10)
