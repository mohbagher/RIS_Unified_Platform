"""Mean Reciprocal Rank (MRR) metric."""

import torch
from typing import Optional
from .base import BaseMetric


class MeanReciprocalRank(BaseMetric):
    """
    Mean Reciprocal Rank (MRR) metric.
    
    Computes the mean of the reciprocal ranks where the target appears
    in the ranked list of predictions. The rank is 1-indexed.
    
    MRR = 1/N * sum(1 / rank_i) where rank_i is the position of the
    true item in the ranked list for sample i.
    """
    
    name: str = "mean_reciprocal_rank"
    description: str = "Mean Reciprocal Rank (MRR) metric"
    higher_is_better: bool = True
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None
    ) -> float:
        """
        Compute Mean Reciprocal Rank.
        
        Args:
            predictions: (batch_size, K) logits or probabilities
            targets: (batch_size,) class indices
            metadata: Optional metadata (not used)
            
        Returns:
            MRR score as a float in (0, 1]
        """
        # Ensure predictions and targets are on the same device
        if predictions.device != targets.device:
            targets = targets.to(predictions.device)
        
        batch_size = predictions.size(0)
        num_classes = predictions.size(1)
        
        # Get sorted indices (descending order by score)
        _, sorted_indices = predictions.sort(dim=1, descending=True)
        
        # Find the rank of the target class for each sample using vectorized operations
        # Create rank positions tensor
        rank_positions = torch.arange(num_classes, device=predictions.device).unsqueeze(0).expand(batch_size, -1)
        
        # Find where each target appears in the sorted indices
        target_ranks = (sorted_indices == targets.unsqueeze(1)).long().argmax(dim=1)
        
        # Convert to 1-indexed ranks and compute reciprocals
        reciprocal_ranks = 1.0 / (target_ranks + 1.0).float()
        
        # Compute mean
        mrr = reciprocal_ranks.mean().item()
        
        return mrr


class MRR(MeanReciprocalRank):
    """Alias for MeanReciprocalRank."""
    
    name: str = "mrr"
    description: str = "Mean Reciprocal Rank (MRR) metric"
