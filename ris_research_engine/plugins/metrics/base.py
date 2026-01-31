"""Base class for all metrics."""

import torch
from abc import ABC, abstractmethod
from typing import Optional


class BaseMetric(ABC):
    """Base class for all metric plugins."""
    
    name: str = "base_metric"
    description: str = "Base metric class"
    higher_is_better: bool = True
    
    @abstractmethod
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None
    ) -> float:
        """
        Compute the metric value.
        
        Args:
            predictions: Model predictions (batch_size, ...)
            targets: Ground truth targets (batch_size, ...)
            metadata: Optional metadata dict with additional information
            
        Returns:
            Computed metric value as a float
        """
        pass
    
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None
    ) -> float:
        """Allow calling the metric as a function."""
        return self.compute(predictions, targets, metadata)
