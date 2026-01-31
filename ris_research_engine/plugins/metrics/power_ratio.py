"""Power ratio metric."""

import torch
from typing import Optional
from .base import BaseMetric


class PowerRatio(BaseMetric):
    """
    Power ratio metric.
    
    Computes the ratio of achieved power to optimal power.
    Measures how close the system operates to optimal power efficiency.
    """
    
    name: str = "power_ratio"
    description: str = "Ratio of achieved power to optimal power"
    higher_is_better: bool = True
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None
    ) -> float:
        """
        Compute power ratio.
        
        Args:
            predictions: Model predictions (not used directly)
            targets: Ground truth targets (not used directly)
            metadata: Must contain 'achieved_powers' and 'optimal_powers' tensors
            
        Returns:
            Mean power ratio as a float
            
        Raises:
            ValueError: If required metadata is missing
        """
        if metadata is None:
            raise ValueError(
                "PowerRatio metric requires metadata with 'achieved_powers' "
                "and 'optimal_powers' keys"
            )
        
        if 'achieved_powers' not in metadata:
            raise ValueError("Metadata must contain 'achieved_powers'")
        
        if 'optimal_powers' not in metadata:
            raise ValueError("Metadata must contain 'optimal_powers'")
        
        achieved_powers = metadata['achieved_powers']
        optimal_powers = metadata['optimal_powers']
        
        # Convert to tensors if needed
        if not isinstance(achieved_powers, torch.Tensor):
            achieved_powers = torch.tensor(achieved_powers)
        if not isinstance(optimal_powers, torch.Tensor):
            optimal_powers = torch.tensor(optimal_powers)
        
        # Ensure same device
        if achieved_powers.device != optimal_powers.device:
            optimal_powers = optimal_powers.to(achieved_powers.device)
        
        # Avoid division by zero
        epsilon = 1e-10
        optimal_powers = optimal_powers.clamp(min=epsilon)
        
        # Compute ratio
        ratio = achieved_powers / optimal_powers
        
        # Return mean ratio
        return ratio.mean().item()
