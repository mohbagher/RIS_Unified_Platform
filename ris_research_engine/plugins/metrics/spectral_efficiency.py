"""Spectral efficiency metric."""

import torch
from typing import Optional
from .base import BaseMetric


class SpectralEfficiency(BaseMetric):
    """
    Spectral efficiency metric.
    
    Computes spectral efficiency based on SNR values.
    SE = log2(1 + SNR) in bits/s/Hz (Shannon capacity formula).
    """
    
    name: str = "spectral_efficiency"
    description: str = "Spectral efficiency metric (bits/s/Hz)"
    higher_is_better: bool = True
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None
    ) -> float:
        """
        Compute spectral efficiency.
        
        Args:
            predictions: Model predictions (not used directly)
            targets: Ground truth targets (not used directly)
            metadata: Must contain 'snr_values' tensor
            
        Returns:
            Mean spectral efficiency in bits/s/Hz
            
        Raises:
            ValueError: If required metadata is missing
        """
        if metadata is None:
            raise ValueError(
                "SpectralEfficiency metric requires metadata with 'snr_values' key"
            )
        
        if 'snr_values' not in metadata:
            raise ValueError("Metadata must contain 'snr_values'")
        
        snr_values = metadata['snr_values']
        
        # Convert to tensor if needed
        if not isinstance(snr_values, torch.Tensor):
            snr_values = torch.tensor(snr_values)
        
        # Compute spectral efficiency using Shannon capacity formula
        # SE = log2(1 + SNR)
        spectral_efficiency = torch.log2(1.0 + snr_values)
        
        # Return mean spectral efficiency
        return spectral_efficiency.mean().item()


class AverageSpectralEfficiency(SpectralEfficiency):
    """Alias for SpectralEfficiency with explicit averaging."""
    
    name: str = "average_spectral_efficiency"
    description: str = "Average spectral efficiency metric (bits/s/Hz)"
