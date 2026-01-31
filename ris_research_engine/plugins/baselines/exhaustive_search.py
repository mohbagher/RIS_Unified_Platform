"""Exhaustive search baseline (not applicable for limited probing scenarios)."""

import numpy as np
from .base import BaseBaseline


class ExhaustiveSearch(BaseBaseline):
    """Exhaustive search baseline.
    
    This baseline represents the ideal case where all configurations have been
    measured. Since it requires measurements for all K configurations, it's not
    truly applicable for the limited probing scenario. Returns uniform scores
    to indicate that no prediction is being made.
    
    Note: In practice, exhaustive search would directly use the measurements
    rather than predicting scores. This implementation serves as a placeholder
    for completeness.
    """
    
    def __init__(self):
        """Initialize exhaustive search baseline."""
        super().__init__()
        self.description = "Not applicable for baseline (requires all K measurements)"
    
    def predict(
        self, 
        probe_measurements: np.ndarray, 
        probe_indices: np.ndarray, 
        K: int
    ) -> np.ndarray:
        """Return uniform scores since exhaustive search is not applicable.
        
        Args:
            probe_measurements: Array of shape (M,) or (M, N).
            probe_indices: Array of shape (M,) containing indices of probed configs.
            K: Total number of configurations in the codebook.
        
        Returns:
            scores: Array of shape (K,) with uniform scores of 1.0.
        """
        # Return uniform scores since we cannot make predictions without all measurements
        return np.ones(K)
