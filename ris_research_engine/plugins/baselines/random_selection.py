"""Random baseline that assigns uniform random scores to all configurations."""

import numpy as np
from .base import BaseBaseline


class RandomSelection(BaseBaseline):
    """Random selection baseline.
    
    Assigns uniform random scores to all configurations, ignoring probe measurements.
    This serves as the simplest possible baseline, representing pure random guessing.
    """
    
    def __init__(self, seed: int = None):
        """Initialize random selection baseline.
        
        Args:
            seed: Random seed for reproducibility. If None, uses random seed.
        """
        super().__init__()
        self.description = "Random uniform scores for all configurations"
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def predict(
        self, 
        probe_measurements: np.ndarray, 
        probe_indices: np.ndarray, 
        K: int
    ) -> np.ndarray:
        """Generate random uniform scores for all configurations.
        
        Args:
            probe_measurements: Ignored.
            probe_indices: Ignored.
            K: Total number of configurations in the codebook.
        
        Returns:
            scores: Array of shape (K,) with uniform random scores in [0, 1].
        """
        return self.rng.uniform(0, 1, size=K)
