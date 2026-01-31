"""Best-of-probed baseline that selects the best measured configuration."""

import numpy as np
from .base import BaseBaseline


class BestOfProbed(BaseBaseline):
    """Best-of-probed baseline.
    
    Assigns high score to the configuration with the best probe measurement,
    and zero scores to all other configurations. This represents the strategy
    of simply selecting the best observed configuration without any prediction
    or generalization.
    """
    
    def __init__(self):
        """Initialize best-of-probed baseline."""
        super().__init__()
        self.description = "Assign high score to best probed config, zero to others"
    
    def predict(
        self, 
        probe_measurements: np.ndarray, 
        probe_indices: np.ndarray, 
        K: int
    ) -> np.ndarray:
        """Assign high score to best probed config, zero to all others.
        
        Args:
            probe_measurements: Array of shape (M,) or (M, N). If 2D, uses mean
                               across features to determine best configuration.
            probe_indices: Array of shape (M,) containing indices of probed configs.
            K: Total number of configurations in the codebook.
        
        Returns:
            scores: Array of shape (K,) with 1.0 for best probed config, 0.0 for others.
        """
        scores = np.zeros(K)
        
        # Handle 2D measurements by taking mean across features
        if probe_measurements.ndim == 2:
            measurement_values = probe_measurements.mean(axis=1)
        else:
            measurement_values = probe_measurements
        
        # Find index of best measurement
        best_probe_idx = np.argmax(measurement_values)
        best_config_idx = probe_indices[best_probe_idx]
        
        # Assign score of 1.0 to best configuration
        scores[best_config_idx] = 1.0
        
        return scores
