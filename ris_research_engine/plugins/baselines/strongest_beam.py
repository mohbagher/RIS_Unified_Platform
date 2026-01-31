"""Strongest beam baseline that scores configs by proximity to best probe."""

import numpy as np
from .base import BaseBaseline


class StrongestBeam(BaseBaseline):
    """Strongest beam baseline.
    
    Scores all configurations based on their proximity (similarity) to the
    strongest measured probe configuration. Assumes that configurations similar
    to the best observed one are likely to also perform well.
    
    Uses cosine similarity for multi-dimensional measurements or correlation
    for single-dimensional measurements based on configuration indices.
    """
    
    def __init__(self, similarity_metric: str = "cosine"):
        """Initialize strongest beam baseline.
        
        Args:
            similarity_metric: Metric to use for computing proximity.
                             Options: "cosine", "correlation", "euclidean"
        """
        super().__init__()
        self.description = "Score configs by proximity to strongest measured probe"
        self.similarity_metric = similarity_metric
    
    def predict(
        self, 
        probe_measurements: np.ndarray, 
        probe_indices: np.ndarray, 
        K: int
    ) -> np.ndarray:
        """Score configs by proximity to strongest probe configuration.
        
        Args:
            probe_measurements: Array of shape (M,) or (M, N).
            probe_indices: Array of shape (M,) containing indices of probed configs.
            K: Total number of configurations in the codebook.
        
        Returns:
            scores: Array of shape (K,) with proximity-based scores.
        """
        # Handle 2D measurements by taking mean across features
        if probe_measurements.ndim == 2:
            measurement_values = probe_measurements.mean(axis=1)
        else:
            measurement_values = probe_measurements
        
        # Find index of best measurement
        best_probe_idx = np.argmax(measurement_values)
        best_config_idx = probe_indices[best_probe_idx]
        
        # Compute proximity based on configuration indices
        # Generate scores based on distance from best configuration
        config_indices = np.arange(K)
        
        if self.similarity_metric == "cosine" or self.similarity_metric == "correlation":
            # For discrete indices, use inverse distance as a proxy for similarity
            # Normalize indices to [0, 1] and compute similarity
            distances = np.abs(config_indices - best_config_idx)
            max_distance = max(best_config_idx, K - 1 - best_config_idx)
            if max_distance > 0:
                # Convert distance to similarity: closer configs get higher scores
                scores = 1.0 - (distances / max_distance)
            else:
                scores = np.ones(K)
        
        elif self.similarity_metric == "euclidean":
            # Inverse Euclidean distance
            distances = np.abs(config_indices - best_config_idx)
            # Use exponential decay for smoother scoring
            scores = np.exp(-distances / (K / 10.0))
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Ensure best configuration has highest score
        scores[best_config_idx] = 1.0
        
        return scores
