"""Base class for baseline RIS configuration selection strategies."""

from abc import ABC, abstractmethod
import numpy as np


class BaseBaseline(ABC):
    """Base class for all baseline strategies.
    
    Baselines are simple heuristic methods for selecting RIS configurations
    based on limited probe measurements, serving as benchmarks for more
    sophisticated learning-based approaches.
    """
    
    def __init__(self):
        """Initialize the baseline."""
        self.name: str = self.__class__.__name__
        self.description: str = ""
    
    @abstractmethod
    def predict(
        self, 
        probe_measurements: np.ndarray, 
        probe_indices: np.ndarray, 
        K: int
    ) -> np.ndarray:
        """Predict scores for all K configurations based on probe measurements.
        
        Args:
            probe_measurements: Array of shape (M,) or (M, N) containing measurements
                               from M probed configurations. If 2D, N is number of 
                               features per measurement.
            probe_indices: Array of shape (M,) containing indices of which configs
                          were probed (values in range [0, K-1]).
            K: Total number of configurations in the codebook.
        
        Returns:
            scores: Array of shape (K,) with predicted scores for all configurations.
                   Higher scores indicate better predicted performance.
        """
        pass
