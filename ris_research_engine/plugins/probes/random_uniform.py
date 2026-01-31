"""Uniform random probe generator."""

import numpy as np
from typing import Dict, Any
from .base import BaseProbe


class RandomUniformProbe(BaseProbe):
    """Generates probes with uniform random phases in [0, 2π)."""
    
    name = "random_uniform"
    description = "Uniform random phases in [0, 2π)"
    
    def generate(self, N: int, M: int, **kwargs) -> np.ndarray:
        """Generate M probes with N uniform random phases each."""
        seed = kwargs.get('seed', None)
        rng = np.random.RandomState(seed)
        return rng.uniform(0, 2 * np.pi, size=(M, N))
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {'seed': None}
    
    def theoretical_diversity(self, N: int, M: int) -> float:
        """
        Theoretical diversity for uniform random phases.
        High diversity as all phases are independent and uniformly distributed.
        """
        return float(N * M)
