"""Halton quasi-random sequence probe generator."""

import numpy as np
from typing import Dict, Any
from scipy.stats import qmc
from .base import BaseProbe


class HaltonProbe(BaseProbe):
    """Generates probes using Halton quasi-random sequences."""
    
    name = "halton"
    description = "Halton quasi-random sequence phases"
    
    def generate(self, N: int, M: int, **kwargs) -> np.ndarray:
        """Generate M probes using Halton sequence in N dimensions."""
        seed = kwargs.get('seed', None)
        
        # Create Halton sampler
        sampler = qmc.Halton(d=N, scramble=True, seed=seed)
        
        # Generate M samples in [0, 1)^N
        samples = sampler.random(M)
        
        # Scale to [0, 2π)
        phases = samples * 2 * np.pi
        
        return phases
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {'seed': None}
    
    def theoretical_diversity(self, N: int, M: int) -> float:
        """
        Theoretical diversity for Halton sequences.
        Quasi-random sequences have better space-filling properties than random.
        """
        return float(N * M * 1.15)  # Good space-filling properties
