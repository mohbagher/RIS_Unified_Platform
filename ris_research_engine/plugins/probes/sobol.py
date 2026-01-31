"""Sobol quasi-random sequence probe generator."""

import numpy as np
from typing import Dict, Any
from scipy.stats import qmc
from .base import BaseProbe


class SobolProbe(BaseProbe):
    """Generates probes using Sobol quasi-random sequences."""
    
    name = "sobol"
    description = "Sobol quasi-random sequence phases"
    
    def generate(self, N: int, M: int, **kwargs) -> np.ndarray:
        """Generate M probes using Sobol sequence in N dimensions."""
        seed = kwargs.get('seed', None)
        
        # Create Sobol sampler
        sampler = qmc.Sobol(d=N, scramble=True, seed=seed)
        
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
        Theoretical diversity for Sobol sequences.
        Quasi-random sequences have better space-filling properties than random.
        """
        return float(N * M * 1.2)  # Slightly better than pure random
