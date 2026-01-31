"""Hadamard matrix probe generator."""

import numpy as np
from typing import Dict, Any
from scipy.linalg import hadamard
from .base import BaseProbe


class HadamardProbe(BaseProbe):
    """Generates probes from Hadamard matrix rows scaled to {0, π}."""
    
    name = "hadamard"
    description = "Hadamard matrix rows scaled to {0, π}"
    
    def generate(self, N: int, M: int, **kwargs) -> np.ndarray:
        """
        Generate M probes from Hadamard matrix.
        
        Note: Hadamard matrices have sizes that are powers of 2.
        If N is not a power of 2, we use the next larger power of 2 and truncate.
        """
        # Find next power of 2 >= N
        n_hadamard = 2 ** int(np.ceil(np.log2(N)))
        
        # Generate Hadamard matrix
        H = hadamard(n_hadamard)
        
        # Take first M rows and first N columns
        # Convert -1 -> π, +1 -> 0
        phases = np.where(H[:M, :N] == 1, 0, np.pi)
        
        return phases
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {}
    
    def theoretical_diversity(self, N: int, M: int) -> float:
        """
        Theoretical diversity for Hadamard probes.
        Hadamard matrices have maximal orthogonality properties.
        """
        return float(min(M, N) * np.log(2))
