"""DFT beamforming probe generator."""

import numpy as np
from typing import Dict, Any
from ...foundation.math_utils import generate_dft_codebook
from .base import BaseProbe


class DFTBeamsProbe(BaseProbe):
    """Generates probes using DFT beamforming vectors."""
    
    name = "dft_beams"
    description = "DFT beamforming vectors"
    
    def generate(self, N: int, M: int, **kwargs) -> np.ndarray:
        """Generate M DFT beamforming vectors for N elements."""
        seed = kwargs.get('seed', None)
        
        # Generate DFT codebook
        codebook = generate_dft_codebook(N, M, seed=seed)
        
        # Extract phases from complex codebook
        # codebook is K × N complex array
        phases = np.angle(codebook)
        
        # Ensure phases are in [0, 2π)
        phases = np.mod(phases, 2 * np.pi)
        
        return phases
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {'seed': None}
    
    def theoretical_diversity(self, N: int, M: int) -> float:
        """
        Theoretical diversity for DFT beams.
        DFT beams are orthogonal and provide structured coverage.
        """
        return float(min(M, N) * np.log(N))
