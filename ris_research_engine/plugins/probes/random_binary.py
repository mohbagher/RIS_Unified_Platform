"""Binary random probe generator."""

import numpy as np
from typing import Dict, Any
from .base import BaseProbe


class RandomBinaryProbe(BaseProbe):
    """Generates probes with binary random phases {0, π}."""
    
    name = "random_binary"
    description = "Binary random phases {0, π}"
    
    def generate(self, N: int, M: int, **kwargs) -> np.ndarray:
        """Generate M probes with N binary random phases each."""
        seed = kwargs.get('seed', None)
        rng = np.random.RandomState(seed)
        return rng.choice([0, np.pi], size=(M, N))
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {'seed': None}
    
    def theoretical_diversity(self, N: int, M: int) -> float:
        """
        Theoretical diversity for binary random phases.
        Lower than uniform since only 2 states per element.
        """
        return float(N * M * np.log(2))
