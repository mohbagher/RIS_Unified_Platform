"""Learned probe generator (stub)."""

import numpy as np
from typing import Dict, Any
from .base import BaseProbe


class LearnedProbe(BaseProbe):
    """Stub for learned probe generation (to be implemented)."""
    
    name = "learned"
    description = "Learned probe patterns (not yet implemented)"
    
    def generate(self, N: int, M: int, **kwargs) -> np.ndarray:
        """Generate learned probes (not yet implemented)."""
        raise NotImplementedError(
            "Learned probe generation is not yet implemented. "
            "This requires training a model to generate optimal probe patterns."
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {}
    
    def theoretical_diversity(self, N: int, M: int) -> float:
        """Theoretical diversity for learned probes."""
        raise NotImplementedError("Learned probe diversity not yet implemented.")
