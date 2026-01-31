"""Probe plugin system with auto-discovery."""

from .base import BaseProbe
from .random_uniform import RandomUniformProbe
from .random_binary import RandomBinaryProbe
from .hadamard import HadamardProbe
from .sobol import SobolProbe
from .halton import HaltonProbe
from .dft_beams import DFTBeamsProbe
from .learned_probe import LearnedProbe


# Auto-import all probes
_PROBES = {}


def register_probe(probe_class):
    """Register a probe class in the registry."""
    _PROBES[probe_class.name] = probe_class


def get_probe(name: str) -> BaseProbe:
    """
    Get a probe instance by name.
    
    Args:
        name: Name of the probe class
        
    Returns:
        Instance of the requested probe class
        
    Raises:
        KeyError: If probe name is not registered
    """
    if name not in _PROBES:
        raise KeyError(
            f"Probe '{name}' not found. Available probes: {list_probes()}"
        )
    return _PROBES[name]()


def list_probes():
    """List all registered probe names."""
    return list(_PROBES.keys())


# Register all built-in probes
register_probe(RandomUniformProbe)
register_probe(RandomBinaryProbe)
register_probe(HadamardProbe)
register_probe(SobolProbe)
register_probe(HaltonProbe)
register_probe(DFTBeamsProbe)
register_probe(LearnedProbe)


__all__ = [
    'BaseProbe',
    'RandomUniformProbe',
    'RandomBinaryProbe',
    'HadamardProbe',
    'SobolProbe',
    'HaltonProbe',
    'DFTBeamsProbe',
    'LearnedProbe',
    'register_probe',
    'get_probe',
    'list_probes',
]
