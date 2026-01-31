"""Data source plugin system with auto-discovery."""

from .base import BaseDataSource
from .hdf5_loader import HDF5DataSource
from .synthetic_rayleigh import SyntheticRayleighDataSource
from .synthetic_rician import SyntheticRicianDataSource


# Auto-import all data sources
_DATA_SOURCES = {}


def register_data_source(data_source_class):
    """Register a data source class in the registry."""
    _DATA_SOURCES[data_source_class.name] = data_source_class


def get_data_source(name: str) -> BaseDataSource:
    """
    Get a data source instance by name.
    
    Args:
        name: Name of the data source class
        
    Returns:
        Instance of the requested data source class
        
    Raises:
        KeyError: If data source name is not registered
    """
    if name not in _DATA_SOURCES:
        raise KeyError(
            f"Data source '{name}' not found. Available data sources: {list_data_sources()}"
        )
    return _DATA_SOURCES[name]()


def list_data_sources():
    """List all registered data source names."""
    return list(_DATA_SOURCES.keys())


# Register all built-in data sources
register_data_source(HDF5DataSource)
register_data_source(SyntheticRayleighDataSource)
register_data_source(SyntheticRicianDataSource)


__all__ = [
    'BaseDataSource',
    'HDF5DataSource',
    'SyntheticRayleighDataSource',
    'SyntheticRicianDataSource',
    'register_data_source',
    'get_data_source',
    'list_data_sources',
]
