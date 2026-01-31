"""Foundation layer initialization."""
from .data_types import SystemConfig, TrainingConfig, ExperimentConfig, ExperimentResult, SearchCampaignResult
from .storage import ResultTracker, detect_hdf5_format, load_hdf5_data, save_hdf5_data, HDF5_AVAILABLE

__all__ = [
    'SystemConfig',
    'TrainingConfig', 
    'ExperimentConfig',
    'ExperimentResult',
    'SearchCampaignResult',
    'ResultTracker',
    'detect_hdf5_format',
    'load_hdf5_data',
    'save_hdf5_data',
    'HDF5_AVAILABLE',
]
