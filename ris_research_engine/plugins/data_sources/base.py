"""Base class for all data source plugins."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

from ris_research_engine.foundation import SystemConfig


class BaseDataSource(ABC):
    """Base class for data source plugins."""
    
    name: str = "base"
    description: str = "Base data source class"
    fidelity: str = "synthetic"  # 'synthetic', 'sionna', or 'hardware'
    
    @abstractmethod
    def load(self, system_config: SystemConfig, **kwargs) -> Dict[str, Any]:
        """
        Load or generate data for the given system configuration.
        
        Args:
            system_config: System configuration
            **kwargs: Additional data source-specific parameters
            
        Returns:
            Dictionary with the following keys:
                - 'train_inputs': (N_train, M*N) probe measurements
                - 'train_targets': (N_train,) optimal config indices
                - 'train_powers': (N_train, K) all config powers
                - 'val_inputs': (N_val, M*N) probe measurements
                - 'val_targets': (N_val,) optimal config indices
                - 'val_powers': (N_val, K) all config powers
                - 'test_inputs': (N_test, M*N) probe measurements
                - 'test_targets': (N_test,) optimal config indices
                - 'test_powers': (N_test, K) all config powers
                - 'codebook': (K, N) phase configurations
                - 'metadata': dict with additional information
        """
        pass
    
    def validate_output(self, data: Dict[str, Any], system_config: SystemConfig) -> bool:
        """
        Validate the output data structure.
        
        Args:
            data: Data dictionary returned by load()
            system_config: System configuration
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        required_keys = [
            'train_inputs', 'train_targets', 'train_powers',
            'val_inputs', 'val_targets', 'val_powers',
            'test_inputs', 'test_targets', 'test_powers',
            'codebook', 'metadata'
        ]
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate shapes
        N, K, M = system_config.N, system_config.K, system_config.M
        
        # Check codebook
        if data['codebook'].shape != (K, N):
            raise ValueError(f"Codebook shape mismatch: expected ({K}, {N}), got {data['codebook'].shape}")
        
        # Check train data
        N_train = data['train_inputs'].shape[0]
        if data['train_inputs'].shape != (N_train, M * N):
            raise ValueError(f"train_inputs shape mismatch: expected (N_train, {M * N}), got {data['train_inputs'].shape}")
        if data['train_targets'].shape != (N_train,):
            raise ValueError(f"train_targets shape mismatch: expected ({N_train},), got {data['train_targets'].shape}")
        if data['train_powers'].shape != (N_train, K):
            raise ValueError(f"train_powers shape mismatch: expected ({N_train}, {K}), got {data['train_powers'].shape}")
        
        # Check val data
        N_val = data['val_inputs'].shape[0]
        if data['val_inputs'].shape != (N_val, M * N):
            raise ValueError(f"val_inputs shape mismatch: expected (N_val, {M * N}), got {data['val_inputs'].shape}")
        if data['val_targets'].shape != (N_val,):
            raise ValueError(f"val_targets shape mismatch: expected ({N_val},), got {data['val_targets'].shape}")
        if data['val_powers'].shape != (N_val, K):
            raise ValueError(f"val_powers shape mismatch: expected ({N_val}, {K}), got {data['val_powers'].shape}")
        
        # Check test data
        N_test = data['test_inputs'].shape[0]
        if data['test_inputs'].shape != (N_test, M * N):
            raise ValueError(f"test_inputs shape mismatch: expected (N_test, {M * N}), got {data['test_inputs'].shape}")
        if data['test_targets'].shape != (N_test,):
            raise ValueError(f"test_targets shape mismatch: expected ({N_test},), got {data['test_targets'].shape}")
        if data['test_powers'].shape != (N_test, K):
            raise ValueError(f"test_powers shape mismatch: expected ({N_test}, {K}), got {data['test_powers'].shape}")
        
        return True
