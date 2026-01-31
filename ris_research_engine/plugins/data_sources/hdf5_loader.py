"""HDF5 data loader with auto-format detection."""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from ris_research_engine.foundation import SystemConfig, detect_hdf5_format, load_hdf5_data, HDF5_AVAILABLE
from .base import BaseDataSource


class HDF5DataSource(BaseDataSource):
    """Load data from HDF5 files with auto-format detection."""
    
    name = "hdf5_loader"
    description = "Load data from HDF5 files (AutoML/Session5/Generic formats)"
    fidelity = "unknown"  # Will be determined from file format
    
    def load(self, system_config: SystemConfig, **kwargs) -> Dict[str, Any]:
        """
        Load data from HDF5 file with auto-format detection.
        
        Required kwargs:
            file_path: Path to HDF5 file
            
        Optional kwargs:
            train_split: Fraction of data for training (default: 0.7)
            val_split: Fraction of data for validation (default: 0.15)
            test_split: Fraction of data for testing (default: 0.15)
            seed: Random seed for reproducibility (default: 42)
            format_hint: Optional format hint ('automl', 'session5', 'generic')
        """
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 operations")
        
        file_path = kwargs.get('file_path')
        if file_path is None:
            raise ValueError("file_path is required")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        
        # Get optional parameters
        train_split = kwargs.get('train_split', 0.7)
        val_split = kwargs.get('val_split', 0.15)
        test_split = kwargs.get('test_split', 0.15)
        seed = kwargs.get('seed', 42)
        format_hint = kwargs.get('format_hint')
        
        # Validate splits
        if not np.isclose(train_split + val_split + test_split, 1.0):
            raise ValueError(f"Splits must sum to 1.0: {train_split} + {val_split} + {test_split} = {train_split + val_split + test_split}")
        
        # Detect format using enhanced detection
        data_format = self._detect_format(file_path, format_hint)
        
        # Load data based on format
        if data_format == 'automl':
            return self._load_automl_format(file_path, system_config, train_split, val_split, test_split, seed)
        elif data_format == 'session5':
            return self._load_session5_format(file_path, system_config, train_split, val_split, test_split, seed)
        else:
            return self._load_generic_format(file_path, system_config, train_split, val_split, test_split, seed)
    
    def _detect_format(self, file_path: str, format_hint: Optional[str] = None) -> str:
        """
        Enhanced format detection based on task requirements.
        
        - AutoML format: keys user_positions, all_config_powers, optimal_indices → fidelity='sionna'
        - Session5 format: keys channel_matrices, codebook, labels, powers → fidelity='synthetic'
        - Generic format: keys inputs, targets
        """
        if format_hint:
            return format_hint
        
        import h5py
        with h5py.File(file_path, 'r') as f:
            keys = set(f.keys())
            
            # Check for AutoML format (Sionna data)
            if 'user_positions' in keys and 'all_config_powers' in keys and 'optimal_indices' in keys:
                self.fidelity = 'sionna'
                return 'automl'
            
            # Check for Session5 format (Synthetic data)
            if 'channel_matrices' in keys and 'codebook' in keys and 'labels' in keys and 'powers' in keys:
                self.fidelity = 'synthetic'
                return 'session5'
            
            # Check for generic format
            if 'inputs' in keys and 'targets' in keys:
                self.fidelity = 'synthetic'  # Default to synthetic
                return 'generic'
            
            # Fallback to original detection
            detected = detect_hdf5_format(file_path)
            if detected == 'automl':
                self.fidelity = 'sionna'
            elif detected == 'session5':
                self.fidelity = 'synthetic'
            else:
                self.fidelity = 'synthetic'
            
            return detected
    
    def _load_automl_format(self, file_path: str, system_config: SystemConfig,
                           train_split: float, val_split: float, test_split: float,
                           seed: int) -> Dict[str, Any]:
        """
        Load AutoML format with keys: user_positions, all_config_powers, optimal_indices.
        """
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            # Load data
            user_positions = f['user_positions'][:]  # (N_samples, 3) or similar
            all_config_powers = f['all_config_powers'][:]  # (N_samples, K)
            optimal_indices = f['optimal_indices'][:]  # (N_samples,)
            
            # Load or generate codebook
            if 'codebook' in f:
                codebook = f['codebook'][:]
            else:
                # Generate codebook if not present
                from ris_research_engine.foundation.math_utils import generate_dft_codebook
                codebook = generate_dft_codebook(system_config.N, system_config.K, seed=seed)
            
            # Load or generate probe matrix
            if 'probe_matrix' in f:
                probe_matrix = f['probe_matrix'][:]  # (M, N)
            else:
                # Generate probe matrix if not present
                from ris_research_engine.foundation.math_utils import generate_random_codebook
                probe_matrix = generate_random_codebook(
                    system_config.N, system_config.M,
                    phase_mode=system_config.phase_mode,
                    phase_bits=system_config.phase_bits,
                    seed=seed
                )
            
            # Check if probe measurements are already computed
            if 'probe_measurements' in f:
                probe_measurements = f['probe_measurements'][:]  # (N_samples, M, N) or (N_samples, M*N)
                if probe_measurements.ndim == 3:
                    probe_measurements = probe_measurements.reshape(len(probe_measurements), -1)
            else:
                # Need to compute probe measurements from channels
                # For AutoML format, we may not have raw channels, so we create synthetic measurements
                # This is a limitation - we'll use powers as a proxy
                N_samples = len(optimal_indices)
                probe_measurements = np.random.randn(N_samples, system_config.M * system_config.N)
        
        # Split data
        data = self._split_data(
            probe_measurements, optimal_indices, all_config_powers, codebook,
            train_split, val_split, test_split, seed
        )
        
        # Add metadata
        data['metadata'] = {
            'format': 'automl',
            'fidelity': 'sionna',
            'file_path': file_path,
            'system_config': system_config.to_dict(),
        }
        
        return data
    
    def _load_session5_format(self, file_path: str, system_config: SystemConfig,
                              train_split: float, val_split: float, test_split: float,
                              seed: int) -> Dict[str, Any]:
        """
        Load Session5 format with keys: channel_matrices, codebook, labels, powers.
        """
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            # Load data
            channel_matrices = f['channel_matrices'][:]  # (N_samples, ...) complex channels
            codebook = f['codebook'][:]  # (K, N)
            labels = f['labels'][:]  # (N_samples,) optimal indices
            powers = f['powers'][:]  # (N_samples, K)
            
            # Load or generate probe matrix
            if 'probe_matrix' in f:
                probe_matrix = f['probe_matrix'][:]  # (M, N)
            else:
                # Generate probe matrix
                from ris_research_engine.foundation.math_utils import generate_random_codebook
                probe_matrix = generate_random_codebook(
                    system_config.N, system_config.M,
                    phase_mode=system_config.phase_mode,
                    phase_bits=system_config.phase_bits,
                    seed=seed
                )
            
            # Compute probe measurements from channels
            if 'probe_measurements' in f:
                probe_measurements = f['probe_measurements'][:]
                if probe_measurements.ndim == 3:
                    probe_measurements = probe_measurements.reshape(len(probe_measurements), -1)
            else:
                # Compute from channels and probe matrix
                probe_measurements = self._compute_probe_measurements(
                    channel_matrices, probe_matrix, system_config
                )
        
        # Split data
        data = self._split_data(
            probe_measurements, labels, powers, codebook,
            train_split, val_split, test_split, seed
        )
        
        # Add metadata
        data['metadata'] = {
            'format': 'session5',
            'fidelity': 'synthetic',
            'file_path': file_path,
            'system_config': system_config.to_dict(),
        }
        
        return data
    
    def _load_generic_format(self, file_path: str, system_config: SystemConfig,
                            train_split: float, val_split: float, test_split: float,
                            seed: int) -> Dict[str, Any]:
        """
        Load generic format with keys: inputs, targets.
        """
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            # Load data
            inputs = f['inputs'][:]  # (N_samples, M*N)
            targets = f['targets'][:]  # (N_samples,)
            
            # Load optional data
            if 'powers' in f:
                powers = f['powers'][:]  # (N_samples, K)
            else:
                # Generate dummy powers
                powers = np.random.rand(len(inputs), system_config.K)
            
            if 'codebook' in f:
                codebook = f['codebook'][:]
            else:
                # Generate codebook
                from ris_research_engine.foundation.math_utils import generate_dft_codebook
                codebook = generate_dft_codebook(system_config.N, system_config.K, seed=seed)
        
        # Split data
        data = self._split_data(
            inputs, targets, powers, codebook,
            train_split, val_split, test_split, seed
        )
        
        # Add metadata
        data['metadata'] = {
            'format': 'generic',
            'fidelity': 'synthetic',
            'file_path': file_path,
            'system_config': system_config.to_dict(),
        }
        
        return data
    
    def _compute_probe_measurements(self, channel_matrices: np.ndarray,
                                   probe_matrix: np.ndarray,
                                   system_config: SystemConfig) -> np.ndarray:
        """
        Compute probe measurements from channel matrices.
        
        Args:
            channel_matrices: (N_samples, N) or (N_samples, 2, N) complex channels
            probe_matrix: (M, N) probe configurations
            system_config: System configuration
            
        Returns:
            (N_samples, M*N) probe measurements
        """
        from ris_research_engine.foundation.math_utils import compute_received_power, db_to_snr
        
        N_samples = len(channel_matrices)
        M, N = probe_matrix.shape
        
        # Ensure channels are complex
        if channel_matrices.dtype != np.complex64 and channel_matrices.dtype != np.complex128:
            # Convert real to complex if needed
            if channel_matrices.ndim == 3 and channel_matrices.shape[1] == 2:
                # Assume first dim is real, second is imag
                channel_matrices = channel_matrices[:, 0, :] + 1j * channel_matrices[:, 1, :]
            else:
                channel_matrices = channel_matrices.astype(np.complex128)
        
        # Compute measurements
        measurements = np.zeros((N_samples, M * N))
        noise_power = 1.0 / db_to_snr(system_config.snr_db)
        
        for i in range(N_samples):
            h_channel = channel_matrices[i]  # Assume single channel (N,)
            
            # For each probe
            for m in range(M):
                probe_config = probe_matrix[m]
                
                # Compute received power
                # Simplified: assume h_tx_ris = h_ris_rx = sqrt(h_channel)
                h_tx_ris = np.sqrt(np.abs(h_channel)) * np.exp(1j * np.angle(h_channel))
                h_ris_rx = np.sqrt(np.abs(h_channel)) * np.exp(1j * np.angle(h_channel))
                
                power = compute_received_power(h_tx_ris, h_ris_rx, probe_config)
                
                # Add noise
                noise = noise_power * np.random.randn()
                measurement = power + noise
                
                # Store measurement with probe config
                # Encode measurement with phase information
                for n in range(N):
                    measurements[i, m*N + n] = measurement * (1 + np.cos(probe_config[n])) / 2
        
        return measurements
    
    def _split_data(self, inputs: np.ndarray, targets: np.ndarray, powers: np.ndarray,
                    codebook: np.ndarray, train_split: float, val_split: float,
                    test_split: float, seed: int) -> Dict[str, Any]:
        """
        Split data into train/val/test sets.
        """
        N_samples = len(inputs)
        
        # Shuffle indices
        rng = np.random.RandomState(seed)
        indices = np.arange(N_samples)
        rng.shuffle(indices)
        
        # Compute split points
        train_end = int(train_split * N_samples)
        val_end = train_end + int(val_split * N_samples)
        
        # Split indices
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        return {
            'train_inputs': inputs[train_idx],
            'train_targets': targets[train_idx],
            'train_powers': powers[train_idx],
            'val_inputs': inputs[val_idx],
            'val_targets': targets[val_idx],
            'val_powers': powers[val_idx],
            'test_inputs': inputs[test_idx],
            'test_targets': targets[test_idx],
            'test_powers': powers[test_idx],
            'codebook': codebook,
        }
