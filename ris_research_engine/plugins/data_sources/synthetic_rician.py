"""Synthetic Rician fading data generator."""

import numpy as np
from typing import Dict, Any, Optional

from ris_research_engine.foundation import SystemConfig
from ris_research_engine.foundation.math_utils import (
    generate_dft_codebook, generate_random_codebook,
    compute_received_power, db_to_snr
)
from .base import BaseDataSource


class SyntheticRicianDataSource(BaseDataSource):
    """Generate synthetic Rician fading channel data."""
    
    name = "synthetic_rician"
    description = "Generate synthetic data with Rician fading channels"
    fidelity = "synthetic"
    
    def load(self, system_config: SystemConfig, **kwargs) -> Dict[str, Any]:
        """
        Generate synthetic Rician fading data.
        
        Optional kwargs:
            n_train: Number of training samples (default: 10000)
            n_val: Number of validation samples (default: 2000)
            n_test: Number of test samples (default: 2000)
            seed: Random seed for reproducibility (default: 42)
            codebook_type: 'dft' or 'random' (default: 'dft')
            probe_type: 'random' or 'dft' (default: 'random')
            K_factor: Rician K-factor (linear scale, default: 5.0)
                     K = Power(LOS) / Power(NLOS)
        """
        # Get parameters
        n_train = kwargs.get('n_train', 10000)
        n_val = kwargs.get('n_val', 2000)
        n_test = kwargs.get('n_test', 2000)
        seed = kwargs.get('seed', 42)
        codebook_type = kwargs.get('codebook_type', 'dft')
        probe_type = kwargs.get('probe_type', 'random')
        K_factor = kwargs.get('K_factor', 5.0)
        
        # Set random seed
        rng = np.random.RandomState(seed)
        
        # Generate codebook
        if codebook_type == 'dft':
            codebook = generate_dft_codebook(system_config.N, system_config.K, seed=seed)
        else:
            codebook = generate_random_codebook(
                system_config.N, system_config.K,
                phase_mode=system_config.phase_mode,
                phase_bits=system_config.phase_bits,
                seed=seed
            )
        
        # Generate probe matrix
        if probe_type == 'dft':
            probe_matrix = generate_dft_codebook(system_config.N, system_config.M, seed=seed+1)
        else:
            probe_matrix = generate_random_codebook(
                system_config.N, system_config.M,
                phase_mode=system_config.phase_mode,
                phase_bits=system_config.phase_bits,
                seed=seed+1
            )
        
        # Generate datasets
        train_data = self._generate_samples(n_train, system_config, codebook, probe_matrix, K_factor, rng)
        val_data = self._generate_samples(n_val, system_config, codebook, probe_matrix, K_factor, rng)
        test_data = self._generate_samples(n_test, system_config, codebook, probe_matrix, K_factor, rng)
        
        return {
            'train_inputs': train_data['inputs'],
            'train_targets': train_data['targets'],
            'train_powers': train_data['powers'],
            'val_inputs': val_data['inputs'],
            'val_targets': val_data['targets'],
            'val_powers': val_data['powers'],
            'test_inputs': test_data['inputs'],
            'test_targets': test_data['targets'],
            'test_powers': test_data['powers'],
            'codebook': codebook,
            'metadata': {
                'format': 'synthetic_rician',
                'fidelity': 'synthetic',
                'n_train': n_train,
                'n_val': n_val,
                'n_test': n_test,
                'seed': seed,
                'codebook_type': codebook_type,
                'probe_type': probe_type,
                'K_factor': K_factor,
                'system_config': system_config.to_dict(),
            }
        }
    
    def _generate_samples(self, n_samples: int, system_config: SystemConfig,
                         codebook: np.ndarray, probe_matrix: np.ndarray,
                         K_factor: float, rng: np.random.RandomState) -> Dict[str, np.ndarray]:
        """
        Generate samples with Rician fading channels.
        
        Args:
            n_samples: Number of samples to generate
            system_config: System configuration
            codebook: (K, N) codebook matrix
            probe_matrix: (M, N) probe matrix
            K_factor: Rician K-factor (linear scale)
            rng: Random number generator
            
        Returns:
            Dictionary with inputs, targets, and powers
        """
        N = system_config.N
        K = system_config.K
        M = system_config.M
        
        # Pre-allocate arrays
        inputs = np.zeros((n_samples, M * N))
        targets = np.zeros(n_samples, dtype=np.int64)
        powers = np.zeros((n_samples, K))
        
        # Noise power from SNR
        noise_power = 1.0 / db_to_snr(system_config.snr_db)
        
        for i in range(n_samples):
            # Generate Rician fading channel
            # h_tx_ris and h_ris_rx are independent Rician fading
            h_tx_ris = self._generate_rician_channel(N, K_factor, rng)
            h_ris_rx = self._generate_rician_channel(N, K_factor, rng)
            
            # Compute powers for all codebook entries
            for k in range(K):
                power = compute_received_power(h_tx_ris, h_ris_rx, codebook[k])
                powers[i, k] = power
            
            # Find optimal configuration
            targets[i] = np.argmax(powers[i])
            
            # Compute probe measurements
            for m in range(M):
                probe_config = probe_matrix[m]
                
                # Measure received power with this probe
                power = compute_received_power(h_tx_ris, h_ris_rx, probe_config)
                
                # Add noise
                noisy_power = power + noise_power * rng.randn()
                
                # Store probe configuration in input
                # Input format: concatenate all probe configs
                inputs[i, m*N:(m+1)*N] = probe_config
        
        return {
            'inputs': inputs,
            'targets': targets,
            'powers': powers,
        }
    
    def _generate_rician_channel(self, N: int, K_factor: float, 
                                rng: np.random.RandomState) -> np.ndarray:
        """
        Generate Rician fading channel coefficients.
        
        Rician fading consists of:
        - LOS component: deterministic line-of-sight path
        - NLOS component: scattered Rayleigh fading
        
        h = sqrt(K/(K+1)) * h_los + sqrt(1/(K+1)) * h_nlos
        
        where:
        - K is the Rician factor (power ratio LOS/NLOS)
        - h_los is the deterministic LOS component
        - h_nlos is the Rayleigh scattered component
        
        Args:
            N: Number of elements
            K_factor: Rician K-factor (linear scale)
            rng: Random number generator
            
        Returns:
            (N,) complex channel coefficients
        """
        # LOS component (deterministic, e.g., uniform phase progression)
        # Assume planar array with uniform angles
        n_indices = np.arange(N)
        angle_of_arrival = np.pi / 4  # 45 degrees
        h_los = np.exp(1j * 2 * np.pi * n_indices * np.sin(angle_of_arrival) / N)
        
        # NLOS component (Rayleigh fading)
        h_nlos_real = rng.randn(N)
        h_nlos_imag = rng.randn(N)
        h_nlos = (h_nlos_real + 1j * h_nlos_imag) / np.sqrt(2)
        
        # Combine with Rician factor
        # Normalize total power to 1
        h = np.sqrt(K_factor / (K_factor + 1)) * h_los + np.sqrt(1 / (K_factor + 1)) * h_nlos
        
        return h
