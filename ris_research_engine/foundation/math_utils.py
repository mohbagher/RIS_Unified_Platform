"""Mathematical utilities for RIS research."""
import numpy as np
from typing import Tuple, Optional


def generate_dft_codebook(N: int, K: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate K DFT beamforming vectors for N elements.
    
    Args:
        N: Number of RIS elements
        K: Codebook size
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (K, N) with phases in [0, 2π)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate K uniformly spaced angles
    angles = np.linspace(0, np.pi, K, endpoint=False)
    
    # Generate DFT vectors
    codebook = np.zeros((K, N))
    for k in range(K):
        # DFT beamforming vector
        n = np.arange(N)
        phases = 2 * np.pi * n * np.sin(angles[k]) / N
        codebook[k] = phases % (2 * np.pi)
    
    return codebook


def generate_random_codebook(N: int, K: int, phase_mode: str = 'continuous', 
                            phase_bits: int = 2, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random phase configurations.
    
    Args:
        N: Number of RIS elements
        K: Codebook size
        phase_mode: 'continuous' or 'discrete'
        phase_bits: Number of bits for phase quantization (if discrete)
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (K, N) with phases in [0, 2π)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if phase_mode == 'continuous':
        # Uniform random phases
        codebook = np.random.uniform(0, 2*np.pi, size=(K, N))
    else:
        # Discrete phases
        num_phases = 2 ** phase_bits
        discrete_phases = np.linspace(0, 2*np.pi, num_phases, endpoint=False)
        indices = np.random.randint(0, num_phases, size=(K, N))
        codebook = discrete_phases[indices]
    
    return codebook


def compute_received_power(h_tx_ris: np.ndarray, h_ris_rx: np.ndarray, 
                          phase_config: np.ndarray) -> float:
    """
    Compute received power for a given phase configuration.
    
    Args:
        h_tx_ris: Channel from transmitter to RIS (N,) complex
        h_ris_rx: Channel from RIS to receiver (N,) complex
        phase_config: Phase configuration (N,) in radians
        
    Returns:
        Received power (scalar)
    """
    # Apply phase shifts
    reflected_signal = h_tx_ris * np.exp(1j * phase_config)
    
    # Combine with RIS-to-receiver channel
    received_signal = np.sum(reflected_signal * h_ris_rx)
    
    # Compute power
    power = np.abs(received_signal) ** 2
    
    return power


def compute_snr(received_power: float, noise_power: float) -> float:
    """
    Compute SNR from received power and noise power.
    
    Args:
        received_power: Received signal power
        noise_power: Noise power
        
    Returns:
        SNR in linear scale
    """
    return received_power / noise_power


def snr_to_db(snr_linear: float) -> float:
    """
    Convert SNR from linear scale to dB.
    
    Args:
        snr_linear: SNR in linear scale
        
    Returns:
        SNR in dB
    """
    return 10 * np.log10(snr_linear + 1e-10)  # Add small value to avoid log(0)


def db_to_snr(snr_db: float) -> float:
    """
    Convert SNR from dB to linear scale.
    
    Args:
        snr_db: SNR in dB
        
    Returns:
        SNR in linear scale
    """
    return 10 ** (snr_db / 10)


def compute_spectral_efficiency(snr_linear: float) -> float:
    """
    Compute spectral efficiency from SNR.
    
    SE = log2(1 + SNR)
    
    Args:
        snr_linear: SNR in linear scale
        
    Returns:
        Spectral efficiency in bits/s/Hz
    """
    return np.log2(1 + snr_linear)


def compute_power_ratio(achieved_power: float, optimal_power: float) -> float:
    """
    Compute power ratio (η) between achieved and optimal power.
    
    Args:
        achieved_power: Achieved received power
        optimal_power: Optimal received power
        
    Returns:
        Power ratio in [0, 1]
    """
    if optimal_power == 0:
        return 0.0
    return achieved_power / optimal_power


def quantize_phases(phases: np.ndarray, bits: int) -> np.ndarray:
    """
    Quantize continuous phases to discrete values.
    
    Args:
        phases: Array of phases in [0, 2π)
        bits: Number of quantization bits
        
    Returns:
        Quantized phases in [0, 2π)
    """
    num_levels = 2 ** bits
    discrete_phases = np.linspace(0, 2*np.pi, num_levels, endpoint=False)
    
    # Find nearest discrete phase for each continuous phase
    phases_normalized = phases % (2 * np.pi)
    indices = np.round(phases_normalized / (2*np.pi) * num_levels).astype(int) % num_levels
    
    return discrete_phases[indices]


def array_response_vector(N: int, angle_rad: float, element_spacing: float = 0.5) -> np.ndarray:
    """
    Generate array response vector for uniform linear array.
    
    Args:
        N: Number of elements
        angle_rad: Angle in radians
        element_spacing: Element spacing in wavelengths
        
    Returns:
        Array response vector (N,) complex
    """
    n = np.arange(N)
    phase_shift = 2 * np.pi * element_spacing * n * np.sin(angle_rad)
    return np.exp(1j * phase_shift) / np.sqrt(N)


def generate_rayleigh_channel(N: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate Rayleigh fading channel coefficients.
    
    Args:
        N: Number of elements
        seed: Random seed for reproducibility
        
    Returns:
        Complex channel coefficients (N,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Complex Gaussian with variance 1
    real_part = np.random.randn(N) / np.sqrt(2)
    imag_part = np.random.randn(N) / np.sqrt(2)
    
    return real_part + 1j * imag_part


def generate_rician_channel(N: int, K_factor_db: float = 10.0, 
                           angle_rad: float = 0.0, element_spacing: float = 0.5,
                           seed: Optional[int] = None) -> np.ndarray:
    """
    Generate Rician fading channel coefficients.
    
    Args:
        N: Number of elements
        K_factor_db: Rician K-factor in dB
        angle_rad: Angle of LOS component
        element_spacing: Element spacing in wavelengths
        seed: Random seed for reproducibility
        
    Returns:
        Complex channel coefficients (N,)
    """
    K_linear = db_to_snr(K_factor_db)
    
    # LOS component
    los_component = array_response_vector(N, angle_rad, element_spacing) * np.sqrt(K_linear)
    
    # NLOS (Rayleigh) component
    nlos_component = generate_rayleigh_channel(N, seed)
    
    # Combine with proper power normalization
    channel = (los_component + nlos_component) / np.sqrt(K_linear + 1)
    
    return channel


def compute_channel_capacity(snr_linear: float) -> float:
    """
    Compute channel capacity (same as spectral efficiency for SISO).
    
    Args:
        snr_linear: SNR in linear scale
        
    Returns:
        Channel capacity in bits/s/Hz
    """
    return compute_spectral_efficiency(snr_linear)


def hadamard_matrix(n: int) -> np.ndarray:
    """
    Generate Hadamard matrix of size n x n (n must be power of 2).
    
    Args:
        n: Matrix size (must be power of 2)
        
    Returns:
        Hadamard matrix (n, n)
    """
    if n == 1:
        return np.array([[1]])
    
    # Check if n is power of 2
    if n & (n - 1) != 0:
        raise ValueError(f"n must be a power of 2, got {n}")
    
    # Recursive construction (Sylvester construction)
    h_half = hadamard_matrix(n // 2)
    return np.block([[h_half, h_half], [h_half, -h_half]])


def normalize_phases(phases: np.ndarray) -> np.ndarray:
    """
    Normalize phases to [0, 2π) range.
    
    Args:
        phases: Array of phases
        
    Returns:
        Normalized phases in [0, 2π)
    """
    return phases % (2 * np.pi)


def phase_difference(phase1: np.ndarray, phase2: np.ndarray) -> np.ndarray:
    """
    Compute phase difference accounting for wrap-around.
    
    Args:
        phase1: First phase array
        phase2: Second phase array
        
    Returns:
        Phase difference in [-π, π)
    """
    diff = phase1 - phase2
    # Wrap to [-π, π)
    return (diff + np.pi) % (2 * np.pi) - np.pi


def compute_beamforming_gain(phase_config: np.ndarray, target_angle: float, 
                             element_spacing: float = 0.5) -> float:
    """
    Compute beamforming gain towards target angle.
    
    Args:
        phase_config: Phase configuration (N,)
        target_angle: Target angle in radians
        element_spacing: Element spacing in wavelengths
        
    Returns:
        Beamforming gain (normalized)
    """
    N = len(phase_config)
    # Desired steering vector
    desired_response = array_response_vector(N, target_angle, element_spacing)
    # Actual response
    actual_response = np.exp(1j * phase_config) / np.sqrt(N)
    # Compute inner product
    gain = np.abs(np.sum(desired_response.conj() * actual_response)) ** 2
    return gain
