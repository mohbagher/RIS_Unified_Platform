"""Test all probe types for shape, range, Hadamard orthogonality, and reproducibility."""

import pytest
import numpy as np

from ris_research_engine.plugins.probes import (
    get_probe, list_probes,
    RandomUniformProbe, RandomBinaryProbe, HadamardProbe,
    SobolProbe, HaltonProbe, DFTBeamsProbe, LearnedProbe
)


# All 7 probe types
PROBE_TYPES = [
    "random_uniform",
    "random_binary", 
    "hadamard",
    "sobol",
    "halton",
    "dft_beams",
    "learned"
]


class TestProbeRegistry:
    """Test probe registration and discovery."""
    
    def test_list_probes(self):
        """Test that all probes are registered."""
        probes = list_probes()
        assert len(probes) >= 7
        for probe_type in PROBE_TYPES:
            assert probe_type in probes
    
    def test_get_probe(self):
        """Test getting probe instances."""
        for probe_type in PROBE_TYPES:
            probe = get_probe(probe_type)
            assert probe is not None
            assert probe.name == probe_type
    
    def test_get_invalid_probe(self):
        """Test that invalid probe name raises error."""
        with pytest.raises(KeyError):
            get_probe("nonexistent_probe")


class TestProbeShape:
    """Test that all probes generate correct output shape."""
    
    @pytest.mark.parametrize("probe_type", PROBE_TYPES)
    @pytest.mark.parametrize("N,M", [(16, 4), (32, 8), (64, 16)])
    def test_probe_shape(self, probe_type, N, M):
        """Test probe output shape is (M, N)."""
        probe = get_probe(probe_type)
        phases = probe.generate(N=N, M=M)
        
        assert phases.shape == (M, N), f"{probe_type} should return (M={M}, N={N})"
        assert isinstance(phases, np.ndarray)
    
    @pytest.mark.parametrize("probe_type", PROBE_TYPES)
    def test_probe_dtype(self, probe_type):
        """Test probe output is numeric."""
        probe = get_probe(probe_type)
        phases = probe.generate(N=16, M=4)
        
        assert np.issubdtype(phases.dtype, np.number)
        assert not np.any(np.isnan(phases))
        assert not np.any(np.isinf(phases))


class TestProbeRange:
    """Test that probe phases are in valid range [0, 2π)."""
    
    @pytest.mark.parametrize("probe_type", PROBE_TYPES)
    def test_phase_range(self, probe_type):
        """Test phases are in [0, 2π)."""
        probe = get_probe(probe_type)
        phases = probe.generate(N=32, M=8)
        
        # Most probes should be in [0, 2π), but some might use different conventions
        # Check that values are reasonable
        assert np.all(phases >= 0.0), f"{probe_type} has negative phases"
        assert np.all(phases < 2 * np.pi + 1e-6), f"{probe_type} phases exceed 2π"


class TestHadamardOrthogonality:
    """Test Hadamard probe orthogonality properties."""
    
    def test_hadamard_orthogonality(self):
        """Test Hadamard probe generates orthogonal patterns."""
        probe = get_probe("hadamard")
        
        # Hadamard requires M to be power of 2
        N, M = 16, 16  # Use M=N as power of 2
        phases = probe.generate(N=N, M=M)
        
        # Convert to complex representation
        complex_probes = np.exp(1j * phases)
        
        # Compute Gram matrix (inner products)
        gram = complex_probes @ complex_probes.conj().T
        
        # Check diagonal (should be N)
        diagonal = np.abs(np.diag(gram))
        assert np.allclose(diagonal, N, rtol=0.1), "Diagonal should be ~N"
        
        # Check off-diagonal (should be small for orthogonal)
        off_diagonal = gram - np.diag(np.diag(gram))
        off_diagonal_norm = np.abs(off_diagonal).max()
        
        # Orthogonal if off-diagonal is much smaller than diagonal
        assert off_diagonal_norm < N * 0.5, "Off-diagonal should be small for orthogonality"
    
    def test_hadamard_power_of_2(self):
        """Test Hadamard with power-of-2 dimensions."""
        probe = get_probe("hadamard")
        
        for M in [4, 8, 16, 32]:
            phases = probe.generate(N=M, M=M)
            assert phases.shape == (M, M)


class TestProbeReproducibility:
    """Test that probes are reproducible with same seed."""
    
    @pytest.mark.parametrize("probe_type", PROBE_TYPES)
    def test_reproducibility(self, probe_type, seed):
        """Test probes produce same output with same seed."""
        probe1 = get_probe(probe_type)
        probe2 = get_probe(probe_type)
        
        # Generate with same parameters
        N, M = 16, 4
        phases1 = probe1.generate(N=N, M=M, seed=seed)
        phases2 = probe2.generate(N=N, M=M, seed=seed)
        
        assert np.allclose(phases1, phases2), f"{probe_type} not reproducible"
    
    @pytest.mark.parametrize("probe_type", PROBE_TYPES)
    def test_different_seeds_differ(self, probe_type):
        """Test probes produce different output with different seeds."""
        probe = get_probe(probe_type)
        
        N, M = 16, 4
        phases1 = probe.generate(N=N, M=M, seed=42)
        phases2 = probe.generate(N=N, M=M, seed=123)
        
        # Should be different (unless deterministic like Hadamard)
        if probe_type not in ["hadamard", "dft_beams"]:
            assert not np.allclose(phases1, phases2), f"{probe_type} should differ with different seeds"


class TestProbeParameters:
    """Test probe parameter handling."""
    
    @pytest.mark.parametrize("probe_type", PROBE_TYPES)
    def test_get_default_params(self, probe_type):
        """Test all probes have default parameters."""
        probe = get_probe(probe_type)
        params = probe.get_default_params()
        
        assert isinstance(params, dict)
    
    @pytest.mark.parametrize("probe_type", PROBE_TYPES)
    def test_theoretical_diversity(self, probe_type):
        """Test theoretical diversity calculation."""
        probe = get_probe(probe_type)
        
        diversity = probe.theoretical_diversity(N=16, M=4)
        
        assert isinstance(diversity, (int, float))
        assert diversity >= 0.0, "Diversity should be non-negative"


class TestSpecificProbes:
    """Test specific probe implementations."""
    
    def test_random_uniform_range(self):
        """Test random uniform generates uniform distribution."""
        probe = get_probe("random_uniform")
        phases = probe.generate(N=1000, M=100, seed=42)
        
        # Should be roughly uniform in [0, 2π)
        assert phases.min() >= 0.0
        assert phases.max() <= 2 * np.pi
        
        # Check approximate uniformity
        hist, _ = np.histogram(phases.flatten(), bins=10, range=(0, 2*np.pi))
        # Bins should be roughly equal (within 50% variance)
        assert hist.std() / hist.mean() < 0.5
    
    def test_random_binary_values(self):
        """Test random binary uses binary phases."""
        probe = get_probe("random_binary")
        phases = probe.generate(N=100, M=10, seed=42)
        
        # Should only have 0 or π values (or similar binary)
        unique_vals = np.unique(phases)
        assert len(unique_vals) <= 3, "Binary should have at most 2-3 unique values"
    
    def test_dft_beams_structure(self):
        """Test DFT beams have proper structure."""
        probe = get_probe("dft_beams")
        phases = probe.generate(N=16, M=8, seed=42)
        
        # DFT beams should have specific structure
        assert phases.shape == (8, 16)
        # Just check they're valid phases
        assert np.all(phases >= 0.0)
        assert np.all(phases <= 2 * np.pi)
