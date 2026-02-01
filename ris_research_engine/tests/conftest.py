"""Pytest fixtures for RIS Research Engine tests."""

import pytest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path

from ris_research_engine.foundation import SystemConfig, TrainingConfig, ExperimentConfig
from ris_research_engine.foundation.storage import ResultTracker


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    tracker = ResultTracker(db_path=db_path)
    yield tracker
    
    # Cleanup
    tracker.close()
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_system_config():
    """Create a sample system configuration."""
    return SystemConfig(
        N=16,
        K=16,
        M=4,
        frequency=28e9,
        snr_db=20.0
    )


@pytest.fixture
def sample_training_config():
    """Create a sample training configuration."""
    return TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        max_epochs=5,
        early_stopping_patience=3,
        optimizer="adam",
        loss_function="cross_entropy",
        val_split=0.15,
        test_split=0.15
    )


@pytest.fixture
def sample_experiment_config(sample_system_config, sample_training_config):
    """Create a sample experiment configuration."""
    return ExperimentConfig(
        name="test_experiment",
        system=sample_system_config,
        training=sample_training_config,
        probe_type="hadamard",
        probe_params={},
        model_type="mlp",
        model_params={},
        data_source="synthetic",
        data_params={"num_samples": 100},
        metrics=["top_1_accuracy"],
        data_fidelity="synthetic"
    )


@pytest.fixture
def sample_probe_data():
    """Generate sample probe data (M x N)."""
    M, N = 4, 16
    return np.random.uniform(0, 2*np.pi, size=(M, N))


@pytest.fixture
def sample_channel_data():
    """Generate sample channel data."""
    K, N = 16, 16
    # Complex channel matrix
    H_real = np.random.randn(K, N)
    H_imag = np.random.randn(K, N)
    return H_real + 1j * H_imag


@pytest.fixture
def sample_model_input():
    """Generate sample model input tensor."""
    batch_size, M, K = 8, 4, 16
    return torch.randn(batch_size, M, K)


@pytest.fixture
def sample_labels():
    """Generate sample labels for classification."""
    batch_size, N = 8, 16
    return torch.randint(0, N, size=(batch_size,))


@pytest.fixture
def perfect_predictions():
    """Generate perfect prediction scores (batch_size x N)."""
    batch_size, N = 8, 16
    scores = torch.zeros(batch_size, N)
    # Make each sample have perfect score at correct index
    for i in range(batch_size):
        scores[i, i % N] = 1.0
    return scores


@pytest.fixture
def random_predictions():
    """Generate random prediction scores."""
    batch_size, N = 8, 16
    return torch.randn(batch_size, N)


@pytest.fixture(scope="session")
def device():
    """Get torch device for testing."""
    return torch.device("cpu")


@pytest.fixture
def seed():
    """Fixed random seed for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed
