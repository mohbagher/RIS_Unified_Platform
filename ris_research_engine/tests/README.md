# RIS Research Engine Test Suite

Complete test suite for the RIS Unified Platform research engine.

## Test Files

### 1. `conftest.py`
Pytest fixtures for test setup:
- Temporary database and directory fixtures
- Sample system, training, and experiment configurations
- Sample data generators (probes, channels, model inputs, labels)
- Device and seed fixtures

### 2. `test_probes.py`
Tests for all 7 probe types:
- **Probe types tested**: random_uniform, random_binary, hadamard, sobol, halton, dft_beams, learned
- **Shape tests**: Output dimensions (M, N)
- **Range tests**: Phases in [0, 2π)
- **Orthogonality tests**: Hadamard probe orthogonality properties
- **Reproducibility tests**: Same seed → same output
- **Parameter tests**: Default params and theoretical diversity

### 3. `test_models.py`
Tests for all 7 model types:
- **Model types tested**: mlp, residual_mlp, cnn_1d, cnn_2d, transformer, set_transformer, lstm
- **Shape tests**: Correct input/output dimensions
- **Forward pass tests**: Execution without errors, finite outputs
- **Gradient flow tests**: Backpropagation works correctly
- **Parameter tests**: Reasonable parameter counts
- **Architecture tests**: Model-specific structural verification

### 4. `test_metrics.py`
Tests for all 6 main metrics:
- **Metrics tested**: top_k_accuracy, top_1_accuracy, power_ratio, hit_at_l, mean_reciprocal_rank, spectral_efficiency
- **Perfect score tests**: Perfect predictions → score = 1.0
- **Random score tests**: Random predictions → score ≈ 1/K
- **Power ratio tests**: Values in [0, 1] range
- **Spectral efficiency tests**: Positive values, SNR relationship
- **Behavior tests**: Edge cases, batch consistency

### 5. `test_engine.py`
Tests for ExperimentRunner:
- **Tiny experiments**: N=8, K=8, M=2, 5 epochs for fast testing
- **Configuration tests**: System and training config validation
- **Progress tracking**: Callback functionality
- **Device handling**: CPU and auto device selection
- **Reproducibility**: Same seed consistency
- **Error handling**: Invalid configurations

### 6. `test_integration.py`
Integration tests:
- **quick_test.yaml**: End-to-end configuration loading and execution
- **Config validation**: System, training, and data configurations
- **Probe-model combinations**: Valid type combinations
- **Metrics configuration**: Primary and secondary metrics
- **Minimal config tests**: Absolute minimum viable configurations

### 7. `test_hdf5_loader.py`
Tests for HDF5 data loader:
- **Format detection**: Session5, AutoML, generic formats
- **Session5 loading**: Channel matrices, codebook, labels, powers
- **AutoML loading**: Train/val/test splits
- **Generic loading**: Any HDF5 structure
- **Auto-detection**: Automatic format identification
- **Error handling**: Nonexistent files, corrupted data
- **Data integrity**: No NaN/Inf, consistent shapes

## Running Tests

### Run all tests:
```bash
pytest ris_research_engine/tests/
```

### Run specific test file:
```bash
pytest ris_research_engine/tests/test_probes.py
```

### Run with verbose output:
```bash
pytest ris_research_engine/tests/ -v
```

### Run specific test class:
```bash
pytest ris_research_engine/tests/test_probes.py::TestProbeShape
```

### Run with coverage:
```bash
pytest ris_research_engine/tests/ --cov=ris_research_engine
```

## Test Statistics

- **Total test files**: 7
- **Total test functions**: 270+
- **Probe tests**: 71
- **Model tests**: 112
- **Metric tests**: 50
- **Engine tests**: 17
- **Integration tests**: 15
- **HDF5 tests**: 29

## Notes

- Tests use pytest fixtures for reusable setup
- Many tests skip gracefully if dependencies are missing
- Integration tests may skip if full environment not available
- Model tests may need API adjustments based on actual implementations
- Power ratio and spectral efficiency metrics may require metadata/configuration adjustments
