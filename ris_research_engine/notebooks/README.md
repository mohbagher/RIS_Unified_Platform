# RIS Auto-Research Engine - Jupyter Notebooks

This directory contains interactive Jupyter notebooks for the RIS Auto-Research Engine.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -e .
   ```

2. **Run quickstart notebook:**
   ```bash
   jupyter notebook ris_research_engine/notebooks/01_quickstart.ipynb
   ```

## Notebooks

### 01_quickstart.ipynb
**Duration:** ~1-2 minutes

Minimal test to verify the system works:
- Install check
- Run a minimal experiment (N=16, K=16, M=4, 500 samples, 10 epochs)
- Display results and training curves
- Compare two probe types

### 02_dashboard.ipynb
**Interactive Dashboard**

Launch the 5-tab interactive interface:
- **Tab 1 (Configure):** Set parameters, select probes/models, manage experiment queue
- **Tab 2 (Run):** Execute experiments with real-time progress monitoring
- **Tab 3 (Results):** View experiment metrics and training curves
- **Tab 4 (Analysis):** Generate comparison plots and campaign analysis
- **Tab 5 (Database):** Browse, filter, and export experiment results

### 03_run_search.ipynb
**Duration:** Variable (2-30 minutes)

Automated search campaigns:
- Run quick test campaign (~2-5 minutes)
- Compare multiple probe types
- Cross-fidelity validation (optional, requires Sionna data)

### 04_analyze_results.ipynb
**Analysis Tool**

Analyze past experiments:
- Load experiments from database
- Compare probes and models
- Sparsity analysis (accuracy vs M/K ratio)
- Cross-fidelity gap analysis
- Export best configurations

## Usage Tips

1. **Start with 01_quickstart.ipynb** to verify your installation
2. **Use 02_dashboard.ipynb** for interactive experimentation
3. **Run 03_run_search.ipynb** for systematic studies
4. **Use 04_analyze_results.ipynb** to analyze past runs

## Configuration

The notebooks use default settings but can be customized:
- System parameters (N, K, M, SNR, frequency)
- Probe types (random_uniform, hadamard, sobol, dft_beams, etc.)
- Model architectures (mlp, cnn_1d, lstm, transformer, etc.)
- Training parameters (learning rate, epochs, batch size)
- Data sources (synthetic_rayleigh, synthetic_rician, hdf5_loader)

## Output

Results are stored in:
- `outputs/experiments/results.db` - SQLite database
- `outputs/plots/` - Generated figures
- `outputs/models/` - Saved model checkpoints (if enabled)

## Troubleshooting

**Import errors:** Run `pip install -e .` from the repository root

**Database locked:** Only one notebook should write to the database at a time

**Out of memory:** Reduce batch_size, n_samples, or model size

**Slow training:** Reduce epochs, n_samples, or use smaller N/K values for testing

## Examples

### Quick Test
```python
from ris_research_engine.ui import RISEngine

engine = RISEngine()
result = engine.run(
    probe="random_uniform",
    model="mlp",
    M=4, K=16, N=16,
    n_samples=500,
    epochs=10
)
engine.show(result)
```

### Compare Probes
```python
results = engine.compare_probes(
    probes=["random_uniform", "hadamard", "sobol"],
    model="mlp",
    M=8, K=64, N=64,
    n_samples=10000,
    epochs=50
)
engine.plot_comparison(results)
```

### Launch Dashboard
```python
from ris_research_engine.ui.jupyter_dashboard import RISDashboard

dashboard = RISDashboard()
dashboard.display()
```

## Citation

If you use this code in your research, please cite:
```
[Citation information to be added]
```
