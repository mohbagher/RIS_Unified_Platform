# RIS Auto-Research Engine

A modular, extensible Auto-Research Engine for PhD research on Reconfigurable Intelligent Surfaces (RIS) with ML-based configuration prediction from sparse probe measurements.

## Overview

This engine provides a systematic framework for exploring:
- **Probe types**: Random, Hadamard, Sobol, DFT beams, and more (7 types)
- **ML models**: MLP, CNN, Transformer, LSTM architectures (7 models)
- **Sparsity levels**: How performance degrades as M/K decreases
- **Cross-fidelity validation**: Validate synthetic results on ray-traced Sionna data
- **Automated search**: Grid search, random search, successive halving, scientific search

## Installation

```bash
# Clone repository
git clone https://github.com/mohbagher/RIS_Unified_Platform.git
cd RIS_Unified_Platform

# Install dependencies
pip install -e .
```

## Quick Start

### 1. Simple Python Script

```python
from ris_research_engine import RISEngine

# Initialize engine
engine = RISEngine(db_path="results.db")

# Run a single experiment
result = engine.run(
    probe='hadamard',
    model='mlp',
    M=8,
    K=64,
    epochs=50
)

# Display results
engine.show(result)
```

### 2. Command-Line Interface

```bash
# Run a single experiment
python -m ris_research_engine.ui.cli run \
    --probe hadamard \
    --model mlp \
    --M 8 \
    --K 64 \
    --epochs 50

# Run automated search
python -m ris_research_engine.ui.cli search \
    --config configs/search_spaces/quick_test.yaml

# List experiments
python -m ris_research_engine.ui.cli list --limit 20

# Compare probes
python -m ris_research_engine.ui.cli compare \
    --type probes \
    --metric top_1_accuracy

# Generate plots
python -m ris_research_engine.ui.cli plot \
    --type all \
    --output-dir outputs/
```

### 3. Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks:
# - notebooks/01_quickstart.ipynb - Basic usage
# - notebooks/02_dashboard.ipynb - Interactive dashboard
# - notebooks/03_run_search.ipynb - Automated searches
# - notebooks/04_analyze_results.ipynb - Results analysis
```

### 4. Example Script

```bash
# Run the example script
python example.py
```

## Project Structure

```
ris_research_engine/
├── foundation/          # Core utilities (data types, math, storage)
├── plugins/             # Swappable components (probes, models, metrics, search)
│   ├── probes/         # 7 probe generation strategies
│   ├── models/         # 7 neural network architectures
│   ├── metrics/        # 14 evaluation metrics
│   ├── data_sources/   # 3 data sources (synthetic, HDF5)
│   ├── baselines/      # 4 baseline methods
│   └── search/         # 4 search strategies
├── engine/              # Orchestration (experiment runner, search controller)
│   ├── experiment_runner.py    # Run individual experiments
│   ├── search_controller.py    # Automated search campaigns
│   ├── scientific_rules.py     # Rule-based experiment control
│   ├── result_analyzer.py      # Statistical analysis
│   └── report_generator.py     # Publication-quality plots
├── ui/                  # User interfaces
│   ├── jupyter_minimal.py      # Simple API for notebooks
│   ├── jupyter_dashboard.py    # Interactive dashboard
│   └── cli.py                  # Command-line interface
├── configs/             # YAML configuration files
│   ├── presets/                # Default parameters
│   ├── search_spaces/          # Search configurations
│   └── scientific_rules/       # Experiment rules
├── notebooks/           # Jupyter notebooks
└── tests/               # Unit tests
```

## Key Features

### 1. Plugin System
- **7 Probe Types**: random_uniform, random_binary, hadamard, sobol, halton, dft_beams, learned_probe
- **7 Models**: mlp, residual_mlp, cnn_1d, cnn_2d, transformer, set_transformer, lstm
- **14 Metrics**: top-k accuracy, power ratio, hit@L, MRR, spectral efficiency, inference time
- **4 Search Strategies**: grid_search, random_search, successive_halving, scientific_search
- **4 Baselines**: random_selection, best_of_probed, exhaustive_search, strongest_beam

### 2. Experiment Orchestration
- Automated data loading/generation
- Probe application and measurement selection
- Model training with early stopping and LR scheduling
- Comprehensive evaluation with multiple metrics
- Baseline method comparison
- Result persistence in SQLite database

### 3. Automated Search
- YAML-based configuration
- Multiple search strategies
- Scientific rules (abandon, early_stop, promote, compare)
- Progress tracking and callbacks
- Cross-fidelity validation

### 4. Analysis & Visualization
- Statistical comparisons (probes, models, sparsity)
- 9 plot types: bar charts, curves, heatmaps, Pareto fronts, distributions
- Publication-quality output (PNG 300 DPI + PDF)
- Comprehensive statistical summaries

## Configuration Files

### Presets
- `configs/presets/default_system.yaml` - Default RIS parameters
- `configs/presets/default_training.yaml` - Default training parameters

### Search Spaces
- `configs/search_spaces/probe_comparison.yaml` - Compare 6 probes
- `configs/search_spaces/sparsity_sweep.yaml` - Test M=[4,8,16,32,48]
- `configs/search_spaces/model_comparison.yaml` - Compare 6 models
- `configs/search_spaces/full_search.yaml` - Comprehensive search
- `configs/search_spaces/quick_test.yaml` - Fast validation
- `configs/search_spaces/cross_fidelity_validation.yaml` - Sionna validation

### Scientific Rules
- `configs/scientific_rules/early_stopping.yaml` - Stop poorly performing runs
- `configs/scientific_rules/pruning_rules.yaml` - Prune configurations
- `configs/scientific_rules/promotion_rules.yaml` - Promote promising configs

## Documentation

- `IMPLEMENTATION.md` - Detailed implementation documentation
- `notebooks/` - Interactive tutorials and examples
- API documentation in docstrings

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0 (for ML models)
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn (for plotting)
- PyYAML (for configs)
- ipywidgets (for dashboard)
- h5py (for HDF5 data)
- scikit-learn

See `requirements.txt` for full list.

## CLI Commands

```bash
# Run experiment
cli run --probe <type> --model <type> --M <int> --K <int>

# Automated search
cli search --config <yaml>

# Cross-fidelity validation
cli validate --campaign <name> --hdf5-path <path>

# List experiments
cli list [--campaign <name>] [--status <status>]

# Compare experiments
cli compare --type <probes|models|sparsity> [--campaign <name>]

# Generate plots
cli plot --type <probes|models|sparsity|heatmap|pareto|all>

# Export results
cli export --output <csv> [--campaign <name>]

# List plugins
cli plugins
```

## License

MIT License