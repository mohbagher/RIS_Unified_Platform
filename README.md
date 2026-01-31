# RIS Auto-Research Engine

A modular, extensible Auto-Research Engine for PhD research on Reconfigurable Intelligent Surfaces (RIS) with ML-based configuration prediction from sparse probe measurements.

## Overview

This engine provides a systematic framework for exploring:
- **Probe types**: Random, Hadamard, Sobol, DFT beams, and more
- **ML models**: MLP, CNN, Transformer, LSTM architectures
- **Sparsity levels**: How performance degrades as M/K decreases
- **Cross-fidelity validation**: Validate synthetic results on ray-traced data

## Quick Start

```bash
# Install
pip install -e .

# Run quick test
python -m ris_research_engine.ui.cli search --config configs/search_spaces/quick_test.yaml

# Or use Jupyter
jupyter notebook notebooks/01_quickstart.ipynb
```

## Project Structure

```
ris_research_engine/
├── foundation/          # Core utilities (data types, math, storage)
├── plugins/             # Swappable components (probes, models, metrics, search)
├── engine/              # Orchestration (experiment runner, search controller)
├── ui/                  # User interfaces (dashboard, CLI)
├── configs/             # YAML configuration files
├── notebooks/           # Jupyter notebooks
└── tests/               # Unit tests
```

## License

MIT License