# RIS Auto-Research Engine Configuration Files

This directory contains 11 production-ready YAML configuration files that define search spaces, scientific rules, and default presets for automated RIS beam prediction experiments.

## 📁 Directory Structure

```
configs/
├── search_spaces/          # Search space definitions (6 files)
│   ├── probe_comparison.yaml
│   ├── sparsity_sweep.yaml
│   ├── model_comparison.yaml
│   ├── full_search.yaml
│   ├── quick_test.yaml
│   └── cross_fidelity_validation.yaml
│
├── scientific_rules/       # Scientific automation rules (3 files)
│   ├── early_stopping.yaml
│   ├── pruning_rules.yaml
│   └── promotion_rules.yaml
│
└── presets/                # Default configurations (2 files)
    ├── default_system.yaml
    └── default_training.yaml
```

## 🔬 Search Spaces

### 1. probe_comparison.yaml
**Purpose:** Systematic comparison of 6 probe design strategies

- **Probes:** random_uniform, random_binary, hadamard, dft_beams, sobol, halton
- **System:** N=64, K=64, M=8
- **Experiments:** 18 (6 probes × 3 seeds)
- **Strategy:** grid_search
- **Runtime:** ~30-60 minutes

**Research Question:** Which probe type provides the most informative training signals?

```bash
python -m ris_research_engine.engine.search_controller \
    --config configs/search_spaces/probe_comparison.yaml
```

### 2. sparsity_sweep.yaml
**Purpose:** Explore measurement sparsity vs. accuracy trade-off

- **Sparsity Levels:** M = [4, 8, 16, 32, 48]
- **Probes:** hadamard, dft_beams, sobol (top 3 from comparison)
- **Experiments:** 45 (5 sparsity × 3 probes × 3 seeds)
- **Strategy:** grid_search
- **Runtime:** ~1-2 hours

**Research Question:** How does measurement overhead affect prediction accuracy?

### 3. model_comparison.yaml
**Purpose:** Compare 6 neural network architectures

- **Models:** MLP, CNN-1D, CNN-2D, LSTM, Transformer, Set Transformer
- **Probe:** hadamard (best from comparison)
- **Experiments:** 18 (6 models × 3 seeds)
- **Strategy:** grid_search
- **Runtime:** ~1-2 hours

**Research Question:** Which architecture best captures probe-to-beam relationships?

### 4. full_search.yaml
**Purpose:** Complete 4-phase scientific search

- **Phase 1:** Probe screening (10 epochs, all probes)
- **Phase 2:** Model selection (best 2 probes, all models, 50 epochs)
- **Phase 3:** Hyperparameter tuning (Bayesian optimization, 100 epochs)
- **Phase 4:** Sparsity validation (best config, M=[4,8,16,32,48])
- **Budget:** Max 200 experiments
- **Strategy:** scientific_search
- **Runtime:** 6-12 hours

**Research Question:** What is the optimal end-to-end configuration?

### 5. quick_test.yaml
**Purpose:** Fast validation for testing and debugging

- **System:** N=16, K=16, M=4 (reduced scale)
- **Experiments:** 2 (hadamard+mlp, random_uniform+mlp)
- **Epochs:** 10 only
- **Runtime:** 2-3 minutes

**Use Case:** Pipeline testing, CI/CD validation, quick sanity checks

### 6. cross_fidelity_validation.yaml
**Purpose:** Validate synthetic results on Sionna ray-tracing data

- **Data Source:** HDF5 file from Sionna simulations
- **Configs:** Top 3 from previous experiments
- **Experiments:** 3 (top 3 configs × 3 seeds)
- **Strategy:** grid_search
- **Runtime:** Depends on HDF5 data size

**Research Question:** Do synthetic insights transfer to realistic scenarios?

## 🧪 Scientific Rules

### 1. early_stopping.yaml
**Purpose:** Rules for abandoning and early-stopping experiments

**Abandon Rules:**
- Below random baseline (accuracy < 1/K)
- Training diverged (loss > 10.0)
- Invalid loss (NaN/Inf)
- Stuck at low accuracy

**Early Stop Rules:**
- Validation loss converged (no improvement for 10 epochs)
- Overfitting detected (train/val divergence for 15 epochs)
- Accuracy plateaued

**Configuration:**
```yaml
early_stopping:
  enabled: true
  patience: 15
  min_delta: 0.001
```

### 2. pruning_rules.yaml
**Purpose:** Rules for pruning inferior configurations

**Comparison Rules:**
- Clear loser (15% accuracy gap from best)
- Unlikely to catch up (trajectory analysis)
- Consistently worse (over 5 checkpoints)

**Efficiency Rules:**
- Prefer simplicity (similar accuracy, 2x more parameters)
- Diminishing returns (marginal gain, 3x slower)

**Statistical Rules:**
- Statistically significantly worse (t-test, p < 0.05)
- Less stable (similar mean, 2x higher variance)

### 3. promotion_rules.yaml
**Purpose:** Rules for promoting promising configurations

**Performance Rules:**
- New best found (+5% improvement)
- Rapid learner (2% improvement per epoch)
- Exceeds expectations

**Scientific Discovery Rules:**
- Unexpected winner (random probe achieving >70%)
- Pareto frontier point (accuracy vs. efficiency)
- Robust performer (low variance across seeds)

**Resource Allocation:**
- Extend training (still improving near max_epochs)
- Hyperparameter tuning (high potential)

## ⚙️ Presets

### 1. default_system.yaml
**Purpose:** Standard RIS system configuration

**Key Parameters:**
- **RIS:** N=64 (8×8), λ/2 spacing, 2-bit phase control
- **Beams:** K=64, DFT codebook
- **Measurements:** M=8, Hadamard default
- **RF:** 28 GHz, 20 dB SNR, 100 MHz bandwidth
- **Channel:** 5-path geometric model
- **Scenario:** Urban micro, 10-100m distance

### 2. default_training.yaml
**Purpose:** Standard training hyperparameters

**Key Parameters:**
- **Optimizer:** Adam, lr=0.001
- **Scheduler:** Cosine annealing, 5-epoch warmup
- **Training:** 100 epochs, batch=64
- **Regularization:** weight_decay=1e-5, dropout=0.1
- **Early Stopping:** patience=15, min_delta=0.001
- **Data:** 70/15/15 train/val/test split

## 🚀 Usage Examples

### Quick Start: Run a Simple Comparison
```bash
# Test the pipeline (2-3 minutes)
python -m ris_research_engine.engine.search_controller \
    --config configs/search_spaces/quick_test.yaml

# Compare probe types (30-60 minutes)
python -m ris_research_engine.engine.search_controller \
    --config configs/search_spaces/probe_comparison.yaml
```

### Advanced: Full Scientific Search
```bash
# Run complete 4-phase search (6-12 hours)
python -m ris_research_engine.engine.search_controller \
    --config configs/search_spaces/full_search.yaml \
    --output_dir outputs/full_search_run_1 \
    --verbose
```

### Custom: Override Parameters
```bash
# Run probe comparison with custom system size
python -m ris_research_engine.engine.search_controller \
    --config configs/search_spaces/probe_comparison.yaml \
    --override system.N=128 system.K=128 system.M=16 \
    --output_dir outputs/large_system
```

### Validation: Cross-Fidelity
```bash
# First generate Sionna data
python -m ris_research_engine.foundation.sionna_data_generator \
    --output outputs/sionna_ris_data.h5 \
    --num_samples 10000

# Then validate top configs
python -m ris_research_engine.engine.search_controller \
    --config configs/search_spaces/cross_fidelity_validation.yaml \
    --source_db results.db \
    --source_experiment probe_comparison
```

## 📊 Loading Configs in Python

```python
import yaml

# Load search space
with open('configs/search_spaces/probe_comparison.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"Strategy: {config['strategy']}")
print(f"Probes: {config['probes']['type']}")
print(f"System: N={config['system']['N']}, K={config['system']['K']}")

# Load scientific rules
with open('configs/scientific_rules/early_stopping.yaml', 'r') as f:
    rules = yaml.safe_load(f)

print(f"Abandon rules: {len(rules['abandon_rules'])}")
print(f"Early stop rules: {len(rules['early_stop_rules'])}")

# Load presets
with open('configs/presets/default_system.yaml', 'r') as f:
    defaults = yaml.safe_load(f)

print(f"Default RIS: {defaults['ris']['N']} elements")
print(f"Default frequency: {defaults['rf']['frequency']/1e9} GHz")
```

## 🔧 Creating Custom Configs

### Minimal Search Space Template
```yaml
name: "my_custom_search"
description: "Custom experiment description"
strategy: "grid_search"  # or "scientific_search"

system:
  N: 64
  K: 64
  M: 8
  frequency: 28e9
  snr_db: 20.0

probes:
  type: ["hadamard", "dft_beams"]

model:
  type: "mlp"
  hidden_dims: [256, 128, 64]

training:
  learning_rate: 0.001
  batch_size: 64
  max_epochs: 100

data:
  num_samples: 10000
  data_source: "synthetic"

random_seeds: [42, 43, 44]

metrics:
  primary: "top_1_accuracy"
  secondary: ["top_5_accuracy", "val_loss"]
```

### Extending Default Presets
```python
import yaml

# Load defaults
with open('configs/presets/default_system.yaml', 'r') as f:
    system = yaml.safe_load(f)

with open('configs/presets/default_training.yaml', 'r') as f:
    training = yaml.safe_load(f)

# Modify as needed
system['ris']['N'] = 128  # Larger RIS
training['training']['max_epochs'] = 200  # More epochs

# Use in your experiment
my_config = {
    'name': 'my_experiment',
    'system': system,
    'training': training,
    # ... other fields
}
```

## 📈 Experiment Tracking

All experiments log to:
- **SQLite Database:** `results.db` (searchable, queryable)
- **TensorBoard:** `outputs/tensorboard/` (training curves)
- **Checkpoints:** `outputs/checkpoints/` (model weights)
- **Reports:** `outputs/reports/` (JSON summaries)

Query results:
```python
import sqlite3

conn = sqlite3.connect('results.db')
cursor = conn.cursor()

# Get best probe type
cursor.execute("""
    SELECT probe_type, AVG(top_1_accuracy) as avg_acc
    FROM experiments
    WHERE experiment_name = 'probe_comparison'
    GROUP BY probe_type
    ORDER BY avg_acc DESC
""")

for row in cursor.fetchall():
    print(f"{row[0]}: {row[1]:.3f}")
```

## 🎯 Recommended Workflow

1. **Quick Test:** Run `quick_test.yaml` to validate pipeline (~3 min)
2. **Probe Selection:** Run `probe_comparison.yaml` to find best probe (~1 hour)
3. **Sparsity Analysis:** Run `sparsity_sweep.yaml` with top probes (~2 hours)
4. **Model Selection:** Run `model_comparison.yaml` with best probe+sparsity (~2 hours)
5. **Full Search:** Run `full_search.yaml` for complete optimization (~12 hours)
6. **Validation:** Run `cross_fidelity_validation.yaml` on Sionna data (~2 hours)

**Total Time:** ~20 hours for complete scientific investigation

## 🛠️ Troubleshooting

### Out of Memory
```yaml
# Reduce batch size
training:
  batch_size: 32  # or 16

# Or reduce model size
model:
  hidden_dims: [128, 64]  # smaller
```

### Too Slow
```yaml
# Use quick_test config
# Or reduce epochs
training:
  max_epochs: 50

# Or reduce samples
data:
  num_samples: 5000
```

### YAML Syntax Errors
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('your_config.yaml'))"
```

## 📚 References

- **Search Space Design:** See `docs/search_space_design.md`
- **Scientific Rules:** See `docs/scientific_rules.md`
- **API Documentation:** See `docs/api_reference.md`

## 🤝 Contributing

To add new configurations:

1. Follow existing naming conventions
2. Include comprehensive comments
3. Validate YAML syntax
4. Test with `quick_test.yaml` first
5. Document in this README

## 📝 Statistics

- **Total Files:** 11
- **Total Lines:** 2,048
- **Total Size:** ~60 KB
- **Search Spaces:** 6
- **Scientific Rules:** 3
- **Presets:** 2

---

**Last Updated:** 2024-02-01  
**Version:** 1.0.0  
**Maintainer:** RIS Research Team
