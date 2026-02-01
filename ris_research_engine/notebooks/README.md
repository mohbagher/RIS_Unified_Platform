# RIS Auto-Research Engine - Jupyter Notebooks

This directory contains 4 comprehensive Jupyter notebooks that guide you through using the RIS Auto-Research Engine for automated experiment management and analysis.

## Notebooks Overview

### 1. `01_quickstart.ipynb` - Quick Start Guide
**Duration:** ~5 minutes  
**Purpose:** Fast introduction and installation verification

**What you'll learn:**
- Verify installation and check dependencies
- Initialize the RIS Engine
- Run your first experiment campaign (quick_test.yaml)
- Visualize training curves
- Compare probe performance
- Understand baseline metrics

**Best for:** First-time users, installation testing, getting familiar with the API

---

### 2. `02_dashboard.ipynb` - Interactive Dashboard
**Duration:** Interactive session  
**Purpose:** GUI-based experiment management

**What you'll learn:**
- Launch the interactive 5-tab dashboard
- Configure experiments with sliders and dropdowns
- Queue and run multiple experiments
- Monitor real-time progress
- View and compare results interactively
- Export data through the GUI

**Best for:** Interactive exploration, parameter tuning, users who prefer GUIs over code

**Features:**
- Tab 1: Configure - Set parameters with intuitive widgets
- Tab 2: Run - Execute experiments and monitor progress
- Tab 3: Results - Browse individual experiment results
- Tab 4: Analyze - Compare multiple experiments
- Tab 5: Export - Save results in various formats

---

### 3. `03_run_search.ipynb` - Run Search Campaigns
**Duration:** Varies (2 min - several hours depending on campaign)  
**Purpose:** Execute comprehensive search campaigns

**What you'll learn:**
- List and explore available search space configurations
- Run multi-experiment campaigns (probe_comparison, sparsity_sweep, etc.)
- Compute statistical summaries across random seeds
- Generate publication-quality comparison plots
- Analyze sparsity vs performance trade-offs
- Export results to CSV for external analysis

**Best for:** Running systematic studies, comparing multiple configurations, production experiments

**Available Campaigns:**
- `quick_test.yaml` - 2 experiments (~2 min)
- `probe_comparison.yaml` - 18 experiments (~15 min)
- `model_comparison.yaml` - 9 experiments (~10 min)
- `sparsity_sweep.yaml` - 45 experiments (~30 min)
- `full_search.yaml` - 200+ experiments (several hours)

---

### 4. `04_analyze_results.ipynb` - Comprehensive Analysis
**Duration:** ~10 minutes  
**Purpose:** Deep dive into experimental results

**What you'll learn:**
- Query and filter the experiment database
- Compute statistical summaries and confidence intervals
- Identify top-performing configurations
- Generate comparison plots (probes, models, Pareto fronts)
- Perform statistical significance tests (t-tests, effect sizes)
- Analyze fidelity gaps and scaling behavior
- Export best configurations for deployment

**Best for:** Post-experiment analysis, statistical validation, report generation, publication prep

**Outputs:**
- Detailed statistical tables
- Publication-ready plots
- Best configuration JSON export
- Actionable recommendations

---

## Quick Start

### Prerequisites
```bash
# From repository root
pip install -e .

# Optional: for statistical tests
pip install scipy

# Optional: for better plots
pip install seaborn
```

### Launch Jupyter
```bash
cd ris_research_engine/notebooks
jupyter notebook
```

### Recommended Order
1. **Start here:** `01_quickstart.ipynb` - Verify everything works
2. **Try the GUI:** `02_dashboard.ipynb` - Interactive experience
3. **Run campaigns:** `03_run_search.ipynb` - Execute systematic studies
4. **Analyze results:** `04_analyze_results.ipynb` - Deep analysis and reporting

## Notebook Features

All notebooks include:
- ✓ **Markdown explanations** - 2-3 sentences per cell explaining what it does
- ✓ **Code comments** - Inline documentation for complex operations
- ✓ **Expected outputs** - Guidance on what to expect
- ✓ **Error handling** - Graceful handling of missing data or dependencies
- ✓ **Progress bars** - Visual feedback for long-running operations
- ✓ **Visualizations** - Inline plots with matplotlib
- ✓ **Self-contained** - Can be run top-to-bottom without errors

## File Structure

```
notebooks/
├── README.md                     # This file
├── 01_quickstart.ipynb          # Quick start (5 min)
├── 02_dashboard.ipynb           # Interactive GUI
├── 03_run_search.ipynb          # Run campaigns (15-30 min)
└── 04_analyze_results.ipynb     # Comprehensive analysis (10 min)
```

## Common Use Cases

### Use Case 1: Quick Validation
"I just installed the package and want to verify it works"
→ Run `01_quickstart.ipynb`

### Use Case 2: Interactive Exploration
"I want to explore different configurations interactively"
→ Use `02_dashboard.ipynb`

### Use Case 3: Systematic Comparison
"I need to compare different probe designs systematically"
→ Run `03_run_search.ipynb` with `probe_comparison.yaml`

### Use Case 4: Publication Analysis
"I need statistical analysis and plots for a paper"
→ Use `04_analyze_results.ipynb` after running campaigns

### Use Case 5: Production Deployment
"I need to find the best configuration for deployment"
1. Run `03_run_search.ipynb` with appropriate campaign
2. Analyze with `04_analyze_results.ipynb`
3. Export best configuration JSON
4. Deploy in production

## Tips and Best Practices

### Performance
- Start with `quick_test.yaml` before running larger campaigns
- Use smaller N, K, M values for initial exploration
- Set fewer epochs (10-20) for parameter sweeps
- Run full-scale experiments only when needed

### Organization
- Results are automatically saved to `results.db`
- Plots are saved to `outputs/` directory
- All experiments are timestamped
- Use meaningful campaign names

### Debugging
- Check `status` field in database for failed experiments
- Review console output for error messages
- Use `verbose=True` in configs for detailed logging
- Start with small batch sizes if memory issues occur

### Reproducibility
- Set `random_seeds` in YAML configs for reproducibility
- Export configurations with results
- Document system specifications for benchmarking
- Save complete experiment metadata

## Troubleshooting

### "Module not found" errors
```bash
# Make sure package is installed in development mode
pip install -e .
```

### "No experiments found" in analysis notebook
```bash
# Run some experiments first
# Use 01_quickstart.ipynb or 03_run_search.ipynb
```

### Jupyter widgets not displaying
```bash
# Install and enable ipywidgets
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Plots not showing
```bash
# Make sure matplotlib backend is correct
# Add to first cell: %matplotlib inline
```

### Database locked errors
```bash
# Close other connections to results.db
# Only one notebook should access database at a time
```

## Advanced Usage

### Custom Campaigns
Create your own YAML configs in `../configs/search_spaces/` and run them:
```python
engine.search("../configs/search_spaces/my_custom_campaign.yaml")
```

### Programmatic API
Use the engine directly without notebooks:
```python
from ris_research_engine.ui import RISEngine
engine = RISEngine()
result = engine.run(probe='hadamard', model='mlp', M=8, K=64)
engine.show(result)
```

### Batch Processing
Run multiple campaigns in sequence:
```python
campaigns = ['probe_comparison', 'model_comparison', 'sparsity_sweep']
for campaign in campaigns:
    engine.search(f"../configs/search_spaces/{campaign}.yaml")
```

### Custom Analysis
Access the database directly for custom queries:
```python
import sqlite3
conn = sqlite3.connect("results.db")
df = pd.read_sql_query("SELECT * FROM experiments WHERE status='completed'", conn)
# Your custom analysis here
```

## Support

- **Documentation:** See `docs/` directory in repository root
- **Examples:** Check `examples/` directory for code examples
- **Issues:** Report bugs on GitHub Issues
- **API Reference:** See `docs/api/` for detailed API documentation

## Citation

If you use these notebooks in your research, please cite:
```bibtex
@software{ris_research_engine,
  title={RIS Auto-Research Engine},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/RIS_Unified_Platform}
}
```

## License

[Your License Here]

---

**Happy Researching! 🚀**

For questions or feedback, please open an issue on GitHub.
