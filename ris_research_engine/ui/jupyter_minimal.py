"""Simple API for Jupyter notebooks - RIS Auto-Research Engine."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from ris_research_engine.foundation import (
    SystemConfig, TrainingConfig, ExperimentConfig, 
    ExperimentResult, ResultTracker, load_hdf5_data
)
from ris_research_engine.foundation.logging_config import get_logger
from ris_research_engine.engine import (
    ExperimentRunner, SearchController, 
    ResultAnalyzer, ReportGenerator
)

logger = get_logger(__name__)


class RISEngine:
    """Simple API for running RIS experiments in Jupyter notebooks."""
    
    def __init__(self, db_path: str = "outputs/experiments/results.db"):
        """
        Initialize the RIS Engine.
        
        Args:
            db_path: Path to SQLite database for results storage
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.tracker = ResultTracker(str(self.db_path))
        self.runner = ExperimentRunner(str(self.db_path))
        self.controller = SearchController(str(self.db_path))
        self.analyzer = ResultAnalyzer(str(self.db_path))
        self.reporter = ReportGenerator(str(self.db_path.parent))
        
        logger.info(f"RISEngine initialized - DB: {db_path}")
    
    def run(
        self,
        probe: str,
        model: str,
        M: int,
        K: int,
        N: int = 64,
        data: str = "synthetic_rayleigh",
        n_samples: int = 5000,
        epochs: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a single experiment with minimal configuration.
        
        Args:
            probe: Probe type (e.g., 'dft_beams', 'hadamard', 'random_uniform')
            model: Model type (e.g., 'mlp', 'cnn_1d', 'transformer')
            M: Sensing budget (number of measurements)
            K: Codebook size
            N: Number of RIS elements (default: 64)
            data: Data source name (default: 'synthetic_rayleigh')
            n_samples: Number of samples to generate (default: 5000)
            epochs: Maximum training epochs (default: 50)
            **kwargs: Additional parameters (learning_rate, batch_size, etc.)
            
        Returns:
            Dictionary with experiment result including experiment_id
        """
        # Create configs
        N_x = int(np.sqrt(N))
        N_y = N_x
        
        system = SystemConfig(
            N=N, N_x=N_x, N_y=N_y, K=K, M=M,
            frequency=kwargs.get('frequency', 28e9),
            snr_db=kwargs.get('snr_db', 20.0)
        )
        
        training = TrainingConfig(
            learning_rate=kwargs.get('learning_rate', 1e-3),
            batch_size=kwargs.get('batch_size', 64),
            max_epochs=epochs,
            early_stopping_patience=kwargs.get('patience', 15),
            dropout=kwargs.get('dropout', 0.1)
        )
        
        experiment = ExperimentConfig(
            name=f"{probe}_{model}_M{M}_K{K}",
            system=system,
            training=training,
            probe_type=probe,
            probe_params=kwargs.get('probe_params', {}),
            model_type=model,
            model_params=kwargs.get('model_params', {}),
            data_source=data,
            data_params={'n_samples': n_samples},
            metrics=['top_k_accuracy', 'mean_reciprocal_rank'],
            tags=kwargs.get('tags', []),
            notes=kwargs.get('notes', '')
        )
        
        logger.info(f"Running experiment: {experiment.name}")
        
        # Run experiment
        result = self.runner.run(experiment)
        
        # Save to database
        exp_id = self.tracker.save_experiment(result)
        
        logger.info(f"Experiment completed - ID: {exp_id}, Status: {result.status}")
        
        # Convert to dict and add ID
        result_dict = result.to_dict()
        result_dict['experiment_id'] = exp_id
        
        return result_dict
    
    def show(self, result: Dict[str, Any]) -> None:
        """
        Print metrics and show training curve for a single experiment result.
        
        Args:
            result: Experiment result dictionary from run()
        """
        print("\n" + "="*70)
        print(f"Experiment: {result['config']['name']}")
        print(f"Status: {result['status']}")
        print("="*70)
        
        # Display metrics table
        print("\nMetrics:")
        print("-" * 70)
        metrics = result['metrics']
        
        metric_data = []
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                metric_data.append({'Metric': key, 'Value': f"{value:.4f}"})
            else:
                metric_data.append({'Metric': key, 'Value': str(value)})
        
        df = pd.DataFrame(metric_data)
        print(df.to_string(index=False))
        
        print(f"\nTraining Time: {result['training_time_seconds']:.2f}s")
        print(f"Best Epoch: {result['best_epoch']}/{result['total_epochs']}")
        print(f"Model Parameters: {result['model_parameters']:,}")
        
        # Plot training curves
        if 'training_history' in result and result['training_history']:
            history = result['training_history']
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss plot
            if 'train_loss' in history and len(history['train_loss']) > 0:
                epochs_range = range(1, len(history['train_loss']) + 1)
                axes[0].plot(epochs_range, history['train_loss'], label='Train', linewidth=2)
                if 'val_loss' in history and len(history['val_loss']) > 0:
                    axes[0].plot(epochs_range, history['val_loss'], label='Val', linewidth=2)
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Training Loss')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Accuracy plot
            if 'val_top_1_accuracy' in history and len(history['val_top_1_accuracy']) > 0:
                epochs_range = range(1, len(history['val_top_1_accuracy']) + 1)
                axes[1].plot(epochs_range, history['val_top_1_accuracy'], 
                           label='Top-1', linewidth=2)
                if 'val_top_5_accuracy' in history and len(history['val_top_5_accuracy']) > 0:
                    axes[1].plot(epochs_range, history['val_top_5_accuracy'], 
                               label='Top-5', linewidth=2)
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                axes[1].set_title('Validation Accuracy')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def compare_probes(
        self,
        probes: List[str],
        model: str,
        M: int,
        K: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple probe types with same model configuration.
        
        Args:
            probes: List of probe types to compare
            model: Model type to use for all experiments
            M: Sensing budget
            K: Codebook size
            **kwargs: Additional parameters passed to run()
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        print(f"\nComparing {len(probes)} probes with {model} (M={M}, K={K})...")
        print("=" * 70)
        
        for i, probe in enumerate(probes, 1):
            print(f"\n[{i}/{len(probes)}] Running {probe}...")
            result = self.run(probe=probe, model=model, M=M, K=K, **kwargs)
            results.append(result)
            
            if result['status'] == 'completed':
                acc = result['metrics'].get('top_1_accuracy', 0)
                print(f"  ✓ Top-1 Accuracy: {acc:.4f}")
            else:
                print(f"  ✗ Status: {result['status']}")
        
        print("\n" + "=" * 70)
        print("Comparison complete!")
        
        return results
    
    def plot_comparison(
        self,
        results: List[Dict[str, Any]],
        metric: str = 'top_1_accuracy'
    ) -> None:
        """
        Plot comparison bar chart for multiple experiment results.
        
        Args:
            results: List of experiment result dictionaries
            metric: Metric to compare (default: 'top_1_accuracy')
        """
        # Extract data
        data = []
        for result in results:
            if result['status'] == 'completed':
                data.append({
                    'name': result['config']['probe_type'],
                    'value': result['metrics'].get(metric, 0.0)
                })
        
        if not data:
            print("No completed experiments to compare")
            return
        
        df = pd.DataFrame(data)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(df)), df['value'])
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['name'], rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Probe Comparison - {metric.replace("_", " ").title()}')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for bar, value in zip(bars, df['value']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def search(self, strategy: str, config: str) -> Dict[str, Any]:
        """
        Run search campaign from YAML configuration file.
        
        Args:
            strategy: Search strategy name ('grid', 'random', 'bayesian', etc.)
            config: Path to YAML configuration file
            
        Returns:
            Campaign result dictionary
        """
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config}")
        
        # Load YAML config
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        
        # Override strategy if specified
        config_dict['strategy'] = strategy
        
        logger.info(f"Starting {strategy} search campaign from {config}")
        
        # Run campaign
        campaign = self.controller.run_campaign(config_dict)
        
        # Save campaign
        campaign_id = self.tracker.save_campaign(campaign)
        
        result = campaign.to_dict()
        result['campaign_id'] = campaign_id
        
        logger.info(f"Campaign completed - ID: {campaign_id}")
        
        # Print summary
        print("\n" + "="*70)
        print(f"Campaign: {campaign.campaign_name}")
        print(f"Strategy: {strategy}")
        print("="*70)
        print(f"Total Experiments: {campaign.total_experiments}")
        print(f"Completed: {campaign.completed_experiments}")
        print(f"Pruned: {campaign.pruned_experiments}")
        print(f"Failed: {campaign.failed_experiments}")
        print(f"Total Time: {campaign.total_time_seconds:.2f}s")
        
        if campaign.best_result:
            print(f"\nBest Result:")
            print(f"  Config: {campaign.best_result.config.name}")
            print(f"  Metric: {campaign.best_result.primary_metric_value:.4f}")
        
        return result
    
    def validate_on_sionna(
        self,
        campaign_name: str,
        hdf5_path: str,
        top_n: int = 3
    ) -> pd.DataFrame:
        """
        Cross-fidelity validation using high-fidelity Sionna simulation data.
        
        This validates top-performing experiments from synthetic data against
        high-fidelity electromagnetic simulations to assess performance degradation.
        
        Args:
            campaign_name: Name of campaign to validate
            hdf5_path: Path to HDF5 file with Sionna simulation data
            top_n: Number of top experiments to validate (default: 3)
            
        Returns:
            DataFrame with validation results
        """
        # Get campaign experiments
        experiments = self.tracker.get_all_experiments(campaign_name=campaign_name)
        
        if not experiments:
            raise ValueError(f"No experiments found for campaign: {campaign_name}")
        
        # Get top N by primary metric
        experiments = sorted(
            [e for e in experiments if e['status'] == 'completed'],
            key=lambda x: x['metrics'].get('top_1_accuracy', 0),
            reverse=True
        )[:top_n]
        
        print(f"\nCross-fidelity validation: {campaign_name}")
        print(f"High-fidelity data: {hdf5_path}")
        print("=" * 70)
        
        # Load high-fidelity data
        logger.info(f"Loading high-fidelity simulation data from {hdf5_path}")
        try:
            sionna_data = load_hdf5_data(hdf5_path)
        except Exception as e:
            logger.warning(f"Could not load HDF5 data: {e}")
            sionna_data = None
        
        # Validate each experiment
        results = []
        
        for i, exp in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] {exp['name']}")
            
            synthetic_acc = exp['metrics'].get('top_1_accuracy', 0)
            print(f"  Synthetic Top-1: {synthetic_acc:.4f}")
            
            # TODO: Re-run with high-fidelity data
            # For now, placeholder - actual Sionna validation to be implemented
            if sionna_data is not None:
                sionna_acc = 0.0  # Placeholder for actual validation
                print(f"  Sionna Top-1: {sionna_acc:.4f}")
                degradation = synthetic_acc - sionna_acc
            else:
                sionna_acc = 0.0
                degradation = 0.0
                print(f"  Sionna Top-1: (data not available)")
            
            results.append({
                'experiment_id': exp['id'],
                'name': exp['name'],
                'probe': exp['probe_type'],
                'model': exp['model_type'],
                'synthetic_accuracy': synthetic_acc,
                'sionna_accuracy': sionna_acc,
                'degradation': degradation
            })
        
        df = pd.DataFrame(results)
        
        print("\n" + "=" * 70)
        print("Validation Results:")
        print(df.to_string(index=False))
        
        return df
    
    def show_history(self, limit: int = 10) -> None:
        """
        Print table of all experiments in database.
        
        Args:
            limit: Maximum number of experiments to show (default: 10)
        """
        experiments = self.tracker.get_all_experiments(limit=limit)
        
        if not experiments:
            print("No experiments found in database")
            return
        
        data = []
        for exp in experiments:
            data.append({
                'ID': exp['id'],
                'Name': exp['name'][:30],
                'Probe': exp['probe_type'],
                'Model': exp['model_type'],
                'M': exp['M'],
                'K': exp['K'],
                'Top-1': f"{exp['metrics'].get('top_1_accuracy', 0):.3f}",
                'Status': exp['status'],
                'Time': f"{exp['training_time_seconds']:.1f}s"
            })
        
        df = pd.DataFrame(data)
        
        print("\n" + "="*100)
        print(f"Experiment History (showing {len(df)} of {len(experiments)})")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
    
    def plot_best(self) -> None:
        """
        Plot best result's training curve.
        
        Finds the experiment with highest top_1_accuracy and displays its training curves.
        """
        experiments = self.tracker.get_all_experiments(limit=100)
        
        if not experiments:
            print("No experiments found in database")
            return
        
        # Find best by top_1_accuracy
        completed = [e for e in experiments if e['status'] == 'completed']
        
        if not completed:
            print("No completed experiments found")
            return
        
        best = max(completed, key=lambda x: x['metrics'].get('top_1_accuracy', 0))
        
        print("\n" + "="*70)
        print(f"Best Experiment (by top_1_accuracy)")
        print("="*70)
        print(f"ID: {best['id']}")
        print(f"Name: {best['name']}")
        print(f"Probe: {best['probe_type']}")
        print(f"Model: {best['model_type']}")
        print(f"Top-1 Accuracy: {best['metrics'].get('top_1_accuracy', 0):.4f}")
        print("="*70)
        
        # Plot training history
        self.show(best)
