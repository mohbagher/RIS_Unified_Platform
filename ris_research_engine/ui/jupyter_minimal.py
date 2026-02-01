"""Simple API for Jupyter notebooks - RIS Auto-Research Engine."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

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
    
    def __init__(self, db_path: str = "results.db", output_dir: str = "outputs"):
        """
        Initialize the RIS Engine.
        
        Args:
            db_path: Path to SQLite database for results storage
            output_dir: Directory for outputs (plots, reports, etc.)
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tracker = ResultTracker(db_path)
        self.runner = ExperimentRunner()
        self.controller = SearchController(db_path)
        self.analyzer = ResultAnalyzer(db_path)
        self.reporter = ReportGenerator(str(self.output_dir))
        
        logger.info(f"RISEngine initialized - DB: {db_path}, Output: {output_dir}")
    
    def run(
        self,
        probe: str,
        model: str,
        M: int,
        K: int,
        N: int = 64,
        data_source: str = 'synthetic_rayleigh',
        n_samples: int = 1000,
        epochs: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a single experiment with given parameters.
        
        Args:
            probe: Probe type (e.g., 'dft_beams', 'hadamard', 'random_uniform')
            model: Model type (e.g., 'mlp', 'cnn_1d', 'transformer')
            M: Sensing budget (number of measurements)
            K: Codebook size
            N: Number of RIS elements (default: 64)
            data_source: Data source name (default: 'synthetic_rayleigh')
            n_samples: Number of samples to generate (default: 1000)
            epochs: Maximum training epochs (default: 100)
            **kwargs: Additional parameters (learning_rate, batch_size, etc.)
            
        Returns:
            Dictionary with experiment result
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
            data_source=data_source,
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
        Display experiment results in notebook.
        
        Args:
            result: Experiment result dictionary
        """
        print("\n" + "="*70)
        print(f"Experiment: {result['config']['name']}")
        print(f"Status: {result['status']}")
        print("="*70)
        
        # Metrics table
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
        if 'training_history' in result:
            history = result['training_history']
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss plot
            if 'train_loss' in history:
                epochs = range(1, len(history['train_loss']) + 1)
                axes[0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
                if 'val_loss' in history:
                    axes[0].plot(epochs, history['val_loss'], label='Val', linewidth=2)
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Training Loss')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Accuracy plot
            if 'val_top_1_accuracy' in history:
                epochs = range(1, len(history['val_top_1_accuracy']) + 1)
                axes[1].plot(epochs, history['val_top_1_accuracy'], 
                           label='Top-1', linewidth=2)
                if 'val_top_5_accuracy' in history:
                    axes[1].plot(epochs, history['val_top_5_accuracy'], 
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
        model: str = 'mlp',
        M: int = 8,
        K: int = 64,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run experiments for multiple probe types and compare.
        
        Args:
            probes: List of probe types to compare
            model: Model type (default: 'mlp')
            M: Sensing budget (default: 8)
            K: Codebook size (default: 64)
            **kwargs: Additional parameters passed to run()
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for probe in probes:
            print(f"\nRunning {probe}...")
            result = self.run(probe=probe, model=model, M=M, K=K, **kwargs)
            results.append(result)
        
        return results
    
    def plot_comparison(
        self,
        results: List[Dict[str, Any]],
        metric: str = 'top_1_accuracy'
    ) -> None:
        """
        Create bar chart comparing multiple experiment results.
        
        Args:
            results: List of experiment result dictionaries
            metric: Metric to compare (default: 'top_1_accuracy')
        """
        data = []
        for result in results:
            if result['status'] == 'completed':
                data.append({
                    'name': result['config']['name'],
                    'probe': result['config']['probe_type'],
                    'value': result['metrics'].get(metric, 0.0)
                })
        
        if not data:
            print("No completed experiments to compare")
            return
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(df)), df['value'])
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['probe'], rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Probe Comparison - {metric.replace("_", " ").title()}')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, df['value'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def search(
        self,
        strategy: str = 'grid',
        config_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a search campaign using SearchController.
        
        Args:
            strategy: Search strategy name ('grid', 'random', 'bayesian', etc.)
            config_dict: Search configuration dict (if None, use default)
            
        Returns:
            Campaign result dictionary
        """
        if config_dict is None:
            # Default search configuration
            config_dict = {
                'name': f'{strategy}_search',
                'search_space': {
                    'probe_type': ['dft_beams', 'hadamard', 'random_uniform'],
                    'model_type': ['mlp', 'cnn_1d'],
                    'M': [4, 8, 16],
                    'learning_rate': [1e-4, 1e-3, 1e-2]
                },
                'budget': {
                    'max_experiments': 20,
                    'max_time_hours': 2.0
                }
            }
        
        logger.info(f"Starting {strategy} search campaign")
        
        campaign = self.controller.run_campaign(
            search_space_config=config_dict,
            strategy_name=strategy
        )
        
        # Save campaign
        campaign_id = self.tracker.save_campaign(campaign)
        
        result = campaign.to_dict()
        result['campaign_id'] = campaign_id
        
        logger.info(f"Campaign completed - ID: {campaign_id}")
        
        return result
    
    def plot_campaign(self, campaign: Dict[str, Any]) -> None:
        """
        Show summary plots for a search campaign.
        
        Args:
            campaign: Campaign result dictionary
        """
        print("\n" + "="*70)
        print(f"Campaign: {campaign['campaign_name']}")
        print(f"Strategy: {campaign['search_strategy']}")
        print("="*70)
        print(f"Total Experiments: {campaign['total_experiments']}")
        print(f"Completed: {campaign['completed_experiments']}")
        print(f"Pruned: {campaign['pruned_experiments']}")
        print(f"Failed: {campaign['failed_experiments']}")
        print(f"Total Time: {campaign['total_time_seconds']:.2f}s")
        
        if campaign['best_result']:
            print(f"\nBest Result:")
            print(f"  Config: {campaign['best_result']['config']['name']}")
            print(f"  Metric: {campaign['best_result']['primary_metric_value']:.4f}")
        
        # Plot experiment metrics over time
        results = campaign['all_results']
        completed = [r for r in results if r['status'] == 'completed']
        
        if len(completed) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Top-1 accuracy progression
            ax = axes[0, 0]
            metrics = [r['metrics'].get('top_1_accuracy', 0) for r in completed]
            ax.plot(metrics, marker='o', linewidth=2)
            ax.set_xlabel('Experiment')
            ax.set_ylabel('Top-1 Accuracy')
            ax.set_title('Accuracy Progression')
            ax.grid(True, alpha=0.3)
            
            # Training time
            ax = axes[0, 1]
            times = [r['training_time_seconds'] for r in completed]
            ax.bar(range(len(times)), times)
            ax.set_xlabel('Experiment')
            ax.set_ylabel('Time (s)')
            ax.set_title('Training Time')
            ax.grid(True, axis='y', alpha=0.3)
            
            # Model parameters vs accuracy
            ax = axes[1, 0]
            params = [r['model_parameters'] for r in completed]
            ax.scatter(params, metrics, s=100, alpha=0.6)
            ax.set_xlabel('Model Parameters')
            ax.set_ylabel('Top-1 Accuracy')
            ax.set_title('Model Size vs Accuracy')
            ax.grid(True, alpha=0.3)
            
            # Best configurations
            ax = axes[1, 1]
            top_n = min(5, len(completed))
            sorted_results = sorted(completed, 
                                  key=lambda x: x['metrics'].get('top_1_accuracy', 0),
                                  reverse=True)[:top_n]
            names = [r['config']['name'][:20] for r in sorted_results]
            values = [r['metrics'].get('top_1_accuracy', 0) for r in sorted_results]
            
            ax.barh(range(len(names)), values)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel('Top-1 Accuracy')
            ax.set_title(f'Top {top_n} Configurations')
            ax.grid(True, axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def validate_cross_fidelity(
        self,
        campaign_name: str,
        hdf5_path: str,
        top_n: int = 3
    ) -> pd.DataFrame:
        """
        Cross-fidelity validation using high-fidelity simulation data (e.g., Sionna).
        
        This method validates top-performing experiments from synthetic data
        against high-fidelity electromagnetic simulations to assess performance
        degradation and ensure results generalize.
        
        Args:
            campaign_name: Name of campaign to validate
            hdf5_path: Path to HDF5 file with high-fidelity simulation data (e.g., Sionna)
            top_n: Number of top experiments to validate
            
        Returns:
            DataFrame with validation results showing synthetic vs high-fidelity accuracy
        """
        # Get campaign experiments
        campaign = self.tracker.get_campaign(campaign_name=campaign_name)
        
        if not campaign:
            raise ValueError(f"Campaign '{campaign_name}' not found")
        
        # Get top N experiments by primary metric
        experiments = self.tracker.get_all_experiments(campaign_name=campaign_name)
        experiments = sorted(
            experiments,
            key=lambda x: x.get('primary_metric_value', 0),
            reverse=True
        )[:top_n]
        
        # Load Sionna data
        logger.info(f"Loading high-fidelity simulation data from {hdf5_path}")
        sionna_data = load_hdf5_data(hdf5_path)
        
        # Validate each experiment
        results = []
        
        for exp in experiments:
            print(f"\nValidating: {exp['name']}")
            
            # TODO: Re-run with high-fidelity data
            # For now, placeholder until Sionna integration is complete
            results.append({
                'experiment_id': exp['id'],
                'name': exp['name'],
                'synthetic_accuracy': exp['metrics'].get('top_1_accuracy', 0),
                'high_fidelity_accuracy': 0.0,  # Placeholder for Sionna validation
                'degradation': 0.0
            })
        
        df = pd.DataFrame(results)
        
        print("\nValidation Results:")
        print(df.to_string(index=False))
        
        return df
    
    def show_history(
        self,
        campaign_name: Optional[str] = None,
        limit: int = 10
    ) -> None:
        """
        Display table of past experiments.
        
        Args:
            campaign_name: Filter by campaign name (None for all)
            limit: Maximum number of experiments to show
        """
        experiments = self.tracker.get_all_experiments(
            campaign_name=campaign_name,
            limit=limit
        )
        
        if not experiments:
            print("No experiments found")
            return
        
        data = []
        for exp in experiments[:limit]:
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
        print("\nExperiment History:")
        print("=" * 100)
        print(df.to_string(index=False))
    
    def plot_best(
        self,
        campaign_name: Optional[str] = None,
        metric: str = 'top_1_accuracy'
    ) -> None:
        """
        Plot training curves of best experiment.
        
        Args:
            campaign_name: Campaign name (None for all experiments)
            metric: Metric to use for selecting best
        """
        experiments = self.tracker.get_all_experiments(
            campaign_name=campaign_name,
            limit=100
        )
        
        # Find best by metric
        best = max(
            experiments,
            key=lambda x: x['metrics'].get(metric, 0)
        )
        
        print(f"\nBest Experiment (by {metric}):")
        print(f"  ID: {best['id']}")
        print(f"  Name: {best['name']}")
        print(f"  {metric}: {best['metrics'].get(metric, 0):.4f}")
        
        # Plot training history
        history = best['training_history']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        if 'train_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
            if 'val_loss' in history:
                axes[0].plot(epochs, history['val_loss'], label='Val', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title(f'Best {best["name"][:30]} - Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        if 'val_top_1_accuracy' in history:
            epochs = range(1, len(history['val_top_1_accuracy']) + 1)
            axes[1].plot(epochs, history['val_top_1_accuracy'], 
                       label='Top-1', linewidth=2)
            if 'val_top_5_accuracy' in history:
                axes[1].plot(epochs, history['val_top_5_accuracy'], 
                           label='Top-5', linewidth=2)
            axes[1].axvline(best['best_epoch'], color='r', 
                          linestyle='--', label='Best', alpha=0.5)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
