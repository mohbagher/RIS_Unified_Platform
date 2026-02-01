"""Minimal Jupyter interface for RIS research engine."""

import logging
import math
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

from ris_research_engine.foundation import (
    SystemConfig, TrainingConfig, ExperimentConfig, ExperimentResult
)
from ris_research_engine.engine import (
    ExperimentRunner, SearchController, ResultAnalyzer, ReportGenerator
)

logger = logging.getLogger(__name__)


class RISEngine:
    """Simple interface for running RIS experiments in Jupyter notebooks."""
    
    def __init__(self, db_path: str = "ris_results.db"):
        """
        Initialize RIS Engine.
        
        Args:
            db_path: Path to SQLite database for storing results
        """
        self.db_path = db_path
        self.runner = ExperimentRunner(db_path)
        self.controller = SearchController(db_path)
        self.analyzer = ResultAnalyzer(db_path)
        self.reporter = ReportGenerator(db_path)
        self.last_result = None
    
    def run(
        self,
        probe: str = "random_uniform",
        model: str = "mlp",
        N: int = 64,
        K: int = 64,
        M: int = 8,
        data_source: str = "synthetic_rayleigh",
        metrics: Optional[List[str]] = None,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        **kwargs
    ) -> ExperimentResult:
        """
        Run a single experiment with simple parameters.
        
        Args:
            probe: Probe type name
            model: Model type name
            N: Number of RIS elements
            K: Codebook size
            M: Sensing budget
            data_source: Data source name
            metrics: List of metric names
            max_epochs: Maximum training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            **kwargs: Additional parameters
            
        Returns:
            ExperimentResult
        """
        # Set default metrics
        if metrics is None:
            metrics = ['top_1_accuracy', 'power_ratio']
        
        # Compute N_x and N_y from N
        N_x = int(math.sqrt(N))
        N_y = N_x
        if N_x * N_y != N:
            # Try to find factors
            for n_x in range(int(math.sqrt(N)), 0, -1):
                if N % n_x == 0:
                    N_x = n_x
                    N_y = N // n_x
                    break
        
        # Create configuration
        system = SystemConfig(N=N, N_x=N_x, N_y=N_y, K=K, M=M)
        training = TrainingConfig(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        
        config = ExperimentConfig(
            name=f"{probe}_{model}_M{M}_K{K}",
            system=system,
            training=training,
            probe_type=probe,
            probe_params=kwargs.get('probe_params', {}),
            model_type=model,
            model_params=kwargs.get('model_params', {}),
            data_source=data_source,
            data_params=kwargs.get('data_params', {'n_samples': 1000}),
            metrics=metrics,
            tags=kwargs.get('tags', []),
            notes=kwargs.get('notes', '')
        )
        
        # Run experiment
        print(f"Running experiment: {config.name}")
        result = self.runner.run(config)
        self.last_result = result
        
        return result
    
    def show(self, result: Optional[ExperimentResult] = None):
        """
        Display metrics and basic plot for an experiment.
        
        Args:
            result: ExperimentResult to display (uses last_result if None)
        """
        if result is None:
            result = self.last_result
        
        if result is None:
            print("No result to display. Run an experiment first.")
            return
        
        # Display status
        print(f"\n{'='*60}")
        print(f"Experiment: {result.config.name}")
        print(f"Status: {result.status}")
        print(f"{'='*60}\n")
        
        if result.status == 'failed':
            print(f"Error: {result.error_message}")
            return
        
        # Display metrics
        print("Metrics:")
        for metric_name, metric_value in result.metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        print(f"\nTraining:")
        print(f"  Time: {result.training_time_seconds:.2f}s")
        print(f"  Epochs: {result.total_epochs}")
        print(f"  Best Epoch: {result.best_epoch}")
        print(f"  Parameters: {result.model_parameters:,}")
        
        # Display baseline comparison
        if result.baseline_results:
            print(f"\nBaseline Comparison:")
            primary_metric = result.primary_metric_name
            if primary_metric in result.metrics:
                ml_value = result.metrics[primary_metric]
                print(f"  ML Model ({primary_metric}): {ml_value:.4f}")
                
                for baseline_name, baseline_metrics in result.baseline_results.items():
                    if primary_metric in baseline_metrics:
                        baseline_value = baseline_metrics[primary_metric]
                        improvement = ((ml_value - baseline_value) / baseline_value * 100) if baseline_value > 0 else 0
                        print(f"  {baseline_name}: {baseline_value:.4f} (improvement: {improvement:+.1f}%)")
        
        print("")
    
    def compare_probes(
        self,
        probes: List[str],
        model: str = "mlp",
        M: int = 8,
        K: int = 64,
        **kwargs
    ) -> List[ExperimentResult]:
        """
        Run and compare multiple probe types.
        
        Args:
            probes: List of probe type names
            model: Model type to use
            M: Sensing budget
            K: Codebook size
            **kwargs: Additional parameters for run()
            
        Returns:
            List of experiment results
        """
        results = []
        
        print(f"Comparing {len(probes)} probe types with {model} model...")
        print("")
        
        for probe in probes:
            print(f"Running {probe}...")
            result = self.run(probe=probe, model=model, M=M, K=K, **kwargs)
            results.append(result)
        
        # Display summary
        print(f"\n{'='*60}")
        print("Comparison Summary")
        print(f"{'='*60}\n")
        
        for result in results:
            if result.status == 'completed':
                primary_metric = result.primary_metric_name
                primary_value = result.primary_metric_value
                print(f"{result.config.probe_type:20s}: {primary_metric}={primary_value:.4f}")
        
        print("")
        
        return results
    
    def plot_comparison(
        self,
        metric: str = 'top_1_accuracy',
        group_by: str = 'probe_type'
    ):
        """
        Plot comparison of experiments.
        
        Args:
            metric: Metric to compare
            group_by: How to group results ('probe_type' or 'model_type')
        """
        if group_by == 'probe_type':
            self.reporter.probe_comparison_bar(metric)
        elif group_by == 'model_type':
            self.reporter.model_comparison_bar(metric)
        else:
            print(f"Unknown group_by: {group_by}")
            return
        
        plt.show()
    
    def search(
        self,
        config_path: str
    ):
        """
        Run a search campaign from YAML configuration.
        
        Args:
            config_path: Path to YAML config file
        """
        print(f"Running search campaign from {config_path}...")
        result = self.controller.run_from_yaml(config_path)
        
        # Display summary
        print(f"\n{'='*60}")
        print(f"Campaign: {result.campaign_name}")
        print(f"{'='*60}\n")
        print(f"Strategy: {result.search_strategy}")
        print(f"Total experiments: {result.total_experiments}")
        print(f"Completed: {result.completed_experiments}")
        print(f"Failed: {result.failed_experiments}")
        print(f"Pruned: {result.pruned_experiments}")
        print(f"Time: {result.total_time_seconds:.2f}s")
        
        if result.best_result:
            print(f"\nBest result:")
            print(f"  Configuration: {result.best_result.config.name}")
            print(f"  {result.best_result.primary_metric_name}: {result.best_result.primary_metric_value:.4f}")
        
        print("")
        
        return result
    
    def plot_campaign(
        self,
        metric: str = 'top_1_accuracy'
    ):
        """
        Plot summary of recent campaign results.
        
        Args:
            metric: Metric to visualize
        """
        # Create multiple plots
        self.reporter.probe_comparison_bar(metric)
        plt.show()
        
        self.reporter.model_comparison_bar(metric)
        plt.show()
        
        self.reporter.heatmap_probe_model(metric)
        plt.show()
    
    def validate_on_sionna(
        self,
        top_n: int = 5
    ):
        """
        Run cross-fidelity validation on top N experiments.
        
        Args:
            top_n: Number of top experiments to validate
        """
        print(f"Running cross-fidelity validation on top {top_n} experiments...")
        results = self.controller.run_cross_fidelity_validation(
            top_n=top_n,
            validation_fidelity='sionna'
        )
        
        print(f"\nValidation completed for {len(results)} experiments")
        return results
    
    def show_history(
        self,
        limit: int = 10
    ):
        """
        Show history of recent experiments.
        
        Args:
            limit: Number of experiments to show
        """
        results = self.runner.tracker.get_all_results()
        
        if not results:
            print("No experiments found.")
            return
        
        # Sort by timestamp descending
        results.sort(key=lambda r: r.timestamp, reverse=True)
        results = results[:limit]
        
        print(f"\n{'='*80}")
        print(f"Recent Experiments (showing {len(results)})")
        print(f"{'='*80}\n")
        
        print(f"{'Name':<30} {'Status':<12} {'Probe':<15} {'Model':<10} {'Metric':<8}")
        print(f"{'-'*80}")
        
        for result in results:
            name = result.config.name[:28] + '..' if len(result.config.name) > 30 else result.config.name
            primary_value = f"{result.primary_metric_value:.3f}" if result.status == 'completed' else "N/A"
            
            print(f"{name:<30} {result.status:<12} {result.config.probe_type:<15} "
                  f"{result.config.model_type:<10} {primary_value:<8}")
        
        print("")
    
    def plot_best(
        self,
        metric: str = 'top_1_accuracy',
        top_n: int = 10
    ):
        """
        Plot top N configurations.
        
        Args:
            metric: Metric to rank by
            top_n: Number of top results to show
        """
        df = self.analyzer.best_configuration(metric, top_n)
        
        if df.empty:
            print("No completed experiments found.")
            return
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = [name[:20] + '..' if len(name) > 20 else name for name in df['name']]
        values = df[metric]
        
        ax.barh(range(len(names)), values, alpha=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(f'Top {top_n} Configurations by {metric.replace("_", " ").title()}')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
