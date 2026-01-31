"""Minimal Jupyter interface for RIS research engine."""

import logging
from typing import Optional, List, Dict, Any
import pandas as pd

from ris_research_engine.foundation.data_types import (
    SystemConfig, TrainingConfig, ExperimentConfig, ExperimentResult
)
from ris_research_engine.foundation.storage import ResultTracker
from ris_research_engine.engine.experiment_runner import ExperimentRunner
from ris_research_engine.engine.search_controller import SearchController
from ris_research_engine.engine.result_analyzer import ResultAnalyzer
from ris_research_engine.engine.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class RISEngine:
    """Simplified interface for RIS research experiments in Jupyter notebooks."""
    
    def __init__(self, db_path: str = "results.db", output_dir: str = "outputs"):
        """Initialize RIS engine.
        
        Args:
            db_path: Path to results database
            output_dir: Directory for outputs and plots
        """
        self.result_tracker = ResultTracker(db_path)
        self.experiment_runner = ExperimentRunner(self.result_tracker)
        self.search_controller = SearchController(self.result_tracker, self.experiment_runner)
        self.analyzer = ResultAnalyzer(self.result_tracker)
        self.reporter = ReportGenerator(output_dir)
        
        logger.info(f"RIS Engine initialized with database: {db_path}")
    
    def run(
        self,
        probe: str,
        model: str,
        M: int,
        K: int = 64,
        N: int = 64,
        data: str = "synthetic_rayleigh",
        n_samples: int = 10000,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        random_seed: int = 42,
        **kwargs
    ) -> ExperimentResult:
        """Run a single experiment with simplified parameters.
        
        Args:
            probe: Probe type (e.g., 'random_uniform', 'hadamard')
            model: Model type (e.g., 'mlp', 'cnn_1d')
            M: Sensing budget (number of probes)
            K: Codebook size
            N: Number of RIS elements
            data: Data source
            n_samples: Number of samples to generate
            epochs: Maximum training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            random_seed: Random seed
            **kwargs: Additional parameters
            
        Returns:
            ExperimentResult
        """
        # Create system config
        N_x = int(N ** 0.5)
        N_y = N_x
        
        system = SystemConfig(
            N=N,
            N_x=N_x,
            N_y=N_y,
            K=K,
            M=M,
            **{k: v for k, v in kwargs.items() if k in SystemConfig.__dataclass_fields__}
        )
        
        # Create training config
        training = TrainingConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=epochs,
            random_seed=random_seed,
            **{k: v for k, v in kwargs.items() if k in TrainingConfig.__dataclass_fields__}
        )
        
        # Create experiment config
        config = ExperimentConfig(
            name=f"{probe}_{model}_M{M}_K{K}",
            system=system,
            training=training,
            probe_type=probe,
            probe_params=kwargs.get('probe_params', {}),
            model_type=model,
            model_params=kwargs.get('model_params', {}),
            data_source=data,
            data_params={'n_samples': n_samples},
            metrics=['top_1_accuracy', 'top_5_accuracy', 'top_10_accuracy'],
        )
        
        # Run experiment
        print(f"Running experiment: {config.name}")
        result = self.experiment_runner.run(config)
        
        return result
    
    def show(self, result: ExperimentResult):
        """Display experiment results with metrics and training curve.
        
        Args:
            result: ExperimentResult to display
        """
        print(f"\n{'='*60}")
        print(f"Experiment: {result.config.name}")
        print(f"{'='*60}")
        print(f"Status: {result.status}")
        print(f"Training time: {result.training_time_seconds:.2f}s")
        print(f"Model parameters: {result.model_parameters:,}")
        print(f"Best epoch: {result.best_epoch + 1}/{result.total_epochs}")
        
        print(f"\nTest Metrics:")
        for metric_name, value in result.metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        if result.baseline_results:
            print(f"\nBaseline Comparisons:")
            for baseline_name, metrics in result.baseline_results.items():
                baseline_acc = metrics.get('top_1_accuracy', 0.0)
                print(f"  {baseline_name}: {baseline_acc:.4f}")
        
        # Plot training curve if matplotlib available
        if MATPLOTLIB_AVAILABLE and result.training_history:
            self.reporter.training_curves(result)
            print(f"\nTraining curves saved to: {self.reporter.output_dir}")
    
    def compare_probes(
        self,
        probes: List[str],
        model: str = "mlp",
        M: int = 8,
        K: int = 64,
        n_runs: int = 3,
        **kwargs
    ) -> List[ExperimentResult]:
        """Compare multiple probe types with the same model.
        
        Args:
            probes: List of probe types to compare
            model: Model type to use
            M: Sensing budget
            K: Codebook size
            n_runs: Number of runs per probe (different seeds)
            **kwargs: Additional parameters for run()
            
        Returns:
            List of experiment results
        """
        results = []
        
        for probe in probes:
            print(f"\nTesting probe: {probe}")
            for seed in range(n_runs):
                result = self.run(
                    probe=probe,
                    model=model,
                    M=M,
                    K=K,
                    random_seed=42 + seed,
                    **kwargs
                )
                results.append(result)
        
        return results
    
    def plot_comparison(self, results: List[ExperimentResult], metric: str = 'top_1_accuracy'):
        """Plot comparison of experiment results.
        
        Args:
            results: List of experiment results
            metric: Metric to compare
        """
        # Analyze results
        comparison_df = self.analyzer.compare_probes(results=results, metric=metric)
        
        # Generate plot
        self.reporter.probe_comparison_bar(comparison_df, metric_name='mean')
        
        # Display summary
        print("\nProbe Comparison:")
        print(comparison_df.to_string(index=False))
    
    def search(
        self,
        strategy: str = "grid_search",
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ) -> 'SearchCampaignResult':
        """Run automated search campaign.
        
        Args:
            strategy: Search strategy name
            config: Search configuration dictionary
            config_path: Path to YAML configuration file
            
        Returns:
            SearchCampaignResult
        """
        if config_path:
            result = self.search_controller.run_from_yaml(config_path)
        elif config:
            result = self.search_controller.run_campaign(config, strategy)
        else:
            raise ValueError("Either config or config_path must be provided")
        
        return result
    
    def plot_campaign(self, campaign: 'SearchCampaignResult'):
        """Generate comprehensive plots for a search campaign.
        
        Args:
            campaign: SearchCampaignResult to visualize
        """
        print(f"Generating plots for campaign: {campaign.campaign_name}")
        
        # Analyze results
        analyzer_results = {}
        
        if campaign.all_results:
            analyzer_results['probe_comparison'] = self.analyzer.compare_probes(
                results=campaign.all_results
            )
            analyzer_results['model_comparison'] = self.analyzer.compare_models(
                results=campaign.all_results
            )
            analyzer_results['sparsity_analysis'] = self.analyzer.sparsity_analysis(
                results=campaign.all_results
            )
        
        # Generate report
        self.reporter.generate_full_report(campaign, analyzer_results)
        
        print(f"Plots saved to: {self.reporter.output_dir}")
    
    def validate_on_sionna(
        self,
        campaign_name: str,
        hdf5_path: str,
        top_n: int = 10
    ) -> pd.DataFrame:
        """Validate synthetic results on Sionna data.
        
        Args:
            campaign_name: Name of synthetic campaign
            hdf5_path: Path to Sionna HDF5 file
            top_n: Number of top configs to validate
            
        Returns:
            DataFrame with validation results
        """
        return self.search_controller.run_cross_fidelity_validation(
            campaign_name, hdf5_path, top_n
        )
    
    def show_history(
        self,
        campaign_name: Optional[str] = None,
        limit: int = 20
    ) -> pd.DataFrame:
        """Show experiment history.
        
        Args:
            campaign_name: Optional campaign name to filter
            limit: Maximum number of results to show
            
        Returns:
            DataFrame with experiment history
        """
        results = self.result_tracker.query(
            campaign_name=campaign_name,
            limit=limit,
            sort_by='timestamp',
            sort_order='desc'
        )
        
        if not results:
            print("No experiments found")
            return pd.DataFrame()
        
        # Create summary DataFrame
        data = []
        for result in results:
            data.append({
                'name': result.config.name,
                'probe': result.config.probe_type,
                'model': result.config.model_type,
                'M': result.config.system.M,
                'K': result.config.system.K,
                'accuracy': result.metrics.get('top_1_accuracy', 0.0),
                'status': result.status,
                'time': result.training_time_seconds,
            })
        
        df = pd.DataFrame(data)
        return df
    
    def plot_best(self, campaign_name: Optional[str] = None, top_k: int = 5):
        """Plot top K best results.
        
        Args:
            campaign_name: Optional campaign name to filter
            top_k: Number of top results to plot
        """
        best_results = self.analyzer.best_configuration(
            campaign_name=campaign_name,
            top_k=top_k
        )
        
        if not best_results:
            print("No completed experiments found")
            return
        
        print(f"\nTop {len(best_results)} Results:")
        for i, result in enumerate(best_results):
            print(f"{i+1}. {result.config.name} - Accuracy: {result.primary_metric_value:.4f}")
        
        # Plot comparison
        self.reporter.ranking_distribution(best_results)
        
        # Plot training curve for best
        if best_results:
            self.reporter.training_curves(best_results[0], title="Best Result Training Curves")
            self.reporter.baseline_comparison(best_results[0])
