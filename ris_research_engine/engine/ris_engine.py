"""High-level API for running experiments and search campaigns."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from ris_research_engine.foundation import (
    SystemConfig, TrainingConfig, ExperimentConfig,
    ExperimentResult, SearchCampaignResult, ResultTracker
)
from .experiment_runner import ExperimentRunner
from .result_analyzer import ResultAnalyzer


class RISEngine:
    """High-level API for the RIS Auto-Research Engine."""
    
    def __init__(self, db_path: str = "outputs/experiments/results.db"):
        """Initialize the engine.
        
        Args:
            db_path: Path to results database
        """
        self.db_path = db_path
        self.runner = ExperimentRunner(db_path)
        self.tracker = ResultTracker(db_path)
        self.analyzer = ResultAnalyzer(self.tracker)
    
    def run(self,
            probe: str,
            model: str,
            M: int = 8,
            K: int = 64,
            N: int = 64,
            data: str = "synthetic_rayleigh",
            n_samples: int = 10000,
            epochs: int = 100,
            learning_rate: float = 1e-3,
            batch_size: int = 64,
            **kwargs) -> ExperimentResult:
        """Run a single experiment with minimal configuration.
        
        Args:
            probe: Probe type name
            model: Model type name
            M: Number of probe measurements
            K: Codebook size
            N: Number of RIS elements
            data: Data source name
            n_samples: Number of training samples
            epochs: Maximum training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            **kwargs: Additional configuration parameters
            
        Returns:
            ExperimentResult
        """
        # Build system config
        system = SystemConfig(
            N=N,
            N_x=int(N**0.5),
            N_y=int(N**0.5),
            K=K,
            M=M,
            frequency=kwargs.get('frequency', 28e9),
            snr_db=kwargs.get('snr_db', 20.0),
        )
        
        # Build training config
        training = TrainingConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=epochs,
            early_stopping_patience=kwargs.get('patience', 15),
            device=kwargs.get('device', 'auto'),
        )
        
        # Build experiment config
        name = f"{probe}_{model}_M{M}_K{K}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config = ExperimentConfig(
            name=name,
            system=system,
            training=training,
            probe_type=probe,
            probe_params=kwargs.get('probe_params', {}),
            model_type=model,
            model_params=kwargs.get('model_params', {}),
            data_source=data,
            data_params={
                'n_train': n_samples,
                'n_val': int(n_samples * 0.15),
                'n_test': int(n_samples * 0.15),
            },
            metrics=['top_1_accuracy', 'hit_at_1', 'power_ratio'],
            tags=kwargs.get('tags', []),
            notes=kwargs.get('notes', ''),
            data_fidelity='synthetic',
        )
        
        # Run experiment
        return self.runner.run(config)
    
    def search(self,
               strategy: str = "grid_search",
               config: Optional[str] = None,
               **kwargs) -> SearchCampaignResult:
        """Run an automated search campaign.
        
        Args:
            strategy: Search strategy name
            config: Path to YAML config file
            **kwargs: Search parameters
            
        Returns:
            SearchCampaignResult
        """
        # Load config if provided
        if config:
            with open(config, 'r') as f:
                search_config = yaml.safe_load(f)
        else:
            search_config = kwargs
        
        # Import search strategy
        from ris_research_engine.plugins.search import get_search_strategy
        
        search_strategy = get_search_strategy(strategy)
        
        # Run search
        campaign_name = search_config.get('campaign_name', f'campaign_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        print(f"Starting search campaign: {campaign_name}")
        print(f"Strategy: {strategy}")
        
        start_time = time.time()
        
        # Execute search
        results = search_strategy.search(
            search_space=search_config,
            runner=self.runner,
            campaign_name=campaign_name
        )
        
        total_time = time.time() - start_time
        
        # Create campaign result
        completed = [r for r in results if r.status == 'completed']
        failed = [r for r in results if r.status == 'failed']
        pruned = [r for r in results if r.status == 'pruned']
        
        # Find best result
        best_result = None
        best_value = -float('inf')
        
        for result in completed:
            value = result.metrics.get('top_1_accuracy', 0.0)
            if value > best_value:
                best_value = value
                best_result = result
        
        campaign_result = SearchCampaignResult(
            campaign_name=campaign_name,
            search_strategy=strategy,
            total_experiments=len(results),
            completed_experiments=len(completed),
            pruned_experiments=len(pruned),
            failed_experiments=len(failed),
            best_result=best_result,
            all_results=results,
            total_time_seconds=total_time,
            search_space_definition=search_config,
            timestamp=datetime.now().isoformat()
        )
        
        # Save campaign
        self.tracker.save_campaign(campaign_result)
        
        print(f"✅ Campaign completed in {total_time:.1f}s")
        print(f"   Completed: {len(completed)}, Failed: {len(failed)}, Pruned: {len(pruned)}")
        if best_result:
            print(f"   Best result: {best_result.metrics.get('top_1_accuracy', 0.0):.3f}")
        
        return campaign_result
    
    def compare_probes(self,
                      probes: List[str],
                      model: str = "mlp",
                      M: int = 8,
                      K: int = 64,
                      **kwargs) -> List[ExperimentResult]:
        """Compare multiple probe types.
        
        Args:
            probes: List of probe type names
            model: Model type name
            M: Number of probe measurements
            K: Codebook size
            **kwargs: Additional configuration
            
        Returns:
            List of ExperimentResults
        """
        results = []
        
        for probe in probes:
            print(f"\n{'='*60}")
            print(f"Testing probe: {probe}")
            print(f"{'='*60}")
            
            result = self.run(
                probe=probe,
                model=model,
                M=M,
                K=K,
                **kwargs
            )
            
            results.append(result)
        
        return results
    
    def validate_on_sionna(self,
                          campaign_name: str,
                          hdf5_path: str,
                          top_n: int = 5) -> Dict[str, Any]:
        """Validate synthetic winners on Sionna data.
        
        Args:
            campaign_name: Name of synthetic campaign
            hdf5_path: Path to Sionna HDF5 data
            top_n: Number of top configurations to validate
            
        Returns:
            Fidelity gap report
        """
        # Get top N experiments from synthetic campaign
        experiments = self.tracker.get_all_experiments(
            campaign_name=campaign_name,
            status='completed'
        )
        
        if not experiments:
            raise ValueError(f"No completed experiments found for campaign: {campaign_name}")
        
        # Sort by top_1_accuracy
        experiments.sort(key=lambda e: e['metrics'].get('top_1_accuracy', 0.0), reverse=True)
        top_experiments = experiments[:top_n]
        
        print(f"Validating top {top_n} configurations on Sionna data...")
        
        # Re-run each configuration with Sionna data
        validation_results = []
        
        for i, exp in enumerate(top_experiments):
            print(f"\n[{i+1}/{top_n}] Validating: {exp['name']}")
            
            # Reconstruct config
            original_config = ExperimentConfig.from_dict(exp['full_config'])
            
            # Modify for Sionna data
            validation_config = ExperimentConfig(
                name=f"{original_config.name}_sionna_validation",
                system=original_config.system,
                training=original_config.training,
                probe_type=original_config.probe_type,
                probe_params=original_config.probe_params,
                model_type=original_config.model_type,
                model_params=original_config.model_params,
                data_source='hdf5_loader',
                data_params={'h5_path': hdf5_path},
                metrics=original_config.metrics,
                tags=original_config.tags + ['cross_fidelity_validation'],
                notes=f"Cross-fidelity validation of {original_config.name}",
                data_fidelity='sionna'
            )
            
            # Run validation
            result = self.runner.run(validation_config, campaign_name=f"{campaign_name}_validation")
            validation_results.append(result)
        
        # Compute fidelity gaps
        gaps = []
        
        for i, (orig_exp, val_result) in enumerate(zip(top_experiments, validation_results)):
            synthetic_acc = orig_exp['metrics'].get('top_1_accuracy', 0.0)
            sionna_acc = val_result.metrics.get('top_1_accuracy', 0.0)
            gap = synthetic_acc - sionna_acc
            
            gaps.append({
                'rank': i + 1,
                'name': orig_exp['name'],
                'probe_type': orig_exp['probe_type'],
                'model_type': orig_exp['model_type'],
                'synthetic_accuracy': synthetic_acc,
                'sionna_accuracy': sionna_acc,
                'absolute_gap': gap,
                'relative_gap_pct': (gap / synthetic_acc * 100) if synthetic_acc > 0 else 0
            })
        
        report = {
            'campaign_name': campaign_name,
            'top_n': top_n,
            'gaps': gaps,
            'mean_absolute_gap': sum(g['absolute_gap'] for g in gaps) / len(gaps),
            'mean_relative_gap_pct': sum(g['relative_gap_pct'] for g in gaps) / len(gaps),
        }
        
        print(f"\n{'='*60}")
        print("Cross-Fidelity Validation Report")
        print(f"{'='*60}")
        print(f"Mean Absolute Gap: {report['mean_absolute_gap']:.4f}")
        print(f"Mean Relative Gap: {report['mean_relative_gap_pct']:.2f}%")
        
        return report
    
    def show(self, result: ExperimentResult):
        """Display experiment result.
        
        Args:
            result: ExperimentResult to display
        """
        print(f"\n{'='*60}")
        print(f"Experiment: {result.config.name}")
        print(f"{'='*60}")
        print(f"Status: {result.status}")
        print(f"Training time: {result.training_time_seconds:.1f}s")
        print(f"Total epochs: {result.total_epochs}")
        print(f"Best epoch: {result.best_epoch}")
        print(f"\nMetrics:")
        for metric, value in sorted(result.metrics.items()):
            print(f"  {metric}: {value:.4f}")
        
        # Plot training curve if history available
        if result.training_history:
            import matplotlib.pyplot as plt
            
            history = result.training_history
            
            if 'val_accuracy' in history:
                plt.figure(figsize=(10, 4))
                
                plt.subplot(1, 2, 1)
                if 'train_loss' in history and 'val_loss' in history:
                    plt.plot(history['train_loss'], label='Train Loss')
                    plt.plot(history['val_loss'], label='Val Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True)
                    plt.title('Training and Validation Loss')
                
                plt.subplot(1, 2, 2)
                plt.plot(history['val_accuracy'], label='Val Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                plt.title('Validation Accuracy')
                
                plt.tight_layout()
                plt.show()
    
    def plot_comparison(self, results: List[ExperimentResult]):
        """Plot comparison of multiple results.
        
        Args:
            results: List of ExperimentResults to compare
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        names = [r.config.probe_type for r in results]
        accuracies = [r.metrics.get('top_1_accuracy', 0.0) for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(names, accuracies)
        plt.xlabel('Configuration')
        plt.ylabel('Top-1 Accuracy')
        plt.title('Performance Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()
    
    def plot_campaign(self, campaign: SearchCampaignResult):
        """Plot campaign results.
        
        Args:
            campaign: SearchCampaignResult to visualize
        """
        # Use analyzer to generate plots
        self.analyzer.plot_probe_comparison(campaign_name=campaign.campaign_name)
        self.analyzer.plot_model_comparison(campaign_name=campaign.campaign_name)
    
    def plot_fidelity_gap(self, gap_report: Dict[str, Any]):
        """Plot fidelity gap report.
        
        Args:
            gap_report: Fidelity gap report from validate_on_sionna
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        gaps = gap_report['gaps']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Synthetic vs Sionna accuracy
        ranks = [g['rank'] for g in gaps]
        synthetic = [g['synthetic_accuracy'] for g in gaps]
        sionna = [g['sionna_accuracy'] for g in gaps]
        
        x = np.arange(len(ranks))
        width = 0.35
        
        axes[0].bar(x - width/2, synthetic, width, label='Synthetic')
        axes[0].bar(x + width/2, sionna, width, label='Sionna')
        axes[0].set_xlabel('Rank')
        axes[0].set_ylabel('Top-1 Accuracy')
        axes[0].set_title('Synthetic vs Sionna Performance')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(ranks)
        axes[0].legend()
        axes[0].grid(True, axis='y')
        
        # Plot 2: Relative gap
        relative_gaps = [g['relative_gap_pct'] for g in gaps]
        names = [g['probe_type'] for g in gaps]
        
        axes[1].barh(names, relative_gaps)
        axes[1].set_xlabel('Relative Gap (%)')
        axes[1].set_ylabel('Configuration')
        axes[1].set_title('Fidelity Gap by Configuration')
        axes[1].grid(True, axis='x')
        
        plt.tight_layout()
        plt.show()
