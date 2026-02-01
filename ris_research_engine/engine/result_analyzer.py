"""Result analyzer for aggregating and visualizing experiment results."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from ris_research_engine.foundation import ResultTracker, ExperimentResult, ExperimentConfig


class ResultAnalyzer:
    """Analyzes and visualizes experiment results from the database."""
    
    def __init__(self, tracker: ResultTracker):
        """Initialize the result analyzer.
        
        Args:
            tracker: ResultTracker instance for database access
        """
        self.tracker = tracker
        sns.set_style('whitegrid')
    
    def best_configuration(self, 
                          metric: str = 'top_1_accuracy',
                          campaign_name: Optional[str] = None,
                          maximize: bool = True) -> Optional[ExperimentResult]:
        """Find the best configuration based on a metric.
        
        Args:
            metric: Metric name to optimize
            campaign_name: Optional campaign filter
            maximize: If True, maximize metric; else minimize
            
        Returns:
            Best ExperimentResult or None
        """
        exp_dict = self.tracker.get_best_experiment(
            metric_name=metric,
            campaign_name=campaign_name,
            maximize=maximize
        )
        
        if exp_dict is None:
            return None
        
        # Convert dict to ExperimentResult
        return self._dict_to_result(exp_dict)
    
    def compare_probes(self, 
                      metric: str = 'top_1_accuracy',
                      campaign_name: Optional[str] = None) -> pd.DataFrame:
        """Compare performance across probe types.
        
        Args:
            metric: Metric to compare
            campaign_name: Optional campaign filter
            
        Returns:
            DataFrame with probe comparison
        """
        experiments = self.tracker.get_all_experiments(
            campaign_name=campaign_name,
            status='completed'
        )
        
        if not experiments:
            return pd.DataFrame()
        
        # Group by probe type
        probe_stats = {}
        
        for exp in experiments:
            probe_type = exp['probe_type']
            
            if probe_type not in probe_stats:
                probe_stats[probe_type] = {
                    'values': [],
                    'M': exp['M'],
                    'K': exp['K'],
                    'N': exp['N'],
                }
            
            if metric in exp['metrics']:
                probe_stats[probe_type]['values'].append(exp['metrics'][metric])
        
        # Compute statistics
        rows = []
        for probe_type, stats in probe_stats.items():
            if stats['values']:
                rows.append({
                    'probe_type': probe_type,
                    'mean': np.mean(stats['values']),
                    'std': np.std(stats['values']),
                    'min': np.min(stats['values']),
                    'max': np.max(stats['values']),
                    'count': len(stats['values']),
                    'M': stats['M'],
                    'K': stats['K'],
                    'N': stats['N'],
                })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('mean', ascending=False)
        
        return df
    
    def compare_models(self,
                      metric: str = 'top_1_accuracy',
                      campaign_name: Optional[str] = None) -> pd.DataFrame:
        """Compare performance across model types.
        
        Args:
            metric: Metric to compare
            campaign_name: Optional campaign filter
            
        Returns:
            DataFrame with model comparison
        """
        experiments = self.tracker.get_all_experiments(
            campaign_name=campaign_name,
            status='completed'
        )
        
        if not experiments:
            return pd.DataFrame()
        
        # Group by model type
        model_stats = {}
        
        for exp in experiments:
            model_type = exp['model_type']
            
            if model_type not in model_stats:
                model_stats[model_type] = {
                    'values': [],
                    'training_times': [],
                    'params': [],
                }
            
            if metric in exp['metrics']:
                model_stats[model_type]['values'].append(exp['metrics'][metric])
                model_stats[model_type]['training_times'].append(exp['training_time_seconds'])
                model_stats[model_type]['params'].append(exp['model_parameters'])
        
        # Compute statistics
        rows = []
        for model_type, stats in model_stats.items():
            if stats['values']:
                rows.append({
                    'model_type': model_type,
                    'mean': np.mean(stats['values']),
                    'std': np.std(stats['values']),
                    'min': np.min(stats['values']),
                    'max': np.max(stats['values']),
                    'count': len(stats['values']),
                    'avg_training_time': np.mean(stats['training_times']),
                    'avg_params': np.mean(stats['params']),
                })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('mean', ascending=False)
        
        return df
    
    def sparsity_analysis(self,
                         metric: str = 'top_1_accuracy',
                         campaign_name: Optional[str] = None) -> pd.DataFrame:
        """Analyze performance vs sparsity (M/K ratio).
        
        Args:
            metric: Metric to analyze
            campaign_name: Optional campaign filter
            
        Returns:
            DataFrame with sparsity analysis
        """
        experiments = self.tracker.get_all_experiments(
            campaign_name=campaign_name,
            status='completed'
        )
        
        if not experiments:
            return pd.DataFrame()
        
        rows = []
        for exp in experiments:
            if metric in exp['metrics']:
                rows.append({
                    'probe_type': exp['probe_type'],
                    'model_type': exp['model_type'],
                    'M': exp['M'],
                    'K': exp['K'],
                    'N': exp['N'],
                    'sparsity_ratio': exp['M'] / exp['K'],
                    metric: exp['metrics'][metric],
                    'training_time': exp['training_time_seconds'],
                })
        
        df = pd.DataFrame(rows)
        return df
    
    def fidelity_gap_analysis(self,
                             metric: str = 'top_1_accuracy') -> pd.DataFrame:
        """Analyze fidelity gap (synthetic vs high-fidelity data).
        
        Args:
            metric: Metric to analyze
            
        Returns:
            DataFrame with fidelity gap analysis
        """
        pairs = self.tracker.get_cross_fidelity_pairs()
        
        if not pairs:
            return pd.DataFrame()
        
        rows = []
        for low_fid, high_fid in pairs:
            if metric in low_fid['metrics'] and metric in high_fid['metrics']:
                gap = low_fid['metrics'][metric] - high_fid['metrics'][metric]
                relative_gap = gap / low_fid['metrics'][metric] if low_fid['metrics'][metric] > 0 else 0
                
                rows.append({
                    'probe_type': low_fid['probe_type'],
                    'model_type': low_fid['model_type'],
                    'M': low_fid['M'],
                    'K': low_fid['K'],
                    'low_fidelity': low_fid['data_fidelity'],
                    'high_fidelity': high_fid['data_fidelity'],
                    f'{metric}_low': low_fid['metrics'][metric],
                    f'{metric}_high': high_fid['metrics'][metric],
                    'absolute_gap': gap,
                    'relative_gap_pct': relative_gap * 100,
                })
        
        df = pd.DataFrame(rows)
        return df
    
    def plot_training_curves(self, experiment_id: int, save_path: Optional[str] = None):
        """Plot training curves for an experiment.
        
        Args:
            experiment_id: Database ID of experiment
            save_path: Optional path to save plot
        """
        exp = self.tracker.get_experiment(experiment_id)
        
        if exp is None:
            print(f"Experiment {experiment_id} not found")
            return
        
        history = exp.get('training_history', {})
        
        if not history:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        if 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0].plot(epochs, history['train_loss'], label='Train Loss')
            axes[0].plot(epochs, history['val_loss'], label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True)
        
        # Plot accuracy
        if 'val_accuracy' in history:
            epochs = range(1, len(history['val_accuracy']) + 1)
            axes[1].plot(epochs, history['val_accuracy'], label='Val Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        
        fig.suptitle(f"Experiment: {exp['name']}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_probe_comparison(self, 
                             metric: str = 'top_1_accuracy',
                             campaign_name: Optional[str] = None,
                             save_path: Optional[str] = None):
        """Plot probe comparison.
        
        Args:
            metric: Metric to compare
            campaign_name: Optional campaign filter
            save_path: Optional path to save plot
        """
        df = self.compare_probes(metric=metric, campaign_name=campaign_name)
        
        if df.empty:
            print("No data available for probe comparison")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar plot with error bars
        ax.bar(df['probe_type'], df['mean'], yerr=df['std'], capsize=5)
        ax.set_xlabel('Probe Type')
        ax.set_ylabel(metric)
        ax.set_title(f'Probe Comparison: {metric}')
        ax.grid(True, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self,
                             metric: str = 'top_1_accuracy',
                             campaign_name: Optional[str] = None,
                             save_path: Optional[str] = None):
        """Plot model comparison.
        
        Args:
            metric: Metric to compare
            campaign_name: Optional campaign filter
            save_path: Optional path to save plot
        """
        df = self.compare_models(metric=metric, campaign_name=campaign_name)
        
        if df.empty:
            print("No data available for model comparison")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar plot with error bars
        ax.bar(df['model_type'], df['mean'], yerr=df['std'], capsize=5)
        ax.set_xlabel('Model Type')
        ax.set_ylabel(metric)
        ax.set_title(f'Model Comparison: {metric}')
        ax.grid(True, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sparsity_analysis(self,
                              metric: str = 'top_1_accuracy',
                              campaign_name: Optional[str] = None,
                              save_path: Optional[str] = None):
        """Plot sparsity analysis.
        
        Args:
            metric: Metric to analyze
            campaign_name: Optional campaign filter
            save_path: Optional path to save plot
        """
        df = self.sparsity_analysis(metric=metric, campaign_name=campaign_name)
        
        if df.empty:
            print("No data available for sparsity analysis")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot by probe type
        for probe_type in df['probe_type'].unique():
            probe_data = df[df['probe_type'] == probe_type]
            ax.plot(probe_data['sparsity_ratio'], probe_data[metric], 
                   marker='o', label=probe_type)
        
        ax.set_xlabel('Sensing Budget (M/K)')
        ax.set_ylabel(metric)
        ax.set_title(f'Accuracy vs Sparsity')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_fidelity_gap(self,
                         metric: str = 'top_1_accuracy',
                         save_path: Optional[str] = None):
        """Plot fidelity gap analysis.
        
        Args:
            metric: Metric to analyze
            save_path: Optional path to save plot
        """
        df = self.fidelity_gap_analysis(metric=metric)
        
        if df.empty:
            print("No cross-fidelity data available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot: synthetic vs high-fidelity
        axes[0].scatter(df[f'{metric}_low'], df[f'{metric}_high'])
        
        # Add diagonal line (perfect agreement)
        min_val = min(df[f'{metric}_low'].min(), df[f'{metric}_high'].min())
        max_val = max(df[f'{metric}_low'].max(), df[f'{metric}_high'].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Agreement')
        
        axes[0].set_xlabel(f'{metric} (Low Fidelity)')
        axes[0].set_ylabel(f'{metric} (High Fidelity)')
        axes[0].set_title('Fidelity Comparison')
        axes[0].legend()
        axes[0].grid(True)
        
        # Bar plot: relative gap by probe type
        gap_by_probe = df.groupby('probe_type')['relative_gap_pct'].mean().sort_values()
        axes[1].barh(gap_by_probe.index, gap_by_probe.values)
        axes[1].set_xlabel('Mean Relative Gap (%)')
        axes[1].set_ylabel('Probe Type')
        axes[1].set_title('Fidelity Gap by Probe Type')
        axes[1].grid(True, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _dict_to_result(self, exp_dict: Dict[str, Any]) -> ExperimentResult:
        """Convert experiment dict to ExperimentResult object."""
        # Reconstruct config
        config = ExperimentConfig.from_dict(exp_dict['full_config'])
        
        # Create result
        result = ExperimentResult(
            config=config,
            metrics=exp_dict['metrics'],
            training_history=exp_dict.get('training_history', {}),
            best_epoch=exp_dict['best_epoch'],
            total_epochs=exp_dict['total_epochs'],
            training_time_seconds=exp_dict['training_time_seconds'],
            model_parameters=exp_dict['model_parameters'],
            timestamp=exp_dict['timestamp'],
            status=exp_dict['status'],
            error_message=exp_dict.get('error_message', ''),
            baseline_results=exp_dict['baseline_results'],
            primary_metric_name='top_1_accuracy',
            primary_metric_value=exp_dict['metrics'].get('top_1_accuracy', 0.0)
        )
        
        return result
