"""Report generator for creating visualizations of experiment results."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from ris_research_engine.foundation import ResultTracker
from ris_research_engine.foundation.logging_config import get_logger

logger = get_logger(__name__)

# Set seaborn style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


class ReportGenerator:
    """Generate publication-quality visualizations of experiment results."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = ResultTracker()
        
        logger.info(f"Report generator initialized. Output directory: {self.output_dir}")
    
    def _save_figure(self, fig: plt.Figure, name: str):
        """Save figure in both PNG and PDF formats."""
        png_path = self.output_dir / f"{name}.png"
        pdf_path = self.output_dir / f"{name}.pdf"
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        
        logger.info(f"Saved figure: {png_path} and {pdf_path}")
        plt.close(fig)
    
    def probe_comparison_bar(
        self, 
        experiment_ids: List[int], 
        metric: str = 'top_1_accuracy'
    ):
        """
        Generate bar chart comparing probe types.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metric: Metric to display (default: 'top_1_accuracy')
        """
        logger.info(f"Generating probe comparison bar chart for metric: {metric}")
        
        # Collect data
        data = []
        for exp_id in experiment_ids:
            exp = self.tracker.get_experiment(exp_id)
            if exp and exp['status'] == 'completed':
                data.append({
                    'probe_type': exp['probe_type'],
                    'metric_value': exp['metrics'].get(metric, 0.0),
                    'M': exp['M']
                })
        
        if not data:
            logger.warning("No data to plot")
            return
        
        df = pd.DataFrame(data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by probe type and compute mean
        grouped = df.groupby('probe_type')['metric_value'].agg(['mean', 'std']).reset_index()
        
        # Create bar plot
        x_pos = np.arange(len(grouped))
        ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], capsize=5, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(grouped['probe_type'], rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xlabel('Probe Type')
        ax.set_title(f'Probe Comparison: {metric.replace("_", " ").title()}')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, f'probe_comparison_{metric}')
    
    def sparsity_curve(self, experiment_ids: List[int]):
        """
        Generate line plot of M/K ratio vs accuracy.
        
        Args:
            experiment_ids: List of experiment IDs to analyze
        """
        logger.info("Generating sparsity curve")
        
        # Collect data
        data = []
        for exp_id in experiment_ids:
            exp = self.tracker.get_experiment(exp_id)
            if exp and exp['status'] == 'completed':
                M = exp['M']
                K = exp['K']
                sparsity = M / K if K > 0 else 0.0
                
                data.append({
                    'sparsity_ratio': sparsity,
                    'top_1_accuracy': exp['metrics'].get('top_1_accuracy', 0.0),
                    'probe_type': exp['probe_type'],
                    'model_type': exp['model_type']
                })
        
        if not data:
            logger.warning("No data to plot")
            return
        
        df = pd.DataFrame(data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot by probe type
        for probe_type in df['probe_type'].unique():
            probe_data = df[df['probe_type'] == probe_type]
            probe_data_sorted = probe_data.sort_values('sparsity_ratio')
            
            ax.plot(
                probe_data_sorted['sparsity_ratio'],
                probe_data_sorted['top_1_accuracy'],
                marker='o',
                label=probe_type,
                linewidth=2
            )
        
        ax.set_xlabel('Sparsity Ratio (M/K)')
        ax.set_ylabel('Top-1 Accuracy')
        ax.set_title('Sparsity vs Accuracy')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'sparsity_curve')
    
    def model_comparison_bar(
        self, 
        experiment_ids: List[int], 
        metric: str = 'top_1_accuracy'
    ):
        """
        Generate bar chart comparing model architectures.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metric: Metric to display
        """
        logger.info(f"Generating model comparison bar chart for metric: {metric}")
        
        # Collect data
        data = []
        for exp_id in experiment_ids:
            exp = self.tracker.get_experiment(exp_id)
            if exp and exp['status'] == 'completed':
                data.append({
                    'model_type': exp['model_type'],
                    'metric_value': exp['metrics'].get(metric, 0.0)
                })
        
        if not data:
            logger.warning("No data to plot")
            return
        
        df = pd.DataFrame(data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by model type
        grouped = df.groupby('model_type')['metric_value'].agg(['mean', 'std']).reset_index()
        
        # Create bar plot
        x_pos = np.arange(len(grouped))
        ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], capsize=5, alpha=0.8, color='skyblue')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(grouped['model_type'], rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xlabel('Model Type')
        ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()}')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, f'model_comparison_{metric}')
    
    def training_curves(self, experiment_id: int):
        """
        Generate training and validation curves (loss and accuracy).
        
        Args:
            experiment_id: Experiment ID to visualize
        """
        logger.info(f"Generating training curves for experiment {experiment_id}")
        
        exp = self.tracker.get_experiment(experiment_id)
        
        if not exp or exp['status'] != 'completed':
            logger.warning(f"Experiment {experiment_id} not found or not completed")
            return
        
        history = exp['training_history']
        
        if not history:
            logger.warning("No training history available")
            return
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        if 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
            ax1.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(alpha=0.3)
        
        # Accuracy curves
        if 'train_acc' in history and 'val_acc' in history:
            epochs = range(1, len(history['train_acc']) + 1)
            ax2.plot(epochs, history['train_acc'], label='Train Accuracy', linewidth=2)
            ax2.plot(epochs, history['val_acc'], label='Val Accuracy', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, f'training_curves_exp{experiment_id}')
    
    def baseline_comparison(self, experiment_id: int):
        """
        Generate bar chart comparing ML model vs baselines.
        
        Args:
            experiment_id: Experiment ID to visualize
        """
        logger.info(f"Generating baseline comparison for experiment {experiment_id}")
        
        exp = self.tracker.get_experiment(experiment_id)
        
        if not exp or exp['status'] != 'completed':
            logger.warning(f"Experiment {experiment_id} not found or not completed")
            return
        
        # Collect data
        labels = []
        top_1_values = []
        top_5_values = []
        top_10_values = []
        
        # ML model results
        labels.append('ML Model')
        top_1_values.append(exp['metrics'].get('top_1_accuracy', 0.0))
        top_5_values.append(exp['metrics'].get('top_5_accuracy', 0.0))
        top_10_values.append(exp['metrics'].get('top_10_accuracy', 0.0))
        
        # Baseline results
        baseline_results = exp['baseline_results']
        for baseline_name, metrics in baseline_results.items():
            labels.append(baseline_name.replace('_', ' ').title())
            top_1_values.append(metrics.get('top_1_accuracy', 0.0))
            top_5_values.append(metrics.get('top_5_accuracy', 0.0))
            top_10_values.append(metrics.get('top_10_accuracy', 0.0))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(labels))
        width = 0.25
        
        ax.bar(x - width, top_1_values, width, label='Top-1', alpha=0.8)
        ax.bar(x, top_5_values, width, label='Top-5', alpha=0.8)
        ax.bar(x + width, top_10_values, width, label='Top-10', alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Accuracy')
        ax.set_title('ML Model vs Baseline Methods')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, f'baseline_comparison_exp{experiment_id}')
    
    def heatmap_probe_model(self, experiment_ids: List[int]):
        """
        Generate 2D heatmap of probe × model performance.
        
        Args:
            experiment_ids: List of experiment IDs to analyze
        """
        logger.info("Generating probe × model heatmap")
        
        # Collect data
        data = []
        for exp_id in experiment_ids:
            exp = self.tracker.get_experiment(exp_id)
            if exp and exp['status'] == 'completed':
                data.append({
                    'probe_type': exp['probe_type'],
                    'model_type': exp['model_type'],
                    'top_1_accuracy': exp['metrics'].get('top_1_accuracy', 0.0)
                })
        
        if not data:
            logger.warning("No data to plot")
            return
        
        df = pd.DataFrame(data)
        
        # Pivot table for heatmap
        pivot = df.pivot_table(
            values='top_1_accuracy',
            index='probe_type',
            columns='model_type',
            aggfunc='mean'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Top-1 Accuracy'},
            ax=ax
        )
        
        ax.set_title('Probe × Model Performance Heatmap')
        ax.set_xlabel('Model Type')
        ax.set_ylabel('Probe Type')
        
        plt.tight_layout()
        self._save_figure(fig, 'heatmap_probe_model')
    
    def pareto_front(
        self, 
        experiment_ids: List[int],
        metric1: str = 'top_1_accuracy',
        metric2: str = 'inference_time'
    ):
        """
        Generate Pareto front scatter plot.
        
        Args:
            experiment_ids: List of experiment IDs to analyze
            metric1: First metric (to maximize)
            metric2: Second metric (to minimize, e.g., inference_time)
        """
        logger.info(f"Generating Pareto front for {metric1} vs {metric2}")
        
        # Collect data
        data = []
        for exp_id in experiment_ids:
            exp = self.tracker.get_experiment(exp_id)
            if exp and exp['status'] == 'completed':
                data.append({
                    'experiment_id': exp_id,
                    'probe_type': exp['probe_type'],
                    'model_type': exp['model_type'],
                    metric1: exp['metrics'].get(metric1, 0.0),
                    metric2: exp['metrics'].get(metric2, 0.0)
                })
        
        if not data:
            logger.warning("No data to plot")
            return
        
        df = pd.DataFrame(data)
        
        # Identify Pareto front using vectorized operations for better performance
        # For metric1: maximize (higher is better)
        # For metric2: minimize (lower is better)
        values = df[[metric1, metric2]].values
        pareto_mask = np.ones(len(df), dtype=bool)
        
        # For small datasets (< 100), use simple O(n²) algorithm
        # For larger datasets, this could be optimized further with spatial indexing
        if len(df) < 100:
            for i in range(len(values)):
                if pareto_mask[i]:
                    # Point i is dominated by point j if:
                    # j[metric1] >= i[metric1] AND j[metric2] <= i[metric2]
                    # with at least one strict inequality
                    dominated = np.logical_and(
                        values[:, 0] >= values[i, 0],  # metric1: higher or equal
                        values[:, 1] <= values[i, 1]   # metric2: lower or equal
                    )
                    # Exclude i itself and check for strict domination
                    dominated[i] = False
                    strictly_better = np.logical_or(
                        values[:, 0] > values[i, 0],
                        values[:, 1] < values[i, 1]
                    )
                    if np.any(np.logical_and(dominated, strictly_better)):
                        pareto_mask[i] = False
        else:
            # For large datasets, use simple iteration (can be optimized further if needed)
            logger.info(f"Computing Pareto front for {len(df)} points (may take a moment)")
            for i in range(len(values)):
                for j in range(len(values)):
                    if i != j:
                        if (values[j, 0] >= values[i, 0] and 
                            values[j, 1] <= values[i, 1] and
                            (values[j, 0] > values[i, 0] or values[j, 1] < values[i, 1])):
                            pareto_mask[i] = False
                            break
        
        df['is_pareto'] = pareto_mask
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot non-Pareto points
        non_pareto = df[~df['is_pareto']]
        ax.scatter(
            non_pareto[metric2],
            non_pareto[metric1],
            alpha=0.5,
            s=50,
            label='Non-Pareto',
            color='gray'
        )
        
        # Plot Pareto points
        pareto = df[df['is_pareto']]
        ax.scatter(
            pareto[metric2],
            pareto[metric1],
            alpha=0.8,
            s=100,
            label='Pareto Front',
            color='red',
            marker='*'
        )
        
        # Draw Pareto front line
        if len(pareto) > 1:
            pareto_sorted = pareto.sort_values(metric2)
            ax.plot(
                pareto_sorted[metric2],
                pareto_sorted[metric1],
                'r--',
                alpha=0.5,
                linewidth=2
            )
        
        ax.set_xlabel(metric2.replace('_', ' ').title())
        ax.set_ylabel(metric1.replace('_', ' ').title())
        ax.set_title(f'Pareto Front: {metric1.replace("_", " ").title()} vs {metric2.replace("_", " ").title()}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, f'pareto_front_{metric1}_{metric2}')
    
    def ranking_distribution(self, experiment_ids: List[int]):
        """
        Generate histogram of top-k accuracies.
        
        Args:
            experiment_ids: List of experiment IDs to analyze
        """
        logger.info("Generating ranking distribution histogram")
        
        # Collect data
        top_1_values = []
        top_5_values = []
        top_10_values = []
        
        for exp_id in experiment_ids:
            exp = self.tracker.get_experiment(exp_id)
            if exp and exp['status'] == 'completed':
                top_1_values.append(exp['metrics'].get('top_1_accuracy', 0.0))
                top_5_values.append(exp['metrics'].get('top_5_accuracy', 0.0))
                top_10_values.append(exp['metrics'].get('top_10_accuracy', 0.0))
        
        if not top_1_values:
            logger.warning("No data to plot")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bins = np.linspace(0, 1, 21)
        
        ax.hist(top_1_values, bins=bins, alpha=0.5, label='Top-1', edgecolor='black')
        ax.hist(top_5_values, bins=bins, alpha=0.5, label='Top-5', edgecolor='black')
        ax.hist(top_10_values, bins=bins, alpha=0.5, label='Top-10', edgecolor='black')
        
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Top-K Accuracies')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'ranking_distribution')
    
    def fidelity_comparison(
        self, 
        synthetic_ids: List[int], 
        sionna_ids: List[int]
    ):
        """
        Generate side-by-side bar chart comparing synthetic vs Sionna results.
        
        Args:
            synthetic_ids: List of synthetic experiment IDs
            sionna_ids: List of Sionna experiment IDs
        """
        logger.info("Generating fidelity comparison")
        
        # Collect synthetic data
        synthetic_data = []
        for exp_id in synthetic_ids:
            exp = self.tracker.get_experiment(exp_id)
            if exp and exp['status'] == 'completed':
                synthetic_data.append({
                    'config': f"{exp['probe_type']}_{exp['model_type']}",
                    'top_1_accuracy': exp['metrics'].get('top_1_accuracy', 0.0)
                })
        
        # Collect Sionna data
        sionna_data = []
        for exp_id in sionna_ids:
            exp = self.tracker.get_experiment(exp_id)
            if exp and exp['status'] == 'completed':
                sionna_data.append({
                    'config': f"{exp['probe_type']}_{exp['model_type']}",
                    'top_1_accuracy': exp['metrics'].get('top_1_accuracy', 0.0)
                })
        
        if not synthetic_data or not sionna_data:
            logger.warning("Insufficient data to plot")
            return
        
        # Create DataFrames
        df_synth = pd.DataFrame(synthetic_data).groupby('config')['top_1_accuracy'].mean().reset_index()
        df_sionna = pd.DataFrame(sionna_data).groupby('config')['top_1_accuracy'].mean().reset_index()
        
        # Merge on config
        df_merged = pd.merge(df_synth, df_sionna, on='config', suffixes=('_synthetic', '_sionna'))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(df_merged))
        width = 0.35
        
        ax.bar(x - width/2, df_merged['top_1_accuracy_synthetic'], width, label='Synthetic', alpha=0.8)
        ax.bar(x + width/2, df_merged['top_1_accuracy_sionna'], width, label='Sionna', alpha=0.8)
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Top-1 Accuracy')
        ax.set_title('Synthetic vs Sionna Fidelity Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(df_merged['config'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'fidelity_comparison')
