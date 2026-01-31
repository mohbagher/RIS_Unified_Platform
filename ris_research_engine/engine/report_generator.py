"""Report generator for creating visualizations and plots."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available, plotting disabled")

from ris_research_engine.foundation.data_types import ExperimentResult, SearchCampaignResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate publication-quality plots and reports."""
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize report generator.
        
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        if MATPLOTLIB_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_context("paper")
    
    def _save_figure(self, fig, name: str):
        """Save figure as PNG and PDF.
        
        Args:
            fig: Matplotlib figure
            name: Base filename (without extension)
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot save")
            return
        
        png_path = self.output_dir / f"{name}.png"
        pdf_path = self.output_dir / f"{name}.pdf"
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        
        logger.info(f"Saved plot to {png_path} and {pdf_path}")
    
    def probe_comparison_bar(
        self,
        comparison_df: pd.DataFrame,
        metric_name: str = 'mean',
        title: str = "Probe Comparison",
        filename: Optional[str] = None
    ):
        """Create bar plot comparing probe types.
        
        Args:
            comparison_df: DataFrame from ResultAnalyzer.compare_probes()
            metric_name: Column name to plot
            title: Plot title
            filename: Optional custom filename
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = comparison_df['probe_type']
        y = comparison_df[metric_name]
        err = comparison_df.get('std', None)
        
        bars = ax.bar(x, y, yerr=err, capsize=5, alpha=0.8)
        
        # Color bars
        colors = sns.color_palette("husl", len(x))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Probe Type', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        fname = filename or f"probe_comparison_{metric_name}"
        self._save_figure(fig, fname)
        plt.close(fig)
    
    def sparsity_curve(
        self,
        sparsity_df: pd.DataFrame,
        title: str = "Performance vs Sparsity",
        filename: Optional[str] = None
    ):
        """Create line plot showing performance vs M/K ratio.
        
        Args:
            sparsity_df: DataFrame from ResultAnalyzer.sparsity_analysis()
            title: Plot title
            filename: Optional custom filename
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = sparsity_df['M_K_ratio']
        y = sparsity_df['mean']
        err = sparsity_df.get('std', None)
        
        ax.plot(x, y, marker='o', linewidth=2, markersize=8)
        
        if err is not None:
            ax.fill_between(x, y - err, y + err, alpha=0.3)
        
        ax.set_xlabel('M/K Ratio (Sparsity)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fname = filename or "sparsity_curve"
        self._save_figure(fig, fname)
        plt.close(fig)
    
    def model_comparison_bar(
        self,
        comparison_df: pd.DataFrame,
        title: str = "Model Architecture Comparison",
        filename: Optional[str] = None
    ):
        """Create bar plot comparing model architectures.
        
        Args:
            comparison_df: DataFrame from ResultAnalyzer.compare_models()
            title: Plot title
            filename: Optional custom filename
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Performance comparison
        x = comparison_df['model_type']
        y = comparison_df['mean_metric']
        err = comparison_df.get('std_metric', None)
        
        bars = ax1.bar(x, y, yerr=err, capsize=5, alpha=0.8)
        colors = sns.color_palette("husl", len(x))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax1.set_xlabel('Model Type', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Performance', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Parameters comparison
        y_params = comparison_df['mean_params']
        bars = ax2.bar(x, y_params, alpha=0.8)
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax2.set_xlabel('Model Type', fontsize=12)
        ax2.set_ylabel('Parameters', fontsize=12)
        ax2.set_title('Model Size', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        fname = filename or "model_comparison"
        self._save_figure(fig, fname)
        plt.close(fig)
    
    def training_curves(
        self,
        result: ExperimentResult,
        title: Optional[str] = None,
        filename: Optional[str] = None
    ):
        """Plot training and validation curves.
        
        Args:
            result: ExperimentResult with training history
            title: Plot title
            filename: Optional custom filename
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot")
            return
        
        history = result.training_history
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        if 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
            ax1.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
            ax1.axvline(x=result.best_epoch + 1, color='r', linestyle='--', label='Best Epoch')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Metric curves
        metric_keys = [k for k in history.keys() if k.startswith('val_') and k != 'val_loss']
        if metric_keys:
            epochs = range(1, len(history[metric_keys[0]]) + 1)
            for metric_key in metric_keys[:3]:  # Plot up to 3 metrics
                metric_name = metric_key.replace('val_', '')
                ax2.plot(epochs, history[metric_key], label=metric_name, linewidth=2)
            
            ax2.axvline(x=result.best_epoch + 1, color='r', linestyle='--', label='Best Epoch')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Metric Value', fontsize=12)
            ax2.set_title('Validation Metrics', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        if title is None:
            title = f"Training Curves - {result.config.name}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        fname = filename or f"training_curves_{result.config.name}"
        self._save_figure(fig, fname)
        plt.close(fig)
    
    def baseline_comparison(
        self,
        result: ExperimentResult,
        metric_name: str = 'top_1_accuracy',
        title: Optional[str] = None,
        filename: Optional[str] = None
    ):
        """Compare model performance with baselines.
        
        Args:
            result: ExperimentResult with baseline results
            metric_name: Metric to compare
            title: Plot title
            filename: Optional custom filename
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        methods = ['Model (Ours)']
        values = [result.metrics.get(metric_name, 0.0)]
        
        for baseline_name, baseline_metrics in result.baseline_results.items():
            methods.append(baseline_name)
            values.append(baseline_metrics.get(metric_name, 0.0))
        
        # Create bar plot
        colors = ['#2ecc71'] + ['#95a5a6'] * (len(methods) - 1)
        bars = ax.bar(methods, values, color=colors, alpha=0.8)
        
        # Highlight best method
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title or 'Model vs Baselines', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        fname = filename or f"baseline_comparison_{result.config.name}"
        self._save_figure(fig, fname)
        plt.close(fig)
    
    def heatmap_probe_model(
        self,
        results: List[ExperimentResult],
        metric_name: str = 'top_1_accuracy',
        title: str = "Probe-Model Performance Heatmap",
        filename: Optional[str] = None
    ):
        """Create heatmap showing probe-model combinations.
        
        Args:
            results: List of experiment results
            metric_name: Metric to display
            title: Plot title
            filename: Optional custom filename
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot")
            return
        
        # Create matrix
        probe_types = sorted(set(r.config.probe_type for r in results))
        model_types = sorted(set(r.config.model_type for r in results))
        
        matrix = np.zeros((len(probe_types), len(model_types)))
        counts = np.zeros((len(probe_types), len(model_types)))
        
        for result in results:
            i = probe_types.index(result.config.probe_type)
            j = model_types.index(result.config.model_type)
            matrix[i, j] += result.metrics.get(metric_name, 0.0)
            counts[i, j] += 1
        
        # Average results
        matrix = np.divide(matrix, counts, where=counts > 0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(model_types)))
        ax.set_yticks(np.arange(len(probe_types)))
        ax.set_xticklabels(model_types)
        ax.set_yticklabels(probe_types)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(metric_name.replace('_', ' ').title(), rotation=-90, va="bottom")
        
        # Add values to cells
        for i in range(len(probe_types)):
            for j in range(len(model_types)):
                if counts[i, j] > 0:
                    text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                                 ha="center", va="center", color="w", fontsize=9)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Model Type', fontsize=12)
        ax.set_ylabel('Probe Type', fontsize=12)
        plt.tight_layout()
        
        fname = filename or "heatmap_probe_model"
        self._save_figure(fig, fname)
        plt.close(fig)
    
    def pareto_front(
        self,
        results: List[ExperimentResult],
        x_metric: str = 'model_parameters',
        y_metric: str = 'top_1_accuracy',
        title: str = "Pareto Front: Accuracy vs Complexity",
        filename: Optional[str] = None
    ):
        """Plot Pareto front for multi-objective optimization.
        
        Args:
            results: List of experiment results
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            title: Plot title
            filename: Optional custom filename
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        x_values = []
        y_values = []
        labels = []
        
        for result in results:
            if x_metric == 'model_parameters':
                x_val = result.model_parameters
            else:
                x_val = result.metrics.get(x_metric, 0.0)
            
            y_val = result.metrics.get(y_metric, 0.0)
            
            x_values.append(x_val)
            y_values.append(y_val)
            labels.append(f"{result.config.probe_type}+{result.config.model_type}")
        
        # Plot all points
        ax.scatter(x_values, y_values, s=100, alpha=0.6)
        
        # Find Pareto front (minimize x, maximize y)
        pareto_indices = []
        for i, (x_i, y_i) in enumerate(zip(x_values, y_values)):
            is_pareto = True
            for j, (x_j, y_j) in enumerate(zip(x_values, y_values)):
                if i != j:
                    if x_j <= x_i and y_j >= y_i and (x_j < x_i or y_j > y_i):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_indices.append(i)
        
        # Highlight Pareto front
        pareto_x = [x_values[i] for i in pareto_indices]
        pareto_y = [y_values[i] for i in pareto_indices]
        
        # Sort for line plot
        sorted_indices = np.argsort(pareto_x)
        pareto_x = [pareto_x[i] for i in sorted_indices]
        pareto_y = [pareto_y[i] for i in sorted_indices]
        
        ax.plot(pareto_x, pareto_y, 'r-', linewidth=2, label='Pareto Front')
        ax.scatter(pareto_x, pareto_y, s=200, c='red', marker='*', 
                  edgecolors='black', linewidths=2, zorder=5, label='Pareto Optimal')
        
        ax.set_xlabel(x_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fname = filename or "pareto_front"
        self._save_figure(fig, fname)
        plt.close(fig)
    
    def ranking_distribution(
        self,
        results: List[ExperimentResult],
        metric_name: str = 'top_1_accuracy',
        title: str = "Result Distribution",
        filename: Optional[str] = None
    ):
        """Create histogram of result distribution.
        
        Args:
            results: List of experiment results
            metric_name: Metric to plot
            title: Plot title
            filename: Optional custom filename
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        values = [r.metrics.get(metric_name, 0.0) for r in results]
        
        ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        
        # Add median line
        median_val = np.median(values)
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
        
        ax.set_xlabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        fname = filename or "ranking_distribution"
        self._save_figure(fig, fname)
        plt.close(fig)
    
    def fidelity_comparison(
        self,
        fidelity_df: pd.DataFrame,
        title: str = "Synthetic vs Sionna Fidelity",
        filename: Optional[str] = None
    ):
        """Plot fidelity gap analysis.
        
        Args:
            fidelity_df: DataFrame from ResultAnalyzer.fidelity_gap_analysis()
            title: Plot title
            filename: Optional custom filename
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot: Synthetic vs Sionna
        ax1.scatter(fidelity_df['synthetic'], fidelity_df['sionna'], s=100, alpha=0.6)
        
        # Add diagonal line
        min_val = min(fidelity_df['synthetic'].min(), fidelity_df['sionna'].min())
        max_val = max(fidelity_df['synthetic'].max(), fidelity_df['sionna'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
        
        ax1.set_xlabel('Synthetic Performance', fontsize=12)
        ax1.set_ylabel('Sionna Performance', fontsize=12)
        ax1.set_title('Correlation', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bar plot: Fidelity gaps
        x = range(len(fidelity_df))
        gaps = fidelity_df['gap']
        colors = ['green' if g >= 0 else 'red' for g in gaps]
        
        ax2.bar(x, gaps, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Configuration Index', fontsize=12)
        ax2.set_ylabel('Fidelity Gap (Synthetic - Sionna)', fontsize=12)
        ax2.set_title('Fidelity Gaps', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        fname = filename or "fidelity_comparison"
        self._save_figure(fig, fname)
        plt.close(fig)
    
    def generate_full_report(
        self,
        campaign_result: SearchCampaignResult,
        analyzer_results: Dict[str, Any]
    ):
        """Generate complete report with all plots.
        
        Args:
            campaign_result: Search campaign result
            analyzer_results: Dictionary with analysis results from ResultAnalyzer
        """
        logger.info(f"Generating full report for campaign: {campaign_result.campaign_name}")
        
        # Generate all applicable plots
        if 'probe_comparison' in analyzer_results:
            self.probe_comparison_bar(
                analyzer_results['probe_comparison'],
                title=f"Probe Comparison - {campaign_result.campaign_name}"
            )
        
        if 'model_comparison' in analyzer_results:
            self.model_comparison_bar(
                analyzer_results['model_comparison'],
                title=f"Model Comparison - {campaign_result.campaign_name}"
            )
        
        if 'sparsity_analysis' in analyzer_results:
            self.sparsity_curve(
                analyzer_results['sparsity_analysis'],
                title=f"Sparsity Analysis - {campaign_result.campaign_name}"
            )
        
        if campaign_result.best_result:
            self.training_curves(campaign_result.best_result)
            self.baseline_comparison(campaign_result.best_result)
        
        if campaign_result.all_results:
            self.heatmap_probe_model(campaign_result.all_results)
            self.pareto_front(campaign_result.all_results)
            self.ranking_distribution(campaign_result.all_results)
        
        logger.info(f"Report generation complete. Plots saved to {self.output_dir}")
