"""Report generator for creating visualizations and reports."""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from ris_research_engine.foundation import ExperimentResult, ResultTracker
from .result_analyzer import ResultAnalyzer

logger = logging.getLogger(__name__)

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")


class ReportGenerator:
    """Generator for creating plots and reports."""
    
    def __init__(self, db_path: str = "ris_results.db", output_dir: str = "outputs/plots"):
        """
        Initialize report generator.
        
        Args:
            db_path: Path to SQLite database
            output_dir: Directory for saving plots
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = ResultAnalyzer(db_path)
        self.tracker = ResultTracker(db_path)
    
    def probe_comparison_bar(
        self,
        metric: str = 'top_1_accuracy',
        model_type: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create bar chart comparing probe types.
        
        Args:
            metric: Metric to compare
            model_type: Optional filter by model type
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        df = self.analyzer.compare_probes(metric, model_type)
        
        if df.empty:
            logger.warning("No data available for probe comparison")
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        x_pos = np.arange(len(df))
        ax.bar(x_pos, df['mean'], yerr=df['std'], capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df['probe_type'], rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Probe Type Comparison - {metric.replace("_", " ").title()}')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"probe_comparison_{metric}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved probe comparison plot to {save_path}")
        return str(save_path)
    
    def sparsity_curve(
        self,
        metric: str = 'top_1_accuracy',
        probe_type: Optional[str] = None,
        model_type: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create curve showing performance vs sparsity (M/K ratio).
        
        Args:
            metric: Metric to plot
            probe_type: Optional filter by probe type
            model_type: Optional filter by model type
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        df = self.analyzer.sparsity_analysis(metric, probe_type, model_type)
        
        if df.empty:
            logger.warning("No data available for sparsity analysis")
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create line plot with error bars
        ax.errorbar(df['M_K_ratio'], df['mean'], yerr=df['std'], 
                   marker='o', capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel('M/K Ratio (Sparsity)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Performance vs Sparsity - {metric.replace("_", " ").title()}')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"sparsity_curve_{metric}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved sparsity curve to {save_path}")
        return str(save_path)
    
    def model_comparison_bar(
        self,
        metric: str = 'top_1_accuracy',
        probe_type: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create bar chart comparing model types.
        
        Args:
            metric: Metric to compare
            probe_type: Optional filter by probe type
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        df = self.analyzer.compare_models(metric, probe_type)
        
        if df.empty:
            logger.warning("No data available for model comparison")
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        x_pos = np.arange(len(df))
        ax.bar(x_pos, df['mean'], yerr=df['std'], capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df['model_type'], rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Model Type Comparison - {metric.replace("_", " ").title()}')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"model_comparison_{metric}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved model comparison plot to {save_path}")
        return str(save_path)
    
    def training_curves(
        self,
        experiment_id: int,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot training curves (loss and accuracy) for a specific experiment.
        
        Args:
            experiment_id: ID of experiment to plot
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        result = self.tracker.get_result(experiment_id)
        
        if not result:
            logger.warning(f"Experiment {experiment_id} not found")
            return ""
        
        history = result.training_history
        
        if not history:
            logger.warning(f"No training history for experiment {experiment_id}")
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        if 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
            ax1.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
            ax1.axvline(result.best_epoch + 1, color='r', linestyle='--', 
                       label=f'Best Epoch ({result.best_epoch + 1})')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(alpha=0.3)
        
        # Plot accuracy
        if 'val_accuracy' in history:
            epochs = range(1, len(history['val_accuracy']) + 1)
            ax2.plot(epochs, history['val_accuracy'], label='Val Accuracy', 
                    linewidth=2, color='green')
            ax2.axvline(result.best_epoch + 1, color='r', linestyle='--',
                       label=f'Best Epoch ({result.best_epoch + 1})')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Validation Accuracy')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"training_curves_{experiment_id}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves to {save_path}")
        return str(save_path)
    
    def baseline_comparison(
        self,
        experiment_id: int,
        metric: str = 'top_1_accuracy',
        save_path: Optional[str] = None
    ) -> str:
        """
        Compare ML model performance against baselines.
        
        Args:
            experiment_id: ID of experiment
            metric: Metric to compare
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        result = self.tracker.get_result(experiment_id)
        
        if not result:
            logger.warning(f"Experiment {experiment_id} not found")
            return ""
        
        # Collect baseline and model performance
        names = []
        values = []
        
        # Add ML model
        if metric in result.metrics:
            names.append('ML Model')
            values.append(result.metrics[metric])
        
        # Add baselines
        for baseline_name, baseline_metrics in result.baseline_results.items():
            if metric in baseline_metrics:
                names.append(baseline_name)
                values.append(baseline_metrics[metric])
        
        if not names:
            logger.warning(f"No data available for metric {metric}")
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart
        x_pos = np.arange(len(names))
        colors = ['#1f77b4'] + ['#d3d3d3'] * (len(names) - 1)
        ax.bar(x_pos, values, color=colors, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'ML Model vs Baselines - {metric.replace("_", " ").title()}')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"baseline_comparison_{experiment_id}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved baseline comparison to {save_path}")
        return str(save_path)
    
    def heatmap_probe_model(
        self,
        metric: str = 'top_1_accuracy',
        save_path: Optional[str] = None
    ) -> str:
        """
        Create heatmap showing performance for probe-model combinations.
        
        Args:
            metric: Metric to visualize
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        df = self.analyzer.get_results_dataframe({'status': 'completed'})
        
        if df.empty or metric not in df.columns:
            logger.warning(f"No data available for heatmap")
            return ""
        
        # Create pivot table
        pivot = df.pivot_table(
            values=metric,
            index='probe_type',
            columns='model_type',
            aggfunc='mean'
        )
        
        if pivot.empty:
            logger.warning("Not enough data for heatmap")
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': metric.replace('_', ' ').title()},
                   ax=ax)
        ax.set_title(f'Probe-Model Performance Heatmap - {metric.replace("_", " ").title()}')
        ax.set_xlabel('Model Type')
        ax.set_ylabel('Probe Type')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"heatmap_probe_model_{metric}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved heatmap to {save_path}")
        return str(save_path)
    
    def pareto_front(
        self,
        metric_x: str = 'training_time',
        metric_y: str = 'top_1_accuracy',
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot Pareto front showing trade-off between two metrics.
        
        Args:
            metric_x: Metric for x-axis (minimize)
            metric_y: Metric for y-axis (maximize)
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        df = self.analyzer.get_results_dataframe({'status': 'completed'})
        
        if df.empty:
            logger.warning("No data available for Pareto front")
            return ""
        
        # Filter to rows with both metrics
        if metric_x not in df.columns or metric_y not in df.columns:
            logger.warning(f"Required metrics not found")
            return ""
        
        df = df[[metric_x, metric_y, 'probe_type', 'model_type']].dropna()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        scatter = ax.scatter(df[metric_x], df[metric_y], 
                           c=pd.Categorical(df['probe_type']).codes,
                           s=100, alpha=0.6, cmap='tab10')
        
        ax.set_xlabel(metric_x.replace('_', ' ').title())
        ax.set_ylabel(metric_y.replace('_', ' ').title())
        ax.set_title(f'Pareto Front: {metric_y.replace("_", " ").title()} vs {metric_x.replace("_", " ").title()}')
        ax.grid(alpha=0.3)
        
        # Add legend
        handles, labels = scatter.legend_elements()
        ax.legend(handles, df['probe_type'].unique(), title='Probe Type', 
                 loc='best', framealpha=0.9)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"pareto_front_{metric_x}_{metric_y}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Pareto front to {save_path}")
        return str(save_path)
    
    def ranking_distribution(
        self,
        metric: str = 'top_1_accuracy',
        group_by: str = 'probe_type',
        save_path: Optional[str] = None
    ) -> str:
        """
        Show distribution of metric values for each group with box plots.
        
        Args:
            metric: Metric to visualize
            group_by: Column to group by
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        df = self.analyzer.get_results_dataframe({'status': 'completed'})
        
        if df.empty or metric not in df.columns:
            logger.warning(f"No data available for ranking distribution")
            return ""
        
        df = df[[group_by, metric]].dropna()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create box plot
        df.boxplot(column=metric, by=group_by, ax=ax, grid=False)
        ax.set_xlabel(group_by.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Distribution of {metric.replace("_", " ").title()} by {group_by.replace("_", " ").title()}')
        plt.suptitle('')  # Remove default title
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"ranking_distribution_{metric}_{group_by}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ranking distribution to {save_path}")
        return str(save_path)
    
    def fidelity_comparison(
        self,
        synthetic_fidelity: str = 'synthetic',
        high_fidelity: str = 'sionna',
        metric: str = 'top_1_accuracy',
        save_path: Optional[str] = None
    ) -> str:
        """
        Compare performance between synthetic and high-fidelity data.
        
        Args:
            synthetic_fidelity: Name of synthetic fidelity
            high_fidelity: Name of high-fidelity level
            metric: Metric to compare
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        df = self.analyzer.fidelity_gap_analysis(
            synthetic_fidelity, high_fidelity, metric
        )
        
        if df.empty:
            logger.warning("No data available for fidelity comparison")
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot comparing fidelities
        synth_col = f'{synthetic_fidelity}_{metric}'
        hf_col = f'{high_fidelity}_{metric}'
        
        ax.scatter(df[synth_col], df[hf_col], s=100, alpha=0.6)
        
        # Add diagonal line (perfect agreement)
        min_val = min(df[synth_col].min(), df[hf_col].min())
        max_val = max(df[synth_col].max(), df[hf_col].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
               label='Perfect Agreement', linewidth=2)
        
        ax.set_xlabel(f'{synthetic_fidelity.title()} {metric.replace("_", " ").title()}')
        ax.set_ylabel(f'{high_fidelity.title()} {metric.replace("_", " ").title()}')
        ax.set_title(f'Fidelity Comparison: {synthetic_fidelity.title()} vs {high_fidelity.title()}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"fidelity_comparison_{metric}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved fidelity comparison to {save_path}")
        return str(save_path)
