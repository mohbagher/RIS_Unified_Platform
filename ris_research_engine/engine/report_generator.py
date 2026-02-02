"""Report generator for creating visualizations of experiment results."""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from matplotlib.backends.backend_pdf import PdfPages

from ris_research_engine.foundation.data_types import ExperimentResult
from ris_research_engine.foundation.logging_config import get_logger

logger = get_logger(__name__)

# Set consistent seaborn style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class ReportGenerator:
    """Generate publication-quality visualizations of experiment results."""
    
    def __init__(self, output_dir: str = "outputs/reports"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ReportGenerator initialized. Output directory: {self.output_dir}")
    
    def _save_figure(self, fig: plt.Figure, save_path: str):
        """
        Save figure in both PNG (300 DPI) and PDF formats.
        
        Args:
            fig: Matplotlib figure object
            save_path: Base path for saving (without extension)
        """
        save_path = Path(save_path)
        
        # Ensure path is relative to output_dir
        if not save_path.is_absolute():
            save_path = self.output_dir / save_path
        
        # Create parent directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        png_path = save_path.with_suffix('.png')
        pdf_path = save_path.with_suffix('.pdf')
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        
        logger.info(f"Saved figure: {png_path} and {pdf_path}")
        plt.close(fig)
    
    def plot_training_curves(self, result: ExperimentResult, save_path: str):
        """
        Generate training and validation curves (loss and accuracy).
        
        Args:
            result: ExperimentResult object containing training history
            save_path: Path to save the plot
        """
        logger.info(f"Generating training curves for experiment: {result.config.name}")
        
        history = result.training_history
        
        if not history:
            logger.warning("No training history available")
            return
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        if 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
            ax1.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=4)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Mark best epoch
            if result.best_epoch > 0 and result.best_epoch <= len(history['val_loss']):
                ax1.axvline(x=result.best_epoch, color='r', linestyle='--', alpha=0.5, label='Best Epoch')
        
        # Accuracy curves
        if 'val_accuracy' in history:
            epochs = range(1, len(history['val_accuracy']) + 1)
            ax2.plot(epochs, history['val_accuracy'], label='Val Accuracy', linewidth=2, marker='o', markersize=4)
            
            # Also plot train accuracy if available
            if 'train_accuracy' in history and len(history['train_accuracy']) == len(history['val_accuracy']):
                ax2.plot(epochs, history['train_accuracy'], label='Train Accuracy', linewidth=2, marker='s', markersize=4)
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # Mark best epoch
            if result.best_epoch > 0 and result.best_epoch <= len(history['val_accuracy']):
                ax2.axvline(x=result.best_epoch, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
    
    def plot_metric_comparison(self, results: List[ExperimentResult], metric_name: str, save_path: str):
        """
        Generate bar chart comparing a specific metric across experiments.
        
        Args:
            results: List of ExperimentResult objects
            metric_name: Name of the metric to compare
            save_path: Path to save the plot
        """
        logger.info(f"Generating metric comparison for: {metric_name}")
        
        if not results:
            logger.warning("No results to plot")
            return
        
        # Extract data
        labels = []
        values = []
        
        for i, result in enumerate(results):
            labels.append(f"{result.config.probe_type}\n{result.config.model_type}\nM={result.config.system.M}")
            values.append(result.metrics.get(metric_name, 0.0))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(10, len(results) * 1.5), 6))
        
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, values, alpha=0.8, color=sns.color_palette("husl", len(labels)))
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_xlabel('Configuration')
        ax.set_title(f'Comparison: {metric_name.replace("_", " ").title()}')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
    
    def plot_probe_comparison(self, df: pd.DataFrame, save_path: str):
        """
        Generate bar chart comparing probe types.
        
        Args:
            df: DataFrame from ResultAnalyzer.compare_probes()
            save_path: Path to save the plot
        """
        logger.info("Generating probe comparison plot")
        
        if df.empty:
            logger.warning("Empty DataFrame, skipping plot")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract mean and std columns (assuming format from compare_probes)
        if df.index.name == 'probe_type':
            # Data is already grouped
            probe_types = df.index.tolist()
            
            # Find the metric column (should be the first column with 'mean' in name)
            mean_col = [col for col in df.columns if 'mean' in col][0]
            std_col = [col for col in df.columns if 'std' in col][0] if any('std' in col for col in df.columns) else None
            
            means = df[mean_col].values
            stds = df[std_col].values if std_col else np.zeros(len(means))
            
            x_pos = np.arange(len(probe_types))
            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, color=sns.color_palette("Set2", len(probe_types)))
            ax.set_xticks(x_pos)
            ax.set_xticklabels(probe_types, rotation=45, ha='right')
            ax.set_ylabel(mean_col.split('_')[0].replace('_', ' ').title())
            ax.set_xlabel('Probe Type')
            ax.set_title('Probe Type Comparison')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
    
    def plot_model_comparison(self, df: pd.DataFrame, save_path: str):
        """
        Generate bar chart comparing model architectures.
        
        Args:
            df: DataFrame from ResultAnalyzer.compare_models()
            save_path: Path to save the plot
        """
        logger.info("Generating model comparison plot")
        
        if df.empty:
            logger.warning("Empty DataFrame, skipping plot")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract mean and std columns
        if df.index.name == 'model_type':
            model_types = df.index.tolist()
            
            # Find the metric column
            mean_col = [col for col in df.columns if 'mean' in col and 'parameters' not in col][0]
            std_col = [col for col in df.columns if 'std' in col][0] if any('std' in col for col in df.columns) else None
            
            means = df[mean_col].values
            stds = df[std_col].values if std_col else np.zeros(len(means))
            
            x_pos = np.arange(len(model_types))
            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, color=sns.color_palette("Set3", len(model_types)))
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_types, rotation=45, ha='right')
            ax.set_ylabel(mean_col.split('_')[0].replace('_', ' ').title())
            ax.set_xlabel('Model Type')
            ax.set_title('Model Architecture Comparison')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
    
    def plot_sparsity_analysis(self, df: pd.DataFrame, save_path: str):
        """
        Generate line plot of M/K ratio vs accuracy.
        
        Args:
            df: DataFrame from ResultAnalyzer.sparsity_analysis()
            save_path: Path to save the plot
        """
        logger.info("Generating sparsity analysis plot")
        
        if df.empty:
            logger.warning("Empty DataFrame, skipping plot")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot by probe type if available
        if 'probe_type' in df.columns:
            for probe_type in df['probe_type'].unique():
                probe_data = df[df['probe_type'] == probe_type]
                probe_data_sorted = probe_data.sort_values('sparsity_ratio')
                
                ax.plot(
                    probe_data_sorted['sparsity_ratio'],
                    probe_data_sorted['top_1_accuracy'],
                    marker='o',
                    label=probe_type,
                    linewidth=2,
                    markersize=8
                )
        else:
            # Plot all data together
            df_sorted = df.sort_values('sparsity_ratio')
            ax.plot(
                df_sorted['sparsity_ratio'],
                df_sorted['top_1_accuracy'],
                marker='o',
                linewidth=2,
                markersize=8
            )
        
        ax.set_xlabel('Sparsity Ratio (M/K)')
        ax.set_ylabel('Top-1 Accuracy')
        ax.set_title('Sparsity vs Accuracy Analysis')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
    
    def plot_baseline_comparison(self, result: ExperimentResult, save_path: str):
        """
        Generate bar chart comparing ML model vs baseline methods.
        
        Args:
            result: ExperimentResult object with baseline results
            save_path: Path to save the plot
        """
        logger.info(f"Generating baseline comparison for: {result.config.name}")
        
        # Collect data
        labels = []
        top_1_values = []
        top_5_values = []
        top_10_values = []
        
        # ML model results
        labels.append('ML Model')
        top_1_values.append(result.metrics.get('top_1_accuracy', 0.0))
        top_5_values.append(result.metrics.get('top_5_accuracy', 0.0))
        top_10_values.append(result.metrics.get('top_10_accuracy', 0.0))
        
        # Baseline results
        for baseline_name, metrics in result.baseline_results.items():
            labels.append(baseline_name.replace('_', ' ').title())
            top_1_values.append(metrics.get('top_1_accuracy', 0.0))
            top_5_values.append(metrics.get('top_5_accuracy', 0.0))
            top_10_values.append(metrics.get('top_10_accuracy', 0.0))
        
        if len(labels) <= 1:
            logger.warning("No baseline results available")
            return
        
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
        self._save_figure(fig, save_path)
    
    def plot_fidelity_gap(self, df: pd.DataFrame, save_path: str):
        """
        Generate visualization of fidelity gap (synthetic vs Sionna).
        
        Args:
            df: DataFrame from ResultAnalyzer.fidelity_gap_analysis()
            save_path: Path to save the plot
        """
        logger.info("Generating fidelity gap plot")
        
        if df.empty:
            logger.warning("Empty DataFrame, skipping plot")
            return
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Create configuration labels
        df['config'] = df['probe_type'] + '\n' + df['model_type'] + f'\nM={df["M"]}'
        
        # Plot 1: Side-by-side comparison
        x = np.arange(len(df))
        width = 0.35
        
        ax1.bar(x - width/2, df['synthetic_top_1'], width, label='Synthetic', alpha=0.8)
        ax1.bar(x + width/2, df['sionna_top_1'], width, label='Sionna', alpha=0.8)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Top-1 Accuracy')
        ax1.set_title('Synthetic vs Sionna Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['config'], rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Gap visualization
        colors = ['green' if gap >= 0 else 'red' for gap in df['gap_top_1']]
        ax2.bar(x, df['gap_top_1'], alpha=0.8, color=colors)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Accuracy Gap (Synthetic - Sionna)')
        ax2.set_title('Fidelity Gap Analysis')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['config'], rotation=45, ha='right', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, save_path)
    
    def plot_metric_heatmap(self, df: pd.DataFrame, save_path: str):
        """
        Generate heatmap of configurations vs metrics.
        
        Args:
            df: DataFrame with configurations and metrics
            save_path: Path to save the plot
        """
        logger.info("Generating metric heatmap")
        
        if df.empty:
            logger.warning("Empty DataFrame, skipping plot")
            return
        
        # Prepare data for heatmap
        # Assuming df has probe_type, model_type as columns and metrics as values
        if 'probe_type' in df.columns and 'model_type' in df.columns:
            # Create a pivot table
            metric_cols = [col for col in df.columns if 'accuracy' in col or 'time' in col]
            
            if not metric_cols:
                logger.warning("No metric columns found for heatmap")
                return
            
            # Use first metric for heatmap
            metric_col = metric_cols[0]
            
            pivot = df.pivot_table(
                values=metric_col,
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
                cbar_kws={'label': metric_col.replace('_', ' ').title()},
                ax=ax
            )
            
            ax.set_title(f'Configuration Performance Heatmap: {metric_col.replace("_", " ").title()}')
            ax.set_xlabel('Model Type')
            ax.set_ylabel('Probe Type')
            
            plt.tight_layout()
            self._save_figure(fig, save_path)
    
    def generate_summary_report(self, campaign_results: List[ExperimentResult], save_path: str):
        """
        Generate a multi-page summary report with multiple plots.
        
        Args:
            campaign_results: List of ExperimentResult objects from a campaign
            save_path: Path to save the report PDF
        """
        logger.info(f"Generating summary report for {len(campaign_results)} experiments")
        
        if not campaign_results:
            logger.warning("No results to generate report")
            return
        
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = self.output_dir / save_path
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path = save_path.with_suffix('.pdf')
        
        with PdfPages(pdf_path) as pdf:
            # Page 1: Summary statistics
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Campaign Summary Report', fontsize=16, fontweight='bold')
            
            # Extract metrics
            top_1_accs = [r.metrics.get('top_1_accuracy', 0.0) for r in campaign_results]
            top_5_accs = [r.metrics.get('top_5_accuracy', 0.0) for r in campaign_results]
            probe_types = [r.config.probe_type for r in campaign_results]
            model_types = [r.config.model_type for r in campaign_results]
            
            # Plot 1: Top-1 accuracy distribution
            axes[0, 0].hist(top_1_accs, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Top-1 Accuracy')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Top-1 Accuracy Distribution')
            axes[0, 0].grid(alpha=0.3)
            
            # Plot 2: Top-5 accuracy distribution
            axes[0, 1].hist(top_5_accs, bins=20, alpha=0.7, edgecolor='black', color='orange')
            axes[0, 1].set_xlabel('Top-5 Accuracy')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Top-5 Accuracy Distribution')
            axes[0, 1].grid(alpha=0.3)
            
            # Plot 3: Probe type distribution
            probe_counts = pd.Series(probe_types).value_counts()
            axes[1, 0].bar(range(len(probe_counts)), probe_counts.values, alpha=0.7)
            axes[1, 0].set_xticks(range(len(probe_counts)))
            axes[1, 0].set_xticklabels(probe_counts.index, rotation=45, ha='right')
            axes[1, 0].set_xlabel('Probe Type')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Probe Type Distribution')
            axes[1, 0].grid(axis='y', alpha=0.3)
            
            # Plot 4: Model type distribution
            model_counts = pd.Series(model_types).value_counts()
            axes[1, 1].bar(range(len(model_counts)), model_counts.values, alpha=0.7, color='green')
            axes[1, 1].set_xticks(range(len(model_counts)))
            axes[1, 1].set_xticklabels(model_counts.index, rotation=45, ha='right')
            axes[1, 1].set_xlabel('Model Type')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Model Type Distribution')
            axes[1, 1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
            
            # Page 2: Performance comparison
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Create comparison data
            labels = [f"{r.config.probe_type[:8]}\n{r.config.model_type[:8]}" for r in campaign_results[:20]]  # Limit to 20
            top_1 = [r.metrics.get('top_1_accuracy', 0.0) for r in campaign_results[:20]]
            top_5 = [r.metrics.get('top_5_accuracy', 0.0) for r in campaign_results[:20]]
            
            x = np.arange(len(labels))
            width = 0.35
            
            ax.bar(x - width/2, top_1, width, label='Top-1', alpha=0.8)
            ax.bar(x + width/2, top_5, width, label='Top-5', alpha=0.8)
            
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Accuracy')
            ax.set_title('Top Experiments Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
            
            # Save metadata
            d = pdf.infodict()
            d['Title'] = 'RIS Research Campaign Summary Report'
            d['Author'] = 'RIS Research Engine'
            d['Subject'] = 'Experiment Results Analysis'
            d['Keywords'] = 'RIS, Machine Learning, Beam Selection'
        
        logger.info(f"Summary report saved: {pdf_path}")
        
        # Also save as PNG for the first page
        png_path = save_path.with_suffix('.png')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Campaign Summary Report', fontsize=16, fontweight='bold')
        
        # Recreate first page plots
        top_1_accs = [r.metrics.get('top_1_accuracy', 0.0) for r in campaign_results]
        top_5_accs = [r.metrics.get('top_5_accuracy', 0.0) for r in campaign_results]
        probe_types = [r.config.probe_type for r in campaign_results]
        model_types = [r.config.model_type for r in campaign_results]
        
        axes[0, 0].hist(top_1_accs, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Top-1 Accuracy')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Top-1 Accuracy Distribution')
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].hist(top_5_accs, bins=20, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 1].set_xlabel('Top-5 Accuracy')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Top-5 Accuracy Distribution')
        axes[0, 1].grid(alpha=0.3)
        
        probe_counts = pd.Series(probe_types).value_counts()
        axes[1, 0].bar(range(len(probe_counts)), probe_counts.values, alpha=0.7)
        axes[1, 0].set_xticks(range(len(probe_counts)))
        axes[1, 0].set_xticklabels(probe_counts.index, rotation=45, ha='right')
        axes[1, 0].set_xlabel('Probe Type')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Probe Type Distribution')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        model_counts = pd.Series(model_types).value_counts()
        axes[1, 1].bar(range(len(model_counts)), model_counts.values, alpha=0.7, color='green')
        axes[1, 1].set_xticks(range(len(model_counts)))
        axes[1, 1].set_xticklabels(model_counts.index, rotation=45, ha='right')
        axes[1, 1].set_xlabel('Model Type')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Model Type Distribution')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Summary report PNG saved: {png_path}")
