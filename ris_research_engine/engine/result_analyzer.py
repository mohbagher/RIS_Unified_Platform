"""Result analyzer for comparing and analyzing experiment results."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from ris_research_engine.foundation import ResultTracker
from ris_research_engine.foundation.logging_config import get_logger

logger = get_logger(__name__)


class ResultAnalyzer:
    """Analyze and compare experiment results from database."""
    
    def __init__(self, db_path: str = "results.db"):
        """
        Initialize the result analyzer.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.tracker = ResultTracker(db_path)
    
    def compare_probes(self, experiment_ids: List[int]) -> pd.DataFrame:
        """
        Compare different probe types across experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            DataFrame with probe comparison
        """
        logger.info(f"Comparing probes for {len(experiment_ids)} experiments")
        
        data = []
        
        for exp_id in experiment_ids:
            exp = self.tracker.get_experiment(exp_id)
            
            if exp is None:
                logger.warning(f"Experiment {exp_id} not found")
                continue
            
            metrics = exp['metrics']
            
            data.append({
                'experiment_id': exp_id,
                'probe_type': exp['probe_type'],
                'M': exp['M'],
                'K': exp['K'],
                'model_type': exp['model_type'],
                'top_1_accuracy': metrics.get('top_1_accuracy', 0.0),
                'top_5_accuracy': metrics.get('top_5_accuracy', 0.0),
                'top_10_accuracy': metrics.get('top_10_accuracy', 0.0),
                'inference_time_ms': metrics.get('inference_time', 0.0),
                'training_time_seconds': exp['training_time_seconds'],
                'model_parameters': exp['model_parameters'],
                'status': exp['status']
            })
        
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            # Add ranking columns
            df['top_1_rank'] = df['top_1_accuracy'].rank(ascending=False)
            df['efficiency_score'] = (
                df['top_1_accuracy'] / (df['inference_time_ms'] + 1e-6)
            )
        
        return df
    
    def compare_models(self, experiment_ids: List[int]) -> pd.DataFrame:
        """
        Compare different model architectures across experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            DataFrame with model comparison
        """
        logger.info(f"Comparing models for {len(experiment_ids)} experiments")
        
        data = []
        
        for exp_id in experiment_ids:
            exp = self.tracker.get_experiment(exp_id)
            
            if exp is None:
                logger.warning(f"Experiment {exp_id} not found")
                continue
            
            metrics = exp['metrics']
            
            data.append({
                'experiment_id': exp_id,
                'model_type': exp['model_type'],
                'probe_type': exp['probe_type'],
                'M': exp['M'],
                'K': exp['K'],
                'model_parameters': exp['model_parameters'],
                'top_1_accuracy': metrics.get('top_1_accuracy', 0.0),
                'top_5_accuracy': metrics.get('top_5_accuracy', 0.0),
                'top_10_accuracy': metrics.get('top_10_accuracy', 0.0),
                'inference_time_ms': metrics.get('inference_time', 0.0),
                'training_time_seconds': exp['training_time_seconds'],
                'best_epoch': exp['best_epoch'],
                'total_epochs': exp['total_epochs'],
                'status': exp['status']
            })
        
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            # Add efficiency metrics
            df['params_per_accuracy'] = df['model_parameters'] / (df['top_1_accuracy'] + 1e-6)
            df['accuracy_rank'] = df['top_1_accuracy'].rank(ascending=False)
        
        return df
    
    def sparsity_analysis(self, experiment_ids: List[int]) -> pd.DataFrame:
        """
        Analyze the relationship between sparsity (M/K) and performance.
        
        Args:
            experiment_ids: List of experiment IDs to analyze
            
        Returns:
            DataFrame with sparsity analysis
        """
        logger.info(f"Performing sparsity analysis for {len(experiment_ids)} experiments")
        
        data = []
        
        for exp_id in experiment_ids:
            exp = self.tracker.get_experiment(exp_id)
            
            if exp is None:
                logger.warning(f"Experiment {exp_id} not found")
                continue
            
            metrics = exp['metrics']
            M = exp['M']
            K = exp['K']
            sparsity_ratio = M / K if K > 0 else 0.0
            
            data.append({
                'experiment_id': exp_id,
                'M': M,
                'K': K,
                'sparsity_ratio': sparsity_ratio,
                'probe_type': exp['probe_type'],
                'model_type': exp['model_type'],
                'top_1_accuracy': metrics.get('top_1_accuracy', 0.0),
                'top_5_accuracy': metrics.get('top_5_accuracy', 0.0),
                'top_10_accuracy': metrics.get('top_10_accuracy', 0.0),
                'power_ratio': metrics.get('power_ratio', 0.0),
                'spectral_efficiency': metrics.get('spectral_efficiency', 0.0),
                'status': exp['status']
            })
        
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            # Sort by sparsity ratio
            df = df.sort_values('sparsity_ratio')
        
        return df
    
    def best_configuration(
        self, 
        metric: str = 'top_1_accuracy',
        campaign_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best experiment configuration by a specific metric.
        
        Args:
            metric: Metric to optimize (default: 'top_1_accuracy')
            campaign_name: Optional campaign name filter
            
        Returns:
            Dictionary with best experiment details, or None if no experiments found
        """
        logger.info(f"Finding best configuration by {metric}")
        
        experiments = self.tracker.get_all_experiments(
            campaign_name=campaign_name,
            status='completed'
        )
        
        if not experiments:
            logger.warning("No completed experiments found")
            return None
        
        # Find best by metric
        best_exp = None
        best_value = -float('inf')
        
        for exp in experiments:
            metrics = exp['metrics']
            value = metrics.get(metric, 0.0)
            
            if value > best_value:
                best_value = value
                best_exp = exp
        
        if best_exp:
            logger.info(
                f"Best configuration: experiment_id={best_exp['id']}, "
                f"{metric}={best_value:.4f}"
            )
            
            return {
                'experiment_id': best_exp['id'],
                'name': best_exp['name'],
                'probe_type': best_exp['probe_type'],
                'model_type': best_exp['model_type'],
                'M': best_exp['M'],
                'K': best_exp['K'],
                'metric_name': metric,
                'metric_value': best_value,
                'full_metrics': best_exp['metrics'],
                'config': best_exp['full_config']
            }
        
        return None
    
    def statistical_summary(self, experiment_ids: List[int]) -> Dict[str, Dict[str, float]]:
        """
        Compute statistical summary (mean, std, min, max) for metrics.
        
        Args:
            experiment_ids: List of experiment IDs to analyze
            
        Returns:
            Dictionary mapping metric names to statistics
        """
        logger.info(f"Computing statistical summary for {len(experiment_ids)} experiments")
        
        # Collect all metrics
        all_metrics = {}
        
        for exp_id in experiment_ids:
            exp = self.tracker.get_experiment(exp_id)
            
            if exp is None or exp['status'] != 'completed':
                continue
            
            metrics = exp['metrics']
            
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Compute statistics
        summary = {}
        
        for metric_name, values in all_metrics.items():
            if len(values) == 0:
                continue
            
            values_array = np.array(values)
            
            summary[metric_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array)),
                'count': len(values)
            }
        
        return summary
    
    def fidelity_gap_analysis(
        self,
        probe_type: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze the fidelity gap between synthetic and Sionna experiments.
        
        Args:
            probe_type: Optional filter by probe type
            model_type: Optional filter by model type
            
        Returns:
            DataFrame with fidelity gap analysis
        """
        logger.info("Performing fidelity gap analysis")
        
        # Get all experiments
        all_experiments = self.tracker.get_all_experiments(status='completed')
        
        # Separate by fidelity
        synthetic_exps = []
        sionna_exps = []
        
        for exp in all_experiments:
            # Apply filters
            if probe_type and exp['probe_type'] != probe_type:
                continue
            if model_type and exp['model_type'] != model_type:
                continue
            
            fidelity = exp.get('data_fidelity', 'synthetic')
            
            if fidelity == 'synthetic':
                synthetic_exps.append(exp)
            elif fidelity == 'sionna':
                sionna_exps.append(exp)
        
        logger.info(
            f"Found {len(synthetic_exps)} synthetic and "
            f"{len(sionna_exps)} Sionna experiments"
        )
        
        # Match experiments by configuration
        comparison_data = []
        
        for synth_exp in synthetic_exps:
            # Find matching Sionna experiment
            matching_sionna = None
            
            for sionna_exp in sionna_exps:
                # Match by probe type, model type, M, K
                if (sionna_exp['probe_type'] == synth_exp['probe_type'] and
                    sionna_exp['model_type'] == synth_exp['model_type'] and
                    sionna_exp['M'] == synth_exp['M'] and
                    sionna_exp['K'] == synth_exp['K']):
                    matching_sionna = sionna_exp
                    break
            
            if matching_sionna:
                synth_metrics = synth_exp['metrics']
                sionna_metrics = matching_sionna['metrics']
                
                comparison_data.append({
                    'probe_type': synth_exp['probe_type'],
                    'model_type': synth_exp['model_type'],
                    'M': synth_exp['M'],
                    'K': synth_exp['K'],
                    'synthetic_exp_id': synth_exp['id'],
                    'sionna_exp_id': matching_sionna['id'],
                    'synthetic_top_1': synth_metrics.get('top_1_accuracy', 0.0),
                    'sionna_top_1': sionna_metrics.get('top_1_accuracy', 0.0),
                    'gap_top_1': synth_metrics.get('top_1_accuracy', 0.0) - sionna_metrics.get('top_1_accuracy', 0.0),
                    'synthetic_top_5': synth_metrics.get('top_5_accuracy', 0.0),
                    'sionna_top_5': sionna_metrics.get('top_5_accuracy', 0.0),
                    'gap_top_5': synth_metrics.get('top_5_accuracy', 0.0) - sionna_metrics.get('top_5_accuracy', 0.0),
                    'synthetic_top_10': synth_metrics.get('top_10_accuracy', 0.0),
                    'sionna_top_10': sionna_metrics.get('top_10_accuracy', 0.0),
                    'gap_top_10': synth_metrics.get('top_10_accuracy', 0.0) - sionna_metrics.get('top_10_accuracy', 0.0),
                })
        
        df = pd.DataFrame(comparison_data)
        
        if len(df) > 0:
            # Add relative gap columns (only where synthetic values are significant)
            df['relative_gap_top_1'] = np.where(
                df['synthetic_top_1'] > 0.01,
                df['gap_top_1'] / df['synthetic_top_1'],
                np.nan
            )
            df['relative_gap_top_5'] = np.where(
                df['synthetic_top_5'] > 0.01,
                df['gap_top_5'] / df['synthetic_top_5'],
                np.nan
            )
            df['relative_gap_top_10'] = np.where(
                df['synthetic_top_10'] > 0.01,
                df['gap_top_10'] / df['synthetic_top_10'],
                np.nan
            )
        
        return df
