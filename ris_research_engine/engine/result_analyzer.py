"""Result analyzer for comparing and analyzing experiment results."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from ris_research_engine.foundation.storage import ResultTracker
from ris_research_engine.foundation.data_types import ExperimentResult
from ris_research_engine.foundation.logging_config import get_logger

logger = get_logger(__name__)


class ResultAnalyzer:
    """Analyze and compare experiment results from database."""
    
    def __init__(self, tracker: Optional[ResultTracker] = None, db_path: str = "results.db"):
        """
        Initialize the result analyzer.
        
        Args:
            tracker: Optional ResultTracker instance. If None, creates one with db_path
            db_path: Path to SQLite database (used if tracker is None)
        """
        self.tracker = tracker if tracker is not None else ResultTracker(db_path)
        logger.info("ResultAnalyzer initialized")
    
    def compare_probes(self, metric: str = 'top_1_accuracy') -> pd.DataFrame:
        """
        Compare different probe types across all experiments.
        
        Args:
            metric: Metric to compare (default: 'top_1_accuracy')
            
        Returns:
            DataFrame with probe_type as index and metric values
        """
        logger.info(f"Comparing probes by metric: {metric}")
        
        experiments = self.tracker.get_all_experiments(status='completed')
        
        if not experiments:
            logger.warning("No completed experiments found")
            return pd.DataFrame()
        
        data = []
        for exp in experiments:
            metrics = exp['metrics']
            data.append({
                'probe_type': exp['probe_type'],
                'model_type': exp['model_type'],
                'M': exp['M'],
                'K': exp['K'],
                'sparsity_ratio': exp['M'] / exp['K'] if exp['K'] > 0 else 0.0,
                metric: metrics.get(metric, 0.0),
                'experiment_id': exp['id']
            })
        
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            # Create summary by probe type
            summary = df.groupby('probe_type').agg({
                metric: ['mean', 'std', 'min', 'max', 'count']
            }).round(4)
            summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
            
            logger.info(f"Probe comparison complete: {len(df)} experiments analyzed")
            return summary
        
        return df
    
    def compare_models(self, metric: str = 'top_1_accuracy') -> pd.DataFrame:
        """
        Compare different model architectures across all experiments.
        
        Args:
            metric: Metric to compare (default: 'top_1_accuracy')
            
        Returns:
            DataFrame with model_type as index and metric values
        """
        logger.info(f"Comparing models by metric: {metric}")
        
        experiments = self.tracker.get_all_experiments(status='completed')
        
        if not experiments:
            logger.warning("No completed experiments found")
            return pd.DataFrame()
        
        data = []
        for exp in experiments:
            metrics = exp['metrics']
            data.append({
                'model_type': exp['model_type'],
                'probe_type': exp['probe_type'],
                'M': exp['M'],
                'K': exp['K'],
                'model_parameters': exp['model_parameters'],
                metric: metrics.get(metric, 0.0),
                'experiment_id': exp['id']
            })
        
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            # Create summary by model type
            summary = df.groupby('model_type').agg({
                metric: ['mean', 'std', 'min', 'max', 'count'],
                'model_parameters': 'mean'
            }).round(4)
            summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
            
            logger.info(f"Model comparison complete: {len(df)} experiments analyzed")
            return summary
        
        return df
    
    def sparsity_analysis(self) -> pd.DataFrame:
        """
        Analyze the relationship between sparsity (M/K ratio) and performance.
        
        Returns:
            DataFrame with sparsity analysis including M, K, sparsity_ratio, and metrics
        """
        logger.info("Performing sparsity analysis")
        
        experiments = self.tracker.get_all_experiments(status='completed')
        
        if not experiments:
            logger.warning("No completed experiments found")
            return pd.DataFrame()
        
        data = []
        for exp in experiments:
            metrics = exp['metrics']
            M = exp['M']
            K = exp['K']
            sparsity_ratio = M / K if K > 0 else 0.0
            
            data.append({
                'experiment_id': exp['id'],
                'probe_type': exp['probe_type'],
                'model_type': exp['model_type'],
                'M': M,
                'K': K,
                'sparsity_ratio': sparsity_ratio,
                'top_1_accuracy': metrics.get('top_1_accuracy', 0.0),
                'top_5_accuracy': metrics.get('top_5_accuracy', 0.0),
                'top_10_accuracy': metrics.get('top_10_accuracy', 0.0),
                'inference_time': metrics.get('inference_time', 0.0)
            })
        
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            # Sort by sparsity ratio
            df = df.sort_values('sparsity_ratio')
            logger.info(f"Sparsity analysis complete: {len(df)} experiments analyzed")
        
        return df
    
    def best_configuration(
        self, 
        metric: str = 'top_1_accuracy',
        top_n: int = 1
    ) -> List[ExperimentResult]:
        """
        Find the best experiment configurations by a specific metric.
        
        Args:
            metric: Metric to optimize (default: 'top_1_accuracy')
            top_n: Number of top configurations to return (default: 1)
            
        Returns:
            List of ExperimentResult objects for top configurations
        """
        logger.info(f"Finding top {top_n} configurations by {metric}")
        
        experiments = self.tracker.get_all_experiments(status='completed')
        
        if not experiments:
            logger.warning("No completed experiments found")
            return []
        
        # Sort experiments by metric
        experiments_with_metric = []
        for exp in experiments:
            metrics = exp['metrics']
            value = metrics.get(metric, 0.0)
            experiments_with_metric.append((value, exp))
        
        # Sort descending by metric value
        experiments_with_metric.sort(key=lambda x: x[0], reverse=True)
        
        # Get top N
        top_experiments = experiments_with_metric[:top_n]
        
        # Convert to ExperimentResult objects
        results = []
        for value, exp in top_experiments:
            result = self.tracker.get_result(exp['id'])
            if result:
                results.append(result)
                logger.info(
                    f"Rank {len(results)}: experiment_id={exp['id']}, "
                    f"{metric}={value:.4f}"
                )
        
        return results
    
    def fidelity_gap_analysis(self) -> pd.DataFrame:
        """
        Analyze the fidelity gap between synthetic and Sionna experiments.
        
        Returns:
            DataFrame with fidelity gap comparison (synthetic vs Sionna)
        """
        logger.info("Performing fidelity gap analysis")
        
        # Get all completed experiments
        all_experiments = self.tracker.get_all_experiments(status='completed')
        
        if not all_experiments:
            logger.warning("No completed experiments found")
            return pd.DataFrame()
        
        # Separate by fidelity
        synthetic_exps = []
        sionna_exps = []
        
        for exp in all_experiments:
            fidelity = exp.get('data_fidelity', 'synthetic')
            
            if fidelity == 'synthetic':
                synthetic_exps.append(exp)
            elif fidelity == 'sionna':
                sionna_exps.append(exp)
        
        logger.info(
            f"Found {len(synthetic_exps)} synthetic and "
            f"{len(sionna_exps)} Sionna experiments"
        )
        
        if not synthetic_exps or not sionna_exps:
            logger.warning("Insufficient data for fidelity gap analysis")
            return pd.DataFrame()
        
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
                })
        
        df = pd.DataFrame(comparison_data)
        
        if len(df) > 0:
            # Add relative gap columns
            df['relative_gap_top_1'] = np.where(
                df['synthetic_top_1'] > 0.01,
                (df['gap_top_1'] / df['synthetic_top_1']) * 100,  # As percentage
                np.nan
            )
            df['relative_gap_top_5'] = np.where(
                df['synthetic_top_5'] > 0.01,
                (df['gap_top_5'] / df['synthetic_top_5']) * 100,  # As percentage
                np.nan
            )
            
            logger.info(f"Fidelity gap analysis complete: {len(df)} matched pairs")
        
        return df
