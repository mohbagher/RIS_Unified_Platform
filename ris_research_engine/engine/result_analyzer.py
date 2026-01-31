"""Result analysis tools for comparing experiments and campaigns."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

from ris_research_engine.foundation.data_types import ExperimentResult, SearchCampaignResult
from ris_research_engine.foundation.storage import ResultTracker

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """Analyze and compare experimental results."""
    
    def __init__(self, result_tracker: Optional[ResultTracker] = None):
        """Initialize result analyzer.
        
        Args:
            result_tracker: Optional result tracker for loading results
        """
        self.result_tracker = result_tracker or ResultTracker()
    
    def compare_probes(
        self,
        campaign_name: Optional[str] = None,
        results: Optional[List[ExperimentResult]] = None,
        metric: str = 'top_1_accuracy'
    ) -> pd.DataFrame:
        """Compare different probe types.
        
        Args:
            campaign_name: Optional campaign name to filter results
            results: Optional list of results to analyze (overrides campaign_name)
            metric: Metric to compare
            
        Returns:
            DataFrame with probe comparison statistics
        """
        logger.info("Comparing probe types")
        
        # Get results
        if results is None:
            if campaign_name:
                results = self.result_tracker.query(
                    campaign_name=campaign_name,
                    status='completed'
                )
            else:
                results = self.result_tracker.query(status='completed')
        
        if not results:
            logger.warning("No results to analyze")
            return pd.DataFrame()
        
        # Group by probe type
        probe_groups = {}
        for result in results:
            probe_type = result.config.probe_type
            if probe_type not in probe_groups:
                probe_groups[probe_type] = []
            probe_groups[probe_type].append(result.metrics.get(metric, 0.0))
        
        # Compute statistics
        comparison_data = []
        for probe_type, values in probe_groups.items():
            comparison_data.append({
                'probe_type': probe_type,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'count': len(values),
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('mean', ascending=False)
        
        return df
    
    def compare_models(
        self,
        campaign_name: Optional[str] = None,
        results: Optional[List[ExperimentResult]] = None,
        metric: str = 'top_1_accuracy'
    ) -> pd.DataFrame:
        """Compare different model architectures.
        
        Args:
            campaign_name: Optional campaign name to filter results
            results: Optional list of results to analyze (overrides campaign_name)
            metric: Metric to compare
            
        Returns:
            DataFrame with model comparison statistics
        """
        logger.info("Comparing model types")
        
        # Get results
        if results is None:
            if campaign_name:
                results = self.result_tracker.query(
                    campaign_name=campaign_name,
                    status='completed'
                )
            else:
                results = self.result_tracker.query(status='completed')
        
        if not results:
            logger.warning("No results to analyze")
            return pd.DataFrame()
        
        # Group by model type
        model_groups = {}
        for result in results:
            model_type = result.config.model_type
            if model_type not in model_groups:
                model_groups[model_type] = []
            model_groups[model_type].append({
                'metric': result.metrics.get(metric, 0.0),
                'params': result.model_parameters,
                'time': result.training_time_seconds,
            })
        
        # Compute statistics
        comparison_data = []
        for model_type, values in model_groups.items():
            metrics = [v['metric'] for v in values]
            params = [v['params'] for v in values]
            times = [v['time'] for v in values]
            
            comparison_data.append({
                'model_type': model_type,
                'mean_metric': np.mean(metrics),
                'std_metric': np.std(metrics),
                'mean_params': np.mean(params),
                'mean_time': np.mean(times),
                'count': len(values),
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('mean_metric', ascending=False)
        
        return df
    
    def sparsity_analysis(
        self,
        campaign_name: Optional[str] = None,
        results: Optional[List[ExperimentResult]] = None,
        metric: str = 'top_1_accuracy'
    ) -> pd.DataFrame:
        """Analyze performance vs sparsity (M/K ratio).
        
        Args:
            campaign_name: Optional campaign name to filter results
            results: Optional list of results to analyze (overrides campaign_name)
            metric: Metric to analyze
            
        Returns:
            DataFrame with sparsity analysis
        """
        logger.info("Analyzing sparsity effects")
        
        # Get results
        if results is None:
            if campaign_name:
                results = self.result_tracker.query(
                    campaign_name=campaign_name,
                    status='completed'
                )
            else:
                results = self.result_tracker.query(status='completed')
        
        if not results:
            logger.warning("No results to analyze")
            return pd.DataFrame()
        
        # Group by M value
        m_groups = {}
        for result in results:
            M = result.config.system.M
            K = result.config.system.K
            ratio = M / K
            
            if M not in m_groups:
                m_groups[M] = []
            
            m_groups[M].append({
                'metric': result.metrics.get(metric, 0.0),
                'ratio': ratio,
                'K': K,
            })
        
        # Compute statistics
        sparsity_data = []
        for M, values in m_groups.items():
            metrics = [v['metric'] for v in values]
            ratio = np.mean([v['ratio'] for v in values])
            
            sparsity_data.append({
                'M': M,
                'M_K_ratio': ratio,
                'mean': np.mean(metrics),
                'std': np.std(metrics),
                'count': len(values),
            })
        
        df = pd.DataFrame(sparsity_data)
        df = df.sort_values('M')
        
        return df
    
    def best_configuration(
        self,
        campaign_name: Optional[str] = None,
        results: Optional[List[ExperimentResult]] = None,
        metric: str = 'top_1_accuracy',
        top_k: int = 1
    ) -> List[ExperimentResult]:
        """Get best performing configuration(s).
        
        Args:
            campaign_name: Optional campaign name to filter results
            results: Optional list of results to analyze (overrides campaign_name)
            metric: Metric to use for ranking
            top_k: Number of top configurations to return
            
        Returns:
            List of top K experiment results
        """
        logger.info(f"Finding top {top_k} configuration(s)")
        
        # Get results
        if results is None:
            if campaign_name:
                results = self.result_tracker.query(
                    campaign_name=campaign_name,
                    status='completed'
                )
            else:
                results = self.result_tracker.query(status='completed')
        
        if not results:
            logger.warning("No results to analyze")
            return []
        
        # Sort by metric
        results_sorted = sorted(
            results,
            key=lambda r: r.metrics.get(metric, 0.0),
            reverse=True
        )
        
        return results_sorted[:top_k]
    
    def statistical_summary(
        self,
        campaign_name: Optional[str] = None,
        results: Optional[List[ExperimentResult]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive statistical summary.
        
        Args:
            campaign_name: Optional campaign name to filter results
            results: Optional list of results to analyze (overrides campaign_name)
            
        Returns:
            Dictionary with statistical summary
        """
        logger.info("Generating statistical summary")
        
        # Get results
        if results is None:
            if campaign_name:
                results = self.result_tracker.query(
                    campaign_name=campaign_name,
                    status='completed'
                )
            else:
                results = self.result_tracker.query(status='completed')
        
        if not results:
            logger.warning("No results to analyze")
            return {}
        
        # Collect metrics
        all_metrics = {}
        for result in results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Compute statistics for each metric
        summary = {
            'total_experiments': len(results),
            'metrics': {}
        }
        
        for metric_name, values in all_metrics.items():
            summary['metrics'][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
            }
        
        # Training statistics
        training_times = [r.training_time_seconds for r in results]
        model_params = [r.model_parameters for r in results]
        
        summary['training'] = {
            'mean_time_seconds': np.mean(training_times),
            'total_time_seconds': np.sum(training_times),
            'mean_params': np.mean(model_params),
        }
        
        # Probe and model distributions
        probe_counts = {}
        model_counts = {}
        for result in results:
            probe_type = result.config.probe_type
            model_type = result.config.model_type
            probe_counts[probe_type] = probe_counts.get(probe_type, 0) + 1
            model_counts[model_type] = model_counts.get(model_type, 0) + 1
        
        summary['distributions'] = {
            'probes': probe_counts,
            'models': model_counts,
        }
        
        return summary
    
    def fidelity_gap_analysis(
        self,
        synthetic_campaign: str,
        sionna_campaign: str,
        metric: str = 'top_1_accuracy'
    ) -> pd.DataFrame:
        """Analyze fidelity gap between synthetic and Sionna results.
        
        Args:
            synthetic_campaign: Name of synthetic campaign
            sionna_campaign: Name of Sionna validation campaign
            metric: Metric to compare
            
        Returns:
            DataFrame with fidelity gap analysis
        """
        logger.info("Analyzing fidelity gap")
        
        # Get synthetic results
        synthetic_results = self.result_tracker.query(
            campaign_name=synthetic_campaign,
            status='completed'
        )
        
        # Get Sionna results
        sionna_results = self.result_tracker.query(
            campaign_name=sionna_campaign,
            status='completed'
        )
        
        if not synthetic_results or not sionna_results:
            logger.warning("Missing results for fidelity analysis")
            return pd.DataFrame()
        
        # Match configurations
        comparison_data = []
        
        for synth_result in synthetic_results:
            # Find matching Sionna result
            matching_sionna = None
            for sionna_result in sionna_results:
                if (sionna_result.config.probe_type == synth_result.config.probe_type and
                    sionna_result.config.model_type == synth_result.config.model_type and
                    sionna_result.config.system.M == synth_result.config.system.M):
                    matching_sionna = sionna_result
                    break
            
            if matching_sionna:
                synth_metric = synth_result.metrics.get(metric, 0.0)
                sionna_metric = matching_sionna.metrics.get(metric, 0.0)
                gap = synth_metric - sionna_metric
                
                comparison_data.append({
                    'probe_type': synth_result.config.probe_type,
                    'model_type': synth_result.config.model_type,
                    'M': synth_result.config.system.M,
                    'synthetic': synth_metric,
                    'sionna': sionna_metric,
                    'gap': gap,
                    'relative_gap': gap / synth_metric if synth_metric > 0 else 0,
                })
        
        df = pd.DataFrame(comparison_data)
        
        if not df.empty:
            logger.info(f"Mean fidelity gap: {df['gap'].mean():.4f}")
            logger.info(f"Mean relative gap: {df['relative_gap'].mean():.4f}")
        
        return df
    
    def significance_test(
        self,
        results_a: List[ExperimentResult],
        results_b: List[ExperimentResult],
        metric: str = 'top_1_accuracy',
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Perform statistical significance test between two groups.
        
        Args:
            results_a: First group of results
            results_b: Second group of results
            metric: Metric to compare
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        values_a = [r.metrics.get(metric, 0.0) for r in results_a]
        values_b = [r.metrics.get(metric, 0.0) for r in results_b]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(values_a, values_b)
        
        # Compute effect size (Cohen's d)
        mean_a, mean_b = np.mean(values_a), np.mean(values_b)
        std_a, std_b = np.std(values_a, ddof=1), np.std(values_b, ddof=1)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        return {
            'mean_a': mean_a,
            'mean_b': mean_b,
            'std_a': std_a,
            'std_b': std_b,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < alpha,
            'cohens_d': cohens_d,
            'effect_size': 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large'),
        }
