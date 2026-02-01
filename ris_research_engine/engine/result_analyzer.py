"""Result analyzer for comparing and analyzing experiment results."""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats

from ris_research_engine.foundation import ExperimentResult, ResultTracker

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """Analyzer for experiment results with pandas-based analysis."""
    
    def __init__(self, db_path: str = "ris_results.db"):
        """
        Initialize result analyzer.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.tracker = ResultTracker(db_path)
    
    def get_results_dataframe(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Get all experiment results as a pandas DataFrame.
        
        Args:
            filters: Optional filters for results
            
        Returns:
            DataFrame with experiment results
        """
        results = self.tracker.get_all_results()
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            for result in results:
                match = True
                for key, value in filters.items():
                    if key == 'status' and result.status != value:
                        match = False
                        break
                    elif key == 'probe_type' and result.config.probe_type != value:
                        match = False
                        break
                    elif key == 'model_type' and result.config.model_type != value:
                        match = False
                        break
                if match:
                    filtered_results.append(result)
            results = filtered_results
        
        # Convert to DataFrame
        data = []
        for result in results:
            row = {
                'name': result.config.name,
                'status': result.status,
                'timestamp': result.timestamp,
                'probe_type': result.config.probe_type,
                'model_type': result.config.model_type,
                'data_source': result.config.data_source,
                'data_fidelity': result.config.data_fidelity,
                'N': result.config.system.N,
                'K': result.config.system.K,
                'M': result.config.system.M,
                'M_K_ratio': result.config.system.M / result.config.system.K,
                'training_time': result.training_time_seconds,
                'model_parameters': result.model_parameters,
                'best_epoch': result.best_epoch,
                'total_epochs': result.total_epochs,
            }
            # Add all metrics
            for metric_name, metric_value in result.metrics.items():
                row[metric_name] = metric_value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def compare_probes(
        self,
        metric: str = 'top_1_accuracy',
        model_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare performance across different probe types.
        
        Args:
            metric: Metric to compare
            model_type: Optional filter by model type
            
        Returns:
            DataFrame grouped by probe type with statistics
        """
        df = self.get_results_dataframe({'status': 'completed'})
        
        if df.empty:
            logger.warning("No completed experiments found")
            return pd.DataFrame()
        
        # Filter by model type if specified
        if model_type:
            df = df[df['model_type'] == model_type]
        
        # Filter to rows that have the metric
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found in results")
            return pd.DataFrame()
        
        df = df[df[metric].notna()]
        
        # Group by probe type and compute statistics
        grouped = df.groupby('probe_type')[metric].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('count', 'count')
        ]).reset_index()
        
        # Sort by mean descending
        grouped = grouped.sort_values('mean', ascending=False)
        
        return grouped
    
    def compare_models(
        self,
        metric: str = 'top_1_accuracy',
        probe_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare performance across different model types.
        
        Args:
            metric: Metric to compare
            probe_type: Optional filter by probe type
            
        Returns:
            DataFrame grouped by model type with statistics
        """
        df = self.get_results_dataframe({'status': 'completed'})
        
        if df.empty:
            logger.warning("No completed experiments found")
            return pd.DataFrame()
        
        # Filter by probe type if specified
        if probe_type:
            df = df[df['probe_type'] == probe_type]
        
        # Filter to rows that have the metric
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found in results")
            return pd.DataFrame()
        
        df = df[df[metric].notna()]
        
        # Group by model type and compute statistics
        grouped = df.groupby('model_type')[metric].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('count', 'count')
        ]).reset_index()
        
        # Sort by mean descending
        grouped = grouped.sort_values('mean', ascending=False)
        
        return grouped
    
    def sparsity_analysis(
        self,
        metric: str = 'top_1_accuracy',
        probe_type: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze how performance changes with sparsity (M/K ratio).
        
        Args:
            metric: Metric to analyze
            probe_type: Optional filter by probe type
            model_type: Optional filter by model type
            
        Returns:
            DataFrame with sparsity analysis
        """
        df = self.get_results_dataframe({'status': 'completed'})
        
        if df.empty:
            logger.warning("No completed experiments found")
            return pd.DataFrame()
        
        # Apply filters
        if probe_type:
            df = df[df['probe_type'] == probe_type]
        if model_type:
            df = df[df['model_type'] == model_type]
        
        # Filter to rows that have the metric
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found in results")
            return pd.DataFrame()
        
        df = df[df[metric].notna()]
        
        # Group by M/K ratio
        grouped = df.groupby('M_K_ratio')[metric].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('count', 'count')
        ]).reset_index()
        
        # Sort by M/K ratio
        grouped = grouped.sort_values('M_K_ratio')
        
        return grouped
    
    def best_configuration(
        self,
        metric: str = 'top_1_accuracy',
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get top N configurations by specified metric.
        
        Args:
            metric: Metric to rank by
            top_n: Number of top results to return
            
        Returns:
            DataFrame with top configurations
        """
        df = self.get_results_dataframe({'status': 'completed'})
        
        if df.empty:
            logger.warning("No completed experiments found")
            return pd.DataFrame()
        
        # Filter to rows that have the metric
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found in results")
            return pd.DataFrame()
        
        df = df[df[metric].notna()]
        
        # Sort by metric and return top N
        df = df.sort_values(metric, ascending=False)
        
        return df.head(top_n)
    
    def statistical_summary(
        self,
        metric: str = 'top_1_accuracy',
        group_by: str = 'probe_type',
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Generate statistical summary with confidence intervals.
        
        Args:
            metric: Metric to analyze
            group_by: Column to group by
            confidence_level: Confidence level for intervals
            
        Returns:
            DataFrame with statistical summary
        """
        df = self.get_results_dataframe({'status': 'completed'})
        
        if df.empty:
            logger.warning("No completed experiments found")
            return pd.DataFrame()
        
        # Filter to rows that have the metric
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found in results")
            return pd.DataFrame()
        
        df = df[df[metric].notna()]
        
        # Group and compute statistics
        results = []
        for group_name, group_df in df.groupby(group_by):
            values = group_df[metric].values
            n = len(values)
            
            if n < 2:
                # Not enough data for confidence interval
                results.append({
                    group_by: group_name,
                    'mean': values[0] if n > 0 else np.nan,
                    'std': 0.0,
                    'count': n,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan
                })
            else:
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                se = std / np.sqrt(n)
                
                # Calculate confidence interval
                ci = stats.t.interval(
                    confidence_level,
                    df=n-1,
                    loc=mean,
                    scale=se
                )
                
                results.append({
                    group_by: group_name,
                    'mean': mean,
                    'std': std,
                    'count': n,
                    'ci_lower': ci[0],
                    'ci_upper': ci[1]
                })
        
        return pd.DataFrame(results)
    
    def fidelity_gap_analysis(
        self,
        synthetic_fidelity: str = 'synthetic',
        high_fidelity: str = 'sionna',
        metric: str = 'top_1_accuracy'
    ) -> pd.DataFrame:
        """
        Analyze performance gap between different data fidelities.
        
        Args:
            synthetic_fidelity: Name of synthetic fidelity level
            high_fidelity: Name of high-fidelity level
            metric: Metric to compare
            
        Returns:
            DataFrame with fidelity gap analysis
        """
        df = self.get_results_dataframe({'status': 'completed'})
        
        if df.empty:
            logger.warning("No completed experiments found")
            return pd.DataFrame()
        
        # Filter to rows that have the metric
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found in results")
            return pd.DataFrame()
        
        # Get synthetic and high-fidelity results
        synthetic_df = df[df['data_fidelity'] == synthetic_fidelity].copy()
        high_fidelity_df = df[df['data_fidelity'] == high_fidelity].copy()
        
        if synthetic_df.empty or high_fidelity_df.empty:
            logger.warning("Not enough data for fidelity gap analysis")
            return pd.DataFrame()
        
        # Try to match experiments by name pattern
        results = []
        for _, synth_row in synthetic_df.iterrows():
            # Look for matching high-fidelity experiment
            base_name = synth_row['name'].replace(f'_{synthetic_fidelity}', '').replace(f'_validation_{high_fidelity}', '')
            
            # Find matching high-fidelity result
            matches = high_fidelity_df[
                high_fidelity_df['name'].str.contains(base_name, regex=False) |
                (high_fidelity_df['probe_type'] == synth_row['probe_type']) &
                (high_fidelity_df['model_type'] == synth_row['model_type']) &
                (high_fidelity_df['M'] == synth_row['M']) &
                (high_fidelity_df['K'] == synth_row['K'])
            ]
            
            if not matches.empty:
                hf_row = matches.iloc[0]
                synth_value = synth_row[metric]
                hf_value = hf_row[metric]
                gap = abs(synth_value - hf_value)
                relative_gap = gap / synth_value if synth_value != 0 else np.nan
                
                results.append({
                    'probe_type': synth_row['probe_type'],
                    'model_type': synth_row['model_type'],
                    'M': synth_row['M'],
                    'K': synth_row['K'],
                    f'{synthetic_fidelity}_{metric}': synth_value,
                    f'{high_fidelity}_{metric}': hf_value,
                    'absolute_gap': gap,
                    'relative_gap': relative_gap
                })
        
        return pd.DataFrame(results)
