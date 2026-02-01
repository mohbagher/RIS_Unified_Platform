"""Search controller for orchestrating multi-experiment campaigns."""

import yaml
import time
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
import pandas as pd
import logging

from ris_research_engine.foundation import (
    SystemConfig, TrainingConfig, ExperimentConfig, 
    ExperimentResult, SearchCampaignResult, ResultTracker
)
from ris_research_engine.foundation.logging_config import get_logger
from ris_research_engine.plugins.search import get_strategy, SEARCH_STRATEGIES
from .experiment_runner import ExperimentRunner

logger = get_logger(__name__)


class SearchController:
    """Orchestrates multi-experiment search campaigns."""
    
    def __init__(self, db_path: str = "results.db"):
        """
        Initialize the search controller.
        
        Args:
            db_path: Path to SQLite database for results
        """
        self.db_path = db_path
        self.tracker = ResultTracker(db_path)
        self.runner = ExperimentRunner()
    
    def run_campaign(
        self,
        search_space_config: Dict[str, Any],
        strategy_name: str,
        rules_config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, float, int, int], None]] = None
    ) -> SearchCampaignResult:
        """
        Run a search campaign using the specified strategy.
        
        Args:
            search_space_config: Dictionary defining search space and budget
            strategy_name: Name of search strategy to use
            rules_config: Optional scientific rules configuration
            progress_callback: Optional callback(message, progress, completed, total)
            
        Returns:
            SearchCampaignResult with all experiment results
        """
        timestamp = datetime.now().isoformat()
        campaign_name = search_space_config.get('name', f'campaign_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        start_time = time.time()
        
        logger.info(f"Starting search campaign: {campaign_name}")
        logger.info(f"Strategy: {strategy_name}")
        
        # Get and initialize search strategy
        if strategy_name not in SEARCH_STRATEGIES:
            raise ValueError(f"Unknown search strategy: {strategy_name}. Available: {list(SEARCH_STRATEGIES.keys())}")
        
        strategy_class = get_strategy(strategy_name)
        strategy = strategy_class()
        
        # Extract search space and budget
        search_space = search_space_config.get('search_space', {})
        budget = search_space_config.get('budget', {})
        
        # Initialize strategy
        strategy.initialize(search_space, budget, rules_config)
        
        # Track results
        all_results = []
        completed = 0
        pruned = 0
        failed = 0
        best_result = None
        best_metric_value = -float('inf')
        
        experiment_count = 0
        max_experiments = budget.get('max_experiments', 100)
        
        while not strategy.is_complete(all_results):
            experiment_count += 1
            
            # Get next configuration
            next_config = strategy.suggest_next(all_results)
            
            if next_config is None:
                logger.info("Search strategy returned no more configurations")
                break
            
            if experiment_count > max_experiments:
                logger.info(f"Reached maximum experiments: {max_experiments}")
                break
            
            logger.info(f"Running experiment {experiment_count}/{max_experiments}")
            
            # Progress callback
            if progress_callback:
                progress = experiment_count / max_experiments
                progress_callback(
                    f"Experiment {experiment_count}/{max_experiments}",
                    progress,
                    completed,
                    experiment_count
                )
            
            # Run experiment
            try:
                result = self.runner.run(next_config, progress_callback=None)
                
                # Save to database immediately
                exp_id = self.tracker.save_experiment(result, campaign_name=campaign_name)
                logger.info(f"Saved experiment {exp_id} to database")
                
                # Update counters
                if result.status == 'completed':
                    completed += 1
                elif result.status == 'failed':
                    failed += 1
                elif result.status == 'pruned':
                    pruned += 1
                
                # Track best result
                if result.status == 'completed':
                    metric_value = result.primary_metric_value
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_result = result
                
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"Experiment {experiment_count} failed with exception: {e}", exc_info=True)
                failed += 1
                
                # Create failed result
                failed_result = ExperimentResult(
                    config=next_config,
                    metrics={},
                    training_history={},
                    best_epoch=0,
                    total_epochs=0,
                    training_time_seconds=0.0,
                    model_parameters=0,
                    timestamp=datetime.now().isoformat(),
                    status='failed',
                    error_message=str(e),
                    baseline_results={},
                    primary_metric_name='top_1_accuracy',
                    primary_metric_value=0.0
                )
                
                # Save failed result
                self.tracker.save_experiment(failed_result, campaign_name=campaign_name)
                all_results.append(failed_result)
        
        total_time = time.time() - start_time
        
        # Create campaign result
        campaign_result = SearchCampaignResult(
            campaign_name=campaign_name,
            search_strategy=strategy_name,
            total_experiments=len(all_results),
            completed_experiments=completed,
            pruned_experiments=pruned,
            failed_experiments=failed,
            best_result=best_result,
            all_results=all_results,
            total_time_seconds=total_time,
            search_space_definition=search_space_config,
            timestamp=timestamp
        )
        
        logger.info(
            f"Campaign completed: {completed} completed, {pruned} pruned, "
            f"{failed} failed in {total_time:.2f}s"
        )
        
        if progress_callback:
            progress_callback("Campaign completed!", 1.0, completed, len(all_results))
        
        return campaign_result
    
    def run_from_yaml(
        self,
        yaml_path: str,
        progress_callback: Optional[Callable[[str, float, int, int], None]] = None
    ) -> SearchCampaignResult:
        """
        Run a search campaign from a YAML configuration file.
        
        Args:
            yaml_path: Path to YAML configuration file
            progress_callback: Optional callback for progress updates
            
        Returns:
            SearchCampaignResult
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        logger.info(f"Loading configuration from {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract components
        search_space_config = config
        strategy_name = config.get('strategy', 'random_search')
        rules_config = config.get('rules', None)
        
        return self.run_campaign(
            search_space_config=search_space_config,
            strategy_name=strategy_name,
            rules_config=rules_config,
            progress_callback=progress_callback
        )
    
    def run_cross_fidelity_validation(
        self,
        synthetic_campaign_name: str,
        hdf5_path: str,
        top_n: int = 3,
        progress_callback: Optional[Callable[[str, float, int, int], None]] = None
    ) -> pd.DataFrame:
        """
        Validate top-N configurations from synthetic campaign on Sionna data.
        
        Args:
            synthetic_campaign_name: Name of completed synthetic campaign
            hdf5_path: Path to Sionna HDF5 data file
            top_n: Number of top configurations to validate
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame comparing synthetic vs Sionna performance
        """
        logger.info(f"Running cross-fidelity validation for campaign: {synthetic_campaign_name}")
        
        # Get all experiments from synthetic campaign
        experiments = self.tracker.get_all_experiments(
            campaign_name=synthetic_campaign_name,
            status='completed'
        )
        
        if not experiments:
            raise ValueError(f"No completed experiments found for campaign: {synthetic_campaign_name}")
        
        # Sort by primary metric
        experiments_sorted = sorted(
            experiments,
            key=lambda x: x['metrics'].get('top_1_accuracy', 0.0),
            reverse=True
        )
        
        # Take top N
        top_experiments = experiments_sorted[:top_n]
        
        logger.info(f"Validating top {top_n} configurations on Sionna data")
        
        # Prepare comparison data
        comparison_data = []
        
        for i, exp in enumerate(top_experiments):
            logger.info(f"Validating configuration {i+1}/{top_n}")
            
            if progress_callback:
                progress_callback(
                    f"Validating {i+1}/{top_n}",
                    i / top_n,
                    i,
                    top_n
                )
            
            # Reconstruct configuration
            full_config = exp['full_config']
            config = ExperimentConfig.from_dict(full_config)
            
            # Modify to use Sionna data
            config.data_source = 'hdf5_loader'
            config.data_params = {'hdf5_path': hdf5_path}
            config.data_fidelity = 'sionna'
            config.name = f"{config.name}_sionna_validation"
            
            # Run on Sionna data
            try:
                result = self.runner.run(config, progress_callback=None)
                
                # Save result
                self.tracker.save_experiment(
                    result, 
                    campaign_name=f"{synthetic_campaign_name}_sionna_validation"
                )
                
                # Record comparison
                comparison_data.append({
                    'experiment_id': exp['id'],
                    'probe_type': config.probe_type,
                    'model_type': config.model_type,
                    'M': config.system.M,
                    'synthetic_top_1': exp['metrics'].get('top_1_accuracy', 0.0),
                    'synthetic_top_5': exp['metrics'].get('top_5_accuracy', 0.0),
                    'synthetic_top_10': exp['metrics'].get('top_10_accuracy', 0.0),
                    'sionna_top_1': result.metrics.get('top_1_accuracy', 0.0),
                    'sionna_top_5': result.metrics.get('top_5_accuracy', 0.0),
                    'sionna_top_10': result.metrics.get('top_10_accuracy', 0.0),
                    'gap_top_1': exp['metrics'].get('top_1_accuracy', 0.0) - result.metrics.get('top_1_accuracy', 0.0),
                    'gap_top_5': exp['metrics'].get('top_5_accuracy', 0.0) - result.metrics.get('top_5_accuracy', 0.0),
                    'gap_top_10': exp['metrics'].get('top_10_accuracy', 0.0) - result.metrics.get('top_10_accuracy', 0.0),
                })
                
            except Exception as e:
                logger.error(f"Failed to validate configuration {i+1}: {e}")
                comparison_data.append({
                    'experiment_id': exp['id'],
                    'probe_type': config.probe_type,
                    'model_type': config.model_type,
                    'M': config.system.M,
                    'synthetic_top_1': exp['metrics'].get('top_1_accuracy', 0.0),
                    'synthetic_top_5': exp['metrics'].get('top_5_accuracy', 0.0),
                    'synthetic_top_10': exp['metrics'].get('top_10_accuracy', 0.0),
                    'sionna_top_1': 0.0,
                    'sionna_top_5': 0.0,
                    'sionna_top_10': 0.0,
                    'gap_top_1': 0.0,
                    'gap_top_5': 0.0,
                    'gap_top_10': 0.0,
                })
        
        df = pd.DataFrame(comparison_data)
        
        if progress_callback:
            progress_callback("Validation complete!", 1.0, top_n, top_n)
        
        logger.info("Cross-fidelity validation completed")
        
        return df
