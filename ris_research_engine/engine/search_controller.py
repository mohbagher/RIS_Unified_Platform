"""Search controller for running campaigns and cross-fidelity validation."""

import logging
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from ris_research_engine.foundation import (
    SystemConfig, TrainingConfig, ExperimentConfig, 
    ExperimentResult, SearchCampaignResult
)
from ris_research_engine.plugins.search import get_search_strategy
from .experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)


class SearchController:
    """Controller for running search campaigns and validation."""
    
    def __init__(self, db_path: str = "ris_results.db"):
        """
        Initialize search controller.
        
        Args:
            db_path: Path to SQLite database for storing results
        """
        self.db_path = db_path
        self.runner = ExperimentRunner(db_path)
    
    def run_campaign(
        self,
        campaign_name: str,
        search_strategy: str,
        search_space: Dict[str, Any],
        **strategy_kwargs
    ) -> SearchCampaignResult:
        """
        Run a search campaign using a specified strategy.
        
        Args:
            campaign_name: Name for this campaign
            search_strategy: Name of search strategy to use
            search_space: Dictionary defining the search space
            **strategy_kwargs: Additional arguments for the search strategy
            
        Returns:
            SearchCampaignResult with all experiment results
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        logger.info(f"Starting campaign: {campaign_name}")
        logger.info(f"Search strategy: {search_strategy}")
        
        # Get search strategy instance
        strategy = get_search_strategy(search_strategy)
        
        # Generate experiment configurations
        configs = strategy.generate_configs(search_space, **strategy_kwargs)
        logger.info(f"Generated {len(configs)} experiment configurations")
        
        # Run experiments
        results = []
        completed = 0
        pruned = 0
        failed = 0
        
        for i, config in enumerate(configs):
            logger.info(f"Running experiment {i+1}/{len(configs)}: {config.name}")
            
            # Add campaign name to config
            config.tags.append(f"campaign:{campaign_name}")
            
            # Run experiment
            result = self.runner.run(config)
            results.append(result)
            
            # Update counters
            if result.status == 'completed':
                completed += 1
            elif result.status == 'pruned':
                pruned += 1
            elif result.status == 'failed':
                failed += 1
            
            # Check if strategy wants to prune remaining experiments
            if hasattr(strategy, 'should_prune'):
                if strategy.should_prune(results):
                    logger.info("Search strategy triggered early pruning")
                    # Mark remaining experiments as pruned
                    pruned += len(configs) - i - 1
                    break
        
        # Find best result
        best_result = None
        best_score = -float('inf')
        for result in results:
            if result.status == 'completed' and result.primary_metric_value > best_score:
                best_score = result.primary_metric_value
                best_result = result
        
        # Create campaign result
        total_time = time.time() - start_time
        campaign_result = SearchCampaignResult(
            campaign_name=campaign_name,
            search_strategy=search_strategy,
            total_experiments=len(configs),
            completed_experiments=completed,
            pruned_experiments=pruned,
            failed_experiments=failed,
            best_result=best_result,
            all_results=results,
            total_time_seconds=total_time,
            search_space_definition=search_space,
            timestamp=timestamp
        )
        
        # Save campaign to database
        self.runner.tracker.save_campaign(campaign_result)
        
        logger.info(f"Campaign completed in {total_time:.2f}s")
        logger.info(f"Completed: {completed}, Pruned: {pruned}, Failed: {failed}")
        if best_result:
            logger.info(f"Best result: {best_result.config.name} "
                       f"({best_result.primary_metric_name}={best_result.primary_metric_value:.4f})")
        
        return campaign_result
    
    def run_from_yaml(self, config_path: str) -> SearchCampaignResult:
        """
        Load configuration from YAML file and run campaign.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            SearchCampaignResult with all experiment results
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        logger.info(f"Loading config from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract campaign settings
        campaign_name = config.get('campaign_name', config_path.stem)
        search_strategy = config.get('search_strategy', 'random_search')
        search_space = config.get('search_space', {})
        strategy_kwargs = config.get('strategy_params', {})
        
        # Run campaign
        return self.run_campaign(
            campaign_name=campaign_name,
            search_strategy=search_strategy,
            search_space=search_space,
            **strategy_kwargs
        )
    
    def run_cross_fidelity_validation(
        self,
        experiment_ids: Optional[List[int]] = None,
        top_n: int = 10,
        validation_fidelity: str = 'sionna',
        validation_data_params: Optional[Dict[str, Any]] = None
    ) -> List[ExperimentResult]:
        """
        Re-run top N experiments on different data fidelity for validation.
        
        Args:
            experiment_ids: Specific experiment IDs to validate (if None, uses top N)
            top_n: Number of top experiments to validate (if experiment_ids not provided)
            validation_fidelity: Target data fidelity ('sionna', 'hardware')
            validation_data_params: Parameters for validation data source
            
        Returns:
            List of validation experiment results
        """
        logger.info(f"Running cross-fidelity validation (target: {validation_fidelity})")
        
        # Get experiments to validate
        if experiment_ids is None:
            # Get top N experiments from database
            all_results = self.runner.tracker.get_all_results()
            # Sort by primary metric
            all_results.sort(key=lambda r: r.primary_metric_value, reverse=True)
            experiments_to_validate = all_results[:top_n]
            logger.info(f"Validating top {top_n} experiments")
        else:
            # Get specific experiments
            experiments_to_validate = []
            for exp_id in experiment_ids:
                result = self.runner.tracker.get_result(exp_id)
                if result:
                    experiments_to_validate.append(result)
            logger.info(f"Validating {len(experiments_to_validate)} specific experiments")
        
        # Run validation experiments
        validation_results = []
        
        for i, original_result in enumerate(experiments_to_validate):
            logger.info(f"Validating experiment {i+1}/{len(experiments_to_validate)}: "
                       f"{original_result.config.name}")
            
            # Create validation config
            validation_config = ExperimentConfig(
                name=f"{original_result.config.name}_validation_{validation_fidelity}",
                system=original_result.config.system,
                training=original_result.config.training,
                probe_type=original_result.config.probe_type,
                probe_params=original_result.config.probe_params,
                model_type=original_result.config.model_type,
                model_params=original_result.config.model_params,
                data_source=validation_fidelity,  # Change data source
                data_params=validation_data_params or {},
                metrics=original_result.config.metrics,
                tags=original_result.config.tags + [f'validation:{validation_fidelity}'],
                notes=f"Cross-fidelity validation of experiment ID {i}",
                data_fidelity=validation_fidelity
            )
            
            # Run validation experiment
            result = self.runner.run(validation_config)
            validation_results.append(result)
            
            # Log comparison
            if result.status == 'completed':
                original_metric = original_result.primary_metric_value
                validation_metric = result.primary_metric_value
                gap = abs(original_metric - validation_metric)
                logger.info(f"Original: {original_metric:.4f}, "
                          f"Validation: {validation_metric:.4f}, "
                          f"Gap: {gap:.4f}")
        
        logger.info(f"Cross-fidelity validation completed for {len(validation_results)} experiments")
        
        return validation_results
