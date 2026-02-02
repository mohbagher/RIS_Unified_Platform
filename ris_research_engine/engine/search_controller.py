"""Search controller for orchestrating multi-experiment campaigns."""

import yaml
import time
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
import pandas as pd

from ris_research_engine.foundation import (
    SystemConfig, TrainingConfig, ExperimentConfig, 
    ExperimentResult, SearchCampaignResult, ResultTracker
)
from ris_research_engine.foundation.logging_config import get_logger
from ris_research_engine.plugins.search import get_strategy, list_strategies, SEARCH_STRATEGIES
from .experiment_runner import ExperimentRunner
from . import scientific_rules

logger = get_logger(__name__)


class SearchController:
    """Orchestrates multi-experiment search campaigns using various search strategies."""
    
    def __init__(self, db_path: str = "outputs/experiments/results.db"):
        """
        Initialize the search controller.
        
        Args:
            db_path: Path to SQLite database for results storage
        """
        self.db_path = Path(db_path)
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.tracker = ResultTracker(str(self.db_path))
        self.runner = ExperimentRunner(str(self.db_path))
        
        logger.info(f"SearchController initialized with database: {self.db_path}")
    
    def run_campaign(
        self,
        config_dict: Dict[str, Any],
        progress_callback: Optional[Callable[[str, float, int, int], None]] = None
    ) -> SearchCampaignResult:
        """
        Run a search campaign using the specified strategy.
        
        This method orchestrates a complete search campaign, iterating through
        the search space using the configured strategy, tracking results, and
        applying scientific rules for early stopping and pruning.
        
        Args:
            config_dict: Dictionary containing complete campaign configuration including:
                - name: Campaign name (default: auto-generated timestamp)
                - strategy: Search strategy name (default: 'random_search')
                - search_space: Dict defining parameters to search
                - budget: Dict with max_experiments, max_time_seconds, etc.
                - rules: Optional dict with scientific rules configuration
                - system: System configuration parameters
                - training: Training configuration parameters
                - data: Data source configuration
            progress_callback: Optional callback(message, progress, completed, total)
                Called periodically with progress updates
            
        Returns:
            SearchCampaignResult containing all experiment results and statistics
            
        Raises:
            ValueError: If strategy name is invalid or configuration is malformed
            RuntimeError: If search campaign fails critically
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # Extract campaign configuration
        campaign_name = config_dict.get('name', f'campaign_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        strategy_name = config_dict.get('strategy', 'random_search')
        search_space = config_dict.get('search_space', {})
        budget = config_dict.get('budget', {'max_experiments': 100})
        rules_config = config_dict.get('rules', None)
        
        logger.info(f"Starting search campaign: {campaign_name}")
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"Budget: {budget}")
        
        # Validate strategy
        if strategy_name not in SEARCH_STRATEGIES:
            available = list(SEARCH_STRATEGIES.keys())
            raise ValueError(
                f"Unknown search strategy: '{strategy_name}'. "
                f"Available strategies: {available}"
            )
        
        # Initialize search strategy
        try:
            strategy_class = get_strategy(strategy_name)
            strategy = strategy_class()
            strategy.initialize(search_space, budget, rules_config)
            logger.info(f"Initialized {strategy_name} strategy")
        except Exception as e:
            logger.error(f"Failed to initialize strategy '{strategy_name}': {e}")
            raise
        
        # Load scientific rules if provided
        rules = None
        if rules_config:
            try:
                if isinstance(rules_config, str):
                    # Path to YAML file
                    rules = scientific_rules.load_rules(rules_config)
                elif isinstance(rules_config, dict):
                    # Rules provided directly
                    rules = rules_config
                logger.info("Scientific rules loaded")
            except Exception as e:
                logger.warning(f"Failed to load scientific rules: {e}. Continuing without rules.")
        
        # Track campaign statistics
        all_results: List[ExperimentResult] = []
        completed = 0
        pruned = 0
        failed = 0
        best_result: Optional[ExperimentResult] = None
        best_metric_value = -float('inf')
        
        experiment_count = 0
        max_experiments = budget.get('max_experiments', 100)
        max_time_seconds = budget.get('max_time_seconds', None)
        
        # Main search loop
        while not strategy.is_complete(all_results):
            experiment_count += 1
            
            # Check budget constraints
            if experiment_count > max_experiments:
                logger.info(f"Reached maximum experiments: {max_experiments}")
                break
            
            if max_time_seconds and (time.time() - start_time) > max_time_seconds:
                logger.info(f"Reached maximum time: {max_time_seconds}s")
                break
            
            # Get next configuration from strategy
            next_config = strategy.suggest_next(all_results)
            
            if next_config is None:
                logger.info("Search strategy returned no more configurations")
                break
            
            logger.info(f"Running experiment {experiment_count}/{max_experiments}")
            
            # Update progress
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
                
                # Update statistics based on status
                if result.status == 'completed':
                    completed += 1
                    
                    # Track best result
                    metric_value = result.primary_metric_value
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_result = result
                        logger.info(f"New best result: {best_metric_value:.4f}")
                        
                elif result.status == 'failed':
                    failed += 1
                    logger.warning(f"Experiment {experiment_count} failed: {result.error_message}")
                    
                elif result.status == 'pruned':
                    pruned += 1
                    logger.info(f"Experiment {experiment_count} was pruned")
                
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
                
                # Save failed result to database
                try:
                    self.tracker.save_experiment(failed_result, campaign_name=campaign_name)
                except Exception as save_error:
                    logger.error(f"Failed to save failed result: {save_error}")
                
                all_results.append(failed_result)
        
        # Calculate campaign statistics
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
            search_space_definition=config_dict,
            timestamp=timestamp
        )
        
        logger.info(
            f"Campaign '{campaign_name}' completed: "
            f"{completed} completed, {pruned} pruned, {failed} failed "
            f"in {total_time:.2f}s"
        )
        
        if best_result:
            logger.info(
                f"Best result: {best_result.primary_metric_name}="
                f"{best_result.primary_metric_value:.4f}"
            )
        
        # Final progress update
        if progress_callback:
            progress_callback(
                "Campaign completed!",
                1.0,
                completed,
                len(all_results)
            )
        
        return campaign_result
    
    def run_from_yaml(
        self,
        yaml_path: str,
        progress_callback: Optional[Callable[[str, float, int, int], None]] = None
    ) -> SearchCampaignResult:
        """
        Run a search campaign from a YAML configuration file.
        
        This is a convenience method that loads configuration from YAML
        and then calls run_campaign().
        
        Args:
            yaml_path: Path to YAML configuration file
            progress_callback: Optional callback for progress updates
            
        Returns:
            SearchCampaignResult
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML file is malformed
            ValueError: If configuration is invalid
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        logger.info(f"Loading configuration from {yaml_path}")
        
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise
        
        if not config:
            raise ValueError(f"Empty or invalid configuration file: {yaml_path}")
        
        logger.info(f"Configuration loaded successfully from {yaml_path}")
        
        # Run campaign with loaded configuration
        return self.run_campaign(
            config_dict=config,
            progress_callback=progress_callback
        )
    
    def run_cross_fidelity_validation(
        self,
        campaign_name: str,
        hdf5_path: str,
        top_n: int = 3,
        progress_callback: Optional[Callable[[str, float, int, int], None]] = None
    ) -> pd.DataFrame:
        """
        Validate top-N configurations from synthetic campaign on high-fidelity Sionna data.
        
        This method takes the best performing configurations from a synthetic data campaign
        and re-evaluates them on real Sionna simulation data to measure the fidelity gap.
        
        Args:
            campaign_name: Name of completed synthetic campaign
            hdf5_path: Path to Sionna HDF5 data file
            top_n: Number of top configurations to validate (default: 3)
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame comparing synthetic vs Sionna performance with columns:
                - experiment_id: Original experiment ID
                - probe_type: Type of probe used
                - model_type: Type of model used
                - M: Number of measurements
                - synthetic_top_1/5/10: Accuracy on synthetic data
                - sionna_top_1/5/10: Accuracy on Sionna data
                - gap_top_1/5/10: Performance gap (synthetic - sionna)
                
        Raises:
            ValueError: If campaign has no completed experiments
            FileNotFoundError: If HDF5 file doesn't exist
        """
        logger.info(f"Running cross-fidelity validation for campaign: {campaign_name}")
        
        # Validate HDF5 path
        hdf5_path_obj = Path(hdf5_path)
        if not hdf5_path_obj.exists():
            raise FileNotFoundError(f"Sionna HDF5 file not found: {hdf5_path}")
        
        # Get all completed experiments from synthetic campaign
        try:
            experiments = self.tracker.get_all_experiments(
                campaign_name=campaign_name,
                status='completed'
            )
        except Exception as e:
            logger.error(f"Error retrieving experiments: {e}")
            raise
        
        if not experiments:
            raise ValueError(
                f"No completed experiments found for campaign: {campaign_name}"
            )
        
        logger.info(f"Found {len(experiments)} completed experiments")
        
        # Sort by primary metric (descending) to get top performers
        try:
            experiments_sorted = sorted(
                experiments,
                key=lambda x: x.get('metrics', {}).get('top_1_accuracy', 0.0),
                reverse=True
            )
        except Exception as e:
            logger.error(f"Error sorting experiments: {e}")
            raise
        
        # Take top N
        top_experiments = experiments_sorted[:top_n]
        
        logger.info(f"Validating top {len(top_experiments)} configurations on Sionna data")
        
        # Prepare comparison data
        comparison_data = []
        
        for i, exp in enumerate(top_experiments):
            logger.info(f"Validating configuration {i+1}/{len(top_experiments)}")
            
            if progress_callback:
                progress_callback(
                    f"Validating {i+1}/{len(top_experiments)}",
                    i / len(top_experiments),
                    i,
                    len(top_experiments)
                )
            
            # Reconstruct configuration
            try:
                full_config = exp.get('full_config', {})
                if not full_config:
                    logger.error(f"Experiment {exp.get('id')} missing full_config")
                    continue
                
                config = ExperimentConfig.from_dict(full_config)
                
                # Modify to use Sionna data
                config.data_source = 'hdf5_loader'
                config.data_params = {'hdf5_path': str(hdf5_path)}
                config.data_fidelity = 'sionna'
                config.name = f"{config.name}_sionna_validation"
                
                # Run on Sionna data
                result = self.runner.run(config, progress_callback=None)
                
                # Save result
                validation_campaign_name = f"{campaign_name}_sionna_validation"
                self.tracker.save_experiment(result, campaign_name=validation_campaign_name)
                
                # Extract metrics safely
                synthetic_metrics = exp.get('metrics', {})
                sionna_metrics = result.metrics
                
                # Record comparison
                comparison_data.append({
                    'experiment_id': exp.get('id'),
                    'probe_type': config.probe_type,
                    'model_type': config.model_type,
                    'M': config.system.M,
                    'synthetic_top_1': synthetic_metrics.get('top_1_accuracy', 0.0),
                    'synthetic_top_5': synthetic_metrics.get('top_5_accuracy', 0.0),
                    'synthetic_top_10': synthetic_metrics.get('top_10_accuracy', 0.0),
                    'sionna_top_1': sionna_metrics.get('top_1_accuracy', 0.0),
                    'sionna_top_5': sionna_metrics.get('top_5_accuracy', 0.0),
                    'sionna_top_10': sionna_metrics.get('top_10_accuracy', 0.0),
                    'gap_top_1': synthetic_metrics.get('top_1_accuracy', 0.0) - sionna_metrics.get('top_1_accuracy', 0.0),
                    'gap_top_5': synthetic_metrics.get('top_5_accuracy', 0.0) - sionna_metrics.get('top_5_accuracy', 0.0),
                    'gap_top_10': synthetic_metrics.get('top_10_accuracy', 0.0) - sionna_metrics.get('top_10_accuracy', 0.0),
                    'status': result.status
                })
                
                logger.info(
                    f"Validation {i+1} complete: "
                    f"synthetic={synthetic_metrics.get('top_1_accuracy', 0.0):.4f}, "
                    f"sionna={sionna_metrics.get('top_1_accuracy', 0.0):.4f}"
                )
                
            except Exception as e:
                logger.error(f"Failed to validate configuration {i+1}: {e}", exc_info=True)
                
                # Add failed entry with available data
                try:
                    synthetic_metrics = exp.get('metrics', {})
                    comparison_data.append({
                        'experiment_id': exp.get('id'),
                        'probe_type': exp.get('probe_type', 'unknown'),
                        'model_type': exp.get('model_type', 'unknown'),
                        'M': exp.get('M', 0),
                        'synthetic_top_1': synthetic_metrics.get('top_1_accuracy', 0.0),
                        'synthetic_top_5': synthetic_metrics.get('top_5_accuracy', 0.0),
                        'synthetic_top_10': synthetic_metrics.get('top_10_accuracy', 0.0),
                        'sionna_top_1': 0.0,
                        'sionna_top_5': 0.0,
                        'sionna_top_10': 0.0,
                        'gap_top_1': 0.0,
                        'gap_top_5': 0.0,
                        'gap_top_10': 0.0,
                        'status': 'failed'
                    })
                except Exception as fallback_error:
                    logger.error(f"Failed to create fallback entry: {fallback_error}")
        
        # Create DataFrame
        if not comparison_data:
            logger.warning("No validation results to report")
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        
        # Log summary statistics
        if len(df) > 0:
            avg_gap = df['gap_top_1'].mean()
            logger.info(f"Average fidelity gap (top-1): {avg_gap:.4f}")
        
        # Final progress update
        if progress_callback:
            progress_callback(
                "Validation complete!",
                1.0,
                len(top_experiments),
                len(top_experiments)
            )
        
        logger.info("Cross-fidelity validation completed")
        
        return df
