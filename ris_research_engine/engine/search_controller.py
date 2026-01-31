"""Search controller for orchestrating automated research campaigns."""

import time
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
import yaml
import pandas as pd

from ris_research_engine.foundation.data_types import (
    ExperimentConfig, ExperimentResult, SearchCampaignResult,
    SystemConfig, TrainingConfig
)
from ris_research_engine.foundation.storage import ResultTracker
from ris_research_engine.plugins.search import BaseSearchStrategy, get_search_strategy
from .experiment_runner import ExperimentRunner
from .scientific_rules import ScientificRules, RuleEngine

logger = logging.getLogger(__name__)


class SearchController:
    """Controller for running automated search campaigns."""
    
    def __init__(
        self, 
        result_tracker: Optional[ResultTracker] = None,
        experiment_runner: Optional[ExperimentRunner] = None
    ):
        """Initialize search controller.
        
        Args:
            result_tracker: Optional result tracker for saving results
            experiment_runner: Optional experiment runner (creates one if not provided)
        """
        self.result_tracker = result_tracker or ResultTracker()
        self.experiment_runner = experiment_runner or ExperimentRunner(result_tracker)
        self.rule_engine = None
    
    def _create_experiment_config(
        self,
        params: Dict[str, Any],
        base_system: SystemConfig,
        base_training: TrainingConfig,
        campaign_name: str,
        data_source: str,
        data_params: Dict[str, Any],
        metrics: List[str]
    ) -> ExperimentConfig:
        """Create an ExperimentConfig from search parameters.
        
        Args:
            params: Parameter dictionary from search strategy
            base_system: Base system configuration
            base_training: Base training configuration
            campaign_name: Name of the search campaign
            data_source: Data source name
            data_params: Data source parameters
            metrics: List of metrics to compute
            
        Returns:
            ExperimentConfig instance
        """
        # Update system config with params
        system = SystemConfig(
            N=params.get('N', base_system.N),
            N_x=params.get('N_x', base_system.N_x),
            N_y=params.get('N_y', base_system.N_y),
            K=params.get('K', base_system.K),
            M=params.get('M', base_system.M),
            frequency=params.get('frequency', base_system.frequency),
            element_spacing=params.get('element_spacing', base_system.element_spacing),
            snr_db=params.get('snr_db', base_system.snr_db),
            phase_mode=params.get('phase_mode', base_system.phase_mode),
            phase_bits=params.get('phase_bits', base_system.phase_bits),
        )
        
        # Update training config with params
        training = TrainingConfig(
            learning_rate=params.get('learning_rate', base_training.learning_rate),
            batch_size=params.get('batch_size', base_training.batch_size),
            max_epochs=params.get('max_epochs', base_training.max_epochs),
            early_stopping_patience=params.get('early_stopping_patience', base_training.early_stopping_patience),
            weight_decay=params.get('weight_decay', base_training.weight_decay),
            dropout=params.get('dropout', base_training.dropout),
            scheduler=params.get('scheduler', base_training.scheduler),
            optimizer=params.get('optimizer', base_training.optimizer),
            loss_function=params.get('loss_function', base_training.loss_function),
            val_split=params.get('val_split', base_training.val_split),
            test_split=params.get('test_split', base_training.test_split),
            random_seed=params.get('random_seed', base_training.random_seed),
            num_workers=params.get('num_workers', base_training.num_workers),
            device=params.get('device', base_training.device),
        )
        
        # Create experiment config
        config = ExperimentConfig(
            name=f"{campaign_name}_{params.get('probe_type')}_{params.get('model_type')}_{params.get('random_seed')}",
            system=system,
            training=training,
            probe_type=params.get('probe_type'),
            probe_params=params.get('probe_params', {}),
            model_type=params.get('model_type'),
            model_params=params.get('model_params', {}),
            data_source=data_source,
            data_params=data_params,
            metrics=metrics,
            tags=[campaign_name] + params.get('tags', []),
            notes=params.get('notes', ''),
            data_fidelity=params.get('data_fidelity', 'synthetic'),
        )
        
        return config
    
    def run_campaign(
        self,
        search_space_config: Dict[str, Any],
        strategy_name: str,
        rules_config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None
    ) -> SearchCampaignResult:
        """Run a complete search campaign.
        
        Args:
            search_space_config: Dictionary defining the search space
            strategy_name: Name of search strategy to use
            rules_config: Optional scientific rules configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            SearchCampaignResult with all experiment results
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        campaign_name = search_space_config.get('name', 'unnamed_campaign')
        
        logger.info(f"Starting search campaign: {campaign_name}")
        logger.info(f"Strategy: {strategy_name}")
        
        # Setup rules engine
        if rules_config:
            rules = ScientificRules(rules_config)
            self.rule_engine = RuleEngine(rules)
        else:
            self.rule_engine = None
        
        # Get search strategy
        strategy = get_search_strategy(strategy_name)
        
        # Initialize strategy
        strategy.initialize(
            search_space=search_space_config.get('search_space', {}),
            budget=search_space_config.get('budget', {}),
            rules=rules_config
        )
        
        # Extract base configs
        base_system = SystemConfig.from_dict(search_space_config.get('system', {}))
        base_training = TrainingConfig.from_dict(search_space_config.get('training', {}))
        data_source = search_space_config.get('data_source', 'synthetic_rayleigh')
        data_params = search_space_config.get('data_params', {})
        metrics = search_space_config.get('metrics', ['top_1_accuracy'])
        
        # Run search
        all_results = []
        completed_count = 0
        pruned_count = 0
        failed_count = 0
        best_result = None
        
        while not strategy.is_complete(all_results):
            # Get next configuration
            next_config_params = strategy.suggest_next(all_results)
            
            if next_config_params is None:
                logger.info("Search strategy returned no more configs")
                break
            
            # Create experiment config
            exp_config = self._create_experiment_config(
                next_config_params,
                base_system,
                base_training,
                campaign_name,
                data_source,
                data_params,
                metrics
            )
            
            logger.info(f"Running experiment {len(all_results) + 1}: {exp_config.name}")
            
            # Run experiment
            result = self.experiment_runner.run(
                exp_config,
                progress_callback=progress_callback
            )
            
            # Add campaign name to result
            result.config.tags.append(campaign_name)
            
            # Update counts
            if result.status == 'completed':
                completed_count += 1
            elif result.status == 'pruned':
                pruned_count += 1
            elif result.status == 'failed':
                failed_count += 1
            
            # Track best result
            if result.status == 'completed':
                if best_result is None or result.primary_metric_value > best_result.primary_metric_value:
                    best_result = result
            
            all_results.append(result)
            
            # Progress callback
            if progress_callback:
                progress_callback({
                    'campaign': campaign_name,
                    'total_experiments': len(all_results),
                    'completed': completed_count,
                    'pruned': pruned_count,
                    'failed': failed_count,
                    'best_metric': best_result.primary_metric_value if best_result else 0.0,
                    'progress': strategy.get_progress(),
                })
        
        # Create campaign result
        total_time = time.time() - start_time
        
        campaign_result = SearchCampaignResult(
            campaign_name=campaign_name,
            search_strategy=strategy_name,
            total_experiments=len(all_results),
            completed_experiments=completed_count,
            pruned_experiments=pruned_count,
            failed_experiments=failed_count,
            best_result=best_result,
            all_results=all_results,
            total_time_seconds=total_time,
            search_space_definition=search_space_config,
            timestamp=timestamp,
        )
        
        logger.info(f"Campaign completed in {total_time:.2f}s")
        logger.info(f"Completed: {completed_count}, Pruned: {pruned_count}, Failed: {failed_count}")
        if best_result:
            logger.info(f"Best result: {best_result.primary_metric_value:.4f}")
        
        return campaign_result
    
    def run_from_yaml(
        self,
        yaml_path: str,
        progress_callback: Optional[Callable] = None
    ) -> SearchCampaignResult:
        """Run search campaign from YAML configuration file.
        
        Args:
            yaml_path: Path to YAML configuration file
            progress_callback: Optional callback for progress updates
            
        Returns:
            SearchCampaignResult with all experiment results
        """
        logger.info(f"Loading search config from {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        strategy_name = config.get('strategy', 'grid_search')
        rules_config = config.get('rules', None)
        
        return self.run_campaign(
            search_space_config=config,
            strategy_name=strategy_name,
            rules_config=rules_config,
            progress_callback=progress_callback
        )
    
    def run_cross_fidelity_validation(
        self,
        synthetic_campaign_name: str,
        hdf5_path: str,
        top_n: int = 10,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """Run cross-fidelity validation.
        
        Validate top N configurations from synthetic search on Sionna data.
        
        Args:
            synthetic_campaign_name: Name of synthetic campaign to validate
            hdf5_path: Path to Sionna HDF5 data file
            top_n: Number of top configurations to validate
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Running cross-fidelity validation for campaign: {synthetic_campaign_name}")
        
        # Get top N results from synthetic campaign
        synthetic_results = self.result_tracker.query(
            campaign_name=synthetic_campaign_name,
            status='completed',
            limit=top_n,
            sort_by='primary_metric_value',
            sort_order='desc'
        )
        
        if not synthetic_results:
            logger.error(f"No completed results found for campaign: {synthetic_campaign_name}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(synthetic_results)} results, validating top {min(top_n, len(synthetic_results))}")
        
        # Run validation experiments on Sionna data
        validation_results = []
        
        for i, synthetic_result in enumerate(synthetic_results[:top_n]):
            # Create validation config with Sionna data
            val_config = synthetic_result.config
            val_config.name = f"{val_config.name}_sionna_validation"
            val_config.data_source = 'hdf5_loader'
            val_config.data_params = {'hdf5_path': hdf5_path}
            val_config.data_fidelity = 'sionna'
            
            logger.info(f"Validating config {i+1}/{top_n}: {val_config.name}")
            
            # Run experiment
            val_result = self.experiment_runner.run(
                val_config,
                progress_callback=progress_callback
            )
            
            validation_results.append({
                'config_name': synthetic_result.config.name,
                'probe_type': synthetic_result.config.probe_type,
                'model_type': synthetic_result.config.model_type,
                'M': synthetic_result.config.system.M,
                'synthetic_metric': synthetic_result.primary_metric_value,
                'sionna_metric': val_result.primary_metric_value if val_result.status == 'completed' else None,
                'fidelity_gap': (synthetic_result.primary_metric_value - val_result.primary_metric_value) 
                                if val_result.status == 'completed' else None,
                'validation_status': val_result.status,
            })
            
            if progress_callback:
                progress_callback({
                    'validation_progress': i + 1,
                    'total': top_n,
                })
        
        # Create DataFrame
        df = pd.DataFrame(validation_results)
        
        logger.info("Cross-fidelity validation complete")
        logger.info(f"Mean fidelity gap: {df['fidelity_gap'].mean():.4f}")
        
        return df
