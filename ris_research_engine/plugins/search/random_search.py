"""Random search strategy - random sampling from search space."""
import random
from typing import List, Tuple, Dict, Any, Optional
from ris_research_engine.foundation.data_types import (
    ExperimentConfig, ExperimentResult, SystemConfig, TrainingConfig
)
from .base import BaseSearchStrategy


class RandomSearchStrategy(BaseSearchStrategy):
    """Random sampling from the search space."""
    
    def __init__(self):
        super().__init__()
        self.name = "random_search"
        self.description = "Random sampling from search space"
        self.experiments_generated: int = 0
        self.random_seed: Optional[int] = None
    
    def _post_initialize(self) -> None:
        """Initialize random number generator."""
        if self.rules and 'random_seed' in self.rules:
            self.random_seed = self.rules['random_seed']
            random.seed(self.random_seed)
    
    def suggest_next(self, past_results: List[ExperimentResult]) -> Optional[ExperimentConfig]:
        """Suggest a random configuration from the search space."""
        if not self.initialized:
            raise RuntimeError("Strategy not initialized. Call initialize() first.")
        
        # Check budget constraints
        if self.is_complete(past_results):
            return None
        
        # Generate random configuration
        config_params = self._sample_random_config()
        
        # Build ExperimentConfig
        experiment_config = self._build_experiment_config(config_params, self.experiments_generated)
        self.experiments_generated += 1
        
        return experiment_config
    
    def _sample_random_config(self) -> Dict[str, Any]:
        """Sample a random configuration from the search space."""
        config = {}
        
        for key, values in self.search_space.items():
            if isinstance(values, list):
                # Discrete choice
                config[key] = random.choice(values)
            elif isinstance(values, dict):
                # Handle ranges with type specification
                if 'type' in values:
                    if values['type'] == 'int':
                        config[key] = random.randint(values['min'], values['max'])
                    elif values['type'] == 'float':
                        if values.get('log_scale', False):
                            # Log-uniform sampling
                            log_min = random.log10(values['min'])
                            log_max = random.log10(values['max'])
                            config[key] = 10 ** random.uniform(log_min, log_max)
                        else:
                            # Uniform sampling
                            config[key] = random.uniform(values['min'], values['max'])
                    elif values['type'] == 'choice':
                        config[key] = random.choice(values['choices'])
                else:
                    # Default: assume uniform float
                    config[key] = random.uniform(values.get('min', 0), values.get('max', 1))
            else:
                # Single value - use as-is
                config[key] = values
        
        return config
    
    def _build_experiment_config(
        self, 
        params: Dict[str, Any], 
        index: int
    ) -> ExperimentConfig:
        """Build an ExperimentConfig from parameter dictionary."""
        # Extract system parameters
        system_config = SystemConfig(
            N=params.get('N', 64),
            N_x=params.get('N_x', 8),
            N_y=params.get('N_y', 8),
            K=params.get('K', 64),
            M=params.get('M', 8),
            frequency=params.get('frequency', 28e9),
            element_spacing=params.get('element_spacing', 0.5),
            snr_db=params.get('snr_db', 20.0),
            phase_mode=params.get('phase_mode', 'continuous'),
            phase_bits=params.get('phase_bits', 2),
        )
        
        # Extract training parameters
        training_config = TrainingConfig(
            learning_rate=params.get('learning_rate', 1e-3),
            batch_size=params.get('batch_size', 64),
            max_epochs=params.get('max_epochs', 100),
            early_stopping_patience=params.get('early_stopping_patience', 15),
            weight_decay=params.get('weight_decay', 1e-5),
            dropout=params.get('dropout', 0.1),
            scheduler=params.get('scheduler', 'cosine'),
            optimizer=params.get('optimizer', 'adam'),
            loss_function=params.get('loss_function', 'cross_entropy'),
            val_split=params.get('val_split', 0.15),
            test_split=params.get('test_split', 0.15),
            random_seed=params.get('random_seed', 42),
            num_workers=params.get('num_workers', 0),
            device=params.get('device', 'auto'),
        )
        
        # Create experiment configuration
        experiment_config = ExperimentConfig(
            name=f"random_search_exp_{index:04d}",
            system=system_config,
            training=training_config,
            probe_type=params.get('probe_type', 'dft'),
            probe_params=params.get('probe_params', {}),
            model_type=params.get('model_type', 'mlp'),
            model_params=params.get('model_params', {}),
            data_source=params.get('data_source', 'synthetic'),
            data_params=params.get('data_params', {}),
            metrics=params.get('metrics', ['top_1_accuracy', 'top_5_accuracy']),
            tags=['random_search'],
            notes=f"Random search configuration {index}",
            data_fidelity=params.get('data_fidelity', 'synthetic'),
        )
        
        return experiment_config
    
    def should_prune(self, partial_result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Optionally prune based on early performance.
        
        Can implement early stopping if performance is below threshold.
        """
        if self.rules and 'prune_threshold' in self.rules:
            current_epoch = partial_result.get('current_epoch', 0)
            min_epoch = self.rules.get('prune_after_epochs', 5)
            
            if current_epoch >= min_epoch:
                current_metric = partial_result.get('val_accuracy', 0.0)
                threshold = self.rules['prune_threshold']
                
                if current_metric < threshold:
                    return True, f"Val accuracy {current_metric:.4f} below threshold {threshold:.4f}"
        
        return False, ""
    
    def get_progress(self) -> Dict[str, Any]:
        """Get progress information."""
        max_experiments = self.budget.get('max_experiments', float('inf'))
        
        return {
            'strategy': self.name,
            'experiments_generated': self.experiments_generated,
            'max_experiments': max_experiments if max_experiments != float('inf') else 'unlimited',
            'random_seed': self.random_seed,
        }
