"""Grid search strategy - exhaustive search over all combinations."""
import itertools
from typing import List, Tuple, Dict, Any, Optional
from ris_research_engine.foundation.data_types import (
    ExperimentConfig, ExperimentResult, SystemConfig, TrainingConfig
)
from .base import BaseSearchStrategy


class GridSearchStrategy(BaseSearchStrategy):
    """Exhaustive grid search over all parameter combinations."""
    
    def __init__(self):
        super().__init__()
        self.name = "grid_search"
        self.description = "Exhaustive grid search over all parameter combinations"
        self.all_configs: List[Dict[str, Any]] = []
        self.current_index: int = 0
    
    def _post_initialize(self) -> None:
        """Generate all possible configurations from search space."""
        if not self.search_space:
            return
        
        # Extract parameter names and their values
        param_names = []
        param_values = []
        
        for key, values in self.search_space.items():
            if isinstance(values, list):
                param_names.append(key)
                param_values.append(values)
        
        # Generate all combinations using Cartesian product
        for combo in itertools.product(*param_values):
            config = dict(zip(param_names, combo))
            self.all_configs.append(config)
        
        self.current_index = 0
    
    def suggest_next(self, past_results: List[ExperimentResult]) -> Optional[ExperimentConfig]:
        """Suggest the next configuration in the grid."""
        if not self.initialized:
            raise RuntimeError("Strategy not initialized. Call initialize() first.")
        
        # Check if we've exhausted all configurations
        if self.current_index >= len(self.all_configs):
            return None
        
        # Check budget constraints
        if self.is_complete(past_results):
            return None
        
        # Get the next configuration
        config_params = self.all_configs[self.current_index]
        self.current_index += 1
        
        # Build ExperimentConfig from parameters
        experiment_config = self._build_experiment_config(config_params, self.current_index - 1)
        
        return experiment_config
    
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
            name=f"grid_search_exp_{index:04d}",
            system=system_config,
            training=training_config,
            probe_type=params.get('probe_type', 'dft'),
            probe_params=params.get('probe_params', {}),
            model_type=params.get('model_type', 'mlp'),
            model_params=params.get('model_params', {}),
            data_source=params.get('data_source', 'synthetic'),
            data_params=params.get('data_params', {}),
            metrics=params.get('metrics', ['top_1_accuracy', 'top_5_accuracy']),
            tags=['grid_search'],
            notes=f"Grid search configuration {index}",
            data_fidelity=params.get('data_fidelity', 'synthetic'),
        )
        
        return experiment_config
    
    def should_prune(self, partial_result: Dict[str, Any]) -> Tuple[bool, str]:
        """Grid search typically doesn't prune - runs all experiments to completion."""
        return False, ""
    
    def get_progress(self) -> Dict[str, Any]:
        """Get progress information."""
        total = len(self.all_configs)
        completed = self.current_index
        
        return {
            'strategy': self.name,
            'total_configurations': total,
            'completed': completed,
            'remaining': total - completed,
            'progress_percent': (completed / total * 100) if total > 0 else 0,
        }
