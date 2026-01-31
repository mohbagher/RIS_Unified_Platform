"""Successive halving strategy - progressive elimination of worst performers."""
import math
from typing import List, Tuple, Dict, Any, Optional
from ris_research_engine.foundation.data_types import (
    ExperimentConfig, ExperimentResult, SystemConfig, TrainingConfig
)
from .base import BaseSearchStrategy


class SuccessiveHalvingStrategy(BaseSearchStrategy):
    """
    Successive Halving: Start with many configurations, train for few epochs,
    eliminate worst half, double training budget for survivors, repeat.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "successive_halving"
        self.description = "Progressive elimination of worst performers"
        
        # Strategy state
        self.n_configs: int = 0
        self.min_epochs: int = 1
        self.max_epochs: int = 100
        self.reduction_factor: int = 2
        self.current_round: int = 0
        self.rounds: List[Dict[str, Any]] = []
        self.all_configs: List[Dict[str, Any]] = []
        self.current_round_configs: List[Dict[str, Any]] = []
        self.current_round_index: int = 0
    
    def _post_initialize(self) -> None:
        """Initialize successive halving rounds."""
        # Extract parameters from rules
        self.n_configs = self.rules.get('n_configs', 16)
        self.min_epochs = self.rules.get('min_epochs', 1)
        self.max_epochs = self.budget.get('max_epochs', 100)
        self.reduction_factor = self.rules.get('reduction_factor', 2)
        
        # Calculate number of rounds
        n_rounds = math.floor(math.log(self.n_configs, self.reduction_factor)) + 1
        
        # Plan rounds
        for round_idx in range(n_rounds):
            n_configs_in_round = self.n_configs // (self.reduction_factor ** round_idx)
            epochs_in_round = self.min_epochs * (self.reduction_factor ** round_idx)
            epochs_in_round = min(epochs_in_round, self.max_epochs)
            
            self.rounds.append({
                'round': round_idx,
                'n_configs': n_configs_in_round,
                'epochs': epochs_in_round,
                'configs': [],
                'results': [],
            })
        
        # Generate initial configurations
        self.all_configs = self._generate_initial_configs(self.n_configs)
        self.current_round_configs = self.all_configs.copy()
        self.current_round = 0
        self.current_round_index = 0
    
    def _generate_initial_configs(self, n: int) -> List[Dict[str, Any]]:
        """Generate n random configurations from search space."""
        import random
        
        if self.rules and 'random_seed' in self.rules:
            random.seed(self.rules['random_seed'])
        
        configs = []
        for _ in range(n):
            config = {}
            for key, values in self.search_space.items():
                if isinstance(values, list):
                    config[key] = random.choice(values)
                elif isinstance(values, dict):
                    if values.get('type') == 'int':
                        config[key] = random.randint(values['min'], values['max'])
                    elif values.get('type') == 'float':
                        if values.get('log_scale', False):
                            log_min = math.log10(values['min'])
                            log_max = math.log10(values['max'])
                            config[key] = 10 ** random.uniform(log_min, log_max)
                        else:
                            config[key] = random.uniform(values['min'], values['max'])
                    elif values.get('type') == 'choice':
                        config[key] = random.choice(values['choices'])
                else:
                    config[key] = values
            configs.append(config)
        
        return configs
    
    def suggest_next(self, past_results: List[ExperimentResult]) -> Optional[ExperimentConfig]:
        """Suggest next configuration in current round or advance to next round."""
        if not self.initialized:
            raise RuntimeError("Strategy not initialized. Call initialize() first.")
        
        # Check if current round is complete
        if self.current_round_index >= len(self.current_round_configs):
            # Advance to next round
            if not self._advance_to_next_round(past_results):
                return None  # Search complete
            
            # Reset index for new round
            self.current_round_index = 0
        
        # Get current configuration
        config_params = self.current_round_configs[self.current_round_index]
        current_epochs = self.rounds[self.current_round]['epochs']
        
        # Override max_epochs for this round
        config_params['max_epochs'] = current_epochs
        
        # Build experiment config
        experiment_config = self._build_experiment_config(
            config_params,
            self.current_round,
            self.current_round_index
        )
        
        self.current_round_index += 1
        
        return experiment_config
    
    def _advance_to_next_round(self, past_results: List[ExperimentResult]) -> bool:
        """Select top performers and advance to next round."""
        # Get results from current round
        current_round_results = [
            r for r in past_results
            if f"round_{self.current_round}" in r.config.tags
        ]
        
        # Check if we have enough results
        if len(current_round_results) < len(self.current_round_configs):
            return True  # Still waiting for results
        
        # Sort by primary metric (descending)
        current_round_results.sort(
            key=lambda r: r.primary_metric_value,
            reverse=True
        )
        
        # Move to next round
        self.current_round += 1
        
        # Check if we've completed all rounds
        if self.current_round >= len(self.rounds):
            return False  # Search complete
        
        # Select top performers
        n_survivors = self.rounds[self.current_round]['n_configs']
        survivors = current_round_results[:n_survivors]
        
        # Update configs for next round
        self.current_round_configs = [
            result.config.to_dict() for result in survivors
        ]
        
        return True
    
    def _build_experiment_config(
        self,
        params: Dict[str, Any],
        round_idx: int,
        config_idx: int
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
            name=f"successive_halving_round_{round_idx}_cfg_{config_idx:04d}",
            system=system_config,
            training=training_config,
            probe_type=params.get('probe_type', 'dft'),
            probe_params=params.get('probe_params', {}),
            model_type=params.get('model_type', 'mlp'),
            model_params=params.get('model_params', {}),
            data_source=params.get('data_source', 'synthetic'),
            data_params=params.get('data_params', {}),
            metrics=params.get('metrics', ['top_1_accuracy', 'top_5_accuracy']),
            tags=['successive_halving', f'round_{round_idx}'],
            notes=f"Successive halving round {round_idx}, config {config_idx}",
            data_fidelity=params.get('data_fidelity', 'synthetic'),
        )
        
        return experiment_config
    
    def should_prune(self, partial_result: Dict[str, Any]) -> Tuple[bool, str]:
        """Successive halving controls epochs per round, no additional pruning."""
        return False, ""
    
    def get_progress(self) -> Dict[str, Any]:
        """Get progress information."""
        if self.current_round < len(self.rounds):
            current_round_info = self.rounds[self.current_round]
        else:
            current_round_info = {'round': -1, 'n_configs': 0, 'epochs': 0}
        
        return {
            'strategy': self.name,
            'current_round': self.current_round,
            'total_rounds': len(self.rounds),
            'configs_in_current_round': len(self.current_round_configs),
            'current_config_index': self.current_round_index,
            'epochs_in_current_round': current_round_info['epochs'],
            'rounds_info': self.rounds,
        }
