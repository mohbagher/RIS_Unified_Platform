"""Scientific search strategy - systematic 4-phase approach for RIS research."""
from typing import List, Tuple, Dict, Any, Optional
from ris_research_engine.foundation.data_types import (
    ExperimentConfig, ExperimentResult, SystemConfig, TrainingConfig
)
from .base import BaseSearchStrategy


class ScientificSearchStrategy(BaseSearchStrategy):
    """
    Scientific search with 4 phases:
    1. Screening: Test all probe types with default MLP, 10 epochs, keep top 3
    2. Sparsity: Sweep M/K ratios for top probes
    3. Models: Compare all model types with best probe+sparsity
    4. Tuning: Grid over lr × batch_size × dropout for best configuration
    """
    
    def __init__(self):
        super().__init__()
        self.name = "scientific_search"
        self.description = "Systematic 4-phase search: screening → sparsity → models → tuning"
        
        # Phase definitions
        self.phases = ['screening', 'sparsity', 'models', 'tuning']
        self.current_phase: int = 0
        self.current_phase_index: int = 0
        
        # Phase-specific state
        self.phase_configs: Dict[str, List[Dict[str, Any]]] = {
            'screening': [],
            'sparsity': [],
            'models': [],
            'tuning': [],
        }
        
        # Results from each phase
        self.phase_results: Dict[str, List[ExperimentResult]] = {
            'screening': [],
            'sparsity': [],
            'models': [],
            'tuning': [],
        }
        
        # Best configurations carried forward
        self.top_probes: List[str] = []
        self.best_probe: str = ""
        self.best_sparsity: Dict[str, int] = {}
        self.best_model: str = ""
    
    def _post_initialize(self) -> None:
        """Initialize phase 1 (screening) configurations."""
        self._initialize_screening_phase()
    
    def _initialize_screening_phase(self) -> None:
        """Phase 1: Screen all probe types with default MLP, 10 epochs."""
        probe_types = self.search_space.get('probe_types', ['dft', 'random', 'hadamard'])
        
        # Get default values for other parameters
        default_M = self.search_space.get('M', [8])[0] if isinstance(
            self.search_space.get('M', [8]), list
        ) else self.search_space.get('M', 8)
        
        default_K = self.search_space.get('K', [64])[0] if isinstance(
            self.search_space.get('K', [64]), list
        ) else self.search_space.get('K', 64)
        
        # Create configs for each probe type
        for probe_type in probe_types:
            config = {
                'probe_type': probe_type,
                'probe_params': {},
                'model_type': 'mlp',
                'model_params': {},
                'M': default_M,
                'K': default_K,
                'max_epochs': 10,
                'learning_rate': 1e-3,
                'batch_size': 64,
                'dropout': 0.1,
            }
            self.phase_configs['screening'].append(config)
    
    def _initialize_sparsity_phase(self) -> None:
        """Phase 2: Sweep M/K ratios for top 3 probes from screening."""
        # Get top 3 probes from screening results
        screening_results = self.phase_results['screening']
        
        if not screening_results:
            # No screening results yet
            return
        
        sorted_results = sorted(
            screening_results,
            key=lambda r: r.primary_metric_value,
            reverse=True
        )
        
        self.top_probes = [r.config.probe_type for r in sorted_results[:3]]
        
        # Define M/K ratios to test
        K_values = self.search_space.get('K', [64])
        M_ratios = self.rules.get('sparsity_ratios', [0.05, 0.1, 0.15, 0.2, 0.25])
        
        # Create configs for each probe and M/K ratio
        for probe_type in self.top_probes:
            for K in (K_values if isinstance(K_values, list) else [K_values]):
                for ratio in M_ratios:
                    M = max(1, int(K * ratio))
                    config = {
                        'probe_type': probe_type,
                        'probe_params': {},
                        'model_type': 'mlp',
                        'model_params': {},
                        'M': M,
                        'K': K,
                        'max_epochs': 30,
                        'learning_rate': 1e-3,
                        'batch_size': 64,
                        'dropout': 0.1,
                    }
                    self.phase_configs['sparsity'].append(config)
    
    def _initialize_models_phase(self) -> None:
        """Phase 3: Compare all models with best probe+sparsity."""
        # Get best probe and sparsity from phase 2
        sparsity_results = self.phase_results['sparsity']
        
        if not sparsity_results:
            # No sparsity results yet
            return
        
        best_result = max(sparsity_results, key=lambda r: r.primary_metric_value)
        
        self.best_probe = best_result.config.probe_type
        self.best_sparsity = {
            'M': best_result.config.system.M,
            'K': best_result.config.system.K,
        }
        
        # Get all model types
        model_types = self.search_space.get('model_types', ['mlp', 'cnn', 'resnet', 'transformer'])
        
        # Create config for each model type
        for model_type in model_types:
            config = {
                'probe_type': self.best_probe,
                'probe_params': {},
                'model_type': model_type,
                'model_params': {},
                'M': self.best_sparsity['M'],
                'K': self.best_sparsity['K'],
                'max_epochs': 50,
                'learning_rate': 1e-3,
                'batch_size': 64,
                'dropout': 0.1,
            }
            self.phase_configs['models'].append(config)
    
    def _initialize_tuning_phase(self) -> None:
        """Phase 4: Grid over lr × batch_size × dropout for best config."""
        # Get best model from phase 3
        models_results = self.phase_results['models']
        
        if not models_results:
            # No model results yet
            return
        
        best_result = max(models_results, key=lambda r: r.primary_metric_value)
        
        self.best_model = best_result.config.model_type
        
        # Define tuning grid
        learning_rates = self.rules.get('tuning_lr', [1e-4, 5e-4, 1e-3, 5e-3])
        batch_sizes = self.rules.get('tuning_batch_size', [32, 64, 128])
        dropouts = self.rules.get('tuning_dropout', [0.0, 0.1, 0.2, 0.3])
        
        # Create grid of configs
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for dropout in dropouts:
                    config = {
                        'probe_type': self.best_probe,
                        'probe_params': {},
                        'model_type': self.best_model,
                        'model_params': {},
                        'M': self.best_sparsity['M'],
                        'K': self.best_sparsity['K'],
                        'max_epochs': 100,
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'dropout': dropout,
                    }
                    self.phase_configs['tuning'].append(config)
    
    def suggest_next(self, past_results: List[ExperimentResult]) -> Optional[ExperimentConfig]:
        """Suggest next experiment based on current phase."""
        if not self.initialized:
            raise RuntimeError("Strategy not initialized. Call initialize() first.")
        
        # Update phase results
        self._update_phase_results(past_results)
        
        # Get current phase name
        if self.current_phase >= len(self.phases):
            return None  # All phases complete
        
        phase_name = self.phases[self.current_phase]
        phase_configs = self.phase_configs[phase_name]
        
        # Check if current phase is complete
        if self.current_phase_index >= len(phase_configs):
            # Check if we have all results from current phase before advancing
            current_phase_results = len(self.phase_results[phase_name])
            expected_results = len(phase_configs)
            
            if current_phase_results < expected_results:
                # Still waiting for results from current phase
                return None
            
            # Move to next phase
            if not self._advance_to_next_phase():
                return None  # All phases complete
            
            phase_name = self.phases[self.current_phase]
            phase_configs = self.phase_configs[phase_name]
            self.current_phase_index = 0
        
        # Get next config in current phase
        config_params = phase_configs[self.current_phase_index]
        
        # Build experiment config
        experiment_config = self._build_experiment_config(
            config_params,
            phase_name,
            self.current_phase_index
        )
        
        self.current_phase_index += 1
        
        return experiment_config
    
    def _update_phase_results(self, all_results: List[ExperimentResult]) -> None:
        """Update results for each phase."""
        for phase_name in self.phases:
            self.phase_results[phase_name] = [
                r for r in all_results if phase_name in r.config.tags
            ]
    
    def _advance_to_next_phase(self) -> bool:
        """Initialize next phase and advance."""
        self.current_phase += 1
        
        if self.current_phase >= len(self.phases):
            return False  # All phases complete
        
        # Initialize next phase
        phase_name = self.phases[self.current_phase]
        
        if phase_name == 'sparsity':
            self._initialize_sparsity_phase()
        elif phase_name == 'models':
            self._initialize_models_phase()
        elif phase_name == 'tuning':
            self._initialize_tuning_phase()
        
        return True
    
    def _build_experiment_config(
        self,
        params: Dict[str, Any],
        phase: str,
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
            name=f"scientific_{phase}_exp_{index:04d}",
            system=system_config,
            training=training_config,
            probe_type=params.get('probe_type', 'dft'),
            probe_params=params.get('probe_params', {}),
            model_type=params.get('model_type', 'mlp'),
            model_params=params.get('model_params', {}),
            data_source=params.get('data_source', 'synthetic'),
            data_params=params.get('data_params', {}),
            metrics=params.get('metrics', ['top_1_accuracy', 'top_5_accuracy']),
            tags=['scientific_search', phase],
            notes=f"Scientific search - {phase} phase, experiment {index}",
            data_fidelity=params.get('data_fidelity', 'synthetic'),
        )
        
        return experiment_config
    
    def should_prune(self, partial_result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Prune experiments that are clearly underperforming.
        
        In screening/sparsity phases, prune if accuracy is very low after min epochs.
        """
        current_phase = self.phases[self.current_phase] if self.current_phase < len(self.phases) else 'done'
        
        if current_phase in ['screening', 'sparsity']:
            current_epoch = partial_result.get('current_epoch', 0)
            if current_epoch >= 5:
                val_accuracy = partial_result.get('val_accuracy', 0.0)
                # Prune if accuracy is below random guess (1/K)
                K = partial_result.get('K', 64)
                random_baseline = 1.0 / K
                
                if val_accuracy < random_baseline * 0.5:  # Less than half of random
                    return True, f"Val accuracy {val_accuracy:.4f} below random baseline"
        
        return False, ""
    
    def get_progress(self) -> Dict[str, Any]:
        """Get progress information."""
        if self.current_phase < len(self.phases):
            phase_name = self.phases[self.current_phase]
            total_in_phase = len(self.phase_configs[phase_name])
        else:
            phase_name = 'complete'
            total_in_phase = 0
        
        return {
            'strategy': self.name,
            'current_phase': self.current_phase + 1,
            'total_phases': len(self.phases),
            'phase_name': phase_name,
            'experiments_in_phase': self.current_phase_index,
            'total_in_phase': total_in_phase,
            'top_probes': self.top_probes,
            'best_probe': self.best_probe,
            'best_sparsity': self.best_sparsity,
            'best_model': self.best_model,
        }
