"""Core data types for the RIS research engine."""
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json


@dataclass
class SystemConfig:
    """Configuration for the RIS system parameters."""
    N: int = 64  # Number of RIS elements
    N_x: int = 8  # Elements in x-dimension
    N_y: int = 8  # Elements in y-dimension
    K: int = 64  # Codebook size
    M: int = 8  # Sensing budget (number of probe measurements)
    frequency: float = 28e9  # Carrier frequency in Hz
    element_spacing: float = 0.5  # Spacing relative to wavelength
    snr_db: float = 20.0  # Signal-to-noise ratio in dB
    phase_mode: str = 'continuous'  # 'continuous' or 'discrete'
    phase_bits: int = 2  # Number of bits for phase quantization (if discrete)
    
    def validate(self):
        """Validate system configuration."""
        assert self.N > 0, "N must be positive"
        assert self.N_x > 0 and self.N_y > 0, "N_x and N_y must be positive"
        assert self.N_x * self.N_y == self.N, f"N_x * N_y must equal N: {self.N_x}*{self.N_y}!={self.N}"
        assert self.K > 0, "K must be positive"
        assert self.M > 0 and self.M <= self.K, f"M must be in range [1, K]: M={self.M}, K={self.K}"
        assert self.frequency > 0, "frequency must be positive"
        assert self.element_spacing > 0, "element_spacing must be positive"
        assert self.phase_mode in ['continuous', 'discrete'], "phase_mode must be 'continuous' or 'discrete'"
        if self.phase_mode == 'discrete':
            assert self.phase_bits > 0, "phase_bits must be positive for discrete phase mode"
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'N': self.N,
            'N_x': self.N_x,
            'N_y': self.N_y,
            'K': self.K,
            'M': self.M,
            'frequency': self.frequency,
            'element_spacing': self.element_spacing,
            'snr_db': self.snr_db,
            'phase_mode': self.phase_mode,
            'phase_bits': self.phase_bits,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SystemConfig':
        """Create from dictionary."""
        return cls(**d)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 15
    weight_decay: float = 1e-5
    dropout: float = 0.1
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau', 'none'
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    loss_function: str = 'cross_entropy'  # 'cross_entropy', 'mse', 'mae'
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    num_workers: int = 0  # DataLoader workers (0 for Windows compatibility)
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    gradient_clip_max_norm: float = 1.0  # Gradient clipping max norm (0 to disable)
    
    def validate(self):
        """Validate training configuration."""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.max_epochs > 0, "max_epochs must be positive"
        assert self.early_stopping_patience > 0, "early_stopping_patience must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.scheduler in ['cosine', 'step', 'plateau', 'none'], "Invalid scheduler"
        assert self.optimizer in ['adam', 'sgd', 'adamw'], "Invalid optimizer"
        assert 0 < self.val_split < 1, "val_split must be in (0, 1)"
        assert 0 < self.test_split < 1, "test_split must be in (0, 1)"
        assert self.val_split + self.test_split < 1, "val_split + test_split must be < 1"
        assert self.num_workers >= 0, "num_workers must be non-negative"
        assert self.device in ['auto', 'cuda', 'cpu'], "Invalid device"
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'weight_decay': self.weight_decay,
            'dropout': self.dropout,
            'scheduler': self.scheduler,
            'optimizer': self.optimizer,
            'loss_function': self.loss_function,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'random_seed': self.random_seed,
            'num_workers': self.num_workers,
            'device': self.device,
            'gradient_clip_max_norm': self.gradient_clip_max_norm,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**d)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    system: SystemConfig
    training: TrainingConfig
    probe_type: str
    probe_params: dict
    model_type: str
    model_params: dict
    data_source: str
    data_params: dict
    metrics: List[str]
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    data_fidelity: str = "synthetic"  # 'synthetic', 'sionna', 'hardware'
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'system': self.system.to_dict(),
            'training': self.training.to_dict(),
            'probe_type': self.probe_type,
            'probe_params': self.probe_params,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'data_source': self.data_source,
            'data_params': self.data_params,
            'metrics': self.metrics,
            'tags': self.tags,
            'notes': self.notes,
            'data_fidelity': self.data_fidelity,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ExperimentConfig':
        """Create from dictionary."""
        d = d.copy()
        d['system'] = SystemConfig.from_dict(d['system'])
        d['training'] = TrainingConfig.from_dict(d['training'])
        return cls(**d)


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    config: ExperimentConfig
    metrics: Dict[str, float]
    training_history: Dict[str, List[float]]
    best_epoch: int
    total_epochs: int
    training_time_seconds: float
    model_parameters: int
    timestamp: str
    status: str  # 'completed', 'failed', 'pruned'
    error_message: str = ""
    baseline_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    primary_metric_name: str = "top_1_accuracy"
    primary_metric_value: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'config': self.config.to_dict(),
            'metrics': self.metrics,
            'training_history': self.training_history,
            'best_epoch': self.best_epoch,
            'total_epochs': self.total_epochs,
            'training_time_seconds': self.training_time_seconds,
            'model_parameters': self.model_parameters,
            'timestamp': self.timestamp,
            'status': self.status,
            'error_message': self.error_message,
            'baseline_results': self.baseline_results,
            'primary_metric_name': self.primary_metric_name,
            'primary_metric_value': self.primary_metric_value,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class SearchCampaignResult:
    """Result from a search campaign."""
    campaign_name: str
    search_strategy: str
    total_experiments: int
    completed_experiments: int
    pruned_experiments: int
    failed_experiments: int
    best_result: ExperimentResult
    all_results: List[ExperimentResult]
    total_time_seconds: float
    search_space_definition: dict
    timestamp: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'campaign_name': self.campaign_name,
            'search_strategy': self.search_strategy,
            'total_experiments': self.total_experiments,
            'completed_experiments': self.completed_experiments,
            'pruned_experiments': self.pruned_experiments,
            'failed_experiments': self.failed_experiments,
            'best_result': self.best_result.to_dict() if self.best_result else None,
            'all_results': [r.to_dict() for r in self.all_results],
            'total_time_seconds': self.total_time_seconds,
            'search_space_definition': self.search_space_definition,
            'timestamp': self.timestamp,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
