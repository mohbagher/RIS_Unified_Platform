"""RIS Auto-Research Engine - A modular framework for RIS research automation."""

__version__ = "0.1.0"

# Foundation layer
from .foundation.data_types import (
    SystemConfig,
    TrainingConfig,
    ExperimentConfig,
    ExperimentResult,
    SearchCampaignResult,
)
from .foundation.storage import ResultTracker
from .foundation.logging_config import setup_logging

# Engine layer
from .engine.experiment_runner import ExperimentRunner
from .engine.search_controller import SearchController
from .engine.scientific_rules import ScientificRules, RuleEngine
from .engine.result_analyzer import ResultAnalyzer
from .engine.report_generator import ReportGenerator

# UI layer
from .ui.jupyter_minimal import RISEngine

# Plugin registries
from .plugins.probes import get_probe, list_probes
from .plugins.models import get_model, list_models
from .plugins.metrics import get_metric, list_metrics
from .plugins.data_sources import get_data_source, list_data_sources
from .plugins.baselines import AVAILABLE_BASELINES

__all__ = [
    # Version
    '__version__',
    
    # Foundation
    'SystemConfig',
    'TrainingConfig',
    'ExperimentConfig',
    'ExperimentResult',
    'SearchCampaignResult',
    'ResultTracker',
    'setup_logging',
    
    # Engine
    'ExperimentRunner',
    'SearchController',
    'ScientificRules',
    'RuleEngine',
    'ResultAnalyzer',
    'ReportGenerator',
    
    # UI
    'RISEngine',
    
    # Plugins
    'get_probe',
    'list_probes',
    'get_model',
    'list_models',
    'get_metric',
    'list_metrics',
    'get_data_source',
    'list_data_sources',
    'AVAILABLE_BASELINES',
]
