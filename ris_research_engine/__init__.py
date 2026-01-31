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

# Plugin registries
from .plugins.probes import get_probe, list_probes
from .plugins.models import get_model, list_models
from .plugins.metrics import get_metric, list_metrics
from .plugins.data_sources import get_data_source, list_data_sources
from .plugins.baselines import AVAILABLE_BASELINES

# Engine layer - import lazily to avoid torch dependency on import
def _lazy_import_engine():
    """Lazy import of engine components."""
    from .engine.experiment_runner import ExperimentRunner
    from .engine.search_controller import SearchController
    from .engine.scientific_rules import ScientificRules, RuleEngine
    from .engine.result_analyzer import ResultAnalyzer
    from .engine.report_generator import ReportGenerator
    return (ExperimentRunner, SearchController, ScientificRules, 
            RuleEngine, ResultAnalyzer, ReportGenerator)

# UI layer - import lazily
def _lazy_import_ui():
    """Lazy import of UI components."""
    from .ui.jupyter_minimal import RISEngine
    return RISEngine,

# Provide lazy imports via __getattr__
def __getattr__(name):
    """Lazy load heavy dependencies."""
    if name == 'ExperimentRunner':
        return _lazy_import_engine()[0]
    elif name == 'SearchController':
        return _lazy_import_engine()[1]
    elif name == 'ScientificRules':
        return _lazy_import_engine()[2]
    elif name == 'RuleEngine':
        return _lazy_import_engine()[3]
    elif name == 'ResultAnalyzer':
        return _lazy_import_engine()[4]
    elif name == 'ReportGenerator':
        return _lazy_import_engine()[5]
    elif name == 'RISEngine':
        return _lazy_import_ui()[0]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

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
