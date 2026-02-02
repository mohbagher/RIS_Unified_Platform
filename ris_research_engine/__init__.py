"""RIS Auto-Research Engine - Main Package."""

from .foundation import (
    SystemConfig, TrainingConfig, ExperimentConfig,
    ExperimentResult, SearchCampaignResult
)
from .foundation.storage import ResultTracker
from .engine import ExperimentRunner, SearchController, ResultAnalyzer, ReportGenerator

__version__ = "1.0.0"

__all__ = [
    'SystemConfig', 'TrainingConfig', 'ExperimentConfig',
    'ExperimentResult', 'SearchCampaignResult', 'ResultTracker',
    'ExperimentRunner', 'SearchController', 'ResultAnalyzer', 'ReportGenerator'
]
