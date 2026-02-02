"""Engine layer for RIS Auto-Research Engine.

This module provides the core execution engine for running experiments,
orchestrating search campaigns, analyzing results, and generating reports.
"""

from .experiment_runner import ExperimentRunner
from .search_controller import SearchController
from .result_analyzer import ResultAnalyzer
from .report_generator import ReportGenerator
from . import scientific_rules

__all__ = [
    'ExperimentRunner',
    'SearchController',
    'ResultAnalyzer',
    'ReportGenerator',
    'scientific_rules',
]
