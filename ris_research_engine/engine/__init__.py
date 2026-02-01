"""Engine layer for RIS Auto-Research Engine.

This module provides the core execution engine for running experiments,
orchestrating search campaigns, analyzing results, and generating reports.
"""

from .experiment_runner import ExperimentRunner
from .search_controller import SearchController
from .result_analyzer import ResultAnalyzer
from .report_generator import ReportGenerator
from .scientific_rules import (
    load_rules,
    evaluate_rule,
    check_abandon_rules,
    check_early_stop_rules,
    check_promote_rules,
    check_compare_rules
)

__all__ = [
    'ExperimentRunner',
    'SearchController',
    'ResultAnalyzer',
    'ReportGenerator',
    'load_rules',
    'evaluate_rule',
    'check_abandon_rules',
    'check_early_stop_rules',
    'check_promote_rules',
    'check_compare_rules',
]
