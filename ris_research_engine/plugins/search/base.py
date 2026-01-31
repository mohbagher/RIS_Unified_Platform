"""Base class for search strategies."""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from ris_research_engine.foundation.data_types import ExperimentConfig, ExperimentResult


class BaseSearchStrategy(ABC):
    """Abstract base class for all search strategies."""
    
    def __init__(self):
        self.name: str = "base"
        self.description: str = "Base search strategy"
        self.search_space: Optional[Dict[str, Any]] = None
        self.budget: Optional[Dict[str, Any]] = None
        self.rules: Optional[Dict[str, Any]] = None
        self.initialized: bool = False
    
    def initialize(
        self, 
        search_space: Dict[str, Any], 
        budget: Dict[str, Any], 
        rules: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the search strategy with search space, budget, and optional rules.
        
        Args:
            search_space: Dictionary defining the search space (e.g., parameters to tune)
            budget: Dictionary defining resource constraints (e.g., max_experiments, max_time)
            rules: Optional dictionary with strategy-specific rules
        """
        self.search_space = search_space
        self.budget = budget
        self.rules = rules or {}
        self.initialized = True
        self._post_initialize()
    
    def _post_initialize(self) -> None:
        """Hook for subclasses to perform additional initialization."""
        pass
    
    @abstractmethod
    def suggest_next(self, past_results: List[ExperimentResult]) -> Optional[ExperimentConfig]:
        """
        Suggest the next experiment configuration based on past results.
        
        Args:
            past_results: List of results from previously run experiments
            
        Returns:
            ExperimentConfig for the next experiment, or None if search is complete
        """
        pass
    
    @abstractmethod
    def should_prune(self, partial_result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if an experiment should be pruned early based on partial results.
        
        Args:
            partial_result: Dictionary containing partial metrics and training info
            
        Returns:
            Tuple of (should_prune: bool, reason: str)
        """
        pass
    
    @abstractmethod
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress information about the search.
        
        Returns:
            Dictionary with progress metrics (e.g., experiments_completed, estimated_remaining)
        """
        pass
    
    def is_complete(self, past_results: List[ExperimentResult]) -> bool:
        """
        Check if the search is complete based on budget and results.
        
        Args:
            past_results: List of results from previously run experiments
            
        Returns:
            True if search is complete, False otherwise
        """
        if not self.initialized or not self.budget:
            return True
        
        # Check max_experiments budget
        if 'max_experiments' in self.budget:
            completed = len([r for r in past_results if r.status == 'completed'])
            if completed >= self.budget['max_experiments']:
                return True
        
        # Check max_time budget
        if 'max_time_hours' in self.budget:
            total_time = sum(r.training_time_seconds for r in past_results) / 3600.0
            if total_time >= self.budget['max_time_hours']:
                return True
        
        return False
