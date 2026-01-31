"""Scientific rules engine for guiding automated search."""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class ScientificRules:
    """Container for scientific rules loaded from YAML."""
    
    def __init__(self, rules_dict: Dict[str, Any]):
        """Initialize with rules dictionary from YAML.
        
        Args:
            rules_dict: Dictionary containing rule definitions
        """
        self.abandon_rules = rules_dict.get('abandon', [])
        self.early_stop_rules = rules_dict.get('early_stop', [])
        self.promote_rules = rules_dict.get('promote', [])
        self.compare_rules = rules_dict.get('compare', [])
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ScientificRules':
        """Load rules from YAML file.
        
        Args:
            yaml_path: Path to YAML file with rules
            
        Returns:
            ScientificRules instance
        """
        with open(yaml_path, 'r') as f:
            rules_dict = yaml.safe_load(f)
        return cls(rules_dict)
    
    def get_all_rules(self) -> List[Dict[str, Any]]:
        """Get all rules as a flat list."""
        all_rules = []
        
        for rule in self.abandon_rules:
            rule['type'] = 'abandon'
            all_rules.append(rule)
        
        for rule in self.early_stop_rules:
            rule['type'] = 'early_stop'
            all_rules.append(rule)
        
        for rule in self.promote_rules:
            rule['type'] = 'promote'
            all_rules.append(rule)
        
        for rule in self.compare_rules:
            rule['type'] = 'compare'
            all_rules.append(rule)
        
        return all_rules


class RuleEngine:
    """Engine for evaluating scientific rules against experiment context."""
    
    def __init__(self, rules: Optional[ScientificRules] = None):
        """Initialize rule engine.
        
        Args:
            rules: Optional ScientificRules instance
        """
        self.rules = rules
    
    def evaluate_condition(
        self, 
        condition: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate a single condition against context.
        
        Args:
            condition: Condition dictionary with 'metric', 'operator', 'threshold'
            context: Context dictionary with available metrics and values
            
        Returns:
            True if condition is met, False otherwise
        """
        metric = condition.get('metric')
        operator = condition.get('operator')
        threshold = condition.get('threshold')
        
        if metric not in context:
            logger.warning(f"Metric {metric} not found in context")
            return False
        
        value = context[metric]
        
        # Evaluate operator
        if operator == '<':
            return value < threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '>':
            return value > threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '==':
            return value == threshold
        elif operator == '!=':
            return value != threshold
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False
    
    def evaluate_conditions(
        self, 
        conditions: List[Dict[str, Any]], 
        context: Dict[str, Any],
        logic: str = 'and'
    ) -> bool:
        """Evaluate multiple conditions with AND/OR logic.
        
        Args:
            conditions: List of condition dictionaries
            context: Context dictionary with available metrics
            logic: 'and' or 'or' logic for combining conditions
            
        Returns:
            True if conditions are met, False otherwise
        """
        if not conditions:
            return True
        
        results = [self.evaluate_condition(cond, context) for cond in conditions]
        
        if logic == 'and':
            return all(results)
        elif logic == 'or':
            return any(results)
        else:
            logger.warning(f"Unknown logic: {logic}, using 'and'")
            return all(results)
    
    def should_abandon(self, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Check if experiment should be abandoned based on rules.
        
        Args:
            context: Context dictionary with metrics
            
        Returns:
            Tuple of (should_abandon, reason)
        """
        if not self.rules:
            return False, None
        
        for rule in self.rules.abandon_rules:
            conditions = rule.get('conditions', [])
            logic = rule.get('logic', 'and')
            
            if self.evaluate_conditions(conditions, context, logic):
                reason = rule.get('reason', 'Abandon rule triggered')
                return True, reason
        
        return False, None
    
    def should_early_stop(self, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Check if training should stop early based on rules.
        
        Args:
            context: Context dictionary with metrics
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if not self.rules:
            return False, None
        
        for rule in self.rules.early_stop_rules:
            conditions = rule.get('conditions', [])
            logic = rule.get('logic', 'and')
            
            if self.evaluate_conditions(conditions, context, logic):
                reason = rule.get('reason', 'Early stop rule triggered')
                return True, reason
        
        return False, None
    
    def should_promote(self, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Check if experiment should be promoted based on rules.
        
        Args:
            context: Context dictionary with metrics
            
        Returns:
            Tuple of (should_promote, reason)
        """
        if not self.rules:
            return False, None
        
        for rule in self.rules.promote_rules:
            conditions = rule.get('conditions', [])
            logic = rule.get('logic', 'and')
            
            if self.evaluate_conditions(conditions, context, logic):
                reason = rule.get('reason', 'Promote rule triggered')
                return True, reason
        
        return False, None
    
    def compare_experiments(
        self, 
        context_a: Dict[str, Any], 
        context_b: Dict[str, Any]
    ) -> Optional[str]:
        """Compare two experiments using comparison rules.
        
        Args:
            context_a: Context for experiment A
            context_b: Context for experiment B
            
        Returns:
            'A' if A is better, 'B' if B is better, None if no rule applies
        """
        if not self.rules:
            return None
        
        for rule in self.rules.compare_rules:
            metric = rule.get('metric')
            preference = rule.get('preference', 'higher')  # 'higher' or 'lower'
            
            if metric not in context_a or metric not in context_b:
                continue
            
            value_a = context_a[metric]
            value_b = context_b[metric]
            
            if preference == 'higher':
                if value_a > value_b:
                    return 'A'
                elif value_b > value_a:
                    return 'B'
            elif preference == 'lower':
                if value_a < value_b:
                    return 'A'
                elif value_b < value_a:
                    return 'B'
        
        return None
    
    def apply_rules(
        self, 
        context: Dict[str, Any], 
        stage: str = 'training'
    ) -> Dict[str, Any]:
        """Apply all relevant rules to context.
        
        Args:
            context: Context dictionary with metrics
            stage: Current stage ('training', 'evaluation', etc.)
            
        Returns:
            Dictionary with rule decisions
        """
        decisions = {
            'should_abandon': False,
            'abandon_reason': None,
            'should_early_stop': False,
            'early_stop_reason': None,
            'should_promote': False,
            'promote_reason': None,
        }
        
        # Check abandon rules
        abandon, reason = self.should_abandon(context)
        decisions['should_abandon'] = abandon
        decisions['abandon_reason'] = reason
        
        # Check early stop rules (only during training)
        if stage == 'training':
            early_stop, reason = self.should_early_stop(context)
            decisions['should_early_stop'] = early_stop
            decisions['early_stop_reason'] = reason
        
        # Check promote rules
        promote, reason = self.should_promote(context)
        decisions['should_promote'] = promote
        decisions['promote_reason'] = reason
        
        return decisions
