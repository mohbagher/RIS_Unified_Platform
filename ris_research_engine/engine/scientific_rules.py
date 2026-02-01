"""Scientific rules for experiment control and decision making."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from ris_research_engine.foundation.logging_config import get_logger

logger = get_logger(__name__)


def load_rules(yaml_path: str) -> Dict[str, Any]:
    """
    Load scientific rules from a YAML file.
    
    Args:
        yaml_path: Path to YAML file containing rules
        
    Returns:
        Dictionary of rules organized by type
        
    Example YAML format:
        rules:
          abandon:
            - name: "poor_early_performance"
              condition: "val_acc < 0.3"
              at_epoch: 10
              reason: "Validation accuracy too low after 10 epochs"
          
          early_stop:
            - name: "excellent_performance"
              condition: "val_acc > 0.95"
              at_epoch: 20
              reason: "Excellent performance achieved early"
          
          promote:
            - name: "promising_result"
              condition: "val_acc > 0.85"
              action: "extend_budget"
              params:
                extra_epochs: 50
          
          compare:
            - name: "beats_baseline"
              condition: "top_1_accuracy > baseline_best_of_probed"
              threshold: 0.05
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Rules file not found: {yaml_path}")
    
    logger.info(f"Loading rules from {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    rules = config.get('rules', {})
    
    # Validate rule structure
    valid_rule_types = ['abandon', 'early_stop', 'promote', 'compare']
    for rule_type in rules.keys():
        if rule_type not in valid_rule_types:
            logger.warning(f"Unknown rule type: {rule_type}")
    
    logger.info(f"Loaded {sum(len(v) for v in rules.values())} rules")
    
    return rules


def evaluate_rule(rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """
    Evaluate a single rule against the current context.
    
    Args:
        rule: Rule dictionary with 'condition' and optional parameters
        context: Context dictionary with current state
        
    Returns:
        True if rule condition is satisfied, False otherwise
        
    Supported operators: <, >, ==, !=, >=, <=
    
    Example context:
        {
            'epoch': 15,
            'val_acc': 0.85,
            'val_loss': 0.42,
            'train_acc': 0.90,
            'train_loss': 0.35,
            'top_1_accuracy': 0.82,
            'top_5_accuracy': 0.95,
            'baseline_random_selection': 0.25,
            'baseline_best_of_probed': 0.65,
            'training_history': {
                'val_acc': [0.3, 0.45, 0.6, 0.75, 0.85],
                'val_loss': [0.8, 0.65, 0.52, 0.45, 0.42],
            }
        }
    
    Example rules:
        {'condition': 'val_acc > 0.8', 'at_epoch': 20}
        {'condition': 'top_1_accuracy > baseline_best_of_probed', 'threshold': 0.05}
        {'condition': 'val_loss < 0.3'}
    """
    condition = rule.get('condition', '')
    
    if not condition:
        logger.warning("Rule has no condition")
        return False
    
    # Check epoch constraint
    if 'at_epoch' in rule:
        required_epoch = rule['at_epoch']
        current_epoch = context.get('epoch', 0)
        if current_epoch < required_epoch:
            return False
    
    # Parse condition
    try:
        result = _evaluate_condition(condition, context)
        
        # Apply threshold if specified
        if 'threshold' in rule and result:
            # For comparison rules with thresholds
            threshold = rule['threshold']
            # This is typically used for "A > B" where we want "A > B + threshold"
            # The condition parser should handle this, but we can add margin here
            pass
        
        return result
        
    except Exception as e:
        logger.error(f"Error evaluating rule condition '{condition}': {e}")
        return False


def _evaluate_condition(condition: str, context: Dict[str, Any]) -> bool:
    """
    Evaluate a condition string against context.
    
    Args:
        condition: Condition string (e.g., "val_acc > 0.8")
        context: Context dictionary
        
    Returns:
        Boolean result of evaluation
    """
    # Supported operators
    operators = ['<=', '>=', '==', '!=', '<', '>']
    
    # Find operator in condition
    operator = None
    for op in operators:
        if op in condition:
            operator = op
            break
    
    if operator is None:
        raise ValueError(f"No valid operator found in condition: {condition}")
    
    # Split by operator
    parts = condition.split(operator)
    if len(parts) != 2:
        raise ValueError(f"Invalid condition format: {condition}")
    
    left_expr = parts[0].strip()
    right_expr = parts[1].strip()
    
    # Evaluate left and right expressions
    left_value = _evaluate_expression(left_expr, context)
    right_value = _evaluate_expression(right_expr, context)
    
    # Compare using operator
    if operator == '<':
        return left_value < right_value
    elif operator == '>':
        return left_value > right_value
    elif operator == '==':
        return left_value == right_value
    elif operator == '!=':
        return left_value != right_value
    elif operator == '<=':
        return left_value <= right_value
    elif operator == '>=':
        return left_value >= right_value
    
    return False


def _evaluate_expression(expr: str, context: Dict[str, Any]) -> Any:
    """
    Evaluate an expression to get its value from context.
    
    Args:
        expr: Expression string (variable name or literal)
        context: Context dictionary
        
    Returns:
        Value of the expression
    """
    expr = expr.strip()
    
    # Try to parse as number
    try:
        return float(expr)
    except ValueError:
        pass
    
    # Try to parse as boolean
    if expr.lower() == 'true':
        return True
    elif expr.lower() == 'false':
        return False
    
    # Try to get from context
    if expr in context:
        return context[expr]
    
    # Try nested keys (e.g., "baseline_random_selection")
    # This is already handled by direct key lookup
    
    # Try accessing training history (e.g., "val_acc[-1]" for last value)
    if '[' in expr and ']' in expr:
        # Handle array indexing
        var_name = expr.split('[')[0].strip()
        index_str = expr.split('[')[1].split(']')[0].strip()
        
        if var_name in context:
            try:
                index = int(index_str)
                value = context[var_name]
                if isinstance(value, (list, tuple)):
                    return value[index]
            except (ValueError, IndexError, TypeError):
                pass
    
    # If not found, raise error
    raise ValueError(f"Cannot resolve expression: {expr}")


def check_abandon_rules(
    rules: Dict[str, Any], 
    context: Dict[str, Any]
) -> tuple:
    """
    Check if any abandon rules are triggered.
    
    Args:
        rules: Dictionary of all rules
        context: Current experiment context
        
    Returns:
        Tuple of (should_abandon: bool, reason: str)
    """
    abandon_rules = rules.get('abandon', [])
    
    for rule in abandon_rules:
        if evaluate_rule(rule, context):
            reason = rule.get('reason', 'Abandon rule triggered')
            logger.info(f"Abandon rule triggered: {rule.get('name', 'unnamed')}")
            return True, reason
    
    return False, ""


def check_early_stop_rules(
    rules: Dict[str, Any], 
    context: Dict[str, Any]
) -> tuple:
    """
    Check if any early stop rules are triggered.
    
    Args:
        rules: Dictionary of all rules
        context: Current experiment context
        
    Returns:
        Tuple of (should_stop: bool, reason: str)
    """
    early_stop_rules = rules.get('early_stop', [])
    
    for rule in early_stop_rules:
        if evaluate_rule(rule, context):
            reason = rule.get('reason', 'Early stop rule triggered')
            logger.info(f"Early stop rule triggered: {rule.get('name', 'unnamed')}")
            return True, reason
    
    return False, ""


def check_promote_rules(
    rules: Dict[str, Any], 
    context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Check if any promote rules are triggered.
    
    Args:
        rules: Dictionary of all rules
        context: Current experiment context
        
    Returns:
        Promotion action dictionary if triggered, None otherwise
    """
    promote_rules = rules.get('promote', [])
    
    for rule in promote_rules:
        if evaluate_rule(rule, context):
            logger.info(f"Promote rule triggered: {rule.get('name', 'unnamed')}")
            return {
                'action': rule.get('action', 'extend_budget'),
                'params': rule.get('params', {})
            }
    
    return None


def check_compare_rules(
    rules: Dict[str, Any], 
    context: Dict[str, Any]
) -> Dict[str, bool]:
    """
    Check all comparison rules and return results.
    
    Args:
        rules: Dictionary of all rules
        context: Current experiment context
        
    Returns:
        Dictionary mapping rule names to boolean results
    """
    compare_rules = rules.get('compare', [])
    results = {}
    
    for rule in compare_rules:
        rule_name = rule.get('name', 'unnamed')
        result = evaluate_rule(rule, context)
        results[rule_name] = result
        
        if result:
            logger.info(f"Compare rule satisfied: {rule_name}")
    
    return results
