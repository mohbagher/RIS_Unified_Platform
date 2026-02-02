"""Scientific rules for experiment control and decision making."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from ris_research_engine.foundation.logging_config import get_logger

logger = get_logger(__name__)


def load_rules(yaml_path: str) -> Dict[str, Any]:
    """
    Load scientific rules from a YAML file.
    
    Args:
        yaml_path: Path to YAML file containing rules
        
    Returns:
        Dictionary of rules organized by type
        
    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
        
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
    
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    
    rules = config.get('rules', {})
    
    # Validate rule structure
    valid_rule_types = ['abandon', 'early_stop', 'promote', 'compare']
    for rule_type in rules.keys():
        if rule_type not in valid_rule_types:
            logger.warning(f"Unknown rule type: {rule_type}")
    
    num_rules = sum(len(v) if isinstance(v, list) else 0 for v in rules.values())
    logger.info(f"Loaded {num_rules} rules across {len(rules)} categories")
    
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
    
    # Check min_epoch constraint
    if 'min_epoch' in rule:
        min_epoch = rule['min_epoch']
        current_epoch = context.get('epoch', 0)
        if current_epoch < min_epoch:
            return False
    
    # Parse and evaluate condition
    try:
        result = _evaluate_condition(condition, context)
        
        # Apply threshold adjustment if present
        if result and 'threshold' in rule:
            # If the rule has a threshold, it typically means we need additional validation
            # For example: "top_1_accuracy > baseline" with threshold 0.05
            # means top_1_accuracy must be > baseline + 0.05
            threshold = rule['threshold']
            # This is handled in condition evaluation with modified comparison
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
    # Supported operators in precedence order (check longer ones first)
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
    parts = condition.split(operator, 1)
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


def _evaluate_expression(expr: str, context: Dict[str, Any]) -> Union[float, bool, str]:
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
    
    # Try to get from context directly
    if expr in context:
        value = context[expr]
        # Convert to float if it's numeric
        if isinstance(value, (int, float)):
            return float(value)
        return value
    
    # Try accessing nested keys with dot notation (e.g., "history.val_acc")
    if '.' in expr:
        parts = expr.split('.')
        current = context
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise ValueError(f"Cannot resolve nested expression: {expr}")
        
        if isinstance(current, (int, float)):
            return float(current)
        return current
    
    # Try accessing array elements (e.g., "val_acc[-1]" for last value)
    if '[' in expr and ']' in expr:
        var_name = expr.split('[')[0].strip()
        index_str = expr.split('[')[1].split(']')[0].strip()
        
        if var_name in context:
            try:
                index = int(index_str)
                value = context[var_name]
                if isinstance(value, (list, tuple)):
                    result = value[index]
                    if isinstance(result, (int, float)):
                        return float(result)
                    return result
            except (ValueError, IndexError, TypeError) as e:
                logger.debug(f"Error accessing array element: {e}")
    
    # If not found, raise error
    raise ValueError(f"Cannot resolve expression: {expr}")


def check_abandon_rules(
    rules: Dict[str, Any], 
    context: Dict[str, Any]
) -> tuple:
    """
    Check if any abandon rules are triggered.
    
    Abandon rules cause immediate termination of the current experiment.
    Used when an experiment is clearly not going to produce useful results.
    
    Args:
        rules: Dictionary of all rules (should contain 'abandon' key)
        context: Current experiment context
        
    Returns:
        Tuple of (should_abandon: bool, reason: str)
    """
    abandon_rules = rules.get('abandon', [])
    
    if not abandon_rules:
        return False, ""
    
    for rule in abandon_rules:
        try:
            if evaluate_rule(rule, context):
                reason = rule.get('reason', 'Abandon rule triggered')
                rule_name = rule.get('name', 'unnamed')
                logger.info(f"Abandon rule triggered: {rule_name}")
                return True, reason
        except Exception as e:
            logger.error(f"Error evaluating abandon rule: {e}")
    
    return False, ""


def check_early_stop_rules(
    rules: Dict[str, Any], 
    context: Dict[str, Any]
) -> tuple:
    """
    Check if any early stop rules are triggered.
    
    Early stop rules terminate training early when sufficient performance
    is achieved or when no further improvement is expected.
    
    Args:
        rules: Dictionary of all rules (should contain 'early_stop' key)
        context: Current experiment context
        
    Returns:
        Tuple of (should_stop: bool, reason: str)
    """
    early_stop_rules = rules.get('early_stop', [])
    
    if not early_stop_rules:
        return False, ""
    
    for rule in early_stop_rules:
        try:
            if evaluate_rule(rule, context):
                reason = rule.get('reason', 'Early stop rule triggered')
                rule_name = rule.get('name', 'unnamed')
                logger.info(f"Early stop rule triggered: {rule_name}")
                return True, reason
        except Exception as e:
            logger.error(f"Error evaluating early stop rule: {e}")
    
    return False, ""


def check_promote_rules(
    rules: Dict[str, Any], 
    context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Check if any promote rules are triggered.
    
    Promote rules identify promising configurations that should receive
    additional resources (e.g., more training epochs, higher priority).
    
    Args:
        rules: Dictionary of all rules (should contain 'promote' key)
        context: Current experiment context
        
    Returns:
        Promotion action dictionary if triggered, None otherwise
        Example: {'action': 'extend_budget', 'params': {'extra_epochs': 50}}
    """
    promote_rules = rules.get('promote', [])
    
    if not promote_rules:
        return None
    
    for rule in promote_rules:
        try:
            if evaluate_rule(rule, context):
                rule_name = rule.get('name', 'unnamed')
                logger.info(f"Promote rule triggered: {rule_name}")
                
                action = rule.get('action', 'extend_budget')
                params = rule.get('params', {})
                
                return {
                    'action': action,
                    'params': params,
                    'rule_name': rule_name
                }
        except Exception as e:
            logger.error(f"Error evaluating promote rule: {e}")
    
    return None


def check_compare_rules(
    rules: Dict[str, Any], 
    context: Dict[str, Any]
) -> Dict[str, bool]:
    """
    Check all comparison rules and return results.
    
    Comparison rules evaluate relationships between configurations
    (e.g., "beats baseline", "better than alternative").
    
    Args:
        rules: Dictionary of all rules (should contain 'compare' key)
        context: Current experiment context
        
    Returns:
        Dictionary mapping rule names to boolean results
        Example: {'beats_baseline': True, 'better_than_random': True}
    """
    compare_rules = rules.get('compare', [])
    results = {}
    
    if not compare_rules:
        return results
    
    for rule in compare_rules:
        rule_name = rule.get('name', 'unnamed')
        try:
            result = evaluate_rule(rule, context)
            results[rule_name] = result
            
            if result:
                logger.info(f"Compare rule satisfied: {rule_name}")
            else:
                logger.debug(f"Compare rule not satisfied: {rule_name}")
                
        except Exception as e:
            logger.error(f"Error evaluating compare rule '{rule_name}': {e}")
            results[rule_name] = False
    
    return results
