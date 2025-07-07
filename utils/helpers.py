"""Helper utilities for the AI Token Factory."""

from typing import Union, Dict, Any, List, Optional
import math
import json
from pathlib import Path


def calculate_growth_rate(current: Union[float, int], previous: Union[float, int], 
                         periods: int = 1) -> float:
    """Calculate growth rate between two values.
    
    Args:
        current: Current value
        previous: Previous value
        periods: Number of periods (default: 1 for simple growth rate)
    
    Returns:
        Growth rate as a percentage
    """
    if previous == 0:
        return 0.0
    
    if periods == 1:
        # Simple growth rate
        return ((current - previous) / previous) * 100
    else:
        # Compound annual growth rate (CAGR)
        return (((current / previous) ** (1 / periods)) - 1) * 100


def calculate_roi(gain: Union[float, int], cost: Union[float, int]) -> float:
    """Calculate Return on Investment (ROI).
    
    Args:
        gain: The gain from investment
        cost: The cost of investment
    
    Returns:
        ROI as a percentage
    """
    if cost == 0:
        return 0.0
    
    return ((gain - cost) / cost) * 100


def calculate_compound_growth(initial_value: Union[float, int], growth_rate: float, 
                            periods: int) -> float:
    """Calculate compound growth over multiple periods.
    
    Args:
        initial_value: Starting value
        growth_rate: Growth rate per period (as percentage)
        periods: Number of periods
    
    Returns:
        Final value after compound growth
    """
    rate_decimal = growth_rate / 100
    return initial_value * ((1 + rate_decimal) ** periods)


def calculate_break_even_point(fixed_costs: Union[float, int], 
                              variable_cost_per_unit: Union[float, int],
                              price_per_unit: Union[float, int]) -> float:
    """Calculate break-even point in units.
    
    Args:
        fixed_costs: Total fixed costs
        variable_cost_per_unit: Variable cost per unit
        price_per_unit: Selling price per unit
    
    Returns:
        Break-even point in units
    """
    contribution_margin = price_per_unit - variable_cost_per_unit
    
    if contribution_margin <= 0:
        return float('inf')  # Cannot break even
    
    return fixed_costs / contribution_margin


def calculate_payback_period(initial_investment: Union[float, int], 
                           annual_cash_flow: Union[float, int]) -> float:
    """Calculate payback period in years.
    
    Args:
        initial_investment: Initial investment amount
        annual_cash_flow: Annual cash flow
    
    Returns:
        Payback period in years
    """
    if annual_cash_flow <= 0:
        return float('inf')  # Never pays back
    
    return initial_investment / annual_cash_flow


def calculate_net_present_value(cash_flows: List[float], discount_rate: float) -> float:
    """Calculate Net Present Value (NPV) of cash flows.
    
    Args:
        cash_flows: List of cash flows (first element is initial investment, negative)
        discount_rate: Discount rate as percentage
    
    Returns:
        Net Present Value
    """
    rate_decimal = discount_rate / 100
    npv = 0
    
    for period, cash_flow in enumerate(cash_flows):
        npv += cash_flow / ((1 + rate_decimal) ** period)
    
    return npv


def calculate_internal_rate_of_return(cash_flows: List[float], 
                                    initial_guess: float = 0.1) -> Optional[float]:
    """Calculate Internal Rate of Return (IRR) using Newton-Raphson method.
    
    Args:
        cash_flows: List of cash flows (first element should be negative initial investment)
        initial_guess: Initial guess for IRR (default: 0.1 or 10%)
    
    Returns:
        IRR as a percentage, or None if not found
    """
    def npv_function(rate):
        return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows))
    
    def npv_derivative(rate):
        return sum(-i * cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows))
    
    rate = initial_guess
    tolerance = 1e-6
    max_iterations = 100
    
    for _ in range(max_iterations):
        npv = npv_function(rate)
        npv_prime = npv_derivative(rate)
        
        if abs(npv) < tolerance:
            return rate * 100  # Convert to percentage
        
        if npv_prime == 0:
            return None  # Cannot find IRR
        
        rate = rate - npv / npv_prime
        
        if rate < -1:  # Prevent negative rates below -100%
            rate = -0.99
    
    return None  # IRR not found


def calculate_efficiency_ratio(output: Union[float, int], input_value: Union[float, int]) -> float:
    """Calculate efficiency ratio (output per unit of input).
    
    Args:
        output: Output value
        input_value: Input value
    
    Returns:
        Efficiency ratio
    """
    if input_value == 0:
        return 0.0
    
    return output / input_value


def calculate_utilization_rate(actual: Union[float, int], capacity: Union[float, int]) -> float:
    """Calculate utilization rate as a percentage.
    
    Args:
        actual: Actual usage/output
        capacity: Maximum capacity
    
    Returns:
        Utilization rate as percentage
    """
    if capacity == 0:
        return 0.0
    
    return (actual / capacity) * 100


def calculate_market_share(company_value: Union[float, int], 
                          total_market_value: Union[float, int]) -> float:
    """Calculate market share as a percentage.
    
    Args:
        company_value: Company's value/revenue/units
        total_market_value: Total market value/revenue/units
    
    Returns:
        Market share as percentage
    """
    if total_market_value == 0:
        return 0.0
    
    return (company_value / total_market_value) * 100


def validate_config(config: Dict[str, Any], required_fields: List[str]) -> Dict[str, List[str]]:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        required_fields: List of required field names
    
    Returns:
        Dictionary with 'errors' and 'warnings' lists
    """
    errors = []
    warnings = []
    
    # Check for required fields
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
        elif config[field] is None:
            errors.append(f"Required field '{field}' cannot be None")
    
    # Check for common data type issues
    for key, value in config.items():
        if isinstance(value, str) and value.strip() == "":
            warnings.append(f"Field '{key}' is an empty string")
        elif isinstance(value, (int, float)) and value < 0 and key.endswith(('_cost', '_price', '_revenue')):
            warnings.append(f"Field '{key}' has negative value: {value}")
    
    return {
        'errors': errors,
        'warnings': warnings
    }


def load_json_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        file_path: Path to JSON configuration file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_config(config: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary to save
        file_path: Path where to save the JSON file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, default=str)


def interpolate_values(start_value: Union[float, int], end_value: Union[float, int], 
                      steps: int) -> List[float]:
    """Generate interpolated values between start and end.
    
    Args:
        start_value: Starting value
        end_value: Ending value
        steps: Number of steps (including start and end)
    
    Returns:
        List of interpolated values
    """
    if steps < 2:
        return [start_value]
    
    step_size = (end_value - start_value) / (steps - 1)
    return [start_value + i * step_size for i in range(steps)]


def calculate_percentile(values: List[Union[float, int]], percentile: float) -> float:
    """Calculate percentile of a list of values.
    
    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)
    
    Returns:
        Percentile value
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if percentile <= 0:
        return sorted_values[0]
    elif percentile >= 100:
        return sorted_values[-1]
    
    index = (percentile / 100) * (n - 1)
    lower_index = int(math.floor(index))
    upper_index = int(math.ceil(index))
    
    if lower_index == upper_index:
        return sorted_values[lower_index]
    
    # Linear interpolation
    weight = index - lower_index
    return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


def normalize_values(values: List[Union[float, int]], 
                    target_min: float = 0.0, target_max: float = 1.0) -> List[float]:
    """Normalize values to a target range.
    
    Args:
        values: List of values to normalize
        target_min: Target minimum value (default: 0.0)
        target_max: Target maximum value (default: 1.0)
    
    Returns:
        List of normalized values
    """
    if not values:
        return []
    
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        # All values are the same
        return [target_min] * len(values)
    
    range_original = max_val - min_val
    range_target = target_max - target_min
    
    return [
        target_min + ((value - min_val) / range_original) * range_target
        for value in values
    ]