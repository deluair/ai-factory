"""Formatting utilities for the AI Token Factory."""

from typing import Union


def format_currency(amount: Union[float, int], currency: str = "USD", precision: int = 0) -> str:
    """Format a number as currency.
    
    Args:
        amount: The amount to format
        currency: Currency code (default: USD)
        precision: Number of decimal places (default: 0)
    
    Returns:
        Formatted currency string
    """
    if amount == 0:
        return "$0"
    
    # Handle large numbers with appropriate suffixes
    if abs(amount) >= 1_000_000_000:
        formatted = f"${amount / 1_000_000_000:.1f}B"
    elif abs(amount) >= 1_000_000:
        formatted = f"${amount / 1_000_000:.1f}M"
    elif abs(amount) >= 1_000:
        formatted = f"${amount / 1_000:.1f}K"
    else:
        if precision > 0:
            formatted = f"${amount:.{precision}f}"
        else:
            formatted = f"${amount:,.0f}"
    
    return formatted


def format_percentage(value: Union[float, int], precision: int = 1) -> str:
    """Format a number as a percentage.
    
    Args:
        value: The value to format (assumed to be in percentage form, e.g., 25.5 for 25.5%)
        precision: Number of decimal places (default: 1)
    
    Returns:
        Formatted percentage string
    """
    if value == 0:
        return "0%"
    
    return f"{value:.{precision}f}%"


def format_number(value: Union[float, int], precision: int = 0, use_separators: bool = True) -> str:
    """Format a number with appropriate separators and precision.
    
    Args:
        value: The number to format
        precision: Number of decimal places (default: 0)
        use_separators: Whether to use thousand separators (default: True)
    
    Returns:
        Formatted number string
    """
    if value == 0:
        return "0"
    
    # Handle large numbers with suffixes
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif abs(value) >= 1_000 and not use_separators:
        return f"{value / 1_000:.1f}K"
    else:
        if use_separators:
            if precision > 0:
                return f"{value:,.{precision}f}"
            else:
                return f"{value:,.0f}"
        else:
            if precision > 0:
                return f"{value:.{precision}f}"
            else:
                return f"{value:.0f}"


def format_ratio(numerator: Union[float, int], denominator: Union[float, int], precision: int = 2) -> str:
    """Format a ratio as a string.
    
    Args:
        numerator: The numerator value
        denominator: The denominator value
        precision: Number of decimal places (default: 2)
    
    Returns:
        Formatted ratio string
    """
    if denominator == 0:
        return "∞"
    
    ratio = numerator / denominator
    return f"{ratio:.{precision}f}:1"


def format_duration(seconds: Union[float, int]) -> str:
    """Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def format_bytes(bytes_value: Union[float, int], precision: int = 1) -> str:
    """Format bytes to human-readable format.
    
    Args:
        bytes_value: Size in bytes
        precision: Number of decimal places (default: 1)
    
    Returns:
        Formatted bytes string
    """
    if bytes_value == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    unit_index = 0
    size = float(bytes_value)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.{precision}f} {units[unit_index]}"


def format_tokens(token_count: Union[float, int], precision: int = 1) -> str:
    """Format token count to human-readable format.
    
    Args:
        token_count: Number of tokens
        precision: Number of decimal places (default: 1)
    
    Returns:
        Formatted token count string
    """
    if token_count == 0:
        return "0 tokens"
    
    if abs(token_count) >= 1_000_000_000:
        return f"{token_count / 1_000_000_000:.{precision}f}B tokens"
    elif abs(token_count) >= 1_000_000:
        return f"{token_count / 1_000_000:.{precision}f}M tokens"
    elif abs(token_count) >= 1_000:
        return f"{token_count / 1_000:.{precision}f}K tokens"
    else:
        return f"{token_count:,.0f} tokens"


def format_metric_change(current: Union[float, int], previous: Union[float, int], 
                        is_percentage: bool = False, precision: int = 1) -> str:
    """Format the change between two metric values.
    
    Args:
        current: Current value
        previous: Previous value
        is_percentage: Whether the values are already percentages
        precision: Number of decimal places (default: 1)
    
    Returns:
        Formatted change string with direction indicator
    """
    if previous == 0:
        return "N/A"
    
    change = current - previous
    change_pct = (change / previous) * 100
    
    if change > 0:
        direction = "↗"
        color_indicator = "+"
    elif change < 0:
        direction = "↘"
        color_indicator = ""
    else:
        direction = "→"
        color_indicator = ""
    
    if is_percentage:
        return f"{direction} {color_indicator}{change:.{precision}f}pp"  # percentage points
    else:
        return f"{direction} {color_indicator}{change_pct:.{precision}f}%"


def format_efficiency_score(score: Union[float, int], max_score: Union[float, int] = 100) -> str:
    """Format an efficiency score with visual indicators.
    
    Args:
        score: The efficiency score
        max_score: Maximum possible score (default: 100)
    
    Returns:
        Formatted efficiency score with visual indicator
    """
    percentage = (score / max_score) * 100 if max_score > 0 else 0
    
    if percentage >= 90:
        indicator = "🟢"
        level = "Excellent"
    elif percentage >= 75:
        indicator = "🟡"
        level = "Good"
    elif percentage >= 50:
        indicator = "🟠"
        level = "Fair"
    else:
        indicator = "🔴"
        level = "Poor"
    
    return f"{indicator} {score:.1f}/{max_score} ({level})"