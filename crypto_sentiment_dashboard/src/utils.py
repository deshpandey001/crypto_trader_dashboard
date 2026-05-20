"""
Utility functions for the Crypto Sentiment Dashboard.
"""

import logging
import os
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger("crypto_sentiment_dashboard")
    logger.setLevel(getattr(logging, log_level))
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(log_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler.setLevel(getattr(logging, log_level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def create_output_directories() -> None:
    """Create necessary output directories if they don't exist."""
    directories = [
        "outputs/cleaned_data",
        "outputs/charts",
        "outputs/reports",
        "outputs/models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division to avoid division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero
        
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to 0-1 range.
    
    Args:
        value: Value to normalize
        min_val: Minimum value in the range
        max_val: Maximum value in the range
        
    Returns:
        Normalized value between 0 and 1
    """
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def calculate_percentile(data: list, percentile: float) -> float:
    """
    Calculate percentile of a list.
    
    Args:
        data: List of values
        percentile: Percentile value (0-100)
        
    Returns:
        Percentile value
    """
    if not data:
        return 0.0
    
    sorted_data = sorted(data)
    index = (percentile / 100) * (len(sorted_data) - 1)
    
    lower_index = int(index)
    upper_index = lower_index + 1
    
    if upper_index >= len(sorted_data):
        return sorted_data[-1]
    
    lower_value = sorted_data[lower_index]
    upper_value = sorted_data[upper_index]
    
    return lower_value + (upper_value - lower_value) * (index - lower_index)


def format_large_number(num: float, decimals: int = 2) -> str:
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        num: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    if abs(num) >= 1e9:
        return f"{num / 1e9:.{decimals}f}B"
    elif abs(num) >= 1e6:
        return f"{num / 1e6:.{decimals}f}M"
    elif abs(num) >= 1e3:
        return f"{num / 1e3:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def convert_sentiment_to_numeric(sentiment: str) -> int:
    """
    Convert sentiment label to numeric value.
    
    Args:
        sentiment: Sentiment string ('Fear' or 'Greed')
        
    Returns:
        Numeric value (0 for Fear, 1 for Greed)
    """
    sentiment_map = {
        "fear": 0,
        "greed": 1,
        "Fear": 0,
        "Greed": 1
    }
    return sentiment_map.get(sentiment, -1)


def convert_numeric_to_sentiment(value: int) -> str:
    """
    Convert numeric value to sentiment label.
    
    Args:
        value: Numeric value (0 or 1)
        
    Returns:
        Sentiment string
    """
    return "Fear" if value == 0 else "Greed"


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save the config file
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4, default=str)
    except Exception as e:
        logging.error(f"Error saving config to {filepath}: {str(e)}")


def load_config(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to the config file
        
    Returns:
        Configuration dictionary or None if file doesn't exist
    """
    try:
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Error loading config from {filepath}: {str(e)}")
    return None


def validate_dataframe_columns(df, required_columns: list) -> bool:
    """
    Validate that dataframe has required columns.
    
    Args:
        df: Pandas dataframe
        required_columns: List of required column names
        
    Returns:
        True if all required columns exist, False otherwise
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return False
    return True
