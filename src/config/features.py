from typing import List, Callable, Optional, Any
from pydantic import BaseModel, Field

class FeatureConfig(BaseModel):
    """Configuration for feature selection."""
    
    # Core features always included
    core_features: List[str]
    
    # Optional features (may be missing)
    optional_features: List[str]
    
    # Features to exclude from model
    excluded_features: List[str]
    
    # Minimum non-null ratio to include feature (0.0 - 1.0)
    min_coverage: float = 0.5
    
    # Custom feature filter function
    custom_filter: Optional[Any] = None


# === ACTIVE FEATURE CONFIG ===
FEATURE_CONFIG = FeatureConfig(
    core_features=[
        "total_revenue",
        "gross_profit",
        "ebitda",
        "total_debt",
        "total_cash",
        "free_cash_flow",
        "operating_cash_flow",
        "shares_outstanding",
        "book_value",
        "profit_margins",
        "debt_to_equity",
        "roa",
        "roe",
        "current_ratio",
        "quick_ratio",
    ],
    optional_features=[
        "dividend_yield",
        "dividend_rate",
        "audit_risk",
        "board_risk",
        "compensation_risk",
        "shares_short",
        "short_ratio",
        "held_percent_insiders",
        "held_percent_institutions",
        "full_time_employees",
    ],
    excluded_features=[
        "ticker",
        "snapshot_timestamp",
        "market_cap_t0",  # This is the target, not a feature
        "price_t0",
        "price_t_minus_1",
        "price_t_plus_1",
    ],
    min_coverage=0.5,
)


def get_feature_list(available_columns: List[str]) -> List[str]:
    """
    Get final feature list based on config and available columns.
    
    Args:
        available_columns: Columns available in the dataset
        
    Returns:
        List of feature names to use
    """
    config = FEATURE_CONFIG
    
    features = []
    
    # Add core features (must be present)
    for f in config.core_features:
        if f in available_columns:
            features.append(f)
    
    # Add optional features (if present)
    for f in config.optional_features:
        if f in available_columns and f not in config.excluded_features:
            features.append(f)
    
    # Apply custom filter if provided
    if config.custom_filter:
        features = config.custom_filter(features)
    
    return features
