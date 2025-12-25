from .predicates import (
    market_cap_min,
    currency_filter,
    min_data_quality,
    max_nan_ratio,
    positive_revenue,
    profitable,
)
from .composer import DatasetFilter


# === DEFAULT FILTER CONFIG ===
# Matches current value_analysis.py behavior
DEFAULT_FILTER = (
    DatasetFilter()
    .add(market_cap_min(1e9))       # Market cap > $1B
    .add(currency_filter("USD"))     # USD stocks only
    .add(max_nan_ratio(0.3))         # Max 30% missing values
)


# === ALTERNATIVE CONFIGS ===
# For experimentation

# All stocks (minimal filtering)
MINIMAL_FILTER = (
    DatasetFilter()
    .add(market_cap_min(0))
    .add(positive_revenue())
)

# Large cap only
LARGE_CAP_FILTER = (
    DatasetFilter()
    .add(market_cap_min(10e9))       # Market cap > $10B
    .add(min_data_quality(0.7))
)

# Profitable companies only
PROFITABLE_FILTER = (
    DatasetFilter()
    .add(market_cap_min(1e9))
    .add(positive_revenue())
    .add(profitable())
)


# === ACTIVE FILTER ===
# Change this to swap filter configs
ACTIVE_FILTER = DEFAULT_FILTER
