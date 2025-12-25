"""Pydantic models for the Index module."""

from enum import Enum
from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class WeightingScheme(str, Enum):
    """Available index weighting schemes."""

    EQUAL = "equal"
    MARKET_CAP = "market_cap"
    CUSTOM = "custom"


class IndexResult(BaseModel):
    """Result of index calculation at a single point in time."""

    index_id: str
    as_of_time: datetime

    # Actual index value (weighted actual market caps)
    actual_index: float

    # Estimated index value (weighted predicted market caps)
    estimated_index: float
    estimated_index_std: float = Field(
        description="Propagated uncertainty from per-ticker predictions"
    )

    # Relative mispricing of the index
    index_relative_error: float = Field(
        description="(estimated - actual) / actual"
    )

    # Constituents used
    n_tickers: int = Field(description="Total tickers in index membership")
    n_tickers_with_valuation: int = Field(
        description="Tickers with valuation results available"
    )

    # Weights used
    weights: Dict[str, float] = Field(
        description="Weight per ticker used in calculation"
    )

    # Model metadata
    model_version: Optional[str] = None
    weighting_scheme: WeightingScheme = WeightingScheme.EQUAL

    class Config:
        """Pydantic config."""

        use_enum_values = True
