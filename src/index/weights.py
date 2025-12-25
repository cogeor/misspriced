"""Weighting scheme implementations for index calculation."""

from typing import Dict, List
from src.db.models.valuation import ValuationResult


def compute_equal_weights(tickers: List[str]) -> Dict[str, float]:
    """
    Compute equal weights for all tickers.

    Each ticker gets weight = 1/N.
    """
    n = len(tickers)
    if n == 0:
        return {}
    weight = 1.0 / n
    return {ticker: weight for ticker in tickers}


def compute_market_cap_weights(
    valuations: List[ValuationResult],
) -> Dict[str, float]:
    """
    Compute market-cap-weighted weights.

    Weight = actual_mcap_i / sum(actual_mcap).
    Larger companies have more influence on the index.
    """
    total_mcap = sum(float(v.actual_mcap) for v in valuations if v.actual_mcap)

    if total_mcap == 0:
        # Fallback to equal weight if no market cap data
        return compute_equal_weights([v.ticker for v in valuations])

    weights = {}
    for v in valuations:
        mcap = float(v.actual_mcap) if v.actual_mcap else 0
        weights[v.ticker] = mcap / total_mcap

    return weights


def normalize_custom_weights(custom_weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize custom weights to sum to 1.0.

    Args:
        custom_weights: User-provided weights (don't need to sum to 1)

    Returns:
        Normalized weights summing to 1.0
    """
    total = sum(custom_weights.values())
    if total == 0:
        return custom_weights

    return {ticker: w / total for ticker, w in custom_weights.items()}
