
import pytest
import pandas as pd
import numpy as np
from src.filters.predicates import (
    market_cap_min, 
    currency_filter, 
    min_data_quality, 
    has_required_fields,
    max_nan_ratio
)
from src.filters.composer import DatasetFilter

@pytest.fixture
def sample_df():
    data = {
        "ticker": ["AAPL", "MSFT", "SMALL", "BAD"],
        "market_cap_t0": [3e12, 2.5e12, 1e8, 5e9],
        "currency": ["USD", "USD", "USD", "EUR"], # Note: predicates uses 'stored_currency' or just 'currency' depending on impl. 
        # Checking predicates.py: currency_filter checks df["currency"]. 
        # Wait, predicates.py lines 23-27: return df["currency"] == currency.
        # But normalization design said 'stored_currency'. 
        # I should fix predicates.py to use 'stored_currency' if that's the canonical name. 
        # For now let's use 'currency' as in current code.
        "stored_currency": ["USD", "USD", "USD", "EUR"], 
        "data_quality_score": [0.95, 0.9, 0.4, 0.8],
        "total_revenue": [100, 100, 10, 50],
        "net_income": [50, 40, -5, 10],
        "sector": ["Tech", "Tech", "Ind", "Fin"]
    }
    return pd.DataFrame(data)

def test_market_cap_min(sample_df):
    f = market_cap_min(1e9)
    res = f(sample_df) # Returns boolean series
    filtered = sample_df[res]
    assert len(filtered) == 3 # SMALL excluded
    assert "SMALL" not in filtered["ticker"].values

def test_currency_filter(sample_df):
    # The current implementation checks df["currency"]
    # We should update it to check stored_currency likely, but testing as-is first.
    # Actually, let's fix the implementation first if needed. 
    # Current code: df["currency"] == currency
    f = currency_filter("USD")
    res = f(sample_df)
    # The input dataframe column must match what the predicate expects
    # In my fixture I called it 'currency' to match 'predicates.py' assumption.
    assert res.sum() == 3 # BAD is EUR (Wait, BAD is EUR)
    # Actually 'BAD' has currency='EUR' in fixture?
    # Oh I see "SMALL" is USD. "BAD" is EUR.
    # AAPL, MSFT, SMALL are USD. BAD is EUR.
    # Total 3 USD.
    assert sum(res) == 3

def test_composer_chaining(sample_df):
    # Filter: MarketCap > 1B AND Quality > 0.8
    pipeline = (
        DatasetFilter()
        .add(market_cap_min(1e9))
        .add(min_data_quality(0.85))
    )
    
    res = pipeline.apply(sample_df)
    # AAPL: Cap=3e12 (Pass), Q=0.95 (Pass) -> KEEP
    # MSFT: Cap=2.5e12 (Pass), Q=0.9 (Pass) -> KEEP
    # SMALL: Cap=1e8 (Fail) -> DROP
    # BAD: Cap=5e9 (Pass), Q=0.8 (Fail) -> DROP
    
    assert len(res) == 2
    assert "AAPL" in res["ticker"].values
    assert "MSFT" in res["ticker"].values

def test_summary(sample_df):
    pipeline = DatasetFilter().add(market_cap_min(1e9))
    summary = pipeline.summary(sample_df)
    
    assert summary["original_count"] == 4
    assert summary["final_count"] == 3
    assert summary["total_dropped"] == 1
    assert summary["steps"][0]["dropped"] == 1

def test_is_latest():
    from src.filters.predicates import is_latest
    
    data = {
        "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "snapshot_timestamp": [
            pd.Timestamp("2024-01-01"), 
            pd.Timestamp("2024-04-01"), # Latest AAPL
            pd.Timestamp("2023-12-01"), 
            pd.Timestamp("2024-03-01")  # Latest MSFT
        ],
        "market_cap_t0": [1, 2, 3, 4]
    }
    df = pd.DataFrame(data)
    
    f = is_latest()
    res = f(df)
    
    assert res.sum() == 2
    # Ensure the correct rows were picked
    filtered = df[res]
    assert pd.Timestamp("2024-04-01") in filtered["snapshot_timestamp"].values
    assert pd.Timestamp("2024-03-01") in filtered["snapshot_timestamp"].values
    assert pd.Timestamp("2024-01-01") not in filtered["snapshot_timestamp"].values

def test_date_range():
    from src.filters.predicates import date_range
    
    data = {
        "ticker": ["A", "B", "C"],
        "period_end_date": ["2023-01-01", "2023-06-01", "2024-01-01"]
    }
    df = pd.DataFrame(data)
    
    # Filter for year 2023
    f = date_range("2023-01-01", "2023-12-31")
    res = f(df)
    
    assert res.sum() == 2
    filtered = df[res]
    assert "A" in filtered["ticker"].values
    assert "B" in filtered["ticker"].values
    assert "C" not in filtered["ticker"].values

