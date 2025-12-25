
import pytest
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.valuation.service import ValuationService
from src.valuation.config import baseline_model
from src.filters.composer import DatasetFilter
from src.filters.predicates import market_cap_min

DB_PATH = "test_valuation.db"

@pytest.fixture
def db_session():
    """Create session for test DB."""
    if not os.path.exists(DB_PATH):
        pytest.fail(f"Test DB {DB_PATH} not found. Run scripts/setup_test_db.py first.")
        
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_valuation_e2e(db_session):
    """
    Test End-to-End Valuation Service.
    1. Load data from test DB
    2. Filter it
    3. Run model (few experiments for speed)
    4. Check results
    """
    service = ValuationService(db_session)
    
    # Configure model (reduce experiments for fast test)
    # Note: ModelBuilder creates config, we can modify it or build custom one
    config = baseline_model()
    config.n_experiments = 2
    config.outer_cv_splits = 2
    config.inner_cv_splits = 2
    
    # Filter
    filters = DatasetFilter().add(market_cap_min(0)) # No min cap for test
    
    # Run
    results = service.run_valuation(config, filters)
    
    # Assertions
    assert "mean" in results
    assert "std" in results
    assert "mispricing" in results
    assert len(results["mean"]) == 10 # 10 samples in test DB
    
    # Check if predictions are reasonable (not all zero, though untrained model might be bad)
    # XGBoost should output something non-zero
    assert results["mean"].sum() > 0
    
    print("\nSample Results:")
    df_res = pd.DataFrame({
        "Ticker": results["tickers"],
        "Actual": results["actual_mcap_mm"],
        "Predicted": results["mean"],
        "Std": results["std"],
        "Mispricing": results["mispricing"]
    })
    print(df_res.head())

def test_valuation_filtering(db_session):
    """Test that filters are applied correctly."""
    service = ValuationService(db_session)
    config = baseline_model()
    config.n_experiments = 2
    config.outer_cv_splits = 2
    config.inner_cv_splits = 2
    
    # Filter to exclude everything (should raise error)
    # Create unlikely filter
    filters = DatasetFilter().add(market_cap_min(1e20)) # impossibly high
    
    with pytest.raises(ValueError, match="All snapshots were filtered out"):
        service.run_valuation(config, filters)
