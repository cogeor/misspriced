
import pytest
from datetime import date
from unittest.mock import MagicMock
from src.normalize.currency import CurrencyConverter
from src.normalize.quality_scorer import QualityScorer, QualityReport
from src.normalize.schema_mapper import SchemaMapper

# --- Mock FX Provider ---
class MockFXProvider:
    def get_rate(self, from_c, to_c, date_obj):
        if from_c == "EUR" and to_c == "USD":
            return 1.10
        raise ValueError("Rate not found")

# --- Currency Tests ---

def test_currency_conversion_success():
    converter = CurrencyConverter(MockFXProvider())
    
    # 1. USD -> USD
    val, curr, rate = converter.convert_to_usd(100.0, "USD", date(2025, 1, 1))
    assert val == 100.0
    assert curr == "USD"
    assert rate == 1.0
    
    # 2. EUR -> USD
    val, curr, rate = converter.convert_to_usd(100.0, "EUR", date(2025, 1, 1))
    assert val == pytest.approx(110.0)
    assert curr == "USD"
    assert rate == 1.10

def test_currency_conversion_fallback():
    converter = CurrencyConverter(MockFXProvider())
    
    # 3. JPY -> USD (fails)
    val, curr, rate = converter.convert_to_usd(1000.0, "JPY", date(2025, 1, 1))
    assert val == 1000.0
    assert curr == "JPY"
    assert rate is None
    
# --- Quality Tests ---

def test_quality_scorer_complete():
    scorer = QualityScorer()
    
    # Mock object with dictionary access behavior via attributes
    class MockSnapshot:
        total_revenue = 100
        market_cap_t0 = 500
        shares_outstanding = 10
        ebitda = 50
        free_cash_flow = 20
        total_debt = 10
        net_income = 10
        fx_rate_to_usd = 1.0
        gross_profit = 80
        
    report = scorer.compute_score(MockSnapshot())
    assert report.score == 1.0
    assert len(report.warnings) == 0

def test_quality_scorer_missing():
    scorer = QualityScorer()
    
    class MockSnapshot:
        total_revenue = None # Critical missing
        market_cap_t0 = 500
        shares_outstanding = 10
        ebitda = None # Important missing
        free_cash_flow = 20
        total_debt = 10
        net_income = 10
        fx_rate_to_usd = 1.0
        gross_profit = 80
        
    report = scorer.compute_score(MockSnapshot())
    # 1.0 - 0.2 (crit) - 0.1 (imp) = 0.7
    assert report.score == pytest.approx(0.7)
    assert len(report.warnings) == 2

def test_quality_scorer_inconsistent():
    scorer = QualityScorer()
    
    class MockSnapshot:
        total_revenue = 100
        market_cap_t0 = 500
        shares_outstanding = 10
        ebitda = 50
        free_cash_flow = 20
        total_debt = 10
        net_income = 10
        fx_rate_to_usd = 1.0
        gross_profit = 200 # > Revenue!
        
    report = scorer.compute_score(MockSnapshot())
    assert report.score == pytest.approx(0.9)
    assert "inconsistent" in report.warnings[0]

# --- Mapper Tests ---

def test_schema_mapper():
    mapper = SchemaMapper()
    info = {"marketCap": 1000, "beta": 1.2, "currency": "USD"}
    financials = {"Total Revenue": 500, "netIncome": 50}
    
    res = mapper.map_yfinance(info, financials)
    
    assert res["market_cap_t0"] == 1000
    assert res["beta"] == 1.2
    assert res["total_revenue"] == 500
    assert res["net_income"] == 50
    assert "gross_profit" not in res
