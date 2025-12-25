from datetime import datetime, date
from typing import Dict, Any, Optional
import pandas as pd
from src.db.models.snapshot import FinancialSnapshot
from src.providers.fx_rates import FXRateProvider

class SnapshotBuilder:
    """
    Builds FinancialSnapshot objects from raw provider data.
    Handles currency conversion and normalization.
    """
    
    def __init__(self, fx_provider: FXRateProvider):
        self.fx_provider = fx_provider
        
    def build_snapshot(
        self,
        ticker: str,
        statement_date: date,
        raw_income: Dict[str, Any],
        raw_balance: Dict[str, Any],
        raw_cashflow: Dict[str, Any],
        company_info: Dict[str, Any],
        price_context: Dict[str, Optional[float]],
    ) -> FinancialSnapshot:
        """
        Construct a FinancialSnapshot from raw data parts.
        """
        # currency resolution
        original_curr = company_info.get("currency", "USD")
        
        # Helper to get converted value
        # Note: In real impl, we'd map fields carefully. 
        # Here we do a simplified mapping assumption
        # that raw_x dicts have keys matching model fields or close to it.
        # But yfinance keys are like "Total Revenue", "Gross Profit".
        # We need a mapper.
        
        # Simplified for prototype:
        # Assuming we can access data by standard keys or map them.
        
        # Let's try to convert revenue as example
        revenue = self._get_val(raw_income, "Total Revenue") or self._get_val(raw_income, "totalRevenue")
        converted_rev, stored_curr, fx_rate = self._convert(revenue, original_curr, statement_date)
        
        # Create snapshot object
        snapshot = FinancialSnapshot(
            ticker=ticker,
            snapshot_timestamp=datetime.combine(statement_date, datetime.min.time()),
            period_end_date=statement_date,
            original_currency=original_curr,
            stored_currency=stored_curr,
            fx_rate_to_usd=fx_rate,
            
            total_revenue=converted_rev,
            market_cap_t0=company_info.get("marketCap"), # Use current if historical not avail, or logic
            
            # ... map other fields ...
            price_t0=price_context.get("t0"),
            price_t_minus_1=price_context.get("t-1"),
            price_t_plus_1=price_context.get("t+1"),
            
            beta=company_info.get("beta"),
        )
        return snapshot
        
    def _get_val(self, data: Dict, key: str) -> Optional[float]:
        val = data.get(key)
        if pd.isna(val) or val is None:
            return None
        return float(val)

    def _convert(self, value: Optional[float], currency: str, date_obj: date):
        if value is None:
            return None, currency, None
            
        if currency == "USD":
            return value, "USD", 1.0
            
        try:
            rate = self.fx_provider.get_rate(currency, "USD", date_obj)
            return value * rate, "USD", rate
        except Exception:
            return value, currency, None
