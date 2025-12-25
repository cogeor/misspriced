
from typing import List, Tuple, Any
from pydantic import BaseModel

class QualityReport(BaseModel):
    score: float
    warnings: List[str]

class QualityScorer:
    """
    Compute data quality score (0.0 - 1.0) for a financial snapshot.
    Higher score = more complete/reliable data.
    """
    
    CRITICAL_FIELDS = ["total_revenue", "market_cap_t0", "shares_outstanding"]
    IMPORTANT_FIELDS = ["ebitda", "free_cash_flow", "total_debt", "net_income"]
    
    def compute_score(self, snapshot: Any) -> QualityReport:
        """
        Compute score for any object with financial attributes (CanonicalSnapshot or FinancialSnapshot).
        """
        score = 1.0
        warnings = []
        
        # Critical fields (high penalty)
        for field in self.CRITICAL_FIELDS:
            if not self._has_value(snapshot, field):
                score -= 0.2
                warnings.append(f"Missing critical field: {field}")
        
        # Important fields (medium penalty)
        for field in self.IMPORTANT_FIELDS:
            if not self._has_value(snapshot, field):
                score -= 0.1
                warnings.append(f"Missing important field: {field}")
        
        # Currency conversion penalty
        # Check if stored_currency is different from original but no rate
        # Or simplistic check if we failed to convert
        fx_rate = getattr(snapshot, "fx_rate_to_usd", None)
        orig_curr = getattr(snapshot, "original_currency", "USD")
        
        if fx_rate is None and orig_curr != "USD":
            score -= 0.05
            warnings.append("FX conversion failed, values in local currency")
        
        # Validation checks
        
        # Gross Profit > Revenue
        # Need to handle if either is None
        rev = getattr(snapshot, "total_revenue", None)
        gp = getattr(snapshot, "gross_profit", None)
        
        if rev is not None and gp is not None and gp > rev:
            score -= 0.1
            warnings.append("Gross profit > revenue (inconsistent)")
            
        # Shares <= 0
        shares = getattr(snapshot, "shares_outstanding", None)
        if shares is not None and shares <= 0:
            score -= 0.2
            warnings.append(f"Invalid shares outstanding: {shares}")
            
        return QualityReport(score=max(0.0, score), warnings=warnings)

    def _has_value(self, obj: Any, field: str) -> bool:
        val = getattr(obj, field, None)
        return val is not None
