import httpx
from datetime import date
from typing import Optional, Dict

class FXRateProvider:
    """Abstract base for FX providers if needed, or simple direct class."""
    def get_rate(self, from_curr: str, to_curr: str, date_obj: date) -> float:
        raise NotImplementedError

class TheRatesAPIProvider(FXRateProvider):
    """
    FX rate provider using theratesapi.com
    Free tier: 1000 requests/month
    """
    
    BASE_URL = "https://api.theratesapi.com/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.cache: Dict[str, float] = {}
    
    def get_rate(
        self, 
        from_currency: str, 
        to_currency: str, 
        as_of_date: date
    ) -> float:
        """
        Get exchange rate for a specific date.
        """
        if from_currency == to_currency:
            return 1.0

        cache_key = f"{from_currency}_{to_currency}_{as_of_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # API call
        url = f"{self.BASE_URL}/{as_of_date.isoformat()}"
        params = {
            "base": from_currency,
            "symbols": to_currency,
        }
        if self.api_key:
            params["access_key"] = self.api_key
        
        try:
            response = httpx.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            rate = float(data["rates"][to_currency])
            
            self.cache[cache_key] = rate
            return rate
        except Exception:
            # Fallback or error handling
            raise ValueError(f"Could not fetch rate for {from_currency}->{to_currency} on {as_of_date}")

# === ACTIVE FX PROVIDER ===
FX_PROVIDER = TheRatesAPIProvider()
