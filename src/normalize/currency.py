
from typing import Optional, Tuple
from datetime import date
from src.providers.fx_rates import FXRateProvider

class CurrencyConverter:
    """
    Convert monetary values to USD using historical FX rates.
    Falls back to original currency if rate unavailable.
    """
    
    def __init__(self, fx_provider: FXRateProvider):
        self.fx_provider = fx_provider
    
    def convert_to_usd(
        self, 
        value: Optional[float], 
        from_currency: str, 
        as_of_date: date
    ) -> Tuple[Optional[float], str, Optional[float]]:
        """
        Attempt to convert value to USD with fallback.
        
        Returns:
            (converted_value, stored_currency, fx_rate_to_usd)
            
        If conversion fails or value is None:
            (original_value, original_currency, None)
        """
        if value is None:
            return None, from_currency, None

        if from_currency == "USD":
            return value, "USD", 1.0
        
        try:
            rate = self.fx_provider.get_rate(from_currency, "USD", as_of_date)
            return value * rate, "USD", rate
        except Exception:
            # Fallback: keep original currency
            # We catch generic Exception because provider might raise different errors
            # Ideally we catch specific FXRateUnavailable
            return value, from_currency, None
