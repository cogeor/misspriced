from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import date


class FinancialDataProvider(ABC):
    """Abstract base for all financial data providers."""

    @abstractmethod
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """Fetch company metadata (name, sector, industry, exchange)."""
        pass

    @abstractmethod
    def get_income_statement(
        self, ticker: str, frequency: str = "quarterly"
    ) -> Dict[str, Any]:
        """Fetch income statement history."""
        pass

    @abstractmethod
    def get_balance_sheet(
        self, ticker: str, frequency: str = "quarterly"
    ) -> Dict[str, Any]:
        """Fetch balance sheet history."""
        pass

    @abstractmethod
    def get_cashflow_statement(
        self, ticker: str, frequency: str = "quarterly"
    ) -> Dict[str, Any]:
        """Fetch cash flow statement history."""
        pass

    @abstractmethod
    def get_price_at_date(
        self, ticker: str, target_date: date
    ) -> Optional[float]:
        """Fetch closing price for a specific date."""
        pass

    @abstractmethod
    def get_price_window(
        self, ticker: str, target_date: date, days_before: int = 1, days_after: int = 1
    ) -> Dict[str, Optional[float]]:
        """Fetch price window around a date (t-1, t0, t+1)."""
        pass

    @abstractmethod
    def get_index_constituents(self, index_symbol: str) -> List[str]:
        """
        Fetch the list of tickers that make up an index.

        Args:
            index_symbol: Index identifier (e.g., "^GSPC" for S&P 500)

        Returns:
            List of ticker symbols in the index
        """
        pass
