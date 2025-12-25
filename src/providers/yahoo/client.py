from typing import Dict, Any, Optional, List
from datetime import date, timedelta
import yfinance as yf
import pandas as pd
from ..base import FinancialDataProvider
from ..rate_limiter import YahooRateLimiter


# Mapping of common index symbols to Wikipedia table URLs
# yfinance doesn't provide direct constituent fetching, so we use Wikipedia
INDEX_SOURCES = {
    "^GSPC": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "table_index": 0,
        "ticker_column": "Symbol",
    },
    "SP500": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "table_index": 0,
        "ticker_column": "Symbol",
    },
    "^NDX": {
        "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "table_index": 4,
        "ticker_column": "Ticker",
    },
    "NASDAQ100": {
        "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "table_index": 4,
        "ticker_column": "Ticker",
    },
    # Europe
    "^GDAXI": {
        "url": "https://en.wikipedia.org/wiki/DAX",
        "table_index": 4, 
        "ticker_column": "Ticker",
        "suffix": ".DE"  # DAX components trade on Xetra
    },
    "DAX": {
        "url": "https://en.wikipedia.org/wiki/DAX",
        "table_index": 4,
        "ticker_column": "Ticker",
        "suffix": ".DE"
    },
    "^FTSE": {
        "url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "table_index": 6,
        "ticker_column": "Ticker",
        "suffix": ".L"
    },
    "FTSE100": {
        "url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "table_index": 6,
        "ticker_column": "Ticker",
        "suffix": ".L"
    },
    # Asia
    "^HSI": {
        "url": "https://en.wikipedia.org/wiki/Hang_Seng_Index",
        "table_index": 6, 
        "ticker_column": "Ticker", 
        "suffix": ".HK",
        "pad_digits": 4
    },
    "HSI": {
        "url": "https://en.wikipedia.org/wiki/Hang_Seng_Index",
        "table_index": 6,
        "ticker_column": "Ticker", 
        "suffix": ".HK",
        "pad_digits": 4
    },
    
    # Expanded US
    "SP400": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "table_index": 0,
        "ticker_column": "Symbol"
    },
    "SP600": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        "table_index": 0,
        "ticker_column": "Symbol"
    },
    "Russell1000": {
        "url": "https://en.wikipedia.org/wiki/Russell_1000_Index",
        "table_index": 3,
        "ticker_column": "Symbol"
    },
    
    # Global
    "EuroStoxx50": {
        "url": "https://en.wikipedia.org/wiki/Euro_Stoxx_50",
        "table_index": 4,
        "ticker_column": "Ticker"
    },
    "CAC40": {
        "url": "https://en.wikipedia.org/wiki/CAC_40",
        "table_index": 4,
        "ticker_column": "Ticker",
        "suffix": ".PA"
    },
    "SMI": {
        "url": "https://en.wikipedia.org/wiki/Swiss_Market_Index",
        "table_index": 2,
        "ticker_column": "Ticker",
        "suffix": ".SW"
    },
    "Nifty50": {
        "url": "https://en.wikipedia.org/wiki/NIFTY_50",
        "table_index": 1,
        "ticker_column": "Symbol",
        "suffix": ".NS"
    },
    "SSE50": {
        "url": "https://en.wikipedia.org/wiki/SSE_50_Index",
        "table_index": 1,
        "ticker_column": "Ticker symbol",
        "suffix": ".SS",
        "pad_digits": 6 # Shanghai codes are 6 digits
    }
}


class YFinanceProvider(FinancialDataProvider):
    """yfinance implementation of the provider interface."""

    def __init__(self, rate_limit_delay: float = 0.25):
        self.limiter = YahooRateLimiter(min_interval=rate_limit_delay)

    def _to_dict_safe(self, df: pd.DataFrame) -> Dict[str, Any]:
        return df.to_dict() if not df.empty else {}

    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        self.limiter.wait_if_needed()
        ticker_data = yf.Ticker(ticker)
        info = ticker_data.info
        # Cleanup
        info.pop("companyOfficers", None)
        return info

    def get_income_statement(
        self, ticker: str, frequency: str = "quarterly"
    ) -> Dict[str, Any]:
        self.limiter.wait_if_needed()
        t = yf.Ticker(ticker)
        df = t.quarterly_income_stmt if frequency == "quarterly" else t.income_stmt
        return self._to_dict_safe(df)

    def get_balance_sheet(
        self, ticker: str, frequency: str = "quarterly"
    ) -> Dict[str, Any]:
        self.limiter.wait_if_needed()
        t = yf.Ticker(ticker)
        df = t.quarterly_balance_sheet if frequency == "quarterly" else t.balance_sheet
        return self._to_dict_safe(df)

    def get_cashflow_statement(
        self, ticker: str, frequency: str = "quarterly"
    ) -> Dict[str, Any]:
        self.limiter.wait_if_needed()
        t = yf.Ticker(ticker)
        df = t.quarterly_cashflow if frequency == "quarterly" else t.cashflow
        return self._to_dict_safe(df)

    def get_price_at_date(
        self, ticker: str, target_date: date
    ) -> Optional[float]:
        self.limiter.wait_if_needed()
        t = yf.Ticker(ticker)
        try:
            # yfinance history handles dates slightly differently, use slight range
            start = target_date.strftime("%Y-%m-%d")
            end = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
            hist = t.history(start=start, end=end)
            if not hist.empty:
                return float(hist["Close"].iloc[0])
        except Exception:
            pass
        return None

    def get_price_window(
        self, ticker: str, target_date: date, days_before: int = 1, days_after: int = 1
    ) -> Dict[str, Optional[float]]:
        self.limiter.wait_if_needed()
        t = yf.Ticker(ticker)

        start_date = target_date - timedelta(days=days_before + 5)  # Buffer for weekends
        end_date = target_date + timedelta(days=days_after + 5)

        try:
            hist = t.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )
            results: Dict[str, Optional[float]] = {}

            # t0
            ts_str = target_date.strftime("%Y-%m-%d")
            if ts_str in hist.index:
                results["t0"] = float(hist.loc[ts_str]["Close"])
            else:
                results["t0"] = None

            # t-1 (previous trading day)
            prev_date = target_date - timedelta(days=days_before)
            prev_str = prev_date.strftime("%Y-%m-%d")
            results["t-1"] = (
                float(hist.loc[prev_str]["Close"]) if prev_str in hist.index else None
            )

            next_date = target_date + timedelta(days=days_after)
            next_str = next_date.strftime("%Y-%m-%d")
            results["t+1"] = (
                float(hist.loc[next_str]["Close"]) if next_str in hist.index else None
            )

            return results
        except Exception:
            return {"t0": None, "t-1": None, "t+1": None}

    def get_index_constituents(self, index_symbol: str) -> List[str]:
        """
        Fetch the list of tickers that make up an index.

        Uses Wikipedia tables since yfinance doesn't provide direct
        index constituent API.

        Args:
            index_symbol: Index identifier (e.g., "^GSPC" for S&P 500, "SP500")

        Returns:
            List of ticker symbols in the index

        Raises:
            ValueError: If index_symbol is not supported
        """
        import requests
        from io import StringIO

        if index_symbol not in INDEX_SOURCES:
            supported = ", ".join(INDEX_SOURCES.keys())
            raise ValueError(
                f"Index '{index_symbol}' not supported. "
                f"Supported indices: {supported}"
            )

        source = INDEX_SOURCES[index_symbol]

        try:
            # Fetch with User-Agent to avoid 403 Forbidden
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(source["url"], headers=headers)
            response.raise_for_status()

            # Read tables from HTML
            tables = pd.read_html(StringIO(response.text))
            df = tables[source["table_index"]]

            # Extract ticker column
            tickers = df[source["ticker_column"]].tolist()

            # Clean up tickers
            cleaned = []
            suffix = source.get("suffix", "")
            pad = source.get("pad_digits", 0)
            
            for ticker in tickers:
                # Ensure string (sometimes read as float/int if all digits)
                t_str = str(ticker).strip()
                
                # Specific cleaning for Hong Kong/China (Wikipedia often uses SEHK: or SSE: prefix)
                t_str = t_str.replace("SEHK:", "").replace("SEHK", "").replace("SSE:", "").replace("SSE", "").strip()
                # Also clean generic noise if needed
                
                # Handling Padding (e.g., numeric codes "5" -> "0005")
                if pad > 0 and t_str.isdigit():
                    t_str = t_str.zfill(pad)
                
                # Handling Suffix
                if suffix and not t_str.endswith(suffix):
                    t_str = f"{t_str}{suffix}"
                
                # For European stocks (EuroStoxx, CAC, SMI), Wikipedia uses hyphens (ADS-DE)
                # but YFinance needs dots (ADS.DE). Convert hyphen to dot.
                #  Only do this if ticker contains a hyphen followed by country code
                if '-' in t_str and len(t_str.split('-')) == 2:
                    # Check if it looks like a ticker-countrycode pattern (e.g., ADS-DE, AIR-FR)
                    parts = t_str.split('-')
                    if len(parts[1]) == 2 and parts[1].isupper():  # Country code is 2 uppercase letters
                        t_str = t_str.replace('-', '.')

                
                cleaned.append(t_str)

            return cleaned

        except Exception as e:
            raise ValueError(
                f"Failed to fetch constituents for '{index_symbol}': {e}"
            ) from e

