from typing import List, Optional
from pathlib import Path
import pandas as pd
from datetime import date, datetime
from tqdm import tqdm

from src.providers.base import FinancialDataProvider
from src.providers.fx_rates import FXRateProvider
from src.db.repositories.snapshot_repo import SnapshotRepository
from src.db.repositories.ticker_repo import TickerRepository
from src.db.repositories.index_repo import IndexMembershipRepository
from src.db.models.ticker import Ticker
from .config import IngestionConfig
from .report import IngestionReport, IndexMembershipReport
from .snapshot_builder import SnapshotBuilder

class IngestionService:
    def __init__(
        self,
        provider: FinancialDataProvider,
        fx_provider: FXRateProvider,
        snapshot_repo: SnapshotRepository,
        ticker_repo: TickerRepository,
        index_membership_repo: Optional[IndexMembershipRepository] = None,
        config: IngestionConfig = IngestionConfig(),
    ):
        self.provider = provider
        self.fx_provider = fx_provider
        self.snapshot_repo = snapshot_repo
        self.ticker_repo = ticker_repo
        self.index_membership_repo = index_membership_repo
        self.config = config
        self.builder = SnapshotBuilder(fx_provider)
        
    async def ingest_tickers(self, tickers: List[str]) -> IngestionReport:
        report = IngestionReport()
        
        # Use tqdm for progress bar
        iterator = tqdm(tickers, desc="Ingesting snapshots")
        for ticker in iterator:
            report.add_attempt(ticker)
            try:
                # 1. Fetch Company Info
                info = self.provider.get_company_info(ticker)
                
                # Validation: If critical info is missing, assume delisted/invalid
                # "quoteType" is usually a good indicator, or just "longName"
                if not info or not info.get("longName"):
                    # Discard and remove from Source of Truth
                    self.ticker_repo.delete(ticker)
                    report.add_failure(ticker, "Invalid/Missing Data - Removed from DB")
                    continue

                # Upsert Ticker (defer commit)
                ticker_obj = Ticker(
                    ticker=ticker,
                    company_name=info.get("longName"),
                    original_currency=info.get("currency", "USD"),
                    sector=info.get("sector"),
                    industry=info.get("industry"),
                    exchange=info.get("exchange"),
                    country=info.get("country"),
                    # ipo_year not always in info, skip for now or get from summaryDetail?
                )
                self.ticker_repo.upsert(ticker_obj, commit=False)
                
                # 2. Fetch Financials
                inc = self.provider.get_income_statement(ticker)
                bs = self.provider.get_balance_sheet(ticker)
                cf = self.provider.get_cashflow_statement(ticker)
                
                # 3. Iterate over dates (available in income stmt usually)
                # Convert dict of dicts to dates
                dates = sorted([d for d in inc.keys() if isinstance(d, (date, pd.Timestamp))], reverse=True)
                
                count = 0
                for d in dates:
                    # Resolve data for this date
                    d_inc = inc.get(d, {})
                    d_bs = bs.get(d, {})
                    d_cf = cf.get(d, {})
                    
                    # 4. Fetch Price Context
                    # Note: converting pd.Timestamp to date
                    d_date = d.date() if hasattr(d, "date") else d
                    price_ctx = self.provider.get_price_window(ticker, d_date)
                    
                    # 5. Build Snapshot
                    snapshot = self.builder.build_snapshot(
                        ticker, d_date, d_inc, d_bs, d_cf, info, price_ctx
                    )
                    
                    # 6. Save (defer commit)
                    self.snapshot_repo.upsert(snapshot, commit=False)
                    count += 1
                
                report.add_success(ticker, count)
                
            except Exception as e:
                report.add_failure(ticker, str(e))
        
        # Batch Commit
        # We share the session across repos (initialized in seed/populate script)
        # So one commit saves all pending adds/updates.
        self.ticker_repo.session.commit()
                
        return report

    async def ingest_from_file(self, file_path: Path) -> IngestionReport:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        tickers = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                tickers.append(line.split("#")[0].strip())
                
        return await self.ingest_tickers(tickers)

    async def ingest_index_memberships(
        self,
        index_id: str,
        as_of_time: Optional[datetime] = None,
        source: Optional[str] = None,
    ) -> "IndexMembershipReport":
        """
        Fetch and store current members of an index.

        Args:
            index_id: Index identifier (e.g., "SP500", "NASDAQ100")
            as_of_time: Timestamp for the membership snapshot (default: now)
            source: Data source identifier

        Returns:
            Report with ticker count and any errors
        """
        if not self.index_membership_repo:
            raise RuntimeError(
                "IndexMembershipRepository not configured. "
                "Pass index_membership_repo to IngestionService."
            )

        as_of_time = as_of_time or datetime.now()

        try:
            # Fetch constituents from provider
            # Note: This assumes provider has get_index_constituents method
            tickers = self.provider.get_index_constituents(index_id)

            # Store to index_memberships table
            count = self.index_membership_repo.upsert_memberships(
                index_id=index_id,
                as_of_time=as_of_time,
                tickers=tickers,
                source=source or "yfinance",
            )

            return IndexMembershipReport(
                index_id=index_id,
                ticker_count=count,
                as_of_time=as_of_time,
            )

        except Exception as e:
            return IndexMembershipReport(
                index_id=index_id,
                ticker_count=0,
                as_of_time=as_of_time,
                error=str(e),
            )

    def resolve_universe_from_index(
        self,
        index_id: str,
        as_of_time: Optional[datetime] = None,
    ) -> List[str]:
        """
        Get tickers for an index from stored memberships.

        Used to drive downstream ingestion of financial snapshots.

        Args:
            index_id: Index to query
            as_of_time: Point in time (default: latest)

        Returns:
            List of ticker symbols
        """
        if not self.index_membership_repo:
            raise RuntimeError(
                "IndexMembershipRepository not configured. "
                "Pass index_membership_repo to IngestionService."
            )

        return self.index_membership_repo.get_members(index_id, as_of_time)
