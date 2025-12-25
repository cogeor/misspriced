"""
Run the full Index pipeline:
1. Fetch S&P 500 constituents from Wikipedia via provider
2. Create index definition
3. Store index memberships (which tickers belong to the index)
4. Compute index values from valuation results
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.config import DATABASE_URL
from src.db.models import Ticker, FinancialSnapshot, ValuationResult, Index, IndexMembership
from src.db.repositories.index_repo import IndexRepository, IndexMembershipRepository
from src.index.service import IndexService
from src.providers.yahoo.client import YFinanceProvider


def main() -> None:
    print(f"Connecting to: {DATABASE_URL}")
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # 1. Check what's in the database
        ticker_count = session.query(Ticker).count()
        snapshot_count = session.query(FinancialSnapshot).count()
        valuation_count = session.query(ValuationResult).count()

        print(f"\n📊 Database Status:")
        print(f"   Tickers: {ticker_count}")
        print(f"   Snapshots: {snapshot_count}")
        print(f"   Valuations: {valuation_count}")

        # 2. Fetch S&P 500 constituents from Wikipedia
        print("\n📌 Fetching S&P 500 constituents from Wikipedia...")
        provider = YFinanceProvider()
        sp500_tickers = provider.get_index_constituents("SP500")
        print(f"   Fetched {len(sp500_tickers)} tickers from S&P 500")

        # 3. Check how many of these tickers are in our database
        db_tickers = set(t.ticker for t in session.query(Ticker.ticker).all())
        matched_tickers = [t for t in sp500_tickers if t in db_tickers]
        print(f"   {len(matched_tickers)} tickers found in database")

        if not matched_tickers:
            print("\n❌ No S&P 500 tickers found in database.")
            print("   You may need to ingest these tickers first.")
            return

        # 4. Create or update index definition
        print("\n📌 Setting up SP500 index definition...")
        index_repo = IndexRepository(session)

        sp500_index = Index(
            index_id="SP500",
            name="S&P 500",
            description="S&P 500 Index - constituents fetched from Wikipedia",
            weighting_scheme="equal",  # Using equal weight for simplicity
            base_value=100.0,
        )
        index_repo.upsert(sp500_index)
        print("   ✅ Index 'SP500' created/updated")

        # 5. Store index memberships (linking tickers to index)
        print("\n📌 Storing index memberships...")
        membership_repo = IndexMembershipRepository(session)

        as_of_time = datetime.now()
        count = membership_repo.upsert_memberships(
            index_id="SP500",
            as_of_time=as_of_time,
            tickers=matched_tickers,  # Only tickers we have in DB
            source="wikipedia",
        )
        print(f"   ✅ Stored {count} ticker→index memberships")

        # Show some example memberships
        print(f"\n   Sample members: {matched_tickers[:10]}...")

        # 6. If we have valuations, compute the index
        if valuation_count > 0:
            print("\n📌 Computing index values...")
            index_service = IndexService(session)

            try:
                result = index_service.compute_index(
                    index_id="SP500",
                    as_of_time=as_of_time,
                )

                print("\n" + "=" * 60)
                print("📈 INDEX RESULTS")
                print("=" * 60)
                print(f"   Index ID:              {result.index_id}")
                print(f"   As of:                 {result.as_of_time}")
                print(f"   Weighting:             {result.weighting_scheme}")
                print(f"   Tickers (total):       {result.n_tickers}")
                print(f"   Tickers (with vals):   {result.n_tickers_with_valuation}")
                print("-" * 60)
                print(f"   Actual Index:          ${result.actual_index:,.2f}M")
                print(f"   Estimated Index:       ${result.estimated_index:,.2f}M")
                print(f"   Estimated Std:         ${result.estimated_index_std:,.2f}M")
                print(f"   Relative Error:        {result.index_relative_error:+.2%}")
                print("=" * 60)

                # Show interpretation
                if result.index_relative_error > 0:
                    print("\n💡 Interpretation: Model predicts index is UNDERVALUED")
                    print(f"   Expected upside: {result.index_relative_error:.2%}")
                elif result.index_relative_error < 0:
                    print("\n💡 Interpretation: Model predicts index is OVERVALUED")
                    print(f"   Expected downside: {result.index_relative_error:.2%}")
                else:
                    print("\n💡 Interpretation: Index is fairly valued")

            except ValueError as e:
                print(f"   ⚠️ Could not compute index: {e}")
        else:
            print("\n⚠️ No valuations in database. Run valuation pipeline first.")
            print("   The index has been set up with memberships.")
            print("   After running valuations, you can compute the index.")

    finally:
        session.close()
        print("\n✅ Done!")


if __name__ == "__main__":
    main()

