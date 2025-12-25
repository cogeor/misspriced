"""
Run valuation on database snapshots and store results.
This will populate the valuation_results table needed for index computation.
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.config import DATABASE_URL
from src.db.models import FinancialSnapshot, ValuationResult
from src.valuation.service import ValuationService
from src.valuation.config import baseline_model
from src.filters.composer import DatasetFilter
from src.filters.predicates import market_cap_min


def main() -> None:
    print(f"Connecting to: {DATABASE_URL}")
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Check snapshots
        snapshot_count = session.query(FinancialSnapshot).count()
        valuation_count = session.query(ValuationResult).count()

        print(f"\n📊 Database Status:")
        print(f"   Snapshots: {snapshot_count}")
        print(f"   Existing Valuations: {valuation_count}")

        if snapshot_count == 0:
            print("\n❌ No snapshots in database. Run ingestion first.")
            return

        # Configure model
        print("\n📌 Configuring valuation model...")
        config = baseline_model()
        # Reduce experiments for faster execution (still statistically valid)
        config.n_experiments = 5
        config.outer_cv_splits = 3
        config.inner_cv_splits = 3
        print(f"   Model: {config.name} v{config.version}")
        print(f"   Experiments: {config.n_experiments}")

        # Create filters (minimum market cap to filter out tiny companies)
        print("\n📌 Setting up filters...")
        filters = DatasetFilter().add(market_cap_min(100))  # Min $100M market cap
        print("   Filter: market_cap >= $100M")

        # Run valuation
        print("\n📌 Running valuation (this may take a while)...")
        service = ValuationService(session)
        results = service.run_valuation(config, filters)

        print(f"\n✅ Valuation complete!")
        print(f"   Tickers processed: {len(results['tickers'])}")
        print(f"   Mean prediction: ${results['mean'].mean():,.2f}M")
        print(f"   Mean mispricing: {results['mispricing'].mean():.2%}")

        # Store results to database
        print("\n📌 Storing valuation results to database...")
        model_version = f"{config.name}_v{config.version}"
        stored_count = 0

        for i in range(len(results['tickers'])):
            ticker = results['tickers'][i]
            snapshot_ts = results['snapshot_timestamps'][i]
            
            valuation = ValuationResult(
                ticker=ticker,
                snapshot_timestamp=snapshot_ts,
                model_version=model_version,
                predicted_mcap_mean=float(results['mean'][i]) * 1e6,  # Convert back to raw
                predicted_mcap_std=float(results['std'][i]) * 1e6,
                actual_mcap=float(results['actual_mcap_mm'][i]) * 1e6,
                relative_error=float(results['mispricing'][i]),
                relative_std=float(results['std'][i] / max(results['actual_mcap_mm'][i], 1e-6)),
                n_experiments=config.n_experiments,
            )
            session.merge(valuation)
            stored_count += 1

        session.commit()
        print(f"   ✅ Stored {stored_count} valuation results")

        # Show sample results
        print("\n📊 Sample Results:")
        import pandas as pd
        df = pd.DataFrame({
            "Ticker": results['tickers'][:10],
            "Actual ($M)": results['actual_mcap_mm'][:10],
            "Predicted ($M)": results['mean'][:10],
            "Mispricing": results['mispricing'][:10],
        })
        print(df.to_string(index=False))

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        session.close()
        print("\n✅ Done!")


if __name__ == "__main__":
    main()
