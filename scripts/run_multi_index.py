
import sys
import os
import asyncio
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.session import SessionLocal
from src.db.models import Ticker, Index, IndexMembership, ValuationResult
from src.db.repositories.ticker_repo import TickerRepository
from src.db.repositories.snapshot_repo import SnapshotRepository
from src.db.repositories.index_repo import IndexRepository, IndexMembershipRepository
from src.providers.yahoo.client import YFinanceProvider, INDEX_SOURCES
from src.providers.fx_rates import FX_PROVIDER
from src.ingestion.service import IngestionService
from src.ingestion.config import IngestionConfig
from datetime import datetime

import subprocess

INDICES_TO_TRACK = [
    "SP500", "NASDAQ100", "DAX", "FTSE100", "HSI",
    "SP400", "SP600", "Russell1000",
    "EuroStoxx50", "CAC40", "SMI", "Nifty50", "SSE50"
]

async def ingest_missing_tickers(tickers: list):
    """Ingest missing tickers and their snapshots."""
    session = SessionLocal()
    ticker_repo = TickerRepository(session)
    snapshot_repo = SnapshotRepository(session)
    provider = YFinanceProvider()
    
    # 1. Check what's missing
    existing_tickers = {t.ticker for t in session.query(Ticker.ticker).all()}
    missing = [t for t in tickers if t not in existing_tickers]
    
    print(f"Total Constituents: {len(tickers)}")
    print(f"Existing in DB: {len(existing_tickers)}")
    print(f"Missing: {len(missing)}")
    
    if not missing:
        print("✅ No missing tickers.")
        session.close()
        return

    print(f"Ingesting {len(missing)} missing tickers...")
    
    # Configure Ingestion
    service = IngestionService(
        provider=provider,
        fx_provider=FX_PROVIDER,
        snapshot_repo=snapshot_repo,
        ticker_repo=ticker_repo,
        config=IngestionConfig()
    )
    
    report = await service.ingest_tickers(missing)
    
    print(f"Ingestion Report: Success={report.success_count}, Failures={report.failure_count}")
    if report.failures:
        print(f"Failures: {list(report.failures.keys())[:5]}...")
        
    session.close()

def update_index_memberships(index_symbol: str, constituents: list):
    session = SessionLocal()
    try:
        repo = IndexMembershipRepository(session)
        idx_repo = IndexRepository(session)
        
        # Ensure Index exists
        idx = Index(
            index_id=index_symbol,
            name=index_symbol, 
            description=f"Auto-generated for {index_symbol}",
            weighting_scheme="equal",
            base_value=100.0
        )
        idx_repo.upsert(idx)
        
        valid_tickers = {t.ticker for t in session.query(Ticker.ticker).filter(Ticker.ticker.in_(constituents)).all()}
        final_list = [t for t in constituents if t in valid_tickers]
        
        count = repo.upsert_memberships(
            index_id=index_symbol,
            as_of_time=datetime.now(),
            tickers=final_list,
            source="wikipedia_auto"
        )
        print(f"✅ Updated {count} memberships for {index_symbol}")
        
    finally:
        session.close()

def analyze_overpricing(index_symbol: str, official_count: int = 0):
    session = SessionLocal()
    try:
        query = session.query(
            ValuationResult.ticker, 
            ValuationResult.relative_error,
            ValuationResult.actual_mcap,
            ValuationResult.predicted_mcap_mean
        ).join(
            IndexMembership, 
            IndexMembership.ticker == ValuationResult.ticker
        ).filter(
            IndexMembership.index_id == index_symbol
            # We assume active membership if recent
        )
        
        results = pd.read_sql(query.statement, session.bind)
        
        if results.empty:
            print(f"⚠️ No valuation results for {index_symbol}")
            return None
            
        results = results.drop_duplicates(subset=['ticker'])
        covered_count = len(results)
        
        # Aggregate Calculation (Cap-Weighted Proxy)
        total_actual = results['actual_mcap'].sum()
        total_predicted = results['predicted_mcap_mean'].sum()
        
        if total_actual == 0:
            print(f"⚠️ Total actual market cap is 0 for {index_symbol}")
            return None

        index_mispricing = (total_predicted - total_actual) / total_actual
        
        print(f"\n📊 Index Analysis: {index_symbol}")
        print(f"   Constituents (Official): {official_count}")
        print(f"   Constituents (Covered):  {covered_count} ({covered_count/official_count:.1%} coverage)")
        print(f"   Total Actual Mcap:       ${total_actual/1e9:,.2f}B")
        print(f"   Total Predicted Mcap:    ${total_predicted/1e9:,.2f}B")
        print(f"   Index Mispricing (Cap-W):   {index_mispricing:+.2%}")
        
        status = "UNDERPRICED" if index_mispricing > 0 else "OVERPRICED"
        print(f"   CONCLUSION: Index is {status} by approx {abs(index_mispricing):.2%}")
        
        return {
            "Index": index_symbol,
            "Mispricing": index_mispricing,
            "Count": covered_count,
            "OfficialCount": official_count,
            "Status": status,
            "TotalActual": total_actual,
            "TotalPredicted": total_predicted
        }

    finally:
        session.close()

async def main():
    print(f"\nAvailable Indices in Provider: {list(INDEX_SOURCES.keys())}")
    
    provider = YFinanceProvider()
    all_constituents = []
    index_counts = {}
    
    for idx in INDICES_TO_TRACK:
        print(f"\n--- Processing Index: {idx} ---")
        try:
            tickers = provider.get_index_constituents(idx)
            count = len(tickers)
            print(f"Fetched {count} tickers from source.")
            index_counts[idx] = count
            
            # 1. Ingest Missing
            await ingest_missing_tickers(tickers)
            
            # 2. Update Memberships
            update_index_memberships(idx, tickers)
            
            all_constituents.extend(tickers)
            
        except Exception as e:
            print(f"❌ Failed to process {idx}: {e}")
            index_counts[idx] = 0
            
    # 3. Rerun Prediction Pipeline
    print("\n--- Running Valuation Pipeline ---")
    cmd = [sys.executable, "scripts/step3_full_pipeline.py"]
    proc = subprocess.run(cmd, capture_output=False)
    
    if proc.returncode != 0:
        print("❌ Valuation pipeline failed!")
        return
        
    # 4. Analyze Overpricing
    print("\n--- Final Analysis ---")
    results_list = []
    for idx in INDICES_TO_TRACK:
        official = index_counts.get(idx, 0)
        res = analyze_overpricing(idx, official)
        if res:
            results_list.append(res)
            
    import json
    with open("index_analysis.json", "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"\n✅ Analysis saved to index_analysis.json")

if __name__ == "__main__":
    asyncio.run(main())
