
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.db.session import engine, SessionLocal
from src.db.models.base import Base
from src.db.repositories.ticker_repo import TickerRepository
from src.db.repositories.snapshot_repo import SnapshotRepository
from src.providers.yahoo.client import YFinanceProvider
from src.providers.fx_rates import FX_PROVIDER
from src.ingestion.service import IngestionService
from src.ingestion.config import IngestionConfig

async def seed():
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)
    
    print("Initializing services...")
    db = SessionLocal()
    
    ticker_repo = TickerRepository(db)
    snapshot_repo = SnapshotRepository(db)
    provider = YFinanceProvider()
    
    service = IngestionService(
        provider=provider,
        fx_provider=FX_PROVIDER,
        snapshot_repo=snapshot_repo,
        ticker_repo=ticker_repo,
        config=IngestionConfig()
    )
    
    tickers = ["AAPL", "MSFT"]
    print(f"Ingesting tickers: {tickers}...")
    
    report = await service.ingest_tickers(tickers)
    
    print("\nIngestion Report:")
    print(f"Successes: {report.success_count}")
    print(f"Failures: {report.failure_count}")
    
    if report.failures:
        print("Failures details:", report.failures)
    
    if report.success_count > 0:
        print("✅ Database seeding successful!")
    else:
        print("❌ Database seeding failed/empty.")

if __name__ == "__main__":
    asyncio.run(seed())
