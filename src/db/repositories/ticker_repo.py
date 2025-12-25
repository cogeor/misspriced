from typing import Optional
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import func
from .base import BaseRepository
from src.db.models.ticker import Ticker

class TickerRepository(BaseRepository[Ticker]):
    def __init__(self, session):
        super().__init__(session, Ticker)

    def get(self, ticker: str) -> Optional[Ticker]:
        return self.session.get(Ticker, ticker)
        
    def delete(self, ticker: str) -> bool:
        obj = self.get(ticker)
        if obj:
            self.session.delete(obj)
            self.session.commit()
            return True
        return False
        
    def upsert(self, ticker_obj: Ticker, commit: bool = True) -> Ticker:
        stmt = insert(Ticker).values(
            ticker=ticker_obj.ticker,
            company_name=ticker_obj.company_name,
            original_currency=ticker_obj.original_currency,
            exchange=ticker_obj.exchange,
            sector=ticker_obj.sector,
            industry=ticker_obj.industry,
            country=ticker_obj.country,
            ipo_year=ticker_obj.ipo_year,
            updated_at=func.now()
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[Ticker.ticker],
            set_={
                "company_name": stmt.excluded.company_name,
                "created_at": Ticker.created_at, # Preserve created_at
                "updated_at": func.now(),
                "original_currency": stmt.excluded.original_currency,
                "exchange": stmt.excluded.exchange,
                "sector": stmt.excluded.sector,
                "industry": stmt.excluded.industry,
                "country": stmt.excluded.country,
                "ipo_year": stmt.excluded.ipo_year,
            }
        )
        self.session.execute(stmt)
        if commit:
            self.session.commit()
        return self.get(ticker_obj.ticker)
