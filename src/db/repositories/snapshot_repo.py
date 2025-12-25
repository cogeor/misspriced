from typing import Optional, List
from datetime import datetime
from sqlalchemy import select, and_, func
from sqlalchemy.dialects.postgresql import insert
from .base import BaseRepository
from src.db.models.snapshot import FinancialSnapshot

class SnapshotRepository(BaseRepository[FinancialSnapshot]):
    def __init__(self, session):
        super().__init__(session, FinancialSnapshot)
        
    def get_snapshot(self, ticker: str, timestamp: datetime) -> Optional[FinancialSnapshot]:
        return self.session.query(FinancialSnapshot).filter_by(
            ticker=ticker, snapshot_timestamp=timestamp
        ).first()

    def get_by_ticker(self, ticker: str, limit: int = 100) -> List[FinancialSnapshot]:
        return self.session.query(FinancialSnapshot).filter_by(ticker=ticker)\
            .order_by(FinancialSnapshot.snapshot_timestamp.desc())\
            .limit(limit).all()

    def upsert(self, snapshot: FinancialSnapshot, commit: bool = True) -> FinancialSnapshot:
        """Upsert financial snapshot."""
        # Convert model instance to dict, excluding internal/auto fields if needed
        # For simplicity, we assume we extract columns. 
        # A robust serialization method on the model would be better.
        
        # Manually constructing values dict from the object instance
        # This is a bit verbose but safe for SQLAlchemy models
        data = {}
        for column in FinancialSnapshot.__table__.columns:
            if hasattr(snapshot, column.name):
                val = getattr(snapshot, column.name)
                if val is not None: # only include non-null or handle defaults
                     data[column.name] = val
        
        # Ensure PKs are there
        if "ticker" not in data or "snapshot_timestamp" not in data:
            raise ValueError("Ticker and snapshot_timestamp are required for upsert")

        stmt = insert(FinancialSnapshot).values(**data)
        
        # Update all fields on conflict
        update_dict = {
            col.name: stmt.excluded[col.name]
            for col in FinancialSnapshot.__table__.columns
            if not col.primary_key and col.name != "created_at"
        }
        update_dict["updated_at"] = func.now()
        
        stmt = stmt.on_conflict_do_update(
            index_elements=[FinancialSnapshot.ticker, FinancialSnapshot.snapshot_timestamp],
            set_=update_dict
        )
        
        self.session.execute(stmt)
        if commit:
            self.session.commit()
        
        return self.get_snapshot(snapshot.ticker, snapshot.snapshot_timestamp)
