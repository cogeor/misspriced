from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, func, Index, text
from sqlalchemy.dialects.postgresql import UUID
from .base import Base

class RawPayload(Base):
    __tablename__ = "raw_payloads"
    
    payload_id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    ticker = Column(String(20), nullable=False)
    snapshot_timestamp = Column(DateTime, nullable=False)
    provider = Column(String(50), nullable=False)
    endpoint = Column(String(100), nullable=False)
    payload_body = Column(JSON, nullable=False)
    fetched_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index('idx_payloads_ticker_ts', 'ticker', 'snapshot_timestamp'),
    )
