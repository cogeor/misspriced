from sqlalchemy import Column, String, Integer, Date, DateTime, JSON, ForeignKey, func, Index, text, Text, ForeignKeyConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from .base import Base

class DataVersion(Base):
    __tablename__ = "data_versions"
    
    version_id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    version_name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    
    # What's included
    ticker_count = Column(Integer, nullable=False)
    snapshot_count = Column(Integer, nullable=False)
    date_range_start = Column(Date, nullable=False)
    date_range_end = Column(Date, nullable=False)
    
    # Integrity
    data_hash = Column(String(64), nullable=False)
    filter_config = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    created_by = Column(String(100))
    notes = Column(Text)


class DataVersionSnapshot(Base):
    __tablename__ = "data_version_snapshots"
    
    version_id = Column(UUID(as_uuid=True), ForeignKey("data_versions.version_id"), primary_key=True)
    ticker = Column(String(20), primary_key=True)
    snapshot_timestamp = Column(DateTime, primary_key=True)
    
    # Foreign key for composite primary key of financial_snapshots
    __table_args__ = (
        ForeignKeyConstraint(
            ['ticker', 'snapshot_timestamp'], 
            ['financial_snapshots.ticker', 'financial_snapshots.snapshot_timestamp']
        ),
    )
