"""Index and IndexMembership SQLAlchemy models."""

from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    Numeric,
    Text,
    ForeignKey,
    Index as SQLIndex,
    func,
)
from .base import Base


class Index(Base):
    """Index definition - represents an index like S&P 500, NASDAQ 100, etc."""

    __tablename__ = "indices"

    index_id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    weighting_scheme = Column(
        String(20), nullable=False
    )  # 'equal' | 'market_cap' | 'custom'
    base_value = Column(Numeric, default=100.0)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class IndexMembership(Base):
    """
    Tracks which tickers are in which index at which time.

    Index constituency changes over time (companies added/removed),
    so we store snapshots of membership at specific points in time.
    """

    __tablename__ = "index_memberships"

    index_id = Column(
        String(50), ForeignKey("indices.index_id"), primary_key=True, nullable=False
    )
    as_of_time = Column(DateTime, primary_key=True, nullable=False)
    ticker = Column(
        String(20), ForeignKey("tickers.ticker"), primary_key=True, nullable=False
    )
    is_member = Column(Boolean, default=True)  # Allows explicit removals
    source = Column(String(50))  # Provider that reported membership
    ingested_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        SQLIndex("idx_membership_index_time", "index_id", "as_of_time"),
        SQLIndex("idx_membership_ticker", "ticker"),
    )
