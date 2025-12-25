from sqlalchemy import Column, String, DateTime, func, Integer, Numeric
from .base import Base

class Ticker(Base):
    __tablename__ = "tickers"
    
    ticker = Column(String(20), primary_key=True)
    company_name = Column(String(255))
    original_currency = Column(String(10), nullable=True) # made nullable as CSV ingestion might not have it immediately
    exchange = Column(String(50))
    sector = Column(String(100))
    industry = Column(String(100))
    country = Column(String(100))
    ipo_year = Column(Integer)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
