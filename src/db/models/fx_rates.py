from sqlalchemy import Column, String, Numeric, Date
from .base import Base

class FXRate(Base):
    __tablename__ = "fx_rates"
    
    rate_date = Column(Date, primary_key=True)
    from_currency = Column(String(10), primary_key=True)
    to_currency = Column(String(10), primary_key=True, default="USD")
    rate = Column(Numeric, nullable=False)
