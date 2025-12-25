from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class IngestionReport(BaseModel):
    """Track ingestion progress and errors."""

    attempted: List[str] = Field(default_factory=list)
    successes: Dict[str, int] = Field(default_factory=dict)
    failures: Dict[str, str] = Field(default_factory=dict)

    @property
    def success_count(self) -> int:
        return len(self.successes)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    def add_attempt(self, ticker: str) -> None:
        self.attempted.append(ticker)

    def add_success(self, ticker: str, snapshot_count: int) -> None:
        self.successes[ticker] = snapshot_count

    def add_failure(self, ticker: str, error: str) -> None:
        self.failures[ticker] = error


class IndexMembershipReport(BaseModel):
    """Report for index membership ingestion."""

    index_id: str
    ticker_count: int
    as_of_time: datetime
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None
