"""Repository for Index and IndexMembership operations."""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_

from src.db.models.index import Index, IndexMembership
from .base import BaseRepository


class IndexRepository(BaseRepository[Index]):
    """Repository for Index CRUD operations."""

    def __init__(self, session: Session):
        super().__init__(session, Index)

    def get_index(self, index_id: str) -> Optional[Index]:
        """Get index definition by ID."""
        return self.session.query(Index).filter(Index.index_id == index_id).first()

    def upsert(self, index: Index, commit: bool = True) -> Index:
        """Create or update an index definition."""
        existing = self.get_index(index.index_id)
        if existing:
            existing.name = index.name
            existing.description = index.description
            existing.weighting_scheme = index.weighting_scheme
            existing.base_value = index.base_value
        else:
            self.session.add(index)

        if commit:
            self.session.commit()
            self.session.refresh(existing or index)

        return existing or index

    def get_all_indices(self) -> List[Index]:
        """Get all index definitions."""
        return self.session.query(Index).all()


class IndexMembershipRepository:
    """Repository for IndexMembership operations."""

    def __init__(self, session: Session):
        self.session = session

    def get_members(
        self,
        index_id: str,
        as_of_time: Optional[datetime] = None,
    ) -> List[str]:
        """
        Get tickers that are members of an index at a specific time.

        Args:
            index_id: The index to query
            as_of_time: Point in time (default: latest available)

        Returns:
            List of ticker symbols
        """
        query = self.session.query(IndexMembership.ticker).filter(
            IndexMembership.index_id == index_id,
            IndexMembership.is_member == True,
        )

        if as_of_time:
            # Get membership at or before the specified time
            query = query.filter(IndexMembership.as_of_time <= as_of_time)

        # Get most recent membership snapshot
        # Group by ticker and get the latest as_of_time for each
        subquery = (
            self.session.query(
                IndexMembership.ticker,
                IndexMembership.is_member,
            )
            .filter(
                IndexMembership.index_id == index_id,
            )
            .order_by(IndexMembership.as_of_time.desc())
        )

        if as_of_time:
            subquery = subquery.filter(IndexMembership.as_of_time <= as_of_time)

        # Get distinct tickers where is_member is True for the latest snapshot
        results = (
            self.session.query(IndexMembership.ticker)
            .filter(
                IndexMembership.index_id == index_id,
                IndexMembership.is_member == True,
            )
            .distinct()
            .all()
        )

        return [r[0] for r in results]

    def upsert_memberships(
        self,
        index_id: str,
        as_of_time: datetime,
        tickers: List[str],
        source: Optional[str] = None,
        commit: bool = True,
    ) -> int:
        """
        Store a membership snapshot for an index.

        Args:
            index_id: The index ID
            as_of_time: Snapshot timestamp
            tickers: List of member tickers
            source: Data source/provider
            commit: Whether to commit the transaction

        Returns:
            Number of memberships stored
        """
        count = 0
        for ticker in tickers:
            membership = IndexMembership(
                index_id=index_id,
                as_of_time=as_of_time,
                ticker=ticker,
                is_member=True,
                source=source,
            )
            self.session.merge(membership)
            count += 1

        if commit:
            self.session.commit()

        return count

    def remove_member(
        self,
        index_id: str,
        as_of_time: datetime,
        ticker: str,
        source: Optional[str] = None,
        commit: bool = True,
    ) -> None:
        """
        Record removal of a ticker from an index.

        Instead of deleting, we set is_member=False for audit trail.
        """
        membership = IndexMembership(
            index_id=index_id,
            as_of_time=as_of_time,
            ticker=ticker,
            is_member=False,
            source=source,
        )
        self.session.merge(membership)

        if commit:
            self.session.commit()
