"""IndexService - computes index-level values from per-ticker valuations."""

import logging
import math
from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from src.db.models.index import Index
from src.db.models.valuation import ValuationResult
from src.db.repositories.index_repo import IndexRepository, IndexMembershipRepository
from .models import IndexResult, WeightingScheme
from .weights import (
    compute_equal_weights,
    compute_market_cap_weights,
    normalize_custom_weights,
)

logger = logging.getLogger(__name__)


class IndexService:
    """
    Compute index-level values from per-ticker valuations.

    This service:
    1. Pulls existing valuation results from DB for tickers in an index
    2. Computes estimated index value (weighted sum of predicted market caps) with std
    3. Computes actual index value (weighted sum of actual market caps)
    4. Returns results in-memory (no persistence for now)
    """

    def __init__(self, session: Session):
        """
        Initialize IndexService with database session.

        Args:
            session: SQLAlchemy session for database access
        """
        self.session = session
        self.index_repo = IndexRepository(session)
        self.membership_repo = IndexMembershipRepository(session)

    def compute_index(
        self,
        index_id: str,
        as_of_time: Optional[datetime] = None,
        model_version: Optional[str] = None,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> IndexResult:
        """
        Compute index values at a specific time.

        Args:
            index_id: Index to compute (e.g., "SP500")
            as_of_time: Snapshot time (default: use latest valuations)
            model_version: Filter valuations by model version
            custom_weights: Override index weighting scheme with custom weights

        Returns:
            IndexResult with actual, estimated, and uncertainty values

        Raises:
            ValueError: If index not found or no valuations available
        """
        logger.info(f"Computing index {index_id} as of {as_of_time}")

        # 1. Get index definition
        index_def = self.index_repo.get_index(index_id)
        if not index_def:
            raise ValueError(f"Index '{index_id}' not found")

        # 2. Get index members
        tickers = self.membership_repo.get_members(index_id, as_of_time)
        if not tickers:
            raise ValueError(f"No members found for index '{index_id}'")

        logger.info(f"Index {index_id} has {len(tickers)} members")

        # 3. Get valuations for members
        valuations = self._get_valuations(tickers, as_of_time, model_version)
        if not valuations:
            raise ValueError(
                f"No valuations found for index '{index_id}' members"
            )

        logger.info(
            f"Found valuations for {len(valuations)}/{len(tickers)} tickers"
        )

        # 4. Compute weights
        if custom_weights:
            weights = normalize_custom_weights(custom_weights)
            weighting_scheme = WeightingScheme.CUSTOM
        else:
            weighting_scheme = WeightingScheme(
                index_def.weighting_scheme or "equal"
            )
            weights = self._compute_weights(valuations, weighting_scheme)

        # 5. Aggregate into index values
        return self._aggregate_index(
            index_def=index_def,
            valuations=valuations,
            weights=weights,
            weighting_scheme=weighting_scheme,
            n_total_tickers=len(tickers),
            model_version=model_version,
        )

    def _get_valuations(
        self,
        tickers: List[str],
        as_of_time: Optional[datetime],
        model_version: Optional[str],
    ) -> List[ValuationResult]:
        """Fetch valuation results for the given tickers."""
        query = self.session.query(ValuationResult).filter(
            ValuationResult.ticker.in_(tickers)
        )

        if as_of_time:
            query = query.filter(
                ValuationResult.snapshot_timestamp <= as_of_time
            )

        if model_version:
            query = query.filter(ValuationResult.model_version == model_version)

        # Get most recent valuation per ticker
        # This is a simplified approach - for production, use window functions
        valuations = query.all()

        # Deduplicate: keep most recent per ticker
        latest_by_ticker: Dict[str, ValuationResult] = {}
        for v in valuations:
            existing = latest_by_ticker.get(v.ticker)
            if not existing or v.snapshot_timestamp > existing.snapshot_timestamp:
                latest_by_ticker[v.ticker] = v

        return list(latest_by_ticker.values())

    def _compute_weights(
        self,
        valuations: List[ValuationResult],
        scheme: WeightingScheme,
    ) -> Dict[str, float]:
        """Compute ticker weights based on scheme."""
        if scheme == WeightingScheme.EQUAL:
            return compute_equal_weights([v.ticker for v in valuations])
        elif scheme == WeightingScheme.MARKET_CAP:
            return compute_market_cap_weights(valuations)
        else:
            raise ValueError(f"Unknown weighting scheme: {scheme}")

    def _aggregate_index(
        self,
        index_def: Index,
        valuations: List[ValuationResult],
        weights: Dict[str, float],
        weighting_scheme: WeightingScheme,
        n_total_tickers: int,
        model_version: Optional[str],
    ) -> IndexResult:
        """Aggregate valuations into index-level metrics."""
        # Sum weighted values
        actual_sum = 0.0
        estimated_sum = 0.0
        variance_sum = 0.0

        # Determine as_of_time from valuations
        as_of_time = max(v.snapshot_timestamp for v in valuations)

        for v in valuations:
            w = weights.get(v.ticker, 0.0)
            if w == 0:
                continue

            actual_mcap = float(v.actual_mcap) if v.actual_mcap else 0
            predicted_mcap = float(v.predicted_mcap_mean) if v.predicted_mcap_mean else 0
            predicted_std = float(v.predicted_mcap_std) if v.predicted_mcap_std else 0

            actual_sum += w * actual_mcap
            estimated_sum += w * predicted_mcap
            variance_sum += (w * predicted_std) ** 2

        # Propagated uncertainty (assuming independent errors)
        estimated_std = math.sqrt(variance_sum)

        # Relative error
        if actual_sum != 0:
            relative_error = (estimated_sum - actual_sum) / actual_sum
        else:
            relative_error = 0.0

        return IndexResult(
            index_id=index_def.index_id,
            as_of_time=as_of_time,
            actual_index=actual_sum,
            estimated_index=estimated_sum,
            estimated_index_std=estimated_std,
            index_relative_error=relative_error,
            n_tickers=n_total_tickers,
            n_tickers_with_valuation=len(valuations),
            weights=weights,
            model_version=model_version,
            weighting_scheme=weighting_scheme,
        )

    def compute_index_series(
        self,
        index_id: str,
        start_time: datetime,
        end_time: datetime,
        model_version: Optional[str] = None,
    ) -> List[IndexResult]:
        """
        Compute index values over a time range.

        This iterates through available snapshots and computes
        index values at each point.

        Args:
            index_id: Index to compute
            start_time: Start of time range
            end_time: End of time range
            model_version: Filter by model version

        Returns:
            List of IndexResult, one per snapshot timestamp
        """
        # Get all unique snapshot timestamps in range
        timestamps = (
            self.session.query(ValuationResult.snapshot_timestamp)
            .filter(
                ValuationResult.snapshot_timestamp >= start_time,
                ValuationResult.snapshot_timestamp <= end_time,
            )
            .distinct()
            .order_by(ValuationResult.snapshot_timestamp)
            .all()
        )

        results = []
        for (ts,) in timestamps:
            try:
                result = self.compute_index(
                    index_id=index_id,
                    as_of_time=ts,
                    model_version=model_version,
                )
                results.append(result)
            except ValueError as e:
                logger.warning(f"Skipping timestamp {ts}: {e}")
                continue

        return results
