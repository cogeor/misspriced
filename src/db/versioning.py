import hashlib
import json
from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from src.db.models.data_version import DataVersion, DataVersionSnapshot
from src.db.models.snapshot import FinancialSnapshot

class DataVersionManager:
    """Manage data versions for reproducibility."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def _get_all_snapshots(self) -> List[FinancialSnapshot]:
        return self.session.query(FinancialSnapshot).all()
        
    def _get_filtered_snapshots(self, filter_config: dict) -> List[FinancialSnapshot]:
        # Placeholder for actual filtering logic implementation
        # Ideally this would use the filter module or dynamic query building
        return self._get_all_snapshots()

    def create_version(
        self,
        name: str,
        description: str,
        filter_config: Optional[dict] = None,
    ) -> DataVersion:
        """
        Create a new data version from current snapshots.
        
        This "freezes" the current state of filtered data for reproducibility.
        """
        # Apply filters if provided
        if filter_config:
            snapshots = self._get_filtered_snapshots(filter_config)
        else:
            snapshots = self._get_all_snapshots()
        
        if not snapshots:
             raise ValueError("No snapshots found to version")

        # Compute data hash for integrity
        data_hash = self._compute_hash(snapshots)
        
        # Calculate stats
        tickers = set(s.ticker for s in snapshots)
        dates = [s.snapshot_timestamp for s in snapshots]
        
        # Create version record
        version = DataVersion(
            version_name=name,
            description=description,
            ticker_count=len(tickers),
            snapshot_count=len(snapshots),
            date_range_start=min(dates).date(),
            date_range_end=max(dates).date(),
            data_hash=data_hash,
            filter_config=filter_config,
        )
        self.session.add(version)
        self.session.flush() # flush to get ID
        
        # Link snapshots to version
        # Bulk insert might be more performant for large datasets
        links = [
            DataVersionSnapshot(
                version_id=version.version_id,
                ticker=s.ticker,
                snapshot_timestamp=s.snapshot_timestamp,
            )
            for s in snapshots
        ]
        self.session.bulk_save_objects(links)
        
        self.session.commit()
        return version
    
    def get_version_snapshots(
        self, version_id: UUID
    ) -> List[FinancialSnapshot]:
        """Get all snapshots in a specific version."""
        return (
            self.session.query(FinancialSnapshot)
            .join(DataVersionSnapshot)
            .filter(DataVersionSnapshot.version_id == version_id)
            .all()
        )
    
    def verify_version(self, version_id: UUID) -> bool:
        """Verify data version integrity by recomputing hash."""
        version = self.session.query(DataVersion).get(version_id)
        if not version:
            return False
        snapshots = self.get_version_snapshots(version_id)
        current_hash = self._compute_hash(snapshots)
        return current_hash == version.data_hash
    
    def _compute_hash(self, snapshots: List[FinancialSnapshot]) -> str:
        """Compute SHA256 hash of snapshot data."""
        # Sort for deterministic order
        sorted_snaps = sorted(snapshots, key=lambda s: (s.ticker, s.snapshot_timestamp))
        
        # Serialize to canonical JSON
        data = [
            {
                "ticker": s.ticker,
                "timestamp": s.snapshot_timestamp.isoformat(),
                "revenue": str(s.total_revenue) if s.total_revenue is not None else None,
                "mcap": str(s.market_cap_t0) if s.market_cap_t0 is not None else None,
                # Add other key fields that define the data integrity
            }
            for s in sorted_snaps
        ]
        
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
