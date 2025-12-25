from pydantic import BaseModel

class IngestionConfig(BaseModel):
    """Config for ingestion process."""
    batch_size: int = 10
    retry_count: int = 3
    retry_delay: float = 1.0
    fetch_historical: bool = True
