from typing import List
import pandas as pd
from .predicates import FilterPredicate


class DatasetFilter:
    """
    Compose multiple filter predicates.
    Applies all filters with AND logic (intersection).
    """
    
    def __init__(self, predicates: List[FilterPredicate] = None):
        self.predicates = predicates or []
    
    def add(self, predicate: FilterPredicate) -> "DatasetFilter":
        """Add a filter predicate. Returns self for chaining."""
        self.predicates.append(predicate)
        return self
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all filters to dataframe.
        Returns filtered dataframe.
        """
        if not self.predicates:
            return df
        
        mask = pd.Series(True, index=df.index)
        for predicate in self.predicates:
            mask &= predicate(df)
        
        return df[mask].copy()
    
    def summary(self, df: pd.DataFrame) -> dict:
        """
        Get filtering summary without actually filtering.
        Useful for debugging/logging.
        """
        original_count = len(df)
        final_mask = pd.Series(True, index=df.index)
        
        step_counts = []
        for i, predicate in enumerate(self.predicates):
            mask = predicate(df)
            remaining = (final_mask & mask).sum()
            step_counts.append({
                "step": i + 1,
                "predicate": predicate.__name__ if hasattr(predicate, "__name__") else str(predicate),
                "remaining": remaining,
                "dropped": (final_mask.sum() - remaining),
            })
            final_mask &= mask
        
        return {
            "original_count": original_count,
            "final_count": final_mask.sum(),
            "total_dropped": original_count - final_mask.sum(),
            "steps": step_counts,
        }
