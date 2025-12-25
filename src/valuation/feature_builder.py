
import pandas as pd
import numpy as np
from typing import List, Tuple
from .model_config import FeatureSpec

def build_feature_matrix(
    snapshots: pd.DataFrame, 
    feature_specs: List[FeatureSpec]
) -> pd.DataFrame:
    """
    Convert pandas DataFrame of snapshots into numeric feature matrix X.
    Applies transforms and fill strategies defined in specs.
    
    Args:
        snapshots: Input dataframe containing raw features
        feature_specs: List of features to extract and process
        
    Returns:
        pd.DataFrame: Processed numeric feature matrix
    """
    X = pd.DataFrame(index=snapshots.index)
    
    for spec in feature_specs:
        if spec.name not in snapshots.columns:
            if spec.required:
                raise ValueError(f"Required feature '{spec.name}' missing from input data.")
            else:
                # If optional and missing, create column of NaNs to be filled later
                # Or should we just skip? 
                # Better to fill with NaN so structure is consistent
                col_data = pd.Series(np.nan, index=snapshots.index)
        else:
            col_data = snapshots[spec.name].copy()
            
        # Convert to numeric, forcing errors to functional NaN
        col_data = pd.to_numeric(col_data, errors='coerce')
        
        # Apply Transforms
        if spec.transform == "log":
            # Handle negative/zero values for log?
            # Typically log(x + 1) or handle negatives explicitly
            # Here assuming strictly positive inputs like MarketCap or Revenue if log used
            # Or use numpy log1p which is log(1+x), safer for small values
            # If value <= 0, log is undefined. Mask them?
            # Let's assume log1p for now, and clip negative to 0
            col_data = np.log1p(col_data.clip(lower=0))
        elif spec.transform == "sqrt":
            col_data = np.sqrt(col_data.clip(lower=0))
            
        # Apply Fill Strategy
        if spec.fill_strategy == "zero":
            col_data = col_data.fillna(0.0)
        elif spec.fill_strategy == "mean":
            col_data = col_data.fillna(col_data.mean())
        elif spec.fill_strategy == "median":
            col_data = col_data.fillna(col_data.median())
        # "drop" strategy is hard in columnar processing - rows would need dropping
        
        X[spec.name] = col_data
        
    return X
