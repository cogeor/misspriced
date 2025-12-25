from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import hashlib

class FeatureSpec(BaseModel):
    """Single feature specification."""
    name: str
    required: bool = True
    transform: Optional[str] = None  # "log", "sqrt", "normalize"
    fill_strategy: Optional[str] = "zero"      # "zero", "mean", "median", "drop"

class ModelConfig(BaseModel):
    """
    Complete model configuration for reproducibility.
    Can be serialized to JSON and reconstructed via builder.
    """
    # Model identification
    name: str
    version: str
    description: str
    
    # Model type
    model_type: str  # "xgboost", "random_forest", "lightgbm"
    
    # Features (as list for builder pattern)
    features: List[FeatureSpec]
    
    # Hyperparameters
    hyperparameters: Dict[str, Any]
    param_grid: Dict[str, List[Any]]  # For grid search
    
    # Cross-validation config
    n_experiments: int = 100
    outer_cv_splits: int = 4
    inner_cv_splits: int = 4
    
    # Loss function
    loss_function: str = "reg:squarederror"
    custom_objective: Optional[str] = None
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    def to_json(self) -> str:
        """Serialize config to JSON."""
        return self.model_dump_json(indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ModelConfig":
        """Deserialize config from JSON."""
        return cls.model_validate_json(json_str)
    
    def config_hash(self) -> str:
        """Compute hash for config identification."""
        # Ensure consistent sorting for hash stability
        return hashlib.sha256(self.model_dump_json(indent=2).encode()).hexdigest()[:16]
