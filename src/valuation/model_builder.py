
from typing import Optional, List, Dict, Any
from .model_config import ModelConfig, FeatureSpec

class ModelBuilder:
    """
    Builder pattern for constructing model configurations.
    Allows modular feature addition and configuration.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self._name = name
        self._version = version
        self._description = ""
        self._model_type = "xgboost"
        self._features: List[FeatureSpec] = []
        self._hyperparameters: Dict[str, Any] = {}
        self._param_grid: Dict[str, List[Any]] = {}
        self._n_experiments = 100
        self._outer_cv_splits = 4
        self._inner_cv_splits = 4
        self._loss_function = "reg:squarederror"
        self._random_seed = 42
    
    def description(self, desc: str) -> "ModelBuilder":
        """Set model description."""
        self._description = desc
        return self
    
    def model_type(self, model_type: str) -> "ModelBuilder":
        """Set model type: 'xgboost', 'random_forest', 'lightgbm'."""
        self._model_type = model_type
        return self
    
    # --- Feature Addition Methods ---
    
    def add_feature(
        self,
        name: str,
        required: bool = True,
        transform: Optional[str] = None,
        fill_strategy: str = "zero",
    ) -> "ModelBuilder":
        """Add a single feature to the model."""
        self._features.append(FeatureSpec(
            name=name,
            required=required,
            transform=transform,
            fill_strategy=fill_strategy,
        ))
        return self
    
    def add_core_fundamentals(self) -> "ModelBuilder":
        """Add standard fundamental features."""
        return (
            self.add_feature("total_revenue", required=True)
            .add_feature("gross_profit", required=True)
            .add_feature("ebitda", required=False, fill_strategy="median")
            .add_feature("net_income", required=False)
            .add_feature("total_debt", required=True)
            .add_feature("total_cash", required=True)
            .add_feature("free_cash_flow", required=False)
            .add_feature("operating_cash_flow", required=False)
            .add_feature("book_value", required=False)
        )
    
    def add_ratio_features(self) -> "ModelBuilder":
        """Add financial ratio features."""
        return (
            self.add_feature("profit_margins", required=False)
            .add_feature("debt_to_equity", required=False, transform="log")
            .add_feature("roe", required=False, fill_strategy="median")
            .add_feature("roa", required=False, fill_strategy="median")
            .add_feature("current_ratio", required=False)
            .add_feature("quick_ratio", required=False)
        )
    
    def add_dividend_features(self) -> "ModelBuilder":
        """Add dividend-related features."""
        # dividend_yield is derived from Price, so we exclude it to prevent leakage.
        return (
            self.add_feature("dividend_rate", required=False)
        )
    
    def add_risk_features(self) -> "ModelBuilder":
        """Add risk score features."""
        return (
            self.add_feature("audit_risk", required=False)
            .add_feature("board_risk", required=False)
            .add_feature("compensation_risk", required=False)
            # beta is market-dependent (price covariance), excluded.
        )
    
    def add_short_interest(self) -> "ModelBuilder":
        """Add short interest features."""
        # Short interest is often shares_short / float.
        # shares_short is public info, but short_ratio might be days to cover (volume dep).
        # We'll keep shares_short (fundamental sentinel) but exclude short_ratio if it depends on volume/price dynamics?
        # User said "no direct price/mcap hint".
        # Let's keep these for now unless explicitly asked, except beta/yield which are mathematically price-linked.
        return (
            self.add_feature("shares_short", required=False)
            .add_feature("short_ratio", required=False)
        )
    
    def add_ownership_features(self) -> "ModelBuilder":
        """Add ownership structure features."""
        return (
            self.add_feature("held_percent_insiders", required=False)
            .add_feature("held_percent_institutions", required=False)
        )
    
    # --- Configuration Methods ---
    
    def hyperparameters(self, **kwargs) -> "ModelBuilder":
        """Set base hyperparameters."""
        self._hyperparameters.update(kwargs)
        return self
    
    def param_grid(self, **kwargs) -> "ModelBuilder":
        """Set hyperparameter search grid."""
        self._param_grid.update(kwargs)
        return self
    
    def cv_config(
        self,
        n_experiments: int = 100,
        outer_splits: int = 4,
        inner_splits: int = 4,
    ) -> "ModelBuilder":
        """Configure cross-validation."""
        self._n_experiments = n_experiments
        self._outer_cv_splits = outer_splits
        self._inner_cv_splits = inner_splits
        return self
    
    def loss_function(self, loss: str) -> "ModelBuilder":
        """Set loss function."""
        self._loss_function = loss
        return self
    
    def random_seed(self, seed: int) -> "ModelBuilder":
        """Set random seed for reproducibility."""
        self._random_seed = seed
        return self
    
    # --- Build Methods ---
    
    def build(self) -> ModelConfig:
        """Build the final ModelConfig."""
        return ModelConfig(
            name=self._name,
            version=self._version,
            description=self._description,
            model_type=self._model_type,
            features=self._features,
            hyperparameters=self._hyperparameters,
            param_grid=self._param_grid,
            n_experiments=self._n_experiments,
            outer_cv_splits=self._outer_cv_splits,
            inner_cv_splits=self._inner_cv_splits,
            loss_function=self._loss_function,
            random_seed=self._random_seed,
        )
    
    def to_json(self) -> str:
        """Build and serialize to JSON."""
        return self.build().to_json()
