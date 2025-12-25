from abc import ABC, abstractmethod
from typing import Dict, Any
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

class ModelAdapter(ABC):
    """
    Abstract adapter for ML models.
    Low abstraction - just wraps sklearn-compatible estimators.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier."""
        pass
    
    @property
    @abstractmethod
    def estimator(self) -> BaseEstimator:
        """Return the sklearn-compatible estimator."""
        pass
    
    @property
    @abstractmethod
    def param_grid(self) -> Dict[str, list]:
        """Hyperparameter grid for GridSearchCV."""
        pass
    
    @property
    def cv_config(self) -> Dict[str, Any]:
        """Cross-validation configuration."""
        return {
            "outer_splits": 4,
            "inner_splits": 4,
            "n_experiments": 100,
            "shuffle": True,
        }


class XGBoostAdapter(ModelAdapter):
    """XGBoost regressor adapter - DEFAULT MODEL."""
    
    name = "xgboost_v1"
    
    @property
    def estimator(self) -> XGBRegressor:
        return XGBRegressor(
            objective=self.loss_function,
            random_state=42,
        )
    
    @property
    def loss_function(self) -> str:
        # Easily changeable loss function
        return "reg:squarederror"  # Options: reg:absoluteerror, custom
    
    @property
    def param_grid(self) -> Dict[str, list]:
        return {
            "n_estimators": [100, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8],
        }


class RandomForestAdapter(ModelAdapter):
    """Random Forest regressor adapter - ALTERNATIVE."""
    
    name = "rf_v1"
    
    @property
    def estimator(self) -> RandomForestRegressor:
        return RandomForestRegressor(random_state=42)
    
    @property
    def loss_function(self) -> str:
        return "squared_error"  # RF uses criterion, not objective
    
    @property
    def param_grid(self) -> Dict[str, list]:
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
        }


# === ACTIVE MODEL SELECTION ===
# Change this to swap models easily
ACTIVE_MODEL: ModelAdapter = XGBoostAdapter()
