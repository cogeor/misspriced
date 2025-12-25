import numpy as np

def relative_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Relative MSE: penalizes relative error, not absolute.
    Better for market cap prediction across different scales.
    """
    relative_errors = (y_pred - y_true) / y_true
    return np.mean(relative_errors ** 2)


def relative_mse_obj(y_pred: np.ndarray, dtrain) -> tuple:
    """
    XGBoost custom objective for relative MSE.
    Returns gradient and hessian.
    """
    y_true = dtrain.get_label()
    
    # Gradient: d/dy_pred of ((y_pred - y_true) / y_true)^2
    grad = 2 * (y_pred - y_true) / (y_true ** 2)
    
    # Hessian: second derivative
    hess = 2 / (y_true ** 2)
    
    return grad, hess


# === ACTIVE LOSS FUNCTION ===
# Set to None to use XGBoost default, or provide custom objective
CUSTOM_OBJECTIVE = None  # Options: relative_mse_obj, None
