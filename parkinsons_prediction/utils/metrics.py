import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape_with_offset(true_values, predictions):
    """Calculate SMAPE with offset to handle zeros."""
    return 200 * np.abs(predictions - true_values) / (np.abs(true_values + 1) + np.abs(predictions + 1))

def smape_metric(true_values, predictions):
    """Calculate mean SMAPE."""
    return smape_with_offset(true_values, predictions).mean()


def mae_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE score
    """
    return mean_absolute_error(y_true, y_pred)


def rmse_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mse_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MSE score
    """
    return mean_squared_error(y_true, y_pred)


def mape_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
        
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100