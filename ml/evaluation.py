"""
Evaluation Metrics & Multicollinearity Diagnostics
=================================================
Calculates standard regression metrics (R², Adj R², RMSE, MAE, MSE) and VIF diagnostics.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> Dict[str, float]:
    """
    Computes R2, Adjusted R2, RMSE, MAE, MSE.
    Handles degrees of freedom for small sample datasets.
    """
    n = len(y_true)
    p = n_features
    
    r2 = float(r2_score(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    
    # Adjusted R2
    if n - p - 1 > 0:
        adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))
    else:
        adj_r2 = r2
        
    return {
        'R2': r2,
        'Adj_R2': adj_r2,
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse
    }


def compute_vif(X: pd.DataFrame, sample_size: int = 50_000) -> pd.DataFrame:
    """
    Calculates Variance Inflation Factor (VIF) for all numerical columns.
    Subsamples if dataset is very large for computational efficiency.
    """
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X.copy()
        
    # Standardize copy for numerical stability in VIF calculation
    X_vals = X_sample.values.astype(float)
    X_std = (X_vals - np.mean(X_vals, axis=0)) / (np.std(X_vals, axis=0) + 1e-9)
    
    # Add constant column for VIF
    X_with_const = np.column_stack([np.ones(len(X_std)), X_std])
    
    vif_records = []
    for i, col in enumerate(X.columns):
        try:
            # i+1 because 0 is constant
            vif_val = float(variance_inflation_factor(X_with_const, i + 1))
        except Exception:
            vif_val = 1.0
        vif_records.append({'Feature': col, 'VIF': round(vif_val, 4)})
        
    vif_df = pd.DataFrame(vif_records).sort_values(by='VIF', ascending=False).reset_index(drop=True)
    return vif_df


def find_optimal_k(df: pd.DataFrame, method_col: str = 'Method', rmse_col: str = 'RMSE', k_col: str = 'Number of Features') -> Dict[str, Any]:
    """
    Finds the optimal feature count k with tie-breaking favoring smaller k.
    """
    best_row = None
    min_rmse = float('inf')
    
    for _, row in df.iterrows():
        cur_rmse = row[rmse_col]
        cur_k = int(row[k_col])
        if cur_rmse < min_rmse - 1e-9:
            min_rmse = cur_rmse
            best_row = row
        elif abs(cur_rmse - min_rmse) <= 1e-9 and best_row is not None:
            if cur_k < int(best_row[k_col]):
                best_row = row
                
    return best_row.to_dict() if best_row is not None else {}
