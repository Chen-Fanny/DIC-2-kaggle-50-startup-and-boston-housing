"""
Data Loading, Cleaning, Categorical Encoding & Train/Val/Test Splitting
====================================================================
Ensures strict zero data leakage by fitting transformers exclusively on training data.
"""

from typing import Tuple, Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml.config import DatasetConfig


class PreprocessingResult:
    """Encapsulates split datasets, encoders, scalers, and feature metadata."""
    def __init__(
        self,
        raw_df: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train_val: pd.DataFrame,
        y_train_val: pd.Series,
        feature_names: List[str],
        scaler: StandardScaler,
        X_train_scaled: np.ndarray,
        X_val_scaled: np.ndarray,
        X_test_scaled: np.ndarray,
        X_train_val_scaled: np.ndarray,
        encoding_info: Dict[str, Any]
    ):
        self.raw_df = raw_df
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.X_train_val = X_train_val
        self.y_train_val = y_train_val
        self.feature_names = feature_names
        self.scaler = scaler
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.X_test_scaled = X_test_scaled
        self.X_train_val_scaled = X_train_val_scaled
        self.encoding_info = encoding_info


def load_and_preprocess(config: DatasetConfig) -> PreprocessingResult:
    """
    Executes CRISP-DM Phase 2 & 3: Data Understanding and Data Preparation.
    """
    # 1. Load Raw CSV
    raw_df = pd.read_csv(config.data_path)
    print(f"[{config.name}] Raw dataset loaded from {config.data_path}")
    print(f"  Shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    print(f"  Missing values: {raw_df.isnull().sum().sum()} total NaNs")
    print(f"  Duplicate rows: {raw_df.duplicated().sum()}")
    
    df = raw_df.copy()
    
    # Drop identifier column if specified
    if config.id_col and config.id_col in df.columns:
        df = df.drop(columns=[config.id_col])
        print(f"  Dropped ID column: '{config.id_col}'")
        
    # 2. Categorical One-Hot Encoding (Avoid Dummy Variable Trap)
    encoding_info = {}
    if config.categorical_features:
        for cat_col in config.categorical_features:
            if cat_col not in df.columns:
                continue
            unique_vals = sorted(df[cat_col].unique().tolist())
            ref_cat = config.reference_categories.get(cat_col, unique_vals[0])
            print(f"  One-Hot Encoding '{cat_col}': Categories = {unique_vals}, Reference (dropped) = '{ref_cat}'")
            
            # Create dummies
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, dtype=float)
            ref_col_name = f"{cat_col}_{ref_cat}"
            if ref_col_name in dummies.columns:
                dummies = dummies.drop(columns=[ref_col_name])
                
            encoding_info[cat_col] = {
                'categories': unique_vals,
                'reference_category': ref_cat,
                'encoded_columns': dummies.columns.tolist()
            }
            # Replace categorical column with dummy columns
            df = pd.concat([df.drop(columns=[cat_col]), dummies], axis=1)

    # 3. Separate Features and Target
    if config.target_col not in df.columns:
        raise ValueError(f"Target column '{config.target_col}' not found in dataset columns: {df.columns.tolist()}")
        
    X = df.drop(columns=[config.target_col])
    y = df[config.target_col]
    feature_names = X.columns.tolist()
    print(f"  Final model input features ({len(feature_names)}): {feature_names}")
    
    # 4. Train / Validation / Holdout Test Split (Strict 64% / 16% / 20%)
    # First split: 80% (Train+Val) and 20% (Holdout Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state
    )
    # Second split: From 80%, take 80% Train (64% overall) and 20% Val (16% overall)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=config.val_size,
        random_state=config.random_state
    )
    
    print("  Data Split (Strict Zero-Leakage):")
    print(f"    Train:        {len(X_train):>7,} rows ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    Validation:   {len(X_val):>7,} rows ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    Holdout Test: {len(X_test):>7,} rows ({len(X_test)/len(X)*100:.1f}%) [ISOLATED]")
    
    # 5. Feature Scaling (Fit exclusively on X_train!)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_train_val_scaled = scaler.transform(X_train_val)
    
    return PreprocessingResult(
        raw_df=raw_df,
        X=X,
        y=y,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        feature_names=feature_names,
        scaler=scaler,
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        X_test_scaled=X_test_scaled,
        X_train_val_scaled=X_train_val_scaled,
        encoding_info=encoding_info
    )
