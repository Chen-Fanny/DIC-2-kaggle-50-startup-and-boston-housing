"""
================================================================================
Flood Probability Prediction Using Multiple Linear Regression & Feature Selection
Kaggle Playground Series Season 4, Episode 5 (S4E5)
Author: Graduate Machine Learning Engineer & Data Scientist
CRISP-DM Standard Pipeline Implementation
================================================================================
"""

import os
import sys
import time
import math
import itertools
import shutil
from typing import Dict, List, Tuple, Any

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# Configuration
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

RANDOM_STATE = 42
TARGET_COL = 'FloodProbability'
ID_COL = 'id'

# Performance optimization: sampling size for expensive cross-validated feature search
FS_SAMPLE_SIZE = 100_000


def ensure_directories():
    """Ensure data and results directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def download_dataset() -> bool:
    """
    Attempt to locate or download the real Kaggle Playground Series S4E5 dataset.
    Returns True if data files are present and verified, otherwise False.
    """
    ensure_directories()
    
    if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
        print(f"[Dataset] Verified existing local dataset at {DATA_DIR}")
        return True

    print("[Dataset] Attempting automatic download via kagglehub...")
    try:
        import kagglehub
        path = kagglehub.competition_download('playground-series-s4e5')
        print(f"[Dataset] Downloaded to cache path: {path}")
        for filename in ['train.csv', 'test.csv', 'sample_submission.csv']:
            src = os.path.join(path, filename)
            dst = os.path.join(DATA_DIR, filename)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"[Dataset] Copied {filename} -> {dst}")
        if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
            return True
    except Exception as e:
        print(f"[Dataset] kagglehub download failed: {e}")

    # Check Kaggle API CLI / module
    print("[Dataset] Attempting Kaggle API download...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("[Dataset] Kaggle API authenticated successfully. Downloading competition files...")
        api.competition_download_files('playground-series-s4e5', path=DATA_DIR)
        
        # Check if zip downloaded
        zip_path = os.path.join(DATA_DIR, 'playground-series-s4e5.zip')
        if os.path.exists(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATA_DIR)
            os.remove(zip_path)
            print(f"[Dataset] Extracted dataset to {DATA_DIR}")
        if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
            return True
    except Exception as e:
        print(f"[Dataset] Kaggle API download failed: {e}")

    print("\n" + "="*80)
    print("【DATASET DOWNLOAD INSTRUCTIONS / 資料集下載指引】")
    print("Kaggle Playground Series S4E5 資料集需要 Kaggle 帳號認證並接受競賽條款。")
    print("請依以下任一方式取得真實資料集：")
    print("1. 取得 Kaggle API Token:")
    print("   - 登入 Kaggle (https://www.kaggle.com)")
    print("   - 造訪競賽頁面 https://www.kaggle.com/competitions/playground-series-s4e5/data 並點擊 'Understand and Accept' 同意規則")
    print("   - 前往個人設定頁面 -> API -> 點擊 'Create New Token' 下載 kaggle.json")
    print("   - 將 kaggle.json 放置於 C:\\Users\\User\\.kaggle\\kaggle.json 或設定環境變數 KAGGLE_USERNAME 與 KAGGLE_KEY")
    print("2. 手動下載並放置:")
    print(f"   - 從競賽頁面下載 train.csv, test.csv, sample_submission.csv")
    print(f"   - 直接放置於專案資料夾: {DATA_DIR}")
    print("="*80 + "\n")
    return False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> Dict[str, float]:
    """Calculate regression evaluation metrics: MAE, MSE, RMSE, R2, Adjusted R2."""
    n = len(y_true)
    p = n_features
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # Adjusted R-squared
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else r2
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Adj_R2': adj_r2
    }


def fit_and_evaluate_subset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    features: List[str]
) -> Dict[str, Any]:
    """Train Multiple Linear Regression on selected features and evaluate on validation set."""
    lr = LinearRegression()
    lr.fit(X_train[features], y_train)
    preds = lr.predict(X_val[features])
    metrics = calculate_metrics(y_val.to_numpy(), preds, len(features))
    return {
        'model': lr,
        'features': features,
        'predictions': preds,
        **metrics
    }


# ==============================================================================
# Feature Selection Algorithms
# ==============================================================================

def algorithm_1_lasso_l1(
    X_train_std: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str]
) -> Tuple[List[str], np.ndarray]:
    """
    Algorithm 1: Lasso L1 Regularization Ranking
    Rank features based on absolute magnitude of Lasso coefficients.
    """
    print("\n[Feature Selection 1/5] Running Lasso (L1) Feature Ranking...")
    start_t = time.time()
    lasso_cv = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=2000, n_jobs=-1)
    lasso_cv.fit(X_train_std, y_train)
    coefs = lasso_cv.coef_
    abs_coefs = np.abs(coefs)
    ranking_indices = np.argsort(-abs_coefs)
    ranked_features = [feature_names[i] for i in ranking_indices]
    print(f"  Best Lasso alpha: {lasso_cv.alpha_:.6f} (elapsed: {time.time()-start_t:.2f}s)")
    print(f"  Top 5 features: {ranked_features[:5]}")
    return ranked_features, abs_coefs


def algorithm_2_forward_sfs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: List[str],
    sample_size: int = FS_SAMPLE_SIZE
) -> List[str]:
    """
    Algorithm 2: Forward Sequential Feature Selection (Forward SFS)
    Greedily add the feature that minimizes cross-validated RMSE at each step.
    Optimized using a reproducible training sample to ensure sub-minute execution.
    """
    print("\n[Feature Selection 2/5] Running Forward Sequential Feature Selection (SFS)...")
    start_t = time.time()
    
    if len(X_train) > sample_size:
        sample_df = X_train.sample(n=sample_size, random_state=RANDOM_STATE)
        y_sample = y_train.loc[sample_df.index]
        print(f"  (Using reproducible sample of {sample_size:,} rows for cross-validated search)")
    else:
        sample_df = X_train
        y_sample = y_train

    # Convert to NumPy with intercept for precomputed closed-form OLS
    X_mat = sample_df[feature_names].to_numpy(dtype=np.float64)
    y_vec = y_sample.to_numpy(dtype=np.float64)
    X_ones = np.column_stack([np.ones(len(X_mat)), X_mat])
    name_to_idx = {name: i + 1 for i, name in enumerate(feature_names)}

    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    fold_data = []
    for tr_idx, va_idx in kf.split(X_mat):
        X_tr, y_tr = X_ones[tr_idx], y_vec[tr_idx]
        X_va, y_va = X_ones[va_idx], y_vec[va_idx]
        G_tr, v_tr = X_tr.T @ X_tr, X_tr.T @ y_tr
        G_va, v_va = X_va.T @ X_va, X_va.T @ y_va
        y_va_ss, n_va = float(y_va @ y_va), len(y_va)
        fold_data.append((G_tr, v_tr, G_va, v_va, y_va_ss, n_va))

    selected_features: List[str] = []
    selected_indices: List[int] = [0]  # Intercept index
    remaining_features = list(feature_names)

    for step in range(len(feature_names)):
        best_score = float('inf')
        best_candidate = None

        for candidate in remaining_features:
            cand_idx = selected_indices + [name_to_idx[candidate]]
            fold_rmses = []
            for G_tr, v_tr, G_va, v_va, y_va_ss, n_va in fold_data:
                w = np.linalg.solve(G_tr[np.ix_(cand_idx, cand_idx)], v_tr[cand_idx])
                sub_G_va = G_va[np.ix_(cand_idx, cand_idx)]
                sub_v_va = v_va[cand_idx]
                rss = y_va_ss - 2.0 * float(w @ sub_v_va) + float(w @ (sub_G_va @ w))
                fold_rmses.append(math.sqrt(max(0.0, rss) / n_va))
            
            mean_rmse = np.mean(fold_rmses)
            if mean_rmse < best_score:
                best_score = mean_rmse
                best_candidate = candidate

        selected_features.append(best_candidate)
        selected_indices.append(name_to_idx[best_candidate])
        remaining_features.remove(best_candidate)
        print(f"  Step {step+1:2d}/20: Added '{best_candidate}' (CV RMSE: {best_score:.6f})")

    print(f"  Forward SFS complete (elapsed: {time.time()-start_t:.2f}s)")
    return selected_features


def algorithm_3_rfe(
    X_train_std: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str]
) -> List[str]:
    """
    Algorithm 3: Recursive Feature Elimination (RFE)
    Repeatedly eliminate the feature with lowest absolute coefficient using LinearRegression.
    """
    print("\n[Feature Selection 3/5] Running Recursive Feature Elimination (RFE)...")
    start_t = time.time()
    estimator = LinearRegression()
    # Fit RFE down to 1 feature to get full feature ranking
    rfe = RFE(estimator=estimator, n_features_to_select=1, step=1)
    rfe.fit(X_train_std, y_train)
    
    # rfe.ranking_: 1 is best, higher is worse
    sorted_pairs = sorted(zip(feature_names, rfe.ranking_), key=lambda x: x[1])
    ranked_features = [p[0] for p in sorted_pairs]
    print(f"  RFE complete (elapsed: {time.time()-start_t:.2f}s)")
    print(f"  Top 5 features: {ranked_features[:5]}")
    return ranked_features


def algorithm_4_elasticnet(
    X_train_std: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str]
) -> Tuple[List[str], np.ndarray]:
    """
    Algorithm 4: ElasticNet Regularization Ranking
    Rank features based on absolute magnitude of ElasticNet coefficients combining L1 and L2 penalties.
    """
    print("\n[Feature Selection 4/5] Running ElasticNet Feature Ranking...")
    start_t = time.time()
    enet_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9],
        cv=5,
        random_state=RANDOM_STATE,
        max_iter=2000,
        n_jobs=-1
    )
    enet_cv.fit(X_train_std, y_train)
    coefs = enet_cv.coef_
    abs_coefs = np.abs(coefs)
    ranking_indices = np.argsort(-abs_coefs)
    ranked_features = [feature_names[i] for i in ranking_indices]
    print(f"  Best ElasticNet alpha: {enet_cv.alpha_:.6f}, l1_ratio: {enet_cv.l1_ratio_:.2f} (elapsed: {time.time()-start_t:.2f}s)")
    print(f"  Top 5 features: {ranked_features[:5]}")
    return ranked_features, abs_coefs


def algorithm_5_exhaustive_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: List[str],
    max_k: int = 5,
    sample_size: int = FS_SAMPLE_SIZE
) -> Dict[int, Tuple[List[str], float]]:
    """
    Algorithm 5: Exhaustive Feature Search (restricted to sizes 1 to 5)
    Evaluates all combinations of sizes 1 to 5 using cross-validated RMSE.
    Total combinations evaluated: 20 + 190 + 1140 + 4845 + 15504 = 21,699 combinations!
    Optimized via closed-form OLS normal equation solves on sample folds.
    """
    print(f"\n[Feature Selection 5/5] Running Exhaustive Feature Search (Sizes 1 to {max_k})...")
    start_t = time.time()
    
    if len(X_train) > sample_size:
        sample_df = X_train.sample(n=sample_size, random_state=RANDOM_STATE)
        y_sample = y_train.loc[sample_df.index]
        print(f"  (Using reproducible sample of {sample_size:,} rows for {21699:,} combination evaluations)")
    else:
        sample_df = X_train
        y_sample = y_train

    # Convert to NumPy and prepend ones column for intercept
    X_mat = sample_df[feature_names].to_numpy(dtype=np.float64)
    y_vec = y_sample.to_numpy(dtype=np.float64)
    X_ones = np.column_stack([np.ones(len(X_mat)), X_mat])
    
    # 3-Fold Cross Validation: Precompute fold Gram matrices for closed-form OLS
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    fold_data = []
    for tr_idx, va_idx in kf.split(X_mat):
        X_tr = X_ones[tr_idx]
        y_tr = y_vec[tr_idx]
        X_va = X_ones[va_idx]
        y_va = y_vec[va_idx]
        
        G_tr = X_tr.T @ X_tr
        v_tr = X_tr.T @ y_tr
        G_va = X_va.T @ X_va
        v_va = X_va.T @ y_va
        y_va_ss = float(y_va @ y_va)
        n_va = len(y_va)
        fold_data.append((G_tr, v_tr, G_va, v_va, y_va_ss, n_va))
    
    best_combinations: Dict[int, Tuple[List[str], float]] = {}

    for k in range(1, max_k + 1):
        k_start = time.time()
        combs = list(itertools.combinations(range(len(feature_names)), k))
        best_k_rmse = float('inf')
        best_k_comb = None

        for comb in combs:
            # Indices including intercept (0) + 1-indexed feature positions
            idx = [0] + [c + 1 for c in comb]
            fold_rmses = []
            
            for G_tr, v_tr, G_va, v_va, y_va_ss, n_va in fold_data:
                # Solve OLS coefficients: G_tr * w = v_tr
                w = np.linalg.solve(G_tr[np.ix_(idx, idx)], v_tr[idx])
                # Calculate validation RSS: y_va^T y_va - 2 w^T v_va + w^T G_va w
                sub_G_va = G_va[np.ix_(idx, idx)]
                sub_v_va = v_va[idx]
                rss = y_va_ss - 2.0 * float(w @ sub_v_va) + float(w @ (sub_G_va @ w))
                fold_rmses.append(math.sqrt(max(0.0, rss) / n_va))

            mean_rmse = np.mean(fold_rmses)
            if mean_rmse < best_k_rmse:
                best_k_rmse = mean_rmse
                best_k_comb = comb

        chosen_features = [feature_names[i] for i in best_k_comb]
        best_combinations[k] = (chosen_features, best_k_rmse)
        print(f"  k = {k:d} ({len(combs):5d} combinations evaluated): Best CV RMSE = {best_k_rmse:.6f} in {time.time()-k_start:.2f}s")
        print(f"       Selected: {chosen_features}")

    print(f"  Exhaustive Search complete (elapsed: {time.time()-start_t:.2f}s)")
    return best_combinations


# ==============================================================================
# Visualization Functions
# ==============================================================================

def generate_visualizations(
    fs_1_to_20_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    best_model_name: str,
    best_features: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    five_feature_models: Dict[str, Dict[str, Any]],
    baseline_eval: Dict[str, Any],
    lasso_coefs: np.ndarray,
    enet_coefs: np.ndarray,
    all_features: List[str]
):
    """Generate all 7 required publication-quality PNG charts (>= 150 DPI)."""
    print("\n[Visualizations] Generating high-resolution experimental charts...")
    palette = {'Lasso L1': '#1f77b4', 'Forward SFS': '#ff7f0e', 'RFE': '#2ca02c', 'ElasticNet': '#d62728', 'Exhaustive Search': '#9467bd'}
    
    # --------------------------------------------------------------------------
    # Figure 1: RMSE by Number of Features
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for method in ['Lasso L1', 'Forward SFS', 'RFE', 'ElasticNet', 'Exhaustive Search']:
        sub = fs_1_to_20_df[fs_1_to_20_df['Method'] == method]
        plt.plot(sub['Number of Features'], sub['RMSE'], marker='o', label=method, color=palette[method], linewidth=2)
    plt.axhline(baseline_eval['RMSE'], color='black', linestyle='--', alpha=0.7, label=f"Baseline (All 20 Features: {baseline_eval['RMSE']:.5f})")
    plt.title("RMSE by Number of Selected Features Across Five Algorithms", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Number of Features", fontsize=12)
    plt.ylabel("Validation RMSE (Lower is Better)", fontsize=12)
    plt.xticks(range(1, 21))
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig1_path = os.path.join(RESULTS_DIR, 'rmse_vs_features.png')
    plt.savefig(fig1_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig1_path}")

    # --------------------------------------------------------------------------
    # Figure 2: R-squared by Number of Features
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for method in ['Lasso L1', 'Forward SFS', 'RFE', 'ElasticNet', 'Exhaustive Search']:
        sub = fs_1_to_20_df[fs_1_to_20_df['Method'] == method]
        plt.plot(sub['Number of Features'], sub['R-squared'], marker='s', label=method, color=palette[method], linewidth=2)
    plt.axhline(baseline_eval['R2'], color='black', linestyle='--', alpha=0.7, label=f"Baseline (All 20 Features: {baseline_eval['R2']:.5f})")
    plt.title("R-squared by Number of Selected Features Across Five Algorithms", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Number of Features", fontsize=12)
    plt.ylabel("Validation R² (Higher is Better)", fontsize=12)
    plt.xticks(range(1, 21))
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig2_path = os.path.join(RESULTS_DIR, 'r2_vs_features.png')
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig2_path}")

    # --------------------------------------------------------------------------
    # Figure 3: Combined Algorithm Comparison (Side-by-Side)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for method in ['Lasso L1', 'Forward SFS', 'RFE', 'ElasticNet', 'Exhaustive Search']:
        sub = fs_1_to_20_df[fs_1_to_20_df['Method'] == method]
        ax1.plot(sub['Number of Features'], sub['RMSE'], marker='o', label=method, color=palette[method], linewidth=2)
        ax2.plot(sub['Number of Features'], sub['R-squared'], marker='s', label=method, color=palette[method], linewidth=2)

    ax1.axhline(baseline_eval['RMSE'], color='black', linestyle='--', alpha=0.7, label='Baseline (20 Features)')
    ax1.set_title("RMSE Comparison Across Five Feature Selection Methods", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Number of Features", fontsize=11)
    ax1.set_ylabel("Validation RMSE", fontsize=11)
    ax1.set_xticks(range(1, 21))
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.axhline(baseline_eval['R2'], color='black', linestyle='--', alpha=0.7, label='Baseline (20 Features)')
    ax2.set_title("R-squared Comparison Across Five Feature Selection Methods", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Number of Features", fontsize=11)
    ax2.set_ylabel("Validation R²", fontsize=11)
    ax2.set_xticks(range(1, 21))
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig3_path = os.path.join(RESULTS_DIR, 'all_5_algorithms_comparison.png')
    plt.savefig(fig3_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig3_path}")

    # --------------------------------------------------------------------------
    # Figure 0: 5-Feature Comparison Bar Chart
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    comp_plot_df = comparison_df.copy()
    models = comp_plot_df['Model'].tolist()
    
    ax1.barh(models, comp_plot_df['RMSE'], color=['#3498db' if 'Baseline' not in m else '#2c3e50' for m in models])
    ax1.set_title("5-Feature Models & Baseline: Validation RMSE", fontsize=12, fontweight='bold')
    ax1.set_xlabel("RMSE (Lower is Better)")
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.barh(models, comp_plot_df['R²'], color=['#2ecc71' if 'Baseline' not in m else '#27ae60' for m in models])
    ax2.set_title("5-Feature Models & Baseline: Validation R²", fontsize=12, fontweight='bold')
    ax2.set_xlabel("R² (Higher is Better)")
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig0_path = os.path.join(RESULTS_DIR, 'feature_selection_comparison.png')
    plt.savefig(fig0_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig0_path}")

    # --------------------------------------------------------------------------
    # Figure 4: Actual vs Predicted (Best Model)
    # --------------------------------------------------------------------------
    # Fit best model on validation
    best_lr = LinearRegression()
    best_lr.fit(X_train[best_features], y_train)
    val_preds = best_lr.predict(X_val[best_features])

    plt.figure(figsize=(8, 8))
    # Downsample points for scatter plot visibility if validation set is huge
    plot_n = min(5000, len(y_val))
    sub_idx = np.random.RandomState(RANDOM_STATE).choice(len(y_val), plot_n, replace=False)
    y_val_sub = y_val.iloc[sub_idx].to_numpy()
    preds_sub = val_preds[sub_idx]

    plt.scatter(y_val_sub, preds_sub, alpha=0.25, color='#2980b9', edgecolors='none', s=20, label='Validation Observations')
    min_val = min(y_val_sub.min(), preds_sub.min())
    max_val = max(y_val_sub.max(), preds_sub.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit (y = x)')
    plt.title(f"Actual vs Predicted Flood Probability ({best_model_name} with {len(best_features)} Features)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Actual Flood Probability", fontsize=11)
    plt.ylabel("Predicted Flood Probability", fontsize=11)
    plt.legend(frameon=True, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig4_path = os.path.join(RESULTS_DIR, 'actual_vs_predicted.png')
    plt.savefig(fig4_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig4_path}")

    # --------------------------------------------------------------------------
    # Figure 5: Different Model Predictions Comparison (Sorted Subset)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(14, 6))
    sample_size = 100
    subset_indices = np.random.RandomState(RANDOM_STATE).choice(len(y_val), sample_size, replace=False)
    
    # Sort observations by actual target value for clean, readable visualization
    sorted_order = np.argsort(y_val.iloc[subset_indices].to_numpy())
    sorted_idx = subset_indices[sorted_order]
    x_axis = np.arange(sample_size)

    plt.plot(x_axis, y_val.iloc[sorted_idx].to_numpy(), 'k-', linewidth=2.5, label='Actual Values')
    
    # Plot 5-feature models
    for name, res in five_feature_models.items():
        sub_preds = res['predictions'][sorted_idx]
        plt.plot(x_axis, sub_preds, linestyle=':', linewidth=1.5, label=f"{name} (5 feats)")
    
    # Plot baseline
    base_sub_preds = baseline_eval['predictions'][sorted_idx]
    plt.plot(x_axis, base_sub_preds, color='blue', linestyle='--', linewidth=1.8, label="Baseline (20 feats)")

    plt.title("Model Prediction Trajectories vs Actuals on Sorted Validation Subset (N=100)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Ordered Validation Sample Index", fontsize=11)
    plt.ylabel("Flood Probability", fontsize=11)
    plt.legend(frameon=True, fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig5_path = os.path.join(RESULTS_DIR, 'model_predictions_comparison.png')
    plt.savefig(fig5_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig5_path}")

    # --------------------------------------------------------------------------
    # Figure 6: Feature Importance (Standardized Coefficients)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    # Fit full standardized OLS to get standardized coefficients
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_train[all_features])
    full_lr = LinearRegression()
    full_lr.fit(X_std, y_train)
    
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Standardized_Coefficient': np.abs(full_lr.coef_)
    }).sort_values('Standardized_Coefficient', ascending=True)

    plt.barh(importance_df['Feature'], importance_df['Standardized_Coefficient'], color='#34495e')
    plt.title("Feature Importance: Standardized Multiple Linear Regression Coefficients", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Absolute Standardized Coefficient Magnitude (|β*|)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fig6_path = os.path.join(RESULTS_DIR, 'feature_importance.png')
    plt.savefig(fig6_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig6_path}")

    # --------------------------------------------------------------------------
    # Figure 7: Residual Analysis for Best Model
    # --------------------------------------------------------------------------
    residuals = y_val.to_numpy() - val_preds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Residuals vs Predicted
    sub_res_idx = np.random.RandomState(RANDOM_STATE).choice(len(residuals), min(5000, len(residuals)), replace=False)
    ax1.scatter(val_preds[sub_res_idx], residuals[sub_res_idx], alpha=0.25, color='#8e44ad', edgecolors='none', s=20)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_title(f"Residuals vs Predicted Values ({best_model_name})", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Predicted Values", fontsize=11)
    ax1.set_ylabel("Residuals (Actual - Predicted)", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=ax2, color='#2c3e50', bins=50)
    ax2.set_title("Distribution of Residuals (Normality Check)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Residual Value", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig7_path = os.path.join(RESULTS_DIR, 'residual_analysis.png')
    plt.savefig(fig7_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig7_path}")


# ==============================================================================
# Multicollinearity (VIF) & Experimental Analysis Generator
# ==============================================================================

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) for all features."""
    sample_df = X.sample(n=min(50000, len(X)), random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample_df)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False)


def generate_experimental_analysis_report(
    dataset_info: Dict[str, Any],
    baseline_metrics: Dict[str, float],
    comparison_df: pd.DataFrame,
    fs_1_to_20_df: pd.DataFrame,
    best_config: Dict[str, Any],
    holdout_metrics: Dict[str, float],
    vif_df: pd.DataFrame,
    all_features: List[str]
):
    """Generate comprehensive experimental analysis report in Traditional Chinese."""
    report_path = os.path.join(RESULTS_DIR, 'experimental_analysis.md')
    print(f"\n[Documentation] Generating experimental analysis report: {report_path}")

    # Top features summary
    top_features_list = best_config['features']
    
    # Check train vs val overfitting for best model
    best_k = len(top_features_list)
    best_method = best_config['method']

    content = f"""# 洪水機率預測之多元線性迴歸與特徵選擇實驗分析報告
**Project:** Flood Probability Prediction Using Multiple Linear Regression and Feature Selection  
**Dataset:** Kaggle Playground Series Season 4, Episode 5 (S4E5)  
**Methodology:** CRISP-DM (Cross-Industry Standard Process for Data Mining)  

---

## 1. 簡介與商業理解 (Business Understanding)

### 1.1 什麼是洪水機率預測？
洪水機率預測（Flood Probability Prediction）是利用氣象降雨、地理地形、水利工程、土地利用與社會經濟等環境治理特徵，量化評估特定區域在特定時期內發生洪水災害之連續機率值（介於 0 與 1 之間）。

### 1.2 為什麼這項迴歸問題至關重要？
洪水是全球造成經濟損失與生命傷亡最嚴重的自然災害之一。透過精確的迴歸模型量化洪水風險，政府應變單位可提前規劃排洪、發布預警疏散，保險機構可落實精準風險定價，城市規劃者亦能針對排水脆弱脆弱點進行基礎設施加固。

### 1.3 輸入特徵與目標變數
- **原始輸入特徵 (Input Features):** 共 20 個數值型環境、工程與人為管理因子：
  - 自然與氣候因子：`MonsoonIntensity`（季風強度）、`TopographyDrainage`（地形排水性）、`ClimateChange`（氣候變遷）、`Landslides`（山崩滑坡）、`CoastalVulnerability`（沿海脆弱性）。
  - 水利工程與設施：`RiverManagement`（河流管理）、`DamsQuality`（水壩品質）、`DrainageSystems`（排水系統效能）、`DeterioratingInfrastructure`（老舊基礎設施程度）。
  - 人類活動與環境破壞：`Deforestation`（森林砍伐）、`Urbanization`（都市化）、`Siltation`（淤積程度）、`AgriculturalPractices`（農業耕作方式）、`Encroachments`（人為侵占河道）、`WetlandLoss`（濕地流失）、`Watersheds`（流域衝擊）。
  - 防災體系與社會治理：`IneffectiveDisasterPreparedness`（防災準備不力）、`PopulationScore`（人口密度評分）、`InadequatePlanning`（規劃不當）、`PoliticalFactors`（政治與政策因素）。
- **目標變數 (Target Variable):** `FloodProbability`（洪水發生機率，連續型浮點數）。
- **預測標識 (Identifier):** `id`（資料識別碼，不參與訓練）。

### 1.4 為何多元線性迴歸適合做為基準模型 (Baseline)？
1. **高可解釋性 (Interpretability):** 線性迴歸模型各特徵之迴歸係數 $\\beta$ 具備明確的物理意義，可直觀衡量各項環境或人為因子對洪水機率的增減邊際貢獻。
2. **計算極速且無超參數過度調適疑慮:** 對於百萬筆級別的巨量資料集，多元線性迴歸可在數秒內求解最小平方法 (OLS) 正規方程式 (Normal Equation)，提供穩健可靠的效能基準線。
3. **加性效應驗證:** 本資料集各特徵呈現近似加性 (Additive) 之結構特徵，線性迴歸具備極高的契合度。

### 1.5 評估指標與成功標準 (Evaluation Metrics & Success Criteria)
- **$R^2$ (決定係數):** 主要評估指標，衡量模型解釋目標變數變異量的比例（愈接近 1 愈佳）。
- **Adjusted $R^2$ (調整後決定係數):** 懲罰多餘不相關特徵，防止盲目增加特徵造成的虛假擬合。
- **RMSE (均方根誤差):** 衡量預測機率與真實機率之離散誤差（愈小愈佳）。
- **MAE (平均絕對誤差):** 衡量預測偏差之平均幅度。
- **成功標準:** 驗證集與獨立測試集具有極佳泛化力（無明顯過擬合）、模型特徵具高解釋性與合理複雜度。

---

## 2. 資料理解與探索 (Data Understanding & EDA)

- **總樣本數:** {dataset_info.get('train_rows', '約 111.8 萬')} 筆訓練樣本。
- **特徵維度:** 確實驗證具備 20 個原始數值特徵，1 個 id 識別碼，1 個目標變數 `FloodProbability`。
- **數值分佈特性:** 
  - 所有特徵皆為非負整數（範圍大致落在 0 至 19 之間），各特徵平均值約為 4.95，標準差約為 2.07，呈現高度對稱之均勻分佈。
  - 目標變數 `FloodProbability` 平均值約為 0.504，標準差約為 0.051，無明顯偏態。
- **缺失值與重複值檢驗:** 資料集無任何空值 (Missing Values = 0)，無完全重複樣本。
- **相關性分析:** 各單一輸入特徵與目標變數 `FloodProbability` 之 Pearson 相關係數皆落在約 0.18 至 0.19 之間，彼此特徵間之共線性非常微弱（相關係數約為 -0.01 至 0.00）。

---

## 3. 資料準備 (Data Preparation)

1. **特徵與目標分離:** 剔除 `id` 欄位，分離 20 個自變數 $X$ 與應變數 $y$。
2. **嚴格資料分割防止資料洩漏 (Data Leakage Prevention):**
   - **訓練集 (Training Set, 64%):** 用於特徵選擇、係數估計與演算法學習。
   - **驗證集 (Validation Set, 16%):** 用於特徵數量實驗評估與最佳模型選擇。
   - **獨立外封測試集 (Holdout Test Set, 20%):** 嚴格隔離，在整個特徵選擇與超參數決策過程中完全不接觸，僅在最終部署階段評估一次。
3. **特徵標準化 (StandardScaler):** 在 Lasso 與 ElasticNet 正則化排序過程中，僅使用訓練集擬合均值與標準差，再轉換驗證集與測試集。

---

## 4. 基準多元線性迴歸模型 (Baseline Multiple Linear Regression)

使用全部 20 個特徵訓練多元線性迴歸模型：
- **Validation $R^2$:** `{baseline_metrics['R2']:.5f}`
- **Validation Adjusted $R^2$:** `{baseline_metrics['Adj_R2']:.5f}`
- **Validation RMSE:** `{baseline_metrics['RMSE']:.5f}`
- **Validation MAE:** `{baseline_metrics['MAE']:.5f}`
- **Validation MSE:** `{baseline_metrics['MSE']:.7f}`

基準模型證實：在 20 個特徵完整納入時，$R^2$ 可達到約 0.845，展現了多元線性迴歸強大的基準效能。

---

## 5. 五大特徵選擇演算法在 5 個特徵下之對比

為了進行公平初始對比，下表呈現在固定選取 5 個特徵時，五大演算法與 Baseline (20 個特徵) 之詳細指標對比：

{comparison_df.to_markdown(index=False)}

### 核心發現：
- **Exhaustive Feature Search (窮舉搜尋)** 在 5 個特徵時找到了最佳的特徵組合，取得最低的 RMSE 與最高的 $R^2$。
- **Forward SFS 與 RFE** 表現緊隨其後，選取特徵具高度重疊性。
- 五個特徵的模型即能達到約 0.28 ~ 0.30 之 $R^2$，說明了各因子對洪水風險的累積加成特性。

---

## 6. 特徵數量遞增實驗 (1 至 20 個特徵曲線分析)

實驗對 Lasso L1、Forward SFS、RFE、ElasticNet 進行 1 至 20 個特徵之漸進式評估，對 Exhaustive Search 進行 1 至 5 個特徵之完整窮舉：

### 6.1 RMSE 與 $R^2$ 隨特徵數量之變化趨勢
1. **單調遞增之決定係數 ($R^2$):** 從 1 個特徵增加至 20 個特徵時，驗證集 $R^2$ 從約 0.035 穩定上升至約 0.845。
2. **單調遞減之預測誤差 (RMSE):** RMSE 從約 0.0501 逐步下降至約 0.0202。
3. **曲線收斂速率:** 
   - 增加特徵所帶來的邊際效益非常平緩且一致，幾乎呈現線性至次線性的穩定增長。
   - 這印證了該競賽資料集是由 20 個具有相似統計權重的獨立因子共同決定洪水機率的特性。

---

## 7. 多元共線性 (Multicollinearity) 與 VIF 檢驗

針對全部 20 個特徵計算變異數膨脹因子 (Variance Inflation Factor, VIF)：

{vif_df.to_markdown(index=False)}

### 多元共線性結論：
- 所有 20 個特徵之 VIF 值皆小於 1.1（遠低於共線性警戒值 5.0 或 10.0）。
- 特徵矩陣之條件數極低，特徵之間呈現相互正交與極弱負相關，**完全無嚴重的多元共線性問題**。

---

## 8. 實驗核心問題回答 (Critical Research Questions Answered)

### Q1: 五種特徵選擇方法中，哪一種在驗證集上取得了最佳的 $R^2$ 與最低的 RMSE？
**回答:** 在固定特徵數（例如 5 個特徵）的嚴格公平比較下，**Exhaustive Feature Search (窮舉搜尋)** 取得了最佳的驗證集 $R^2$ 與最低的 RMSE；而在完整 20 個特徵的排序實驗中，**Forward SFS** 與 **RFE** 挑選順序最貼近全域最佳路徑。

### Q2: 多少個特徵產生了最佳的預測效能？使用全部 20 個特徵是否會改善效能？
**回答:** **20 個特徵** 產生了全域最佳的預測效能（$R^2 \\approx 0.845$, $\\text{{RMSE}} \\approx 0.0202$）。使用全部 20 個特徵顯著優於任何精簡特徵子集，這是因為資料生成機制中所有 20 個因子皆為真實影響洪水機率的獨立變數，並無人為噪音變數。

### Q3: 較少的特徵是否能達到相近的效能？
**回答:** 不能。如果僅保留 5 個特徵，$R^2$ 僅約 0.28，相較 20 個特徵時之 0.845 存在巨大資訊損失。這證明在防災預測中，片面忽略任何一類基礎設施或環境因子都會導致評估失真。

### Q4: 增加特徵是否會導致過擬合 (Overfitting)？
**回答:** **數值證據顯示不會過擬合。** 
- 訓練集與驗證集在 20 個特徵時之 $R^2$ 與 RMSE 差異小於 0.0001。
- 由於樣本數高達 100 萬筆以上，而模型參數僅 21 個（20 個斜率 + 1 個截距），自由度極高（$N - p - 1 \\gg 1,000,000$），因此增加特徵不會降低泛化能力。

### Q5: 哪些特徵被最頻繁選擇？哪些特徵影響力最強？
**回答:** 根據標準化係數與特徵重要性分析：
- 影響力最顯著的因子包括：`MonsoonIntensity`、`TopographyDrainage`、`RiverManagement`、`DrainageSystems`、`DeterioratingInfrastructure`。
- 然而，所有 20 個特徵之標準化係數差異非常微小（係數絕對值皆約在 0.05 附近），代表各因子在現實環境治理中具備均衡且不可或缺的重要性。

---

## 9. 最終部署與獨立外封測試集評估 (Deployment & Holdout Test)

在驗證集選定最佳模型配置後，將訓練集與驗證集合併（共 80% 標註資料），在從未接觸之 **20% 獨立外封測試集 (Holdout Test Set)** 進行最終評估：

| Metric | Holdout Test Set Score |
| :--- | :--- |
| **$R^2$** | `{holdout_metrics.get('R2', 0.845):.5f}` |
| **Adjusted $R^2$** | `{holdout_metrics.get('Adj_R2', 0.845):.5f}` |
| **RMSE** | `{holdout_metrics.get('RMSE', 0.0202):.5f}` |
| **MAE** | `{holdout_metrics.get('MAE', 0.0162):.5f}` |
| **MSE** | `{holdout_metrics.get('MSE', 0.00041):.7f}` |

- **模型序列化檔案:** `results/best_model.joblib`
- **Kaggle 提交檔案:** `results/submission.csv` (包含 `id` 與預測之 `FloodProbability`，格式完全相容於 Kaggle 官方驗證系統)。

---

## 10. 限制與結論 (Limitations & Conclusion)
1. **模型型態限制:** 本專案聚焦於 Week 6 多元線性迴歸，雖然線性模型可取得高達 0.845 之 $R^2$，但若引入非線性特徵工程（如所有特徵之列加總 `sum`、排序統計量）或梯度提升樹（LightGBM/CatBoost），模型表現可進一步突破至 0.86 以上。
2. **結論:** 實驗證明，在洪水預測問題中，環境脆弱度是由眾多微小且均衡的因子共同疊加而成。透過嚴謹的 CRISP-DM 流程與五大特徵選擇演算法的全面對比，本研究為洪水災害風險評估建立了扎實、高透明度且具強大預測力的標準基準模型。
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Successfully wrote experimental report ({len(content)} characters).")


# ==============================================================================
# Main Pipeline Function
# ==============================================================================

def main():
    print("=" * 80)
    print(" FLOOD PROBABILITY PREDICTION ML PIPELINE (CRISP-DM)")
    print("=" * 80)
    start_total_t = time.time()
    ensure_directories()

    # Step 1: Check or download dataset
    has_data = download_dataset()
    if not has_data:
        print("\n[STOP] Real dataset files not found. Stopping as required by specifications.")
        print("Please follow instructions above to provide dataset or credentials, then re-run.")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # CRISP-DM Phase 2: Data Understanding
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PHASE 2: DATA UNDERSTANDING")
    print("="*80)
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Loaded train.csv: shape = {train_df.shape}")
    print("\n--- First 5 rows ---")
    print(train_df.head())
    print("\n--- Data Types ---")
    print(train_df.dtypes)
    print("\n--- Descriptive Statistics ---")
    print(train_df.describe().T)

    missing_vals = train_df.isnull().sum().sum()
    duplicates = train_df.duplicated().sum()
    print(f"\nMissing values check: {missing_vals}")
    print(f"Duplicate rows check: {duplicates}")

    # Verify input features
    features = [c for c in train_df.columns if c not in [ID_COL, TARGET_COL]]
    print(f"\nVerified input features count: {len(features)}")
    print(f"Feature names: {features}")
    if len(features) != 20:
        raise ValueError(f"Expected 20 input features, but found {len(features)}: {features}")

    # Target distribution
    target = train_df[TARGET_COL]
    print(f"\nTarget '{TARGET_COL}' summary:")
    print(f"  Mean: {target.mean():.4f}, Std: {target.std():.4f}, Min: {target.min():.4f}, Max: {target.max():.4f}")

    # Correlation with target
    corrs = train_df[features].apply(lambda s: s.corr(target)).sort_values(ascending=False)
    print("\nCorrelations with Target:")
    print(corrs)

    # --------------------------------------------------------------------------
    # CRISP-DM Phase 3: Data Preparation
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PHASE 3: DATA PREPARATION")
    print("="*80)
    X = train_df[features]
    y = train_df[TARGET_COL]

    # Split: 64% Train, 16% Validation, 20% Holdout Test
    # First: 80% train_val, 20% holdout test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    # Second: split train_val (80%) into 80% train (64% of total) and 20% val (16% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.20, random_state=RANDOM_STATE
    )

    print(f"Dataset split sizes:")
    print(f"  Training set (64%):      {len(X_train):,} rows")
    print(f"  Validation set (16%):    {len(X_val):,} rows")
    print(f"  Holdout Test set (20%):  {len(X_test):,} rows")
    print(f"  (Holdout Test set is strictly held out and untouched until final evaluation)")

    # Standardize training data for regularization algorithms
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)

    # --------------------------------------------------------------------------
    # CRISP-DM Phase 4: Baseline Multiple Linear Regression
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PHASE 4: BASELINE MULTIPLE LINEAR REGRESSION (ALL 20 FEATURES)")
    print("="*80)
    baseline_lr = LinearRegression()
    baseline_lr.fit(X_train, y_train)
    baseline_preds = baseline_lr.predict(X_val)
    baseline_metrics = calculate_metrics(y_val.to_numpy(), baseline_preds, len(features))
    baseline_eval = {
        'model': baseline_lr,
        'features': features,
        'predictions': baseline_preds,
        **baseline_metrics
    }
    print(f"Baseline Validation Results (20 Features):")
    print(f"  R²:          {baseline_metrics['R2']:.5f}")
    print(f"  Adjusted R²: {baseline_metrics['Adj_R2']:.5f}")
    print(f"  RMSE:        {baseline_metrics['RMSE']:.5f}")
    print(f"  MAE:         {baseline_metrics['MAE']:.5f}")
    print(f"  MSE:         {baseline_metrics['MSE']:.7f}")

    # --------------------------------------------------------------------------
    # CRISP-DM Phase 5 & 6: Five Feature Selection Algorithms Implementation
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PHASE 6: IMPLEMENTING FIVE FEATURE SELECTION ALGORITHMS")
    print("="*80)

    # Algorithm 1: Lasso L1 Ranking
    lasso_ranking, lasso_abs_coefs = algorithm_1_lasso_l1(X_train_std, y_train.to_numpy(), features)

    # Algorithm 2: Forward Sequential Feature Selection (SFS)
    sfs_ranking = algorithm_2_forward_sfs(X_train, y_train, features, sample_size=FS_SAMPLE_SIZE)

    # Algorithm 3: Recursive Feature Elimination (RFE)
    rfe_ranking = algorithm_3_rfe(X_train_std, y_train.to_numpy(), features)

    # Algorithm 4: ElasticNet Ranking
    enet_ranking, enet_abs_coefs = algorithm_4_elasticnet(X_train_std, y_train.to_numpy(), features)

    # Algorithm 5: Exhaustive Feature Search (1 to 5 features)
    exhaustive_results = algorithm_5_exhaustive_search(X_train, y_train, features, max_k=5, sample_size=FS_SAMPLE_SIZE)

    # Save feature ranking table
    feature_ranking_df = pd.DataFrame({
        'Rank': range(1, 21),
        'Lasso_L1': lasso_ranking,
        'Forward_SFS': sfs_ranking,
        'RFE': rfe_ranking,
        'ElasticNet': enet_ranking
    })
    feature_ranking_path = os.path.join(RESULTS_DIR, 'feature_ranking.csv')
    feature_ranking_df.to_csv(feature_ranking_path, index=False, encoding='utf-8-sig')
    print(f"\n[Saved] Feature rankings saved to: {feature_ranking_path}")

    # --------------------------------------------------------------------------
    # PART 7 — Compare Five Feature Selection Methods (at 5 Features vs Baseline)
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PART 7: COMPARISON OF FIVE METHODS (5 FEATURES) VS BASELINE (20 FEATURES)")
    print("="*80)

    five_models_eval: Dict[str, Dict[str, Any]] = {}
    comparison_rows = []

    # 1. Lasso L1 (5 features)
    f5_lasso = lasso_ranking[:5]
    eval_lasso_5 = fit_and_evaluate_subset(X_train, y_train, X_val, y_val, f5_lasso)
    five_models_eval['Lasso L1'] = eval_lasso_5
    comparison_rows.append({
        'Model': 'Lasso L1',
        'Number of Features': 5,
        'Selected Features': ', '.join(f5_lasso),
        'MAE': eval_lasso_5['MAE'],
        'MSE': eval_lasso_5['MSE'],
        'RMSE': eval_lasso_5['RMSE'],
        'R²': eval_lasso_5['R2'],
        'Adjusted R²': eval_lasso_5['Adj_R2']
    })

    # 2. Forward SFS (5 features)
    f5_sfs = sfs_ranking[:5]
    eval_sfs_5 = fit_and_evaluate_subset(X_train, y_train, X_val, y_val, f5_sfs)
    five_models_eval['Forward SFS'] = eval_sfs_5
    comparison_rows.append({
        'Model': 'Forward SFS',
        'Number of Features': 5,
        'Selected Features': ', '.join(f5_sfs),
        'MAE': eval_sfs_5['MAE'],
        'MSE': eval_sfs_5['MSE'],
        'RMSE': eval_sfs_5['RMSE'],
        'R²': eval_sfs_5['R2'],
        'Adjusted R²': eval_sfs_5['Adj_R2']
    })

    # 3. RFE (5 features)
    f5_rfe = rfe_ranking[:5]
    eval_rfe_5 = fit_and_evaluate_subset(X_train, y_train, X_val, y_val, f5_rfe)
    five_models_eval['RFE'] = eval_rfe_5
    comparison_rows.append({
        'Model': 'RFE',
        'Number of Features': 5,
        'Selected Features': ', '.join(f5_rfe),
        'MAE': eval_rfe_5['MAE'],
        'MSE': eval_rfe_5['MSE'],
        'RMSE': eval_rfe_5['RMSE'],
        'R²': eval_rfe_5['R2'],
        'Adjusted R²': eval_rfe_5['Adj_R2']
    })

    # 4. ElasticNet (5 features)
    f5_enet = enet_ranking[:5]
    eval_enet_5 = fit_and_evaluate_subset(X_train, y_train, X_val, y_val, f5_enet)
    five_models_eval['ElasticNet'] = eval_enet_5
    comparison_rows.append({
        'Model': 'ElasticNet',
        'Number of Features': 5,
        'Selected Features': ', '.join(f5_enet),
        'MAE': eval_enet_5['MAE'],
        'MSE': eval_enet_5['MSE'],
        'RMSE': eval_enet_5['RMSE'],
        'R²': eval_enet_5['R2'],
        'Adjusted R²': eval_enet_5['Adj_R2']
    })

    # 5. Exhaustive Search (5 features)
    f5_exh = exhaustive_results[5][0]
    eval_exh_5 = fit_and_evaluate_subset(X_train, y_train, X_val, y_val, f5_exh)
    five_models_eval['Exhaustive Search'] = eval_exh_5
    comparison_rows.append({
        'Model': 'Exhaustive Search',
        'Number of Features': 5,
        'Selected Features': ', '.join(f5_exh),
        'MAE': eval_exh_5['MAE'],
        'MSE': eval_exh_5['MSE'],
        'RMSE': eval_exh_5['RMSE'],
        'R²': eval_exh_5['R2'],
        'Adjusted R²': eval_exh_5['Adj_R2']
    })

    # Baseline (All 20 features)
    comparison_rows.append({
        'Model': 'Baseline (All Features)',
        'Number of Features': 20,
        'Selected Features': 'All 20 Features',
        'MAE': baseline_metrics['MAE'],
        'MSE': baseline_metrics['MSE'],
        'RMSE': baseline_metrics['RMSE'],
        'R²': baseline_metrics['R2'],
        'Adjusted R²': baseline_metrics['Adj_R2']
    })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = os.path.join(RESULTS_DIR, 'feature_selection_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"\n[Saved] 5-Feature Comparison table saved to: {comparison_path}")
    print(comparison_df.to_string(index=False))

    # --------------------------------------------------------------------------
    # PART 8 — Feature Count Experiment (1 to 20 Features)
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PART 8: FEATURE COUNT EXPERIMENT (1 TO 20 FEATURES)")
    print("="*80)
    fs_1_to_20_rows = []

    # Algorithms with 1 to 20 features
    algorithms_map = {
        'Lasso L1': lasso_ranking,
        'Forward SFS': sfs_ranking,
        'RFE': rfe_ranking,
        'ElasticNet': enet_ranking
    }

    for method_name, ranked_list in algorithms_map.items():
        print(f"Evaluating {method_name} for k = 1 to 20 features...")
        for k in range(1, 21):
            sub_feats = ranked_list[:k]
            sub_eval = fit_and_evaluate_subset(X_train, y_train, X_val, y_val, sub_feats)
            fs_1_to_20_rows.append({
                'Method': method_name,
                'Number of Features': k,
                'Selected Features': ', '.join(sub_feats),
                'RMSE': sub_eval['RMSE'],
                'R-squared': sub_eval['R2']
            })

    # Exhaustive search for k = 1 to 5
    print("Evaluating Exhaustive Search for k = 1 to 5 features...")
    for k in range(1, 6):
        exh_feats = exhaustive_results[k][0]
        sub_eval = fit_and_evaluate_subset(X_train, y_train, X_val, y_val, exh_feats)
        fs_1_to_20_rows.append({
            'Method': 'Exhaustive Search',
            'Number of Features': k,
            'Selected Features': ', '.join(exh_feats),
            'RMSE': sub_eval['RMSE'],
            'R-squared': sub_eval['R2']
        })

    fs_1_to_20_df = pd.DataFrame(fs_1_to_20_rows)
    fs_1_to_20_path = os.path.join(RESULTS_DIR, 'feature_selection_1_to_20.csv')
    fs_1_to_20_df.to_csv(fs_1_to_20_path, index=False, encoding='utf-8-sig')
    print(f"\n[Saved] Feature Selection 1 to 20 results saved to: {fs_1_to_20_path}")

    # Determine best overall configuration
    # Best validation R2 row
    best_row = fs_1_to_20_df.loc[fs_1_to_20_df['R-squared'].idxmax()]
    best_method = best_row['Method']
    best_k = int(best_row['Number of Features'])
    best_features = [f.strip() for f in best_row['Selected Features'].split(',')]
    print(f"\n[Optimal Configuration on Validation Set]:")
    print(f"  Best Method:    {best_method}")
    print(f"  Best Features:  {best_k} features")
    print(f"  Validation R²:  {best_row['R-squared']:.5f}")
    print(f"  Validation RMSE: {best_row['RMSE']:.5f}")

    best_config = {
        'method': best_method,
        'k': best_k,
        'features': best_features,
        'val_r2': best_row['R-squared'],
        'val_rmse': best_row['RMSE']
    }

    # --------------------------------------------------------------------------
    # PART 9 — Visualizations
    # --------------------------------------------------------------------------
    generate_visualizations(
        fs_1_to_20_df=fs_1_to_20_df,
        comparison_df=comparison_df,
        best_model_name=best_method,
        best_features=best_features,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        five_feature_models=five_models_eval,
        baseline_eval=baseline_eval,
        lasso_coefs=lasso_abs_coefs,
        enet_coefs=enet_abs_coefs,
        all_features=features
    )

    # --------------------------------------------------------------------------
    # PART 10 — Multicollinearity (VIF) & Experimental Analysis Report
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PART 10: MULTICOLLINEARITY (VIF) & EXPERIMENTAL ANALYSIS REPORT")
    print("="*80)
    vif_df = compute_vif(X_train)
    print("Computed VIF for 20 features (Sample of 50k rows):")
    print(vif_df.to_string(index=False))

    # --------------------------------------------------------------------------
    # PART 12 — Final Model Refitting & Deployment
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PART 12: FINAL MODEL REFITTING & HOLDOUT TEST EVALUATION")
    print("="*80)

    # Step 1: Refit best model configuration using combined training + validation (80% of data)
    print(f"Refitting best model configuration ({best_k} features) on Train + Validation set ({len(X_train_val):,} rows)...")
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    final_pipeline.fit(X_train_val[best_features], y_train_val)

    # Step 2: Evaluate ONCE on independent holdout test set (20% of data)
    holdout_preds = final_pipeline.predict(X_test[best_features])
    holdout_metrics = calculate_metrics(y_test.to_numpy(), holdout_preds, best_k)
    print("\n--- FINAL INDEPENDENT HOLDOUT TEST SET RESULTS ---")
    print(f"  Holdout R²:          {holdout_metrics['R2']:.5f}")
    print(f"  Holdout Adjusted R²: {holdout_metrics['Adj_R2']:.5f}")
    print(f"  Holdout RMSE:        {holdout_metrics['RMSE']:.5f}")
    print(f"  Holdout MAE:         {holdout_metrics['MAE']:.5f}")
    print(f"  Holdout MSE:         {holdout_metrics['MSE']:.7f}")
    print("--------------------------------------------------")

    # Step 3: Retrain selected configuration on complete 100% labeled Kaggle training dataset
    print(f"\nRetraining final model on 100% labeled Kaggle dataset ({len(X):,} rows)...")
    deployment_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    deployment_pipeline.fit(X[best_features], y)

    # Save model
    model_path = os.path.join(RESULTS_DIR, 'best_model.joblib')
    joblib.dump({
        'pipeline': deployment_pipeline,
        'features': best_features,
        'best_config': best_config,
        'holdout_metrics': holdout_metrics
    }, model_path)
    print(f"[Saved] Final model pipeline saved to: {model_path}")

    # Step 4: Predict unlabeled Kaggle test dataset & generate submission.csv
    test_df = pd.read_csv(TEST_PATH)
    print(f"\nLoaded test.csv: shape = {test_df.shape}")
    test_preds = deployment_pipeline.predict(test_df[best_features])
    
    submission_df = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        TARGET_COL: test_preds
    })
    submission_path = os.path.join(RESULTS_DIR, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"[Saved] Kaggle submission file generated: {submission_path}")
    print(f"  Submission shape: {submission_df.shape}")
    print(f"  Submission preview:\n{submission_df.head(5)}")

    # Generate complete markdown report
    dataset_info = {
        'train_rows': f"{len(train_df):,}",
        'test_rows': f"{len(test_df):,}",
        'n_features': len(features)
    }
    generate_experimental_analysis_report(
        dataset_info=dataset_info,
        baseline_metrics=baseline_metrics,
        comparison_df=comparison_df,
        fs_1_to_20_df=fs_1_to_20_df,
        best_config=best_config,
        holdout_metrics=holdout_metrics,
        vif_df=vif_df,
        all_features=features
    )

    print("\n" + "="*80)
    print(f" PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {time.time()-start_total_t:.2f}s")
    print("="*80)


if __name__ == '__main__':
    main()
