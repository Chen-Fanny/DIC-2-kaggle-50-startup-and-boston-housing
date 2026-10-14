"""
Five Feature Selection Algorithms & k=1..k_max Evaluation Suite
==============================================================
Implements:
1. Lasso L1 Ranking (LassoCV)
2. Forward Sequential Feature Selection (Forward SFS)
3. Recursive Feature Elimination (RFE)
4. ElasticNet Ranking (ElasticNetCV)
5. Exhaustive Feature Search (All combinations up to exhaustive_k_max)

All five methods use sklearn.linear_model.LinearRegression as the final prediction model.
"""

import itertools
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from ml.config import DatasetConfig
from ml.preprocessing import PreprocessingResult
from ml.evaluation import calculate_metrics, find_optimal_k


def run_feature_selection_suite(
    config: DatasetConfig,
    prep: PreprocessingResult
) -> Dict[str, Any]:
    """
    Executes all 5 feature selection algorithms and evaluates k from 1 to config.k_max.
    """
    X_train = prep.X_train
    y_train = prep.y_train
    X_val = prep.X_val
    y_val = prep.y_val
    feature_names = prep.feature_names
    total_features = len(feature_names)
    k_max = min(config.k_max, total_features)
    exhaustive_k_max = min(config.exhaustive_k_max, total_features)
    
    print(f"\n[{config.name}] Executing Feature Selection Suite (Total Features = {total_features}, k_max = {k_max})...")
    
    # --------------------------------------------------------------------------
    # 1. Algorithm 1: Lasso L1 Ranking
    # --------------------------------------------------------------------------
    print("  Running Algorithm 1: Lasso L1 Ranking (LassoCV)...")
    t0 = time.time()
    cv_folds = min(config.cv_folds, len(X_train))
    lasso_cv = LassoCV(cv=cv_folds, random_state=config.random_state, max_iter=10000)
    lasso_cv.fit(prep.X_train_scaled, y_train)
    lasso_abs_coefs = np.abs(lasso_cv.coef_)
    lasso_ranking_idx = np.argsort(lasso_abs_coefs)[::-1]
    lasso_ranked_features = [feature_names[i] for i in lasso_ranking_idx]
    print(f"    Lasso completed in {time.time()-t0:.2f}s (alpha={lasso_cv.alpha_:.6f})")
    
    # --------------------------------------------------------------------------
    # 2. Algorithm 2: Forward Sequential Feature Selection (SFS)
    # --------------------------------------------------------------------------
    print("  Running Algorithm 2: Forward Sequential Feature Selection (SFS)...")
    t0 = time.time()
    sfs_selected_features = []
    remaining_features = list(feature_names)
    
    # Subsample if large for CV greedy search
    if config.sample_size_fs and len(X_train) > config.sample_size_fs:
        idx_sample = np.random.RandomState(config.random_state).choice(len(X_train), config.sample_size_fs, replace=False)
        X_sfs_train = X_train.iloc[idx_sample]
        y_sfs_train = y_train.iloc[idx_sample].to_numpy()
    else:
        X_sfs_train = X_train
        y_sfs_train = y_train.to_numpy()
        
    sfs_kf = KFold(n_splits=min(3, len(X_sfs_train)), shuffle=True, random_state=config.random_state)
    
    for step in range(k_max):
        best_candidate = None
        best_cv_rmse = float('inf')
        
        for cand in remaining_features:
            trial_cols = sfs_selected_features + [cand]
            trial_X = X_sfs_train[trial_cols].to_numpy()
            
            fold_rmses = []
            for tr_idx, val_idx in sfs_kf.split(trial_X):
                reg = LinearRegression()
                reg.fit(trial_X[tr_idx], y_sfs_train[tr_idx])
                preds = reg.predict(trial_X[val_idx])
                fold_rmses.append(np.sqrt(np.mean((y_sfs_train[val_idx] - preds) ** 2)))
                
            mean_rmse = np.mean(fold_rmses)
            if mean_rmse < best_cv_rmse:
                best_cv_rmse = mean_rmse
                best_candidate = cand
                
        sfs_selected_features.append(best_candidate)
        remaining_features.remove(best_candidate)
        
    print(f"    Forward SFS completed in {time.time()-t0:.2f}s")
    
    # --------------------------------------------------------------------------
    # 3. Algorithm 3: Recursive Feature Elimination (RFE)
    # --------------------------------------------------------------------------
    print("  Running Algorithm 3: Recursive Feature Elimination (RFE)...")
    t0 = time.time()
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=1, step=1)
    rfe.fit(prep.X_train_scaled, y_train)
    # Sort features by rfe.ranking_ (1 = best)
    rfe_ranking_idx = np.argsort(rfe.ranking_)
    rfe_ranked_features = [feature_names[i] for i in rfe_ranking_idx]
    print(f"    RFE completed in {time.time()-t0:.2f}s")
    
    # --------------------------------------------------------------------------
    # 4. Algorithm 4: ElasticNet Ranking
    # --------------------------------------------------------------------------
    print("  Running Algorithm 4: ElasticNet Ranking (ElasticNetCV)...")
    t0 = time.time()
    enet_cv = ElasticNetCV(
        cv=cv_folds,
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        random_state=config.random_state,
        max_iter=10000
    )
    enet_cv.fit(prep.X_train_scaled, y_train)
    enet_abs_coefs = np.abs(enet_cv.coef_)
    enet_ranking_idx = np.argsort(enet_abs_coefs)[::-1]
    enet_ranked_features = [feature_names[i] for i in enet_ranking_idx]
    print(f"    ElasticNet completed in {time.time()-t0:.2f}s (alpha={enet_cv.alpha_:.6f}, l1_ratio={enet_cv.l1_ratio_:.2f})")
    
    # --------------------------------------------------------------------------
    # 5. Algorithm 5: Exhaustive Feature Search (k = 1 to exhaustive_k_max)
    # --------------------------------------------------------------------------
    print(f"  Running Algorithm 5: Exhaustive Feature Search (k = 1 to {exhaustive_k_max})...")
    t0 = time.time()
    exhaustive_best_subsets = {}
    
    for k in range(1, exhaustive_k_max + 1):
        best_combo = None
        best_cv_rmse = float('inf')
        
        # Evaluate all combinations of size k
        combos = list(itertools.combinations(feature_names, k))
        for combo in combos:
            combo_cols = list(combo)
            trial_X = X_sfs_train[combo_cols].to_numpy()
            
            fold_rmses = []
            for tr_idx, val_idx in sfs_kf.split(trial_X):
                reg = LinearRegression()
                reg.fit(trial_X[tr_idx], y_sfs_train[tr_idx])
                preds = reg.predict(trial_X[val_idx])
                fold_rmses.append(np.sqrt(np.mean((y_sfs_train[val_idx] - preds) ** 2)))
                
            mean_rmse = np.mean(fold_rmses)
            if mean_rmse < best_cv_rmse:
                best_cv_rmse = mean_rmse
                best_combo = combo_cols
                
        exhaustive_best_subsets[k] = best_combo
        
    print(f"    Exhaustive search completed in {time.time()-t0:.2f}s")
    
    # --------------------------------------------------------------------------
    # Build Feature Rankings DataFrame
    # --------------------------------------------------------------------------
    ranking_records = []
    for rank in range(1, total_features + 1):
        idx = rank - 1
        ranking_records.append({
            'Rank': rank,
            'Lasso_L1': lasso_ranked_features[idx],
            'Forward_SFS': sfs_selected_features[idx] if idx < len(sfs_selected_features) else "N/A",
            'RFE': rfe_ranked_features[idx],
            'ElasticNet': enet_ranked_features[idx]
        })
    ranking_df = pd.DataFrame(ranking_records)
    
    # --------------------------------------------------------------------------
    # Evaluate All Methods on Validation Set for k = 1 to k_max
    # --------------------------------------------------------------------------
    print("  Evaluating all methods on Holdout Validation Set across k...")
    eval_rows = []
    feature_subsets = {}
    
    methods_ranked = [
        ('Lasso L1', lasso_ranked_features),
        ('Forward SFS', sfs_selected_features),
        ('RFE', rfe_ranked_features),
        ('ElasticNet', enet_ranked_features),
    ]
    
    y_val_arr = y_val.to_numpy()
    
    for method_name, ranked_list in methods_ranked:
        for k in range(1, k_max + 1):
            chosen = ranked_list[:k]
            feature_subsets[(method_name, k)] = chosen
            
            reg = LinearRegression()
            reg.fit(X_train[chosen], y_train)
            val_preds = reg.predict(X_val[chosen])
            
            m = calculate_metrics(y_val_arr, val_preds, k)
            eval_rows.append({
                'Method': method_name,
                'Number of Features': k,
                'Selected Features': ", ".join(chosen),
                'MAE': m['MAE'],
                'MSE': m['MSE'],
                'RMSE': m['RMSE'],
                'R-squared': m['R2'],
                'Adjusted R-squared': m['Adj_R2']
            })
            
    # Exhaustive search evaluations
    for k in range(1, exhaustive_k_max + 1):
        chosen = exhaustive_best_subsets[k]
        feature_subsets[('Exhaustive Search', k)] = chosen
        
        reg = LinearRegression()
        reg.fit(X_train[chosen], y_train)
        val_preds = reg.predict(X_val[chosen])
        
        m = calculate_metrics(y_val_arr, val_preds, k)
        eval_rows.append({
            'Method': 'Exhaustive Search',
            'Number of Features': k,
            'Selected Features': ", ".join(chosen),
            'MAE': m['MAE'],
            'MSE': m['MSE'],
            'RMSE': m['RMSE'],
            'R-squared': m['R2'],
            'Adjusted R-squared': m['Adj_R2']
        })
        
    fs_curves_df = pd.DataFrame(eval_rows)
    
    # --------------------------------------------------------------------------
    # Evaluate Baseline (All Features)
    # --------------------------------------------------------------------------
    baseline_reg = LinearRegression()
    baseline_reg.fit(X_train[feature_names], y_train)
    baseline_val_preds = baseline_reg.predict(X_val[feature_names])
    baseline_metrics = calculate_metrics(y_val_arr, baseline_val_preds, total_features)
    
    # --------------------------------------------------------------------------
    # Benchmark Comparison Table (at benchmark_k)
    # --------------------------------------------------------------------------
    benchmark_k = min(5, k_max)
    comp_rows = []
    for method_name in ['Forward SFS', 'Exhaustive Search', 'RFE', 'Lasso L1', 'ElasticNet']:
        subset = fs_curves_df[(fs_curves_df['Method'] == method_name) & (fs_curves_df['Number of Features'] == benchmark_k)]
        if not subset.empty:
            row = subset.iloc[0].to_dict()
            comp_rows.append(row)
            
    # Add Baseline
    comp_rows.append({
        'Method': f'Baseline (All {total_features} Features)',
        'Number of Features': total_features,
        'Selected Features': f"All {total_features} features included",
        'MAE': baseline_metrics['MAE'],
        'MSE': baseline_metrics['MSE'],
        'RMSE': baseline_metrics['RMSE'],
        'R-squared': baseline_metrics['R2'],
        'Adjusted R-squared': baseline_metrics['Adj_R2']
    })
    comparison_df = pd.DataFrame(comp_rows)
    
    return {
        'ranking_df': ranking_df,
        'fs_curves_df': fs_curves_df,
        'comparison_df': comparison_df,
        'feature_subsets': feature_subsets,
        'baseline_metrics': baseline_metrics,
        'baseline_model': baseline_reg,
        'lasso_coefs': lasso_abs_coefs,
        'enet_coefs': enet_abs_coefs
    }
