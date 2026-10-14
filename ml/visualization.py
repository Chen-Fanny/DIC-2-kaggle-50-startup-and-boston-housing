"""
Publication-Quality Visualization Generator (300 DPI)
=====================================================
Generates 8 research-grade analytical figures for regression and feature selection analysis.
"""

import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from ml.config import DatasetConfig
from ml.preprocessing import PreprocessingResult

# Visual configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
DPI = 300

METHOD_COLORS = {
    'Lasso L1': '#e11d48',       # Rose
    'Forward SFS': '#2563eb',    # Royal Blue
    'RFE': '#16a34a',            # Green
    'ElasticNet': '#d97706',     # Amber
    'Exhaustive Search': '#9333ea', # Purple
    'Baseline': '#475569'        # Slate
}


def generate_all_visualizations(
    config: DatasetConfig,
    prep: PreprocessingResult,
    fs_results: Dict[str, Any],
    best_features: List[str],
    output_dir: str
) -> Dict[str, str]:
    """
    Generates all 8 research figures and saves them as 300-DPI PNGs into output_dir.
    Returns a dict mapping chart keys to their saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}
    
    fs_curves_df = fs_results['fs_curves_df']
    comparison_df = fs_results['comparison_df']
    X_train = prep.X_train
    y_train = prep.y_train
    X_val = prep.X_val
    y_val = prep.y_val
    
    # --------------------------------------------------------------------------
    # 1. Feature Selection Method Comparison Bar Chart
    # --------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=DPI)
    methods = comparison_df['Method'].tolist()
    rmses = comparison_df['RMSE'].tolist()
    r2s = comparison_df['R-squared'].tolist()
    
    x = np.arange(len(methods))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, rmses, width, label='Validation RMSE', color='#2563eb', alpha=0.85)
    ax1.set_ylabel('RMSE (Lower is Better)', color='#2563eb', fontweight='bold', fontsize=11)
    ax1.set_title(f'{config.name} - Feature Selection Methods Comparison', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace(' (All', '\n(All') for m in methods], rotation=15, ha='right', fontsize=9)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, r2s, width, label='Validation R²', color='#10b981', alpha=0.85)
    ax2.set_ylabel('R² Score (Higher is Better)', color='#10b981', fontweight='bold', fontsize=11)
    ax2.grid(False)
    
    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#1e3a8a')
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#065f46')
        
    fig.tight_layout()
    p1 = os.path.join(output_dir, 'feature_selection_comparison.png')
    fig.savefig(p1, bbox_inches='tight')
    plt.close(fig)
    saved_paths['feature_selection_comparison'] = p1
    
    # --------------------------------------------------------------------------
    # 2. RMSE vs Number of Features
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)
    methods_list = ['Lasso L1', 'Forward SFS', 'RFE', 'ElasticNet', 'Exhaustive Search']
    
    for method in methods_list:
        sub = fs_curves_df[fs_curves_df['Method'] == method]
        if not sub.empty:
            color = METHOD_COLORS.get(method, '#333333')
            ax.plot(sub['Number of Features'], sub['RMSE'], marker='o', markersize=4,
                    linewidth=2, label=method, color=color)
            # Mark optimal
            opt_idx = sub['RMSE'].idxmin()
            opt_row = sub.loc[opt_idx]
            ax.scatter([opt_row['Number of Features']], [opt_row['RMSE']],
                       color=color, s=120, marker='*', zorder=5, edgecolor='black')
            
    ax.set_xlabel('Number of Selected Features (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Validation RMSE (Lower is Better)', fontsize=11, fontweight='bold')
    ax.set_title(f'{config.name} - Validation RMSE vs Number of Features', fontsize=13, fontweight='bold')
    ax.legend(title='Algorithm (★ = Min RMSE)', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    p2 = os.path.join(output_dir, 'rmse_vs_features.png')
    fig.savefig(p2, bbox_inches='tight')
    plt.close(fig)
    saved_paths['rmse_vs_features'] = p2
    
    # --------------------------------------------------------------------------
    # 3. R2 vs Number of Features
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)
    for method in methods_list:
        sub = fs_curves_df[fs_curves_df['Method'] == method]
        if not sub.empty:
            color = METHOD_COLORS.get(method, '#333333')
            ax.plot(sub['Number of Features'], sub['R-squared'], marker='s', markersize=4,
                    linewidth=2, label=method, color=color)
            
    ax.set_xlabel('Number of Selected Features (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Validation R² Score (Higher is Better)', fontsize=11, fontweight='bold')
    ax.set_title(f'{config.name} - Validation R² Score vs Number of Features', fontsize=13, fontweight='bold')
    ax.legend(title='Algorithm', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    p3 = os.path.join(output_dir, 'r2_vs_features.png')
    fig.savefig(p3, bbox_inches='tight')
    plt.close(fig)
    saved_paths['r2_vs_features'] = p3
    
    # --------------------------------------------------------------------------
    # 4. Dual Subplots: All 5 Algorithms Comparison (RMSE & R2)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=DPI)
    for method in methods_list:
        sub = fs_curves_df[fs_curves_df['Method'] == method]
        if not sub.empty:
            color = METHOD_COLORS.get(method, '#333333')
            ax1.plot(sub['Number of Features'], sub['RMSE'], marker='o', markersize=3,
                     linewidth=1.8, label=method, color=color)
            ax2.plot(sub['Number of Features'], sub['R-squared'], marker='s', markersize=3,
                     linewidth=1.8, label=method, color=color)
            
    ax1.set_xlabel('Number of Features (k)', fontweight='bold')
    ax1.set_ylabel('Validation RMSE', fontweight='bold')
    ax1.set_title('(A) Validation RMSE Progression', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.set_xlabel('Number of Features (k)', fontweight='bold')
    ax2.set_ylabel('Validation R²', fontweight='bold')
    ax2.set_title('(B) Validation R² Score Progression', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    fig.suptitle(f'{config.name} - Comprehensive Feature Selection Dynamics', fontsize=14, fontweight='bold')
    fig.tight_layout()
    p4 = os.path.join(output_dir, 'all_5_algorithms_comparison.png')
    fig.savefig(p4, bbox_inches='tight')
    plt.close(fig)
    saved_paths['all_5_algorithms_comparison'] = p4
    
    # --------------------------------------------------------------------------
    # Train Best Model for Diagnostic Visualizations
    # --------------------------------------------------------------------------
    best_reg = LinearRegression()
    best_reg.fit(X_train[best_features], y_train)
    val_preds = best_reg.predict(X_val[best_features])
    residuals = y_val.to_numpy() - val_preds
    
    # --------------------------------------------------------------------------
    # 5. Actual vs Predicted Scatter Plot (with y = x line)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=DPI)
    ax.scatter(y_val, val_preds, alpha=0.7, color='#2563eb', edgecolors='white', s=50, label='Validation Samples')
    
    min_val = min(y_val.min(), val_preds.min())
    max_val = max(y_val.max(), val_preds.max())
    pad = (max_val - min_val) * 0.05
    line_range = np.linspace(min_val - pad, max_val + pad, 100)
    ax.plot(line_range, line_range, 'r--', linewidth=2, label='Ideal Prediction (y = x)')
    
    ax.set_xlabel(f'Actual {config.target_col}', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Predicted {config.target_col}', fontsize=11, fontweight='bold')
    ax.set_title(f'{config.name} - Actual vs Predicted (Best Model, k={len(best_features)})', fontsize=12, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    p5 = os.path.join(output_dir, 'actual_vs_predicted.png')
    fig.savefig(p5, bbox_inches='tight')
    plt.close(fig)
    saved_paths['actual_vs_predicted'] = p5
    
    # --------------------------------------------------------------------------
    # 6. Model Predictions Comparison (Sorted Samples Trajectory)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=DPI)
    sort_idx = np.argsort(y_val.to_numpy())
    sorted_actual = y_val.to_numpy()[sort_idx]
    sorted_preds = val_preds[sort_idx]
    
    sample_indices = np.arange(len(sorted_actual))
    ax.plot(sample_indices, sorted_actual, 'k-', linewidth=2, label='Actual Values (Sorted)')
    ax.plot(sample_indices, sorted_preds, 'o-', color='#2563eb', markersize=4, linewidth=1.5,
            alpha=0.8, label=f'Best Model Prediction (k={len(best_features)})')
    
    ax.set_xlabel('Sample Index (Sorted by Target Value)', fontsize=11, fontweight='bold')
    ax.set_ylabel(config.target_col, fontsize=11, fontweight='bold')
    ax.set_title(f'{config.name} - Sorted Target vs Predicted Trajectory', fontsize=12, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    p6 = os.path.join(output_dir, 'model_predictions_comparison.png')
    fig.savefig(p6, bbox_inches='tight')
    plt.close(fig)
    saved_paths['model_predictions_comparison'] = p6
    
    # --------------------------------------------------------------------------
    # 7. Feature Importance (Standardized Coefficients)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, max(4, len(prep.feature_names) * 0.45)), dpi=DPI)
    # Fit OLS on standardized full training features
    full_std_reg = LinearRegression()
    full_std_reg.fit(prep.X_train_scaled, y_train)
    coefs = full_std_reg.coef_
    
    imp_df = pd.DataFrame({
        'Feature': prep.feature_names,
        'Importance': np.abs(coefs),
        'Signed_Coef': coefs
    }).sort_values(by='Importance', ascending=True)
    
    colors = ['#10b981' if c >= 0 else '#ef4444' for c in imp_df['Signed_Coef']]
    y_pos = np.arange(len(imp_df))
    ax.barh(y_pos, imp_df['Importance'], color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_df['Feature'], fontsize=10)
    ax.set_xlabel('Standardized Coefficient Magnitude (|β|)', fontsize=11, fontweight='bold')
    ax.set_title(f'{config.name} - Standardized Feature Importance (All Features)', fontsize=12, fontweight='bold')
    
    # Legend for sign
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#10b981', label='Positive Impact (+β)'),
        Patch(facecolor='#ef4444', label='Negative Impact (-β)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    p7 = os.path.join(output_dir, 'feature_importance.png')
    fig.savefig(p7, bbox_inches='tight')
    plt.close(fig)
    saved_paths['feature_importance'] = p7
    
    # --------------------------------------------------------------------------
    # 8. Residual Analysis (Residuals vs Fitted & Distribution)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    
    # Subplot A: Residuals vs Fitted
    ax1.scatter(val_preds, residuals, alpha=0.7, color='#6366f1', edgecolors='white', s=50)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Fitted Values (Predicted)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Residuals (Actual - Predicted)', fontsize=10, fontweight='bold')
    ax1.set_title('(A) Residuals vs Fitted Values', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Subplot B: Residual Distribution
    sns.histplot(residuals, kde=True, ax=ax2, color='#2563eb', stat='density', bins=min(20, max(5, len(residuals)//2)))
    ax2.set_xlabel('Residual Value', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax2.set_title('(B) Residual Normality Distribution', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    fig.suptitle(f'{config.name} - Model Residual Diagnostics', fontsize=13, fontweight='bold')
    fig.tight_layout()
    p8 = os.path.join(output_dir, 'residual_analysis.png')
    fig.savefig(p8, bbox_inches='tight')
    plt.close(fig)
    saved_paths['residual_analysis'] = p8
    
    print(f"[{config.name}] Successfully generated 8 publication-quality visualizations into {output_dir}")
    return saved_paths
