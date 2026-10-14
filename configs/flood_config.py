"""
Flood Probability Prediction Configuration
=========================================
Configures the CRISP-DM Multiple Linear Regression and Feature Selection pipeline for Kaggle S4E5 Flood Prediction.
"""

import os
from ml.config import DatasetConfig

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

flood_config = DatasetConfig(
    name="Flood Probability Prediction",
    project_id="flood",
    data_path=os.path.join(BASE_DIR, "data", "train.csv"),
    target_col="FloodProbability",
    id_col="id",
    categorical_features=[],
    reference_categories={},
    k_max=20,
    exhaustive_k_max=5,
    sample_size_fs=100_000,
    cv_folds=3,
    output_dir=os.path.join(BASE_DIR, "results"),
    model_filename="best_model.joblib",
    random_state=42,
    test_size=0.20,
    val_size=0.20,
    unit_symbol="",
    unit_position="suffix",
    target_name_zh="洪水發生機率 (FloodProbability)",
    description_zh="基於 20 項水文氣候、防洪基礎設施、土地利用及社會治理因子預測區域洪水發生機率（Kaggle S4E5）。"
)
