"""
50 Startups Profit Prediction Configuration
==========================================
Configures the CRISP-DM Multiple Linear Regression and Feature Selection pipeline for 50 Startups.
"""

import os
from ml.config import DatasetConfig

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

startup_config = DatasetConfig(
    name="50 Startups Profit Prediction",
    project_id="startup",
    data_path=os.path.join(BASE_DIR, "data", "50_Startups.csv"),
    target_col="Profit",
    id_col=None,
    categorical_features=["State"],
    reference_categories={"State": "California"},
    k_max=5,
    exhaustive_k_max=5,
    sample_size_fs=None,
    cv_folds=5,
    output_dir=os.path.join(BASE_DIR, "results", "startup"),
    model_filename="startup_model.joblib",
    random_state=42,
    test_size=0.20,
    val_size=0.20,
    unit_symbol="$",
    unit_position="prefix",
    target_name_zh="企業淨利潤 (Profit)",
    description_zh="依據 R&D Spend (研發支出)、Administration (管理費用)、Marketing Spend (市場行銷費用) 及 State (所在州別) 預測 50 家新創企業之淨利潤。"
)
