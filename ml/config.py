"""
Dataset and Workflow Configuration Dataclasses
=============================================
Provides reusable and extensible configuration structures for CRISP-DM regression projects.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class DatasetConfig:
    """Configuration structure for a regression dataset and its CRISP-DM pipeline."""
    # Identification
    name: str                           # Display name (e.g. "50 Startups Profit Prediction")
    project_id: str                     # Unique identifier (e.g. "startup", "flood")
    data_path: str                      # Path to the primary CSV dataset
    target_col: str                     # Name of target continuous variable
    id_col: Optional[str] = None        # Identifier column (e.g. "id") to drop if present
    
    # Feature Specifications
    categorical_features: List[str] = field(default_factory=list)
    reference_categories: Dict[str, str] = field(default_factory=dict)
    
    # Experiment Constraints
    k_max: int = 5                      # Maximum features to evaluate (1 to k_max)
    exhaustive_k_max: int = 5           # Maximum k for Exhaustive Search
    sample_size_fs: Optional[int] = None  # Subsample size for expensive CV search (e.g. 100k for flood)
    cv_folds: int = 5                   # Cross-validation folds for feature ranking / search
    
    # Output and Artifact Paths
    output_dir: str = ""                # Destination folder for CSVs, PNGs, reports
    model_filename: str = "model.joblib"
    
    # Train / Val / Test Splits (Prevent Data Leakage)
    random_state: int = 42
    test_size: float = 0.20             # 20% independent holdout test
    val_size: float = 0.20              # 20% of remaining 80% (16% of total) -> 64/16/20 split
    
    # Visual & Reporting Metas
    unit_symbol: str = ""               # e.g. "$" or ""
    unit_position: str = "prefix"       # "prefix" for $100k or "suffix" for 0.85
    target_name_zh: str = ""            # e.g. "企業淨利潤 (Profit)" or "洪水機率 (FloodProbability)"
    description_zh: str = ""           # Detailed dataset overview in Traditional Chinese
