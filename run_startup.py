"""
50 Startups Profit Prediction Pipeline Runner
============================================
Executes the complete CRISP-DM workflow for the 50 Startups dataset.
"""

import os
import sys

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from configs.startup_config import startup_config
from ml.workflow import run_crisp_dm_pipeline


def main():
    print("=" * 80)
    print("STARTING 50 STARTUPS PROFIT PREDICTION CRISP-DM PIPELINE")
    print("=" * 80)
    
    payload = run_crisp_dm_pipeline(startup_config)
    
    print("\n--- 50 Startups Execution Summary ---")
    print(f"Dataset:            {payload['name']}")
    print(f"Target:             {payload['target_col']}")
    print(f"Total Samples:      {payload['total_samples']}")
    print(f"Input Features (p): {len(payload['feature_names'])} features: {payload['feature_names']}")
    print(f"Best Method:        {payload['best_overall']['method']}")
    print(f"Optimal k:          {payload['best_overall']['k']}")
    print(f"Selected Features:  {payload['best_overall']['features']}")
    print(f"Holdout R²:         {payload['best_overall']['holdout_metrics']['R2']:.5f}")
    print(f"Holdout RMSE:       ${payload['best_overall']['holdout_metrics']['RMSE']:,.2f}")
    print(f"Holdout MAE:        ${payload['best_overall']['holdout_metrics']['MAE']:,.2f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
