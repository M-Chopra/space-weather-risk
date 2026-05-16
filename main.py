"""
=============================================================================
  SPACE WEATHER PREDICTOR — Main Training Pipeline
  Run: python main.py
=============================================================================

Steps
-----
1. Load / generate dataset
2. Preprocess + feature engineering
3. Train all ML models on three targets
4. Generate all static visualisations
5. Print performance summary table
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Ensure project root is on path ────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from utils.data_collector  import load_or_generate_dataset
from utils.preprocessor    import preprocess, get_X_y
from models.trainer        import train_all_models
from utils.visualizer      import generate_all_visuals

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Prediction targets
TARGETS = {
    "satellite_disruption_risk": "Satellite Disruption Risk (0-100)",
    "gps_blackout_prob"        : "GPS Blackout Probability (0-100)",
    "kp_index"                 : "Kp Geomagnetic Index (0-9)",
}


def main():
    banner = r"""
  ╔══════════════════════════════════════════════════════════════╗
  ║   🛰  SPACE WEATHER PREDICTION & SATELLITE RISK ANALYSIS    ║
  ║           ML Pipeline  |  v1.0  |  Scikit-learn + XGBoost   ║
  ╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    logger.info("Step 1/4 — Loading dataset …")
    raw_df = load_or_generate_dataset(n_years=5, save_path="data/space_weather.csv")
    logger.info("Dataset shape: %s", raw_df.shape)
    logger.info("Date range  : %s → %s",
                raw_df["timestamp"].min().date(),
                raw_df["timestamp"].max().date())

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    logger.info("Step 2/4 — Preprocessing …")
    df_proc, scaler, encoders = preprocess(raw_df)
    logger.info("Processed shape: %s", df_proc.shape)

    # ── 3. Train models ───────────────────────────────────────────────────────
    logger.info("Step 3/4 — Training models …")
    all_results = {}
    for target, label in TARGETS.items():
        print(f"\n{'─'*60}")
        print(f"  Target: {label}")
        print(f"{'─'*60}")
        X, y = get_X_y(df_proc, target)
        results = train_all_models(X, y, target_name=target)
        all_results[target] = results

    # ── 4. Visualisations ─────────────────────────────────────────────────────
    logger.info("Step 4/4 — Generating visualisations …")
    generate_all_visuals(raw_df)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "═" * 75)
    print("  MODEL PERFORMANCE SUMMARY")
    print("═" * 75)
    print(f"  {'Target':<35} {'Model':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
    print("  " + "─" * 70)

    for target, results in all_results.items():
        best = max(results.items(), key=lambda kv: kv[1]["r2"])
        bname, bscores = best
        print(f"  {target[:35]:<35} {bname:<22} "
              f"{bscores['mae']:>7.3f} {bscores['rmse']:>7.3f} {bscores['r2']:>7.4f}")

    print("═" * 75)
    print("\n  ✅  Training complete!")
    print("  📊  Visuals saved to  /visuals/")
    print("  🤖  Models saved to   /models/")
    print("\n  ➤  Launch dashboard:  streamlit run app/dashboard.py\n")


if __name__ == "__main__":
    main()
