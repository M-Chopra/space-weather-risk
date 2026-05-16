"""
=============================================================================
  SPACE WEATHER PREDICTOR — Model Training & Evaluation
=============================================================================
Trains:
  • Linear Regression       (baseline)
  • Random Forest Regressor (ensemble)
  • XGBoost Regressor       (gradient boosting)
  • LSTM                    (deep time-series, optional)

Evaluates each model with MAE, RMSE, R², and saves artefacts to /models/.
"""

import os
import time
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost                 import XGBRegressor

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MODEL_DIR  = "models"
VISUAL_DIR = "visuals"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def _build_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression" : Ridge(alpha=1.0),
        "Random Forest"    : RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_leaf=4, n_jobs=-1, random_state=42),
        "XGBoost"          : XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=42),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(X: pd.DataFrame, y: pd.Series,
                     target_name: str = "satellite_disruption_risk") -> dict:
    """
    Train all models, evaluate, save best model.

    Returns
    -------
    results : dict  {model_name: {mae, rmse, r2, train_time}}
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    logger.info("Training set: %d rows | Test set: %d rows", len(X_train), len(X_test))

    models  = _build_models()
    results = {}
    best_r2 = -np.inf
    best_name = None

    for name, model in models.items():
        logger.info("  ▸ Training %s …", name)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = model.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)

        results[name] = {
            "mae"        : round(mae,  3),
            "rmse"       : round(rmse, 3),
            "r2"         : round(r2,   4),
            "train_time" : round(elapsed, 2),
            "model"      : model,
            "y_test"     : y_test,
            "y_pred"     : y_pred,
        }

        logger.info("    MAE=%.3f  RMSE=%.3f  R²=%.4f  (%.1fs)", mae, rmse, r2, elapsed)

        if r2 > best_r2:
            best_r2  = r2
            best_name = name

    # Save best model
    best_model = results[best_name]["model"]
    save_path  = os.path.join(MODEL_DIR, f"{target_name}_best_model.pkl")
    joblib.dump(best_model, save_path)
    logger.info("Best model '%s' saved → %s", best_name, save_path)

    # Persist all models individually
    for name, data in results.items():
        slug = name.lower().replace(" ", "_")
        joblib.dump(data["model"], os.path.join(MODEL_DIR, f"{target_name}_{slug}.pkl"))

    # Generate evaluation visuals
    _plot_model_comparison(results, target_name)
    _plot_feature_importance(results, X.columns.tolist(), target_name)
    _plot_predictions(results[best_name]["y_test"],
                      results[best_name]["y_pred"], best_name, target_name)

    return results


def load_best_model(target_name: str = "satellite_disruption_risk"):
    """Load saved best model from disk."""
    path = os.path.join(MODEL_DIR, f"{target_name}_best_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at {path}. Run training first.")
    return joblib.load(path)


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_DARK = "#0d1117"
_ACCENT = "#00d4ff"
_WARN   = "#ff6b35"

def _dark_fig(figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=_DARK)
    ax.set_facecolor(_DARK)
    return fig, ax


def _plot_model_comparison(results: dict, target_name: str):
    names  = list(results.keys())
    maes   = [results[n]["mae"]  for n in names]
    rmses  = [results[n]["rmse"] for n in names]
    r2s    = [results[n]["r2"]   for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=_DARK)
    metrics = [("MAE ↓", maes, "#00d4ff"), ("RMSE ↓", rmses, "#ff6b35"), ("R² ↑", r2s, "#39ff14")]
    for ax, (title, vals, color) in zip(axes, metrics):
        ax.set_facecolor(_DARK)
        bars = ax.barh(names, vals, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", color="white", fontsize=9)
    plt.suptitle(f"Model Comparison — {target_name}", color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, f"{target_name}_model_comparison.png"),
                dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close()


def _plot_feature_importance(results: dict, feature_names: list, target_name: str):
    for model_name in ["XGBoost", "Random Forest"]:
        if model_name not in results:
            continue
        model = results[model_name]["model"]
        imp = getattr(model, "feature_importances_", None)
        if imp is None:
            continue
        top_n = 20
        indices = np.argsort(imp)[-top_n:][::-1]
        top_feats = [feature_names[i] for i in indices]
        top_vals  = imp[indices]

        fig, ax = _dark_fig(figsize=(10, 7))
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, top_n))
        ax.barh(range(top_n), top_vals[::-1], color=colors, edgecolor="none")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_feats[::-1], color="white", fontsize=9)
        ax.set_title(f"{model_name} — Top {top_n} Feature Importances\n({target_name})",
                     color="white", fontsize=12)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        slug = model_name.lower().replace(" ", "_")
        plt.savefig(os.path.join(VISUAL_DIR, f"{target_name}_{slug}_feature_importance.png"),
                    dpi=150, bbox_inches="tight", facecolor=_DARK)
        plt.close()
        break   # just the first tree-based model found


def _plot_predictions(y_test, y_pred, model_name: str, target_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_DARK)

    # Actual vs Predicted scatter
    ax = axes[0]
    ax.set_facecolor(_DARK)
    ax.scatter(y_test, y_pred, alpha=0.4, color=_ACCENT, s=12, edgecolors="none")
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color=_WARN, lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual", color="white")
    ax.set_ylabel("Predicted", color="white")
    ax.set_title(f"Actual vs Predicted — {model_name}", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Residuals
    ax = axes[1]
    ax.set_facecolor(_DARK)
    residuals = np.array(y_test) - np.array(y_pred)
    ax.hist(residuals, bins=40, color=_ACCENT, alpha=0.7, edgecolor="none")
    ax.axvline(0, color=_WARN, lw=1.5, linestyle="--")
    ax.set_xlabel("Residuals", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title("Residual Distribution", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.suptitle(target_name, color="white", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, f"{target_name}_predictions.png"),
                dpi=150, bbox_inches="tight", facecolor=_DARK)
    plt.close()
