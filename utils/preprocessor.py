"""
=============================================================================
  SPACE WEATHER PREDICTOR — Preprocessing & Feature Engineering
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import logging

logger = logging.getLogger(__name__)


# ── Categorical maps ──────────────────────────────────────────────────────────
FLARE_CLASS_MAP  = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
SEVERITY_MAP     = {"Low": 0, "Moderate": 1, "High": 2, "Extreme": 3}


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, dict]:
    """
    Full preprocessing pipeline.

    Returns
    -------
    df_clean   : cleaned & feature-engineered DataFrame
    scaler     : fitted StandardScaler (for inverse transform later)
    encoders   : dict of label encoders / maps used
    """
    df = df.copy()

    # 1. Parse timestamps ────────────────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 2. Handle missing values (KNN for numerical cols) ─────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputer  = KNNImputer(n_neighbors=5)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    logger.info("Missing values imputed via KNN")

    # 3. Encode categorical features ─────────────────────────────────────────
    df["flare_class_enc"] = df["flare_class"].map(FLARE_CLASS_MAP).fillna(0).astype(int)
    df["severity_enc"]    = df["severity"].map(SEVERITY_MAP).fillna(0).astype(int)

    # 4. Time-based features ─────────────────────────────────────────────────
    df["day_of_year"]   = df["timestamp"].dt.dayofyear
    df["month"]         = df["timestamp"].dt.month
    df["year"]          = df["timestamp"].dt.year
    df["day_sin"]       = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"]       = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # 5. Lag features (yesterday / 3-day / 7-day rolling) ───────────────────
    for col in ["kp_index", "solar_wind_speed", "imf_bz", "sunspot_number"]:
        df[f"{col}_lag1"]    = df[col].shift(1)
        df[f"{col}_lag3"]    = df[col].shift(3)
        df[f"{col}_roll7"]   = df[col].rolling(7,  min_periods=1).mean()
        df[f"{col}_roll30"]  = df[col].rolling(30, min_periods=1).mean()
        df[f"{col}_std7"]    = df[col].rolling(7,  min_periods=1).std().fillna(0)

    # 6. Interaction features ─────────────────────────────────────────────────
    df["wind_bz_interaction"]   = df["solar_wind_speed"] * (-df["imf_bz"].clip(upper=0))
    df["storm_energy"]          = df["solar_wind_speed"] ** 2 * df["solar_wind_density"]
    df["flare_kp_interaction"]  = df["flare_class_enc"] * df["kp_index"]
    df["southward_bz"]          = (-df["imf_bz"]).clip(lower=0)

    # 7. Log-transform skewed columns ─────────────────────────────────────────
    for col in ["xray_flux", "proton_flux"]:
        df[f"log_{col}"] = np.log1p(df[col])

    # 8. Outlier detection (IQR clipping) ─────────────────────────────────────
    clip_cols = ["solar_wind_speed", "proton_flux", "xray_flux"]
    for col in clip_cols:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(q1, q3)

    # 9. Drop rows with NaNs introduced by lag ────────────────────────────────
    df = df.dropna().reset_index(drop=True)

    # 10. Scale numerical features ────────────────────────────────────────────
    feature_cols = _get_feature_cols(df)
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    encoders = {
        "flare_class_map" : FLARE_CLASS_MAP,
        "severity_map"    : SEVERITY_MAP,
        "feature_cols"    : feature_cols,
    }

    logger.info("Preprocessing complete — %d features, %d rows", len(feature_cols), len(df_scaled))
    return df_scaled, scaler, encoders


def _get_feature_cols(df: pd.DataFrame) -> list:
    """Return all numeric feature columns (exclude targets & identifiers)."""
    exclude = {
        "timestamp", "flare_class", "severity",
        "satellite_disruption_risk", "gps_blackout_prob",
        "storm_flag", "severity_enc", "year",
    }
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def get_X_y(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split features / target for a given target column."""
    feature_cols = _get_feature_cols(df)
    # Remove the target itself from features if it slipped in
    feature_cols = [c for c in feature_cols if c != target]
    X = df[feature_cols]
    y = df[target]
    return X, y
