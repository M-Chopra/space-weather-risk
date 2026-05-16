"""
=============================================================================
  SPACE WEATHER PREDICTOR — LSTM Time-Series Forecasting
=============================================================================
Builds a multi-step LSTM to forecast:
  • Kp index          (next 48 hours)
  • Satellite risk    (next 48 hours)

Uses Keras / TensorFlow.  Gracefully skips if TF not installed.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

LOOK_BACK  = 30   # days of history fed to LSTM
FORECAST_H = 2    # days ahead to predict (48 h)
MODEL_DIR  = "models"


# ── Try importing TensorFlow ──────────────────────────────────────────────────
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
    logger.info("TensorFlow %s detected — LSTM training enabled.", tf.__version__)
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not found — LSTM module disabled.")


def build_sequences(series: np.ndarray, look_back: int, horizon: int):
    """Create sliding-window (X, y) pairs for LSTM training."""
    X, y = [], []
    for i in range(len(series) - look_back - horizon + 1):
        X.append(series[i : i + look_back])
        y.append(series[i + look_back : i + look_back + horizon])
    return np.array(X), np.array(y)


def train_lstm(df: pd.DataFrame,
               target_col: str = "kp_index",
               epochs: int = 50,
               batch_size: int = 32) -> dict:
    """
    Train a Bidirectional LSTM forecaster.

    Returns
    -------
    dict with keys: model, history, rmse, mae
    """
    if not TF_AVAILABLE:
        logger.warning("Skipping LSTM training (TensorFlow unavailable).")
        return {}

    series = df[target_col].values.astype(np.float32)

    # Normalise
    mu, sigma = series.mean(), series.std() + 1e-8
    series_norm = (series - mu) / sigma

    X, y = build_sequences(series_norm, LOOK_BACK, FORECAST_H)
    X    = X[..., np.newaxis]          # (samples, timesteps, 1)

    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # ── Architecture ──────────────────────────────────────────────────────────
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(LOOK_BACK, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(FORECAST_H),
    ])
    model.compile(optimizer="adam", loss="huber")
    logger.info("LSTM model built — %d parameters", model.count_params())

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    y_pred_norm = model.predict(X_val, verbose=0)
    y_pred = y_pred_norm * sigma + mu
    y_true = y_val * sigma + mu

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    logger.info("LSTM  MAE=%.3f  RMSE=%.3f", mae, rmse)

    # Persist
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, f"lstm_{target_col}.keras"))
    joblib.dump({"mu": mu, "sigma": sigma}, os.path.join(MODEL_DIR, f"lstm_{target_col}_scaler.pkl"))

    return {"model": model, "history": history.history,
            "mae": mae, "rmse": rmse, "mu": mu, "sigma": sigma}


def forecast_next_48h(df: pd.DataFrame,
                       target_col: str = "kp_index") -> np.ndarray:
    """
    Load saved LSTM and predict next 48 h (2 daily steps).
    Falls back to a simple rolling-mean extrapolation if LSTM unavailable.
    """
    model_path  = os.path.join(MODEL_DIR, f"lstm_{target_col}.keras")
    scaler_path = os.path.join(MODEL_DIR, f"lstm_{target_col}_scaler.pkl")

    if TF_AVAILABLE and os.path.exists(model_path):
        model = load_model(model_path)
        sc    = joblib.load(scaler_path)
        mu, sigma = sc["mu"], sc["sigma"]

        series = df[target_col].values[-LOOK_BACK:].astype(np.float32)
        series_norm = (series - mu) / sigma
        X = series_norm[np.newaxis, :, np.newaxis]
        pred_norm = model.predict(X, verbose=0)[0]
        return pred_norm * sigma + mu

    # ── Fallback: repeat last 2-day trend ────────────────────────────────────
    recent = df[target_col].values[-14:]
    trend  = (recent[-1] - recent[0]) / len(recent)
    return np.array([recent[-1] + trend, recent[-1] + 2 * trend])
