"""
=============================================================================
  SPACE WEATHER PREDICTOR — Data Collection Module
  Author : Space-Weather ML Project
  Purpose: Fetch / generate realistic space weather datasets
=============================================================================

Data sources used (when network is available):
  • NOAA SWPC  : https://services.swpc.noaa.gov/json/
  • NASA DONKI : https://api.nasa.gov/DONKI/
  • SILSO      : https://www.sidc.be/silso/datafiles (sunspot numbers)

When network is unavailable the module falls back to a high-fidelity
synthetic dataset that mirrors real solar-cycle statistics.
"""

import os
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
NOAA_SOLAR_WIND  = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
NOAA_MAG         = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
NOAA_KP          = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
NASA_DONKI_CME   = "https://api.nasa.gov/DONKI/CME"
SEED             = 42

np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def load_or_generate_dataset(n_years: int = 5, save_path: str = "data/space_weather.csv") -> pd.DataFrame:
    """
    Primary entry-point.
    Returns a DataFrame covering *n_years* of space-weather observations.
    If *save_path* already exists the file is loaded directly.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        logger.info("Loading cached dataset from %s", save_path)
        df = pd.read_csv(save_path, parse_dates=["timestamp"])
        return df

    logger.info("Generating synthetic space-weather dataset (%d years)…", n_years)
    df = _generate_synthetic_dataset(n_years=n_years)
    df.to_csv(save_path, index=False)
    logger.info("Dataset saved → %s  (%d rows)", save_path, len(df))
    return df


def fetch_live_noaa_data() -> Optional[pd.DataFrame]:
    """
    Try to pull the latest real-time solar-wind & Kp data from NOAA SWPC.
    Returns None on any network / parse error.
    """
    try:
        kp_resp = requests.get(NOAA_KP, timeout=10)
        kp_resp.raise_for_status()
        kp_json = kp_resp.json()

        rows = []
        for entry in kp_json[-60:]:          # last 60 minutes
            rows.append({
                "timestamp": pd.to_datetime(entry.get("time_tag")),
                "kp_index" : float(entry.get("kp", 0)),
            })

        df_live = pd.DataFrame(rows).dropna()
        logger.info("Live NOAA Kp data fetched — %d rows", len(df_live))
        return df_live

    except Exception as exc:
        logger.warning("Live NOAA fetch failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATASET GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_dataset(n_years: int = 5) -> pd.DataFrame:
    """
    Build a realistic synthetic space-weather time series.

    Solar activity follows an ~11-year Schwabe cycle.  We simulate:
      • Sunspot number  (SSN)
      • Solar flux F10.7
      • X-ray flux (goes A→X scale mapped to float)
      • Proton flux (>10 MeV)
      • Geomagnetic Kp / Dst indices
      • Solar wind speed & density & Bz component
      • Derived risk scores for ML targets
    """
    n_days    = n_years * 365
    dates     = pd.date_range(end=datetime.utcnow(), periods=n_days, freq="D")

    # ── Solar cycle phase (11-year period, peak at ~5.5 years from start) ──
    cycle_phase = np.linspace(0, 2 * np.pi * n_years / 11, n_days)
    solar_cycle = (np.sin(cycle_phase - np.pi / 2) + 1) / 2   # 0→1

    # ── Base physical quantities ───────────────────────────────────────────
    ssn          = _bounded(70 * solar_cycle + 30 * np.random.beta(2, 5, n_days) + np.random.normal(0, 8, n_days), 0, 250)
    f107         = _bounded(70 + 130 * solar_cycle + np.random.normal(0, 10, n_days), 68, 300)
    solar_wind_v = _bounded(400 + 150 * solar_cycle + np.random.normal(0, 40, n_days), 280, 900)
    solar_wind_d = _bounded(6 - 2 * solar_cycle + np.random.normal(0, 1.5, n_days), 1, 30)

    # Bz (southward IMF drives storms)
    bz           = np.random.normal(0, 5, n_days) - 8 * solar_cycle * np.random.beta(1, 4, n_days)

    # X-ray flux (log-normal)
    xray_base    = 1e-8 * (1 + 100 * solar_cycle)
    xray_flux    = np.abs(np.random.lognormal(np.log(xray_base), 1.5, n_days))

    # Proton flux (> 10 MeV pfu)
    proton_flux  = np.abs(np.random.lognormal(np.log(1 + 50 * solar_cycle), 2, n_days))

    # ── Derived geomagnetic indices ────────────────────────────────────────
    # Kp increases with fast solar wind & southward Bz
    kp_raw  = (0.01 * solar_wind_v + (-bz).clip(0) * 0.3 +
               3 * solar_cycle + np.random.exponential(0.5, n_days))
    kp      = _bounded(kp_raw, 0, 9)

    # Dst (nT) — negative during storms
    dst     = _bounded(-20 - 25 * kp / 9 * solar_cycle * 4 +
                       np.random.normal(0, 10, n_days), -400, 50)

    # ── Flare classification ───────────────────────────────────────────────
    flare_intensity = np.log10(xray_flux + 1e-9) + 9   # re-scale to 0-8
    flare_class     = _classify_flare(flare_intensity)

    # ── Risk scores (0-100) ───────────────────────────────────────────────
    sat_disruption  = _bounded(
        10 * (kp / 9) + 5 * solar_cycle + 0.05 * proton_flux.clip(0, 1000) +
        np.random.normal(0, 5, n_days), 0, 100)

    gps_blackout    = _bounded(
        15 * (kp / 9) + 10 * solar_cycle + 0.1 * flare_intensity +
        np.random.normal(0, 4, n_days), 0, 100)

    # ── Severity label ─────────────────────────────────────────────────────
    severity = _severity_label(kp)

    # ── Storm event flag ──────────────────────────────────────────────────
    storm_flag = (kp >= 5).astype(int)

    # ── CME arrival flag ──────────────────────────────────────────────────
    cme_flag = np.zeros(n_days, dtype=int)
    # Randomly scatter CMEs correlated with solar maximum
    cme_probs = 0.01 + 0.05 * solar_cycle
    cme_mask  = np.random.random(n_days) < cme_probs
    cme_flag[cme_mask] = 1

    df = pd.DataFrame({
        "timestamp"           : dates,
        "sunspot_number"      : ssn.round(1),
        "f107_solar_flux"     : f107.round(2),
        "solar_wind_speed"    : solar_wind_v.round(1),
        "solar_wind_density"  : solar_wind_d.round(2),
        "imf_bz"              : bz.round(2),
        "xray_flux"           : xray_flux,
        "proton_flux"         : proton_flux.round(2),
        "kp_index"            : kp.round(1),
        "dst_index"           : dst.round(1),
        "flare_class"         : flare_class,
        "flare_intensity"     : flare_intensity.round(3),
        "satellite_disruption_risk" : sat_disruption.round(2),
        "gps_blackout_prob"   : gps_blackout.round(2),
        "storm_flag"          : storm_flag,
        "cme_flag"            : cme_flag,
        "severity"            : severity,
        "solar_cycle_phase"   : solar_cycle.round(4),
    })

    # ── Add some realistic NaNs (~2 %) to simulate sensor outages ─────────
    for col in ["solar_wind_speed", "imf_bz", "xray_flux", "proton_flux", "dst_index"]:
        mask = np.random.random(n_days) < 0.02
        df.loc[mask, col] = np.nan

    return df


# ── Helper utilities ──────────────────────────────────────────────────────────

def _bounded(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def _classify_flare(intensity: np.ndarray) -> list:
    classes = []
    for v in intensity:
        if   v < 1: classes.append("A")
        elif v < 2: classes.append("B")
        elif v < 3: classes.append("C")
        elif v < 4: classes.append("M")
        else:       classes.append("X")
    return classes


def _severity_label(kp: np.ndarray) -> list:
    labels = []
    for k in kp:
        if   k < 3: labels.append("Low")
        elif k < 5: labels.append("Moderate")
        elif k < 7: labels.append("High")
        else:       labels.append("Extreme")
    return labels
